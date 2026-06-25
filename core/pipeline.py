"""
KGCS - Full Detection Pipeline Orchestrator

Connects SDM → SAM (mask generation) → OPM → ISM → Output

Reference: KGCS_R2.pdf Fig.2 (Workflow of KGCS)
"""

import os
import shutil
import cv2
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from config.settings import (
    SAM_MODEL_TYPE, SAM_CHECKPOINT, OUTPUT_BASE, TEMP_DIRS,
    OPM as OPM_CFG, TARGET_DESCRIPTIONS, DISTRACTOR_DESCRIPTIONS,
    CONTOUR_DESCRIPTIONS, FIXED_BOUNDARY_CATEGORIES
)
from core.sdm import SceneDescriptionModule
from core.opm import ObjectProposalModule, mask_to_hbbs, non_max_suppression
from core.ism import ImageTextSimilarityModule, classify_masks_with_clip


# ---------------------------------------------------------------------------
# SAM Mask Generation Wrapper
# ---------------------------------------------------------------------------

def _init_sam(model_type: str = SAM_MODEL_TYPE,
              checkpoint_path: Optional[str] = None):
    """Load SAM model for automatic mask generation."""
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    if checkpoint_path is None:
        checkpoint_path = SAM_CHECKPOINT
    ckpt = checkpoint_path

    print(f"[SAM] Loading {model_type} from {ckpt} ...")
    sam = sam_model_registry[model_type](checkpoint=ckpt)
    import torch
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")

    generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=OPM_CFG.get("pred_iou_thresh", 0.86),
        stability_score_thresh=OPM_CFG.get("stability_score_thresh", 0.92),
        crop_n_layers=OPM_CFG.get("crop_n_layers", 1),
        crop_n_points_downscale_factor=2,
        min_mask_region_area=OPM_CFG.get("min_mask_region_area", 100),
    )
    return generator


def write_masks_to_folder(masks, path):
    """Save SAM masks to folder (inlined from segment_anything.utils.amg)."""
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"
    metadata = [header]
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        cv2.imwrite(os.path.join(path, f"{i}.png"), mask * 255)
        meta = [
            str(i), str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        metadata.append(",".join(meta))
    with open(os.path.join(path, "metadata.csv"), "w") as f:
        f.write("\n".join(metadata))


def generate_sam_masks(generator, image_path: str,
                       image_id: str, output_dir: str = "images_masks"):
    """
    Generate SAM masks for one image and save to output_dir / {image_id}/.
    Returns the path to the mask folder.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    masks = generator.generate(image_rgb)

    mask_dir = os.path.join(output_dir, image_id)
    if os.path.exists(mask_dir):
        shutil.rmtree(mask_dir)
    os.makedirs(mask_dir, exist_ok=True)

    write_masks_to_folder(masks, mask_dir)
    return mask_dir


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class KGCS_Pipeline:
    """
    End-to-end KGCS detection pipeline.

    Steps:
      1. Image preparation (format conversion, resize)
      2. SAM mask generation
      3. SDM dictionary construction
      4. Mask → crop → clip_remove (patch-level classification)
      5. OPM proposal generation
      6. ISM two-stage screening (gradient screening)
      7. Bbox output & evaluation
    """

    def __init__(self,
                 target_category: str = "ship",
                 sam_mode: str = SAM_MODEL_TYPE,
                 output_base: str = OUTPUT_BASE,
                 use_expert_descriptions: bool = True,
                 llm_api_key: Optional[str] = None,
                 llm_api_url: Optional[str] = None):
        self.category = target_category
        self.sam_mode = sam_mode
        self.output_base = output_base
        self.llm_api_key = llm_api_key
        self.llm_api_url = llm_api_url

        # Determine boundary type
        self.is_boundary_clear = target_category in FIXED_BOUNDARY_CATEGORIES

        # Initialize modules
        self.sam_generator = _init_sam(sam_mode)

        # SDM: load expert descriptions for this category
        target_dict = {}
        if use_expert_descriptions and target_category in TARGET_DESCRIPTIONS:
            target_dict = {target_category: TARGET_DESCRIPTIONS[target_category]}
        else:
            target_dict = {target_category: f"A photo of a {target_category}"}

        self.sdm = SceneDescriptionModule(
            target_descriptions=target_dict,
            distractor_descriptions=DISTRACTOR_DESCRIPTIONS,
            contour_descriptions=CONTOUR_DESCRIPTIONS,
            llm_api_key=llm_api_key,
            llm_api_url=llm_api_url,
        )

        # OPM
        self.opm = ObjectProposalModule()

        # ISM
        self.ism = ImageTextSimilarityModule()

        # Tracking
        self.timings = {}

    # ---- Clean working directories ----------------------------------------

    def _clean_workdirs(self):
        for d in TEMP_DIRS:
            if os.path.exists(d):
                shutil.rmtree(d)
        os.makedirs("images", exist_ok=True)
        os.makedirs("images_masks", exist_ok=True)
        os.makedirs("seg_img", exist_ok=True)
        os.makedirs("label", exist_ok=True)

    # ---- Image preparation ------------------------------------------------

    def _prepare_image(self, src_path: str) -> Tuple[str, str]:
        """
        Copy image to workspace, convert to PNG if needed.
        Returns (workspace_path, image_id).
        """
        fname = os.path.basename(src_path)
        name, ext = os.path.splitext(fname)
        target_name = name + ".png"
        target_path = os.path.join("images", target_name)

        # Copy and convert
        if ext.lower() in (".jpg", ".jpeg", ".tif", ".tiff"):
            from PIL import Image
            img = Image.open(src_path)
            img.save(target_path, "PNG")
        else:
            shutil.copy2(src_path, target_path)

        # Also keep a copy in images_orgin
        os.makedirs("images_orgin", exist_ok=True)
        shutil.copy2(target_path, os.path.join("images_orgin", target_name))

        return target_path, name

    # ---- Convert JPG to PNG in a folder -----------------------------------

    @staticmethod
    def _convert_to_png(folder: str, resize: Optional[Tuple[int, int]] = None):
        from PIL import Image
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            name, ext = os.path.splitext(fname)
            if ext.lower() in (".jpg", ".jpeg", ".tif", ".tiff"):
                img = Image.open(fpath)
                if resize:
                    img = img.resize(resize)
                img.save(os.path.join(folder, name + ".png"), "PNG")
                os.remove(fpath)

    # ---- Mask-to-crop (seg_img generation) --------------------------------

    def _masks_to_crops(self, image_path: str, image_id: str):
        """Generate cropped object images from each mask, skipping bad crops."""
        from PIL import Image
        import numpy as np

        img_pil = Image.open(os.path.join("images_orgin", image_id + ".png")).convert("RGB")
        img_np = np.array(img_pil)
        mask_dir = os.path.join("images_masks", image_id)
        seg_dir = "seg_img"
        seg_back_dir = "seg_img_back"
        os.makedirs(seg_dir, exist_ok=True)
        os.makedirs(seg_back_dir, exist_ok=True)

        for fname in sorted(os.listdir(mask_dir)):
            if fname.endswith(("csv", "txt")):
                continue
            mask_path = os.path.join(mask_dir, fname)
            try:
                mask_pil = Image.open(mask_path).convert("L")
                mask_np = np.array(mask_pil) / 255
                coords = np.column_stack(np.where(mask_np == 1))
                if len(coords) == 0:
                    continue
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                h, w = y_max - y_min, x_max - x_min
                if h < 2 or w < 2:
                    continue

                # foreground crop (with padding)
                pad_y, pad_x = max(int(0.05 * h), 1), max(int(0.05 * w), 1)
                y1 = max(0, y_min - pad_y)
                y2 = min(img_np.shape[0], y_max + 1 + pad_y)
                x1 = max(0, x_min - pad_x)
                x2 = min(img_np.shape[1], x_max + 1 + pad_x)
                if y2 <= y1 or x2 <= x1:
                    continue
                crop_fg = Image.fromarray(img_np[y1:y2, x1:x2].astype(np.uint8)).resize((56, 56))
                crop_fg.save(os.path.join(seg_dir, fname))

                # background crop (larger context)
                y1b = max(0, y_min - h)
                y2b = min(img_np.shape[0], y_max + 1 + h)
                x1b = max(0, x_min - w)
                x2b = min(img_np.shape[1], x_max + 1 + w)
                if y2b > y1b and x2b > x1b:
                    crop_bg = Image.fromarray(img_np[y1b:y2b, x1b:x2b].astype(np.uint8)).resize((14, 14))
                    crop_bg.save(os.path.join(seg_back_dir, fname))
            except Exception:
                continue  # skip problematic masks

    # ---- Mask-to-txt (YOLO-format label generation) -----------------------

    def _mask_to_txt(self, image_id: str):
        """Write YOLO-format .txt for valid masks."""
        from PIL import Image

        maskid_list = []
        mask_dir = os.path.join("images_masks", image_id)
        if not os.path.exists(mask_dir):
            return
        for f in os.listdir(mask_dir):
            if f.endswith(("csv", "txt")):
                continue
            maskid_list.append(f.split(".")[0])

        img = Image.open(os.path.join("images", f"{image_id}.png"))
        img_w, img_h = img.size

        entries = []
        csv_path = os.path.join(mask_dir, "metadata.csv")
        if not os.path.exists(csv_path):
            return

        with open(csv_path, "r") as cf:
            header = cf.readline().strip().split(",")
            for line in cf.readlines():
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                mid = parts[0]
                if mid not in maskid_list:
                    continue
                bx0 = float(parts[2])
                by0 = float(parts[3])
                bw = float(parts[4])
                bh = float(parts[5])
                xc = (bx0 + bw / 2) / img_w
                yc = (by0 + bh / 2) / img_h
                nw = bw / img_w
                nh = bh / img_h
                entries.append(f"{mid} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")

        with open(os.path.join(mask_dir, f"{image_id}.txt"), "w") as f:
            f.writelines(entries)

    # ---- CLIP classify & prune (replaces clipcompare.py:clip_remove) ----

    def _clip_classify_masks(self, image_id: str):
        """
        Classify cropped mask images with CLIP using the SDM dictionary,
        prune non-target masks, and update mask dir accordingly.
        """
        # Build SDM dictionary for this image
        img_path = os.path.join("images", f"{image_id}.png")
        fused_dict = self.sdm.build_dictionary(
            image_path=img_path if os.path.exists(img_path) else None,
            target_category=self.category,
            max_entries=5,
        )
        text_lang = self.sdm.text_language

        # Run per-mask classification
        from core.ism import classify_masks_with_clip
        kept, removed = classify_masks_with_clip(
            seg_img_dir="seg_img",
            target_label=self.category,
            descriptions=fused_dict,
            text_language=text_lang,
            top_k=0,
        )

        # Move removed masks from images_masks/{img_id}/
        mask_dir = os.path.join("images_masks", image_id)
        wrong_dir = os.path.join("wrongobjectMASK", image_id)
        for rm_file in removed:
            src = os.path.join(mask_dir, rm_file)
            if os.path.exists(src):
                os.makedirs(wrong_dir, exist_ok=True)
                shutil.move(src, os.path.join(wrong_dir, rm_file))

        # Update txt to remove wrong entries
        txt_path = os.path.join(mask_dir, f"{image_id}.txt")
        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                lines = f.readlines()
            removed_ids = {f.split(".")[0] for f in removed}
            filtered = [l for l in lines if l.split()[0] not in removed_ids]
            with open(txt_path, "w") as f:
                f.writelines(filtered)

        return kept, removed, fused_dict

    # ---- Bbox drawing & output --------------------------------------------

    def _draw_and_save_bboxes(self, proposals: List[List[int]],
                               image_id: str, image: np.ndarray):
        """Draw bounding boxes and save output files."""
        mask_dir = os.path.join("images_masks", image_id)
        os.makedirs(mask_dir, exist_ok=True)

        # Save bbox txt
        txt_path = os.path.join(mask_dir, f"{image_id}_hbb.txt")
        with open(txt_path, "w") as f:
            for bbox in proposals:
                f.write(f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

        # Also copy to label/ directory for ACC evaluation
        label_dir = "label"
        os.makedirs(label_dir, exist_ok=True)
        shutil.copy2(txt_path, os.path.join(label_dir, f"{image_id}.txt"))

        # Draw on image
        img_draw = image.copy()
        for bbox in proposals:
            cv2.rectangle(img_draw, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]), (0, 255, 0), 2)

        out_path = os.path.join(mask_dir, f"{image_id}_hbb_image.png")
        cv2.imwrite(out_path, img_draw)
        cv2.imwrite("output_image_hbb.png", img_draw)

    # ---- Evaluation --------------------------------------------------------

    def _evaluate(self, gt_folder: str):
        """Run accuracy evaluation (requires COMPARE module, optional)."""
        try:
            from COMPARE.txtcomparemapall import ACC
        except ImportError:
            print("  [ACC] COMPARE module not available, skipping evaluation")
            return
        for iou_thr in [0.4, 0.5, 0.6]:
            try:
                ACC(gt_folder, "label", iou_threshold=iou_thr)
                if os.path.exists(f"acc_result_{iou_thr}.txt"):
                    shutil.copy(
                        f"acc_result_{iou_thr}.txt",
                        os.path.join(self.output_base, f"acc_result_{iou_thr}.txt"),
                    )
            except Exception as e:
                print(f"  [ACC@{iou_thr}] Error: {e}")

    # ---- Main per-image detection ------------------------------------------

    def detect_image(self, image_path: str,
                     gt_folder: Optional[str] = None,
                     origin_image_folder: Optional[str] = None,
                     output_subdir: Optional[str] = None
                     ) -> Dict:
        """
        Run full KGCS pipeline on a single image.

        Args:
            image_path:   Path to input image
            gt_folder:    Optional ground-truth folder for evaluation
            origin_image_folder: Original high-res image folder
            output_subdir:     Sub-directory under output_base for results

        Returns:
            dict with keys: image_id, n_proposals, n_final, timings, debug
        """
        t_start = time.time()
        result = {"image_path": image_path}

        # 0. Clean workspace
        self._clean_workdirs()
        t0 = time.time()

        # 1. Prepare image
        ws_path, image_id = self._prepare_image(image_path)
        img = cv2.imread(ws_path)
        result["image_id"] = image_id
        print(f"\n{'='*50}")
        print(f"[KGCS] Processing {image_id} (boundary={'clear' if self.is_boundary_clear else 'ambiguous'})")
        print(f"{'='*50}")

        # 2. Generate SAM masks
        t1 = time.time()
        print("[Step 1/6] SAM mask generation...")
        try:
            mask_dir = generate_sam_masks(
                self.sam_generator, ws_path, image_id, output_dir="images_masks")
        except Exception as e:
            print(f"  SAM failed: {e}")
            return {**result, "error": str(e)}
        t2 = time.time()
        self.timings["sam"] = t2 - t1
        print(f"  Masks saved to {mask_dir}  ({t2-t1:.1f}s)")

        # 3. SDM dictionary
        print("[Step 2/6] SDM: building description dictionary...")
        img_path_full = os.path.join("images", f"{image_id}.png")
        fused_dict = self.sdm.build_dictionary(
            image_path=img_path_full,
            target_category=self.category,
            max_entries=5,
        )
        print(f"  Fused dictionary: {len(fused_dict)} entries")
        t3 = time.time()

        # 4. Masks → crops → CLIP classification
        print("[Step 3/6] Cropping masks & CLIP classification...")
        self._masks_to_crops(ws_path, image_id)
        self._mask_to_txt(image_id)
        kept, removed, fused_dict = self._clip_classify_masks(image_id)
        print(f"  Kept: {len(kept)}, Removed: {len(removed)}")
        t4 = time.time()
        self.timings["clip_classify"] = t4 - t3

        # 5. OPM: generate proposals from remaining masks
        print("[Step 4/6] OPM: generating proposals...")
        mask_id_dir = os.path.join("images_masks", image_id)
        contour_desc = self.sdm.get_contour_description(
            "compact" if self.is_boundary_clear else "complex")
        proposals, path_type = self.opm.generate_proposals(
            image=img,
            masks_dir=mask_id_dir,
            is_boundary_clear=self.is_boundary_clear,
            contour_description="",  # shape validation already done in CLIP classification
        )
        print(f"  {len(proposals)} proposals generated ({path_type} path)")
        t5 = time.time()
        self.timings["opm"] = t5 - t4

        # 6. ISM: two-stage gradient screening
        print("[Step 5/6] ISM: gradient screening (two-stage adaptive filtering)...")
        self.ism.load_reference_descriptions(
            target_dict={self.category: TARGET_DESCRIPTIONS.get(self.category, f"A {self.category}")},
            distractor_dict=DISTRACTOR_DESCRIPTIONS,
        )
        final_proposals, final_scores, debug = self.ism.screen_proposals(
            proposals=proposals,
            image=img,
            masks_dir=mask_id_dir,
            category=self.category,
            is_boundary_clear=self.is_boundary_clear,
        )
        print(f"  Stage1 threshold: {debug['threshold']:.4f}  |  "
              f"Stage1: {debug['n_stage1']}/{debug['n_total']}  |  "
              f"Stage2: {debug['n_stage2']}")
        t6 = time.time()
        self.timings["ism"] = t6 - t5

        # 7. Draw and save results
        print("[Step 6/6] Drawing bounding boxes & saving...")
        self._draw_and_save_bboxes(final_proposals, image_id, img)

        # Move results to output folder
        if output_subdir:
            out_dir = os.path.join(self.output_base, output_subdir)
        else:
            out_dir = os.path.join(self.output_base, self.category, image_id)
        os.makedirs(out_dir, exist_ok=True)
        self._copy_results(out_dir, image_id)

        t_end = time.time()
        self.timings["total"] = t_end - t_start

        print(f"\n[KGCS] Finished {image_id} in {t_end-t_start:.1f}s")
        for k, v in self.timings.items():
            print(f"  {k}: {v:.1f}s")

        result.update({
            "n_proposals": len(proposals),
            "n_final": len(final_proposals),
            "path_type": path_type,
            "timings": dict(self.timings),
            "debug": debug,
        })
        return result

    # ---- Utility: copy results to output dir ------------------------------

    def _copy_results(self, out_dir: str, image_id: str):
        mask_dir = os.path.join("images_masks", image_id)
        for fname in os.listdir(mask_dir):
            shutil.copy2(os.path.join(mask_dir, fname), out_dir)
        for f in ["output_image_hbb.png"]:
            if os.path.exists(f):
                shutil.copy2(f, out_dir)
        for f in os.listdir("label"):
            shutil.copy2(os.path.join("label", f), out_dir)

    # ---- Batch processing --------------------------------------------------

    def detect_batch(self, image_folder: str,
                     gt_folder: Optional[str] = None,
                     origin_image_folder: Optional[str] = None,
                     max_images: int = 0):
        """
        Run KGCS on all images in a folder.

        Args:
            image_folder:       Path to folder with input images
            gt_folder:          Optional ground-truth folder path
            origin_image_folder: Original hi-res images (for DOTA/DIOR)
            max_images:         Max images to process (0 = all)
        """
        extensions = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
        images = sorted([
            f for f in os.listdir(image_folder)
            if f.lower().endswith(extensions)
        ])

        if max_images > 0:
            images = images[:max_images]

        print(f"\n{'#'*60}")
        print(f"# KGCS Batch Detection: {len(images)} images")
        print(f"# Category: {self.category}")
        print(f"# Output:   {self.output_base}")
        print(f"{'#'*60}\n")

        results = []
        total_t0 = time.time()

        for fname in tqdm(images, desc="KGCS Detection"):
            fpath = os.path.join(image_folder, fname)
            try:
                res = self.detect_image(
                    image_path=fpath,
                    gt_folder=gt_folder,
                    origin_image_folder=origin_image_folder,
                    output_subdir=self.category,
                )
                results.append(res)
            except Exception as e:
                print(f"\n  [ERROR] {fname}: {e}")
                import traceback
                traceback.print_exc()
                continue

        total_t = time.time() - total_t0
        n_ok = sum(1 for r in results if "error" not in r)
        n_final = sum(r.get("n_final", 0) for r in results if "error" not in r)

        print(f"\n{'='*50}")
        print(f"Batch complete: {n_ok}/{len(images)} OK in {total_t:.1f}s")
        print(f"Total detections: {n_final}")
        print(f"Avg time/image: {total_t/max(n_ok,1):.1f}s")
        print(f"{'='*50}")

        return results
