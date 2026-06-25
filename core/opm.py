"""
KGCS - Object Proposal Module (OPM)

Structure-aware dual-path proposal strategy that generates high-quality
candidate regions from SAM segmentation masks.

Path 1 — Contour-Clear Targets:  Direct mask-to-bbox with CLIP shape validation
Path 2 — Boundary-Ambiguous / Compositional Targets:
          Structure-guided mask clustering + adaptive sliding window

Reference: KGCS_R2.pdf § II-B (Object Proposal Module)
"""

import os
import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict

from config.settings import OPM as OPM_CFG


# ---------------------------------------------------------------------------
# 1. Mask Utilities
# ---------------------------------------------------------------------------

def read_mask(mask_path: str) -> np.ndarray:
    """Read a single-channel mask, binarize."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary


def mask_to_bbox(mask: np.ndarray,
                 min_area: int = 100,
                 expand_ratio: float = 0.0) -> Optional[List[int]]:
    """
    Convert a binary mask to its HBB (horizontal bounding box).
    Returns [x1, y1, x2, y2] or None if no valid contour found.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    best_bbox, max_area = None, 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area >= min_area and area > max_area:
            max_area = area
            expand_x = int(w * expand_ratio)
            expand_y = int(h * expand_ratio)
            best_bbox = [
                max(0, x - expand_x),
                max(0, y - expand_y),
                x + w + expand_x,
                y + h + expand_y,
            ]
    return best_bbox


def mask_to_hbbs(mask: np.ndarray,
                 min_area: int = 100,
                 expand_ratio: float = 0.05) -> List[List[int]]:
    """
    Extract ALL valid HBBs from a mask (for multi-contour cases).
    Returns list of [x1, y1, x2, y2].
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    hbbs = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area >= min_area:
            ex = int(w * expand_ratio)
            ey = int(h * expand_ratio)
            hbbs.append([max(0, x - ex), max(0, y - ey),
                         x + w + ex, y + h + ey])
    return hbbs


def non_max_suppression(hbbs: List[List[int]],
                        iou_threshold: float = 0.5) -> List[List[int]]:
    """Standard NMS on a list of HBBs."""
    if not hbbs:
        return []
    boxes = np.array(hbbs, dtype=np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = areas.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-10)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return boxes[keep].astype(int).tolist()


# ---------------------------------------------------------------------------
# 2. Mask Clustering (for compositional targets)
# ---------------------------------------------------------------------------

def cluster_masks_by_connectivity(masks_dir: str,
                                  min_area: int = 100
                                  ) -> List[Dict]:
    """
    Cluster individual mask fragments into structure-consistent groups
    based on spatial connectivity (overlap / adjacency).

    Returns:
        List of dicts: [{"bbox": [x1,y1,x2,y2],
                         "mask_ids": [str,...],
                         "n_masks": int}, ...]
    """
    # Load all masks and their bounding boxes
    mask_bboxes = []  # (mask_id, bbox)
    for fname in os.listdir(masks_dir):
        if not fname.endswith(".png"):
            continue
        fpath = os.path.join(masks_dir, fname)
        mask = read_mask(fpath)
        bbox = mask_to_bbox(mask, min_area=min_area)
        if bbox is not None:
            mask_bboxes.append((fname, bbox, mask))

    if not mask_bboxes:
        return []

    # Simple greedy clustering: merge if IoU > 0
    clusters = []
    assigned = set()

    for i, (mid_i, bbox_i, _) in enumerate(mask_bboxes):
        if i in assigned:
            continue
        # Start a new cluster
        cluster_ids = [mid_i]
        cx1, cy1, cx2, cy2 = bbox_i
        assigned.add(i)

        for j, (mid_j, bbox_j, _) in enumerate(mask_bboxes):
            if j in assigned:
                continue
            # Check overlap
            ox1 = max(cx1, bbox_j[0])
            oy1 = max(cy1, bbox_j[1])
            ox2 = min(cx2, bbox_j[2])
            oy2 = min(cy2, bbox_j[3])
            if ox2 > ox1 and oy2 > oy1:
                cluster_ids.append(mid_j)
                cx1 = min(cx1, bbox_j[0])
                cy1 = min(cy1, bbox_j[1])
                cx2 = max(cx2, bbox_j[2])
                cy2 = max(cy2, bbox_j[3])
                assigned.add(j)

        clusters.append({
            "bbox": [int(cx1), int(cy1), int(cx2), int(cy2)],
            "mask_ids": cluster_ids,
            "n_masks": len(cluster_ids),
        })

    return clusters


def adaptive_sliding_windows(cluster_bbox: List[int],
                             n_masks: int,
                             max_masks: int,
                             img_shape: Tuple[int, int]
                             ) -> List[List[int]]:
    """
    Generate adaptive sliding windows for a mask cluster.

    Eq.(6)-(8) from KGCS paper:
      Window size S = L * W  (long side * short side)
      Step size δ = (S/3) * (1 - N_cluster/N_max)
      Multi-scale: W_std = S, W_large = 1.05*S, W_small = 0.95*S
    """
    h, w = img_shape[:2]
    cx1, cy1, cx2, cy2 = cluster_bbox
    cw = max(cx2 - cx1, 1)
    ch = max(cy2 - cy1, 1)

    # Window size: use the larger side
    S = max(cw, ch)
    # Center point
    cx = (cx1 + cx2) // 2
    cy = (cy1 + cy2) // 2

    n_max = max(max_masks, 1)
    step_factor = max(0.1, 1.0 - n_masks / n_max)
    step = max(int(S * step_factor / 3), 10)

    scales = [1.0, 1.05, 0.95]
    windows = []

    for scale in scales:
        ws = int(S * scale)
        # Generate windows in a grid around the cluster center
        for dy in [-step, 0, step]:
            for dx in [-step, 0, step]:
                x1 = max(0, cx + dx - ws // 2)
                y1 = max(0, cy + dy - ws // 2)
                x2 = min(w, cx + dx + ws // 2)
                y2 = min(h, cy + dy + ws // 2)
                if x2 > x1 and y2 > y1:
                    windows.append([x1, y1, x2, y2])

    return windows


# ---------------------------------------------------------------------------
# 3. CLIP-based Shape Validation
# ---------------------------------------------------------------------------

def shape_validate_with_clip(image_crop: np.ndarray,
                             contour_description: str,
                             clip_model,
                             clip_preprocess,
                             device: str,
                             threshold: float = 0.5) -> bool:
    """
    Validate whether a cropped region matches the expected contour
    description using CLIP (Path 1 shape validation, Eq.(4)).

    Returns True if CLIP similarity > threshold.
    """
    from PIL import Image
    import torch

    # Convert crop to PIL & preprocess
    crop_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
    crop_pil = Image.fromarray(crop_rgb)
    img_tensor = clip_preprocess(crop_pil).unsqueeze(0).to(device)

    import clip
    text_tokens = clip.tokenize([contour_description]).to(device)

    with torch.no_grad():
        img_feat = clip_model.encode_image(img_tensor)
        txt_feat = clip_model.encode_text(text_tokens)
        img_feat = torch.nn.functional.normalize(img_feat, p=2, dim=-1)
        txt_feat = torch.nn.functional.normalize(txt_feat, p=2, dim=-1)
        sim = torch.matmul(img_feat, txt_feat.T).item()

    return sim > threshold


# ---------------------------------------------------------------------------
# 4. OPM Core
# ---------------------------------------------------------------------------

class ObjectProposalModule:
    """
    Object Proposal Module (OPM).

    Dual-path strategy:
      - Path 1: contour-clear targets → bbox + shape validation
      - Path 2: boundary-ambiguous targets → clustering + adaptive windows

    Usage:
        opm = ObjectProposalModule()
        proposals = opm.generate_proposals(image, masks_dir, is_composite=False)
    """

    def __init__(self, config: Optional[Dict] = None):
        self.cfg = config or OPM_CFG
        self.clip_model = None
        self.clip_preprocess = None
        self.device = "cuda"
        self._init_clip()

    def _init_clip(self):
        """Lazy-load CLIP model for shape validation."""
        try:
            import clip  # noqa
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model, self.clip_preprocess = clip.load(
                "ViT-L/14@336px", device=self.device)
        except Exception:
            self.clip_model = None
            self.clip_preprocess = None

    # ---- PATH 1: Contour-Clear -------------------------------------------

    def path_contour_clear(self, image: np.ndarray,
                           masks_dir: str,
                           contour_description: str = ""
                           ) -> List[List[int]]:
        """
        Direct proposal generation for well-defined objects.

        1. Convert each mask → bbox
        2. (Optional) CLIP shape validation
        3. NMS cleanup
        """
        proposals = []
        min_area = self.cfg.get("min_contour_area", 100)

        for fname in sorted(os.listdir(masks_dir)):
            if not fname.endswith(".png"):
                continue
            mpath = os.path.join(masks_dir, fname)
            try:
                mask = read_mask(mpath)
            except Exception:
                continue
            bbox = mask_to_bbox(mask, min_area=min_area)
            if bbox is None:
                continue

            # Optional CLIP shape validation (Eq.4)
            if contour_description and self.clip_model is not None:
                crop = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                if not shape_validate_with_clip(
                        crop, contour_description,
                        self.clip_model, self.clip_preprocess,
                        self.device):
                    continue

            proposals.append(bbox)

        return non_max_suppression(proposals,
                                   iou_threshold=self.cfg.get("iou_threshold", 0.5))

    # ---- PATH 2: Boundary-Ambiguous ---------------------------------------

    def path_boundary_ambiguous(self, image: np.ndarray,
                                masks_dir: str
                                ) -> List[List[int]]:
        """
        Structure-aware proposal generation for composite/ambiguous targets.

        1. Cluster masks by spatial connectivity (Eq.5)
        2. Generate adaptive sliding windows per cluster (Eq.6-8)
        3. Collect unique windows
        """
        clusters = cluster_masks_by_connectivity(
            masks_dir, min_area=self.cfg.get("min_contour_area", 100))
        if not clusters:
            return []

        max_masks = max((c["n_masks"] for c in clusters), default=1)
        img_shape = image.shape
        all_windows = []

        for cluster in clusters:
            windows = adaptive_sliding_windows(
                cluster["bbox"], cluster["n_masks"],
                max_masks, img_shape)
            all_windows.extend(windows)

        # Deduplicate: keep only unique windows (threshold IoU < 0.8)
        deduped = []
        for w in all_windows:
            if not any(mask_to_bbox_ioa(w, d) > 0.8 for d in deduped):
                deduped.append(w)

        return deduped

    # ---- Main entry -------------------------------------------------------

    def generate_proposals(self, image: np.ndarray,
                           masks_dir: str,
                           is_boundary_clear: bool = True,
                           contour_description: str = ""
                           ) -> Tuple[List[List[int]], str]:
        """
        Generate candidate proposals from SAM mask output.

        Args:
            image:          Original image (BGR numpy array)
            masks_dir:      Path to per-image mask folder
            is_boundary_clear: True for Path 1, False for Path 2
            contour_description: Text description for shape validation

        Returns:
            (proposals, path_name):
                proposals  = list of [x1,y1,x2,y2]
                path_name  = "clear" | "ambiguous"
        """
        if is_boundary_clear:
            props = self.path_contour_clear(
                image, masks_dir, contour_description)
            return props, "clear"
        else:
            props = self.path_boundary_ambiguous(image, masks_dir)
            return props, "ambiguous"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def mask_to_bbox_ioa(bbox_a: List[int], bbox_b: List[int]) -> float:
    """Intersection over Area of the smaller bbox."""
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_min = min((bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1]),
                   (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1]))
    return inter / area_min if area_min > 0 else 0.0
