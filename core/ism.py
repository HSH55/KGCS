"""
KGCS - Image-Text Similarity Module (ISM)

Two-stage adaptive filtering for zero-shot classification & localization:

  Stage 1 — Confidence-Distribution-Guided Dynamic Threshold Calibration
  Stage 2 — Boundary-Type-Aware Differentiated Ratio Filtering (Gradient Screening)

Reference: KGCS_R2.pdf § II-C (Image-Text Similarity Module)
"""

import os
import torch
import clip
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple

from config.settings import ISM as ISM_CFG
from config.settings import FIXED_BOUNDARY_CATEGORIES


# ---------------------------------------------------------------------------
# CLIP helper (singleton pattern for reuse)
# ---------------------------------------------------------------------------

_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_CLIP_DEVICE = None


def _load_clip(model_name: str = "ViT-L/14@336px"):
    global _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE
    if _CLIP_MODEL is None:
        _CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        _CLIP_MODEL, _CLIP_PREPROCESS = clip.load(model_name, device=_CLIP_DEVICE)
    return _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE


def clip_encode_text(texts: List[str]) -> np.ndarray:
    """Encode text descriptions → normalized feature vectors."""
    model, preproc, device = _load_clip()
    text_tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        features = model.encode_text(text_tokens)
        features = torch.nn.functional.normalize(features, p=2, dim=-1)
    return features.cpu().numpy()


def clip_encode_image(image_path: str) -> np.ndarray:
    """Encode a single image → normalized feature vector."""
    model, preproc, device = _load_clip()
    img = preproc(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(img)
        features = torch.nn.functional.normalize(features, p=2, dim=-1)
    return features.cpu().numpy()


def compute_similarity(image_feat: np.ndarray,
                       text_feats: np.ndarray) -> np.ndarray:
    """Cosine similarity between one image and N text features."""
    return np.dot(image_feat, text_feats.T).flatten()


# ---------------------------------------------------------------------------
# ISM Core
# ---------------------------------------------------------------------------

class ImageTextSimilarityModule:
    """
    Image-Text Similarity Module (ISM).

    Two-stage adaptive filtering (Gradient Screening):
      Stage 1: Dynamic threshold calibration from confidence distribution
      Stage 2: Boundary-type-aware ratio filtering safeguard

    Usage:
        ism = ImageTextSimilarityModule()
        ism.load_reference_descriptions(target_dict, distractor_dict)
        results = ism.screen_proposals(
            proposals, image, masks_dir, category="ship", is_boundary_clear=True)
    """

    def __init__(self, config: Optional[Dict] = None):
        self.cfg = config or ISM_CFG

        # SDM dictionary content (loaded before screening)
        self.target_dict: Dict[str, str] = {}
        self.distractor_dict: Dict[str, str] = {}
        self.fused_texts: List[str] = []          # ordered text labels
        self.text_features: Optional[np.ndarray] = None  # cached features

        # Storage for screening results
        self.confidence_scores: List[float] = []

    # ---- Setup ------------------------------------------------------------

    def load_reference_descriptions(self,
                                    target_dict: Dict[str, str],
                                    distractor_dict: Optional[Dict[str, str]] = None,
                                    text_language: Optional[List[str]] = None):
        """
        Load the SDM output dictionaries.

        Args:
            target_dict:     {category: description}  (D_target)
            distractor_dict: {name: description}      (D_distractor)
            text_language:   ordered keys for fused dict (optional)
        """
        self.target_dict = dict(target_dict)
        self.distractor_dict = dict(distractor_dict or {})

        # Build fused text list
        if text_language:
            self.fused_texts = list(text_language)
        else:
            self.fused_texts = list(self.target_dict.keys()) + \
                list(self.distractor_dict.keys())

        # Build fused description list (in order)
        self.fused_descriptions = []
        for key in self.fused_texts:
            if key in self.target_dict:
                self.fused_descriptions.append(self.target_dict[key])
            elif key in self.distractor_dict:
                self.fused_descriptions.append(self.distractor_dict[key])

        # Pre-compute text features
        if self.fused_descriptions:
            self.text_features = clip_encode_text(self.fused_descriptions)

    # ---- Stage 1: Dynamic Threshold Calibration ---------------------------

    def _stage1_calibrate(self, scores: List[float],
                          sigma_factor: float = 0.5) -> Tuple[float, List[int]]:
        """
        Confidence-distribution-guided dynamic threshold calibration.

        Eqs.(11)-(13):
          μ = mean(scores),  σ = std(scores)
          T_dynamic = μ - σ_factor · σ

        Returns:
            (threshold, valid_indices)
        """
        scores_arr = np.array(scores)
        mu = np.mean(scores_arr)
        sigma = np.std(scores_arr)

        threshold = mu - sigma_factor * sigma
        valid = np.where(scores_arr > threshold)[0].tolist()

        return float(threshold), valid

    # ---- Stage 2: Ratio Filtering Safeguard -------------------------------

    def _stage2_ratio_filter(self, proposals: List, scores: List[float],
                             retention_ratio: float = 0.8
                             ) -> Tuple[List, List[float]]:
        """
        Boundary-type-aware differentiated ratio filtering.

        Keep the top `retention_ratio` proportion of proposals ranked
        by descending confidence score.

        Returns:
            (filtered_proposals, filtered_scores)
        """
        if not proposals:
            return [], []

        # Sort by score descending
        sorted_idx = np.argsort(scores)[::-1]
        n_keep = max(1, int(len(proposals) * retention_ratio))

        kept_proposals = [proposals[i] for i in sorted_idx[:n_keep]]
        kept_scores = [scores[i] for i in sorted_idx[:n_keep]]

        return kept_proposals, kept_scores

    # ---- Full Screening Pipeline ------------------------------------------

    def screen_proposals(self,
                         proposals: List[List[int]],
                         image: np.ndarray,
                         masks_dir: str,
                         category: str,
                         is_boundary_clear: bool = True
                         ) -> Tuple[List[List[int]], List[float], Dict]:
        """
        Run the full two-stage adaptive filtering (Gradient Screening).

        Args:
            proposals:          List of [x1,y1,x2,y2] from OPM
            image:              Original image (BGR numpy array)
            masks_dir:          Path to mask directory (for feature reference)
            category:           Target category name
            is_boundary_clear:  True = contour-clear, False = boundary-ambiguous

        Returns:
            (final_proposals, final_scores, debug_info)
        """
        if not proposals or self.text_features is None:
            return [], [], {"n_stage1": 0, "n_stage2": 0, "threshold": 0,
                            "n_total": len(proposals) if proposals else 0,
                            "mean_score": 0, "std_score": 0}

        # --- Compute confidence scores (Eqs.9-10) ---
        # For each proposal, encode the cropped region and compute
        # max cosine similarity against target descriptions.

        # We compute in batch for efficiency
        cropped_images = []
        for [x1, y1, x2, y2] in proposals:
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            cropped_images.append(Image.fromarray(crop_rgb))

        if not cropped_images:
            return [], [], {"n_stage1": 0, "n_stage2": 0}

        # Batch CLIP encoding
        model, preproc, device = _load_clip()
        batch_imgs = torch.cat([
            preproc(img).unsqueeze(0).to(device) for img in cropped_images
        ], dim=0)

        # Target text features (only target descriptions)
        target_keys = list(self.target_dict.keys())
        target_texts = [self.target_dict[k] for k in target_keys]
        text_tokens = clip.tokenize(target_texts).to(device)

        with torch.no_grad():
            img_feats = model.encode_image(batch_imgs)
            txt_feats = model.encode_text(text_tokens)
            img_feats = torch.nn.functional.normalize(img_feats, p=2, dim=-1)
            txt_feats = torch.nn.functional.normalize(txt_feats, p=2, dim=-1)

            # Similarity: [N_proposals, N_targets]
            sim_matrix = torch.matmul(img_feats, txt_feats.T)
            max_scores, pred_indices = sim_matrix.max(dim=1)
            scores = max_scores.cpu().numpy().tolist()
            labels = [target_keys[idx] for idx in pred_indices.cpu().numpy()]

        # --- Stage 1: Dynamic threshold calibration ---
        sigma_factor = (self.cfg.get("sigma_factor_clear", 0.5)
                        if is_boundary_clear
                        else self.cfg.get("sigma_factor_ambiguous", 0.8))

        threshold, valid_idx = self._stage1_calibrate(
            scores, sigma_factor=sigma_factor)

        if not valid_idx:
            return [], [], {"n_stage1": 0, "n_stage2": 0, "threshold": threshold}

        stage1_proposals = [proposals[i] for i in valid_idx]
        stage1_scores = [scores[i] for i in valid_idx]

        # --- Stage 2: Ratio filtering safeguard ---
        retention_ratio = (self.cfg.get("retention_ratio_clear", 0.8)
                           if is_boundary_clear
                           else self.cfg.get("retention_ratio_ambiguous", 0.3))

        final_proposals, final_scores = self._stage2_ratio_filter(
            stage1_proposals, stage1_scores, retention_ratio=retention_ratio)

        debug = {
            "n_total": len(proposals),
            "threshold": threshold,
            "n_stage1": len(stage1_proposals),
            "n_stage2": len(final_proposals),
            "mean_score": float(np.mean(scores)) if scores else 0,
            "std_score": float(np.std(scores)) if scores else 0,
        }

        return final_proposals, final_scores, debug


# ---------------------------------------------------------------------------
# Patch-level ISM for per-mask classification (clip_remove equivalent)
# ---------------------------------------------------------------------------

def classify_masks_with_clip(seg_img_dir: str,
                              target_label: str,
                              descriptions: Dict[str, str],
                              text_language: List[str],
                              top_k: int = 0
                              ) -> Tuple[List[str], List[str]]:
    """
    Classify each cropped mask image using CLIP against the SDM dictionary.

    This replaces clipcompare.py:clip_remove() functionality.

    Args:
        seg_img_dir:   Directory with per-mask crops (from masktoimg)
        target_label:  Target category to select
        descriptions:  {label: description} fused dictionary
        text_language: Ordered label list

    Returns:
        (kept_masks, removed_masks) — filenames
    """
    import os

    # Gather image paths
    img_paths = sorted([
        os.path.join(seg_img_dir, f) for f in os.listdir(seg_img_dir)
        if f.endswith(".png")
    ])

    if not img_paths or not text_language:
        return [], []

    # Build description text list matching text_language order
    text_descriptions = [descriptions.get(label, label) for label in text_language]

    # Batch CLIP classification (process in batches to avoid OOM)
    model, preproc, device = _load_clip()

    BATCH_SIZE = 32
    all_probs = []
    text = clip.tokenize(text_descriptions).to(device)

    for i in range(0, len(img_paths), BATCH_SIZE):
        batch_paths = img_paths[i:i + BATCH_SIZE]
        batch_imgs = []
        for p in batch_paths:
            try:
                tensor = preproc(Image.open(p)).unsqueeze(0).to(device)
                batch_imgs.append(tensor)
            except Exception:
                continue
        if not batch_imgs:
            all_probs.append(np.zeros((0, len(text_descriptions))))
            continue
        batch_tensor = torch.cat(batch_imgs, dim=0)
        with torch.no_grad():
            logits_patch, _ = model(batch_tensor, text)
            probs_patch = logits_patch.softmax(dim=-1).cpu().numpy()
        all_probs.append(probs_patch)
        # Free GPU memory
        del batch_tensor, logits_patch, probs_patch

    probs = np.concatenate(all_probs, axis=0) if all_probs else np.array([])

    # Select masks predicted as target_label, sorted by similarity
    results = []
    for i, img_path in enumerate(img_paths):
        idx = np.argmax(probs[i])
        label = text_language[idx]
        similarity = probs[i, idx]
        results.append((img_path, label, similarity, os.path.basename(img_path)))

    # Filter by target label
    matching = [(p, s, n) for p, l, s, n in results if l == target_label]

    if top_k > 0:
        matching.sort(key=lambda x: x[1], reverse=True)
        matching = matching[:top_k]

    kept = [n for _, _, n in matching]
    removed = [n for _, _, _, n in results
               if n not in kept]

    return kept, removed


# Circular import safe
import cv2  # noqa: E402
