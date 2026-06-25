"""
KGCS - Scene Description Module (SDM)

Builds hierarchical description dictionaries to enhance CLIP's fine-grained
discrimination for remote-sensing aerial objects.

Three dictionary types:
  1. Target Semantic Description Dictionary   D_target
  2. Distractor Description Dictionary        D_distractor
  3. Contour Feature Dictionary               D_contour

Reference: KGCS_R2.pdf § II-A (Scene Description Module)
"""

import json
import os
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# 1. Helpers
# ---------------------------------------------------------------------------

def load_descriptions(desc_dict: Dict[str, str]) -> Dict[str, str]:
    """Load & validate description dictionary entries."""
    cleaned = {}
    for key, val in desc_dict.items():
        key_clean = key.strip().replace(" ", "_")
        cleaned[key_clean] = val.strip()
    return cleaned


def _call_gpt4o_for_scene(image_path: str, api_key: str,
                          api_url: str) -> Dict[str, str]:
    """
    One-shot scene parsing with GPT-4o.
    Extracts scene-level entities from the whole image.
    """
    from utils.obj_judge import ImageAnalyzer
    prompt = (
        "You are a remote sensing expert. For the provided aerial image, "
        "describe all possible targets and the features of their easily "
        "confusable distractors. Return only a JSON dictionary mapping "
        "object names to concise descriptions (≤15 words each)."
    )
    analyzer = ImageAnalyzer(api_key=api_key, api_url=api_url,
                             detect_prompt=prompt)
    response = analyzer.get_img_txt(os.path.basename(image_path),
                                    os.path.dirname(image_path))
    # Parse the JSON response
    response = response.strip().replace("'", '"')
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# 2. SDM Core
# ---------------------------------------------------------------------------

class SceneDescriptionModule:
    """
    Scene Description Module (SDM).

    Constructs a fused hierarchical dictionary from expert knowledge and
    optional LLM one-shot parsing, then provides it to ISM for matching.
    """

    def __init__(self,
                 target_descriptions: Dict[str, str],
                 distractor_descriptions: Optional[Dict[str, str]] = None,
                 contour_descriptions: Optional[Dict[str, str]] = None,
                 llm_api_key: Optional[str] = None,
                 llm_api_url: Optional[str] = None):
        """
        Args:
            target_descriptions:  Expert-defined {category: description}
            distractor_descriptions: Common distractor {name: description}
            contour_descriptions:  Geometric contour prototypes
            llm_api_key / llm_api_url:  Optional GPT-4o credentials
        """
        self.target_dict = load_descriptions(target_descriptions)
        self.distractor_dict = load_descriptions(
            distractor_descriptions or {})
        self.contour_dict = load_descriptions(contour_descriptions or {})
        self.llm_api_key = llm_api_key
        self.llm_api_url = llm_api_url

        # Fused semantic dictionary (target + selected distractors)
        self.fused_dict: Dict[str, str] = {}
        self._text_language: List[str] = []

    # ---- Public API ----

    @property
    def text_language(self) -> List[str]:
        """Ordered list of keys in the fused dictionary."""
        return self._text_language

    def build_dictionary(self,
                         image_path: Optional[str] = None,
                         target_category: Optional[str] = None,
                         max_entries: int = 5) -> Dict[str, str]:
        """
        Build the fused hierarchical description dictionary.

        Strategy (KGCS § II-A4):
          - Start with expert target description(s)
          - Optionally fuse with LLM scene parsing
          - Append 3-4 common distractor descriptions
          - Keep total ≤ `max_entries` for CLIP efficiency

        Returns:
            Dict[str, str]  {label: description}
        """
        d_target: Dict[str, str] = {}
        d_distractor: Dict[str, str] = {}

        # --- Target descriptions ---
        if target_category and target_category in self.target_dict:
            d_target[target_category] = self.target_dict[target_category]
        else:
            # Use all target descriptions (multi-class scenario)
            d_target = dict(self.target_dict)

        # --- Optional LLM fusion (one-shot scene parsing) ---
        if image_path and self.llm_api_key:
            try:
                llm_dict = _call_gpt4o_for_scene(
                    image_path, self.llm_api_key, self.llm_api_url)
                # Merge LLM output into target dict (new keys only)
                for k, v in llm_dict.items():
                    if k not in d_target and k not in self.distractor_dict:
                        d_target[k] = v
            except Exception:
                pass  # fallback to expert descriptions only

        # --- Distractor selection (quality over quantity) ---
        distractor_keys = list(self.distractor_dict.keys())
        # Pick up to (max_entries - len(d_target)) distractors
        n_distractors = min(len(distractor_keys),
                            max(2, max_entries - len(d_target)))
        for key in distractor_keys[:n_distractors]:
            d_distractor[key] = self.distractor_dict[key]

        # --- Fusion ---
        self.fused_dict = {}
        self.fused_dict.update(d_target)
        self.fused_dict.update(d_distractor)
        self._text_language = list(self.fused_dict.keys())

        return self.fused_dict

    def get_contour_description(self, shape_type: str = "compact") -> str:
        """Retrieve contour description for shape validation."""
        return self.contour_dict.get(shape_type,
                                     self.contour_dict.get("compact", ""))

    def get_fused_dict(self) -> Dict[str, str]:
        return self.fused_dict

    def update_descriptions(self,
                            target_descriptions: Optional[Dict[str, str]] = None,
                            distractor_descriptions: Optional[Dict[str, str]] = None):
        """Update the expert dictionaries at runtime."""
        if target_descriptions:
            self.target_dict.update(load_descriptions(target_descriptions))
        if distractor_descriptions:
            self.distractor_dict.update(load_descriptions(distractor_descriptions))
