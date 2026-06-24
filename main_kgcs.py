#!/usr/bin/env python
"""
KGCS: Zero-Annotation Expert-Knowledge Injection for Object Detection
                     in Aerial Images

Main entry point — the ONLY main function.
All other modules are called as orchestrated sub-modules.

Architecture (cf. KGCS_R2.pdf):
  ┌─────────────────────────────────────────────────────────────┐
  │   SDM (Scene Description Module)   — Semantic Dictionaries  │
  │   OPM (Object Proposal Module)     — Dual-path Proposals    │
  │   ISM (Image-Text Similarity)      — Gradient Screening     │
  └─────────────────────────────────────────────────────────────┘

Usage:
    python main_kgcs.py --image test.jpg --category ship
    python main_kgcs.py --folder ./images --category airplane --max 10
    python main_kgcs.py --eval --category ship --gt D:/DIOR/hbbtxt/ship
"""

import os
import sys
import argparse
import shutil
import time

# Add project root and original SAM project to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
ORIG_PROJECT = r"d:/KGCS/SAM-fine-tune-main (2)"
if os.path.exists(ORIG_PROJECT):
    sys.path.insert(0, ORIG_PROJECT)

from core.pipeline import KGCS_Pipeline


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="KGCS: Zero-Annotation Object Detection in Aerial Images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --image test.jpg --category ship
  %(prog)s --folder ./images --category airplane --max 5
  %(prog)s --image 12895.jpg --category airplane --output E:/results
        """,
    )

    # Input source (one of these)
    group_input = parser.add_mutually_exclusive_group(required=True)
    group_input.add_argument("--image", "-i", type=str,
                             help="Path to a single input image")
    group_input.add_argument("--folder", "-f", type=str,
                             help="Path to folder with input images")
    group_input.add_argument("--eval", action="store_true",
                             help="Evaluation mode with ground-truth comparison")

    # Detection parameters
    parser.add_argument("--category", "-c", type=str, default="ship",
                        choices=[
                            "airplane", "airport", "baseballfield",
                            "basketballcourt", "bridge", "chimney", "dam",
                            "Expressway-Service-area", "Expressway-toll-station",
                            "golffield", "groundtrackfield", "harbor",
                            "overpass", "ship", "stadium", "storagetank",
                            "tenniscourt", "trainstation", "vehicle", "windmill",
                        ],
                        help="Target detection category")
    parser.add_argument("--sam-mode", type=str, default="vit_h",
                        choices=["vit_h", "vit_l", "vit_b"],
                        help="SAM model variant")

    # Output
    parser.add_argument("--output", "-o", type=str, default="E:/test/KGCS_output",
                        help="Output directory for results")
    parser.add_argument("--max", "-m", type=int, default=0,
                        help="Max images to process (0 = all)")

    # Ground truth (for evaluation)
    parser.add_argument("--gt", "-g", type=str,
                        help="Ground-truth label folder (for --eval or --folder)")

    # Origin images (for high-res restoration)
    parser.add_argument("--origin", type=str,
                        help="Original high-res image folder (for DOTA/DIOR)")

    # GPT-4o integration
    parser.add_argument("--llm-api-key", type=str, default=None,
                        help="API key for GPT-4o scene parsing (optional)")
    parser.add_argument("--llm-api-url", type=str, default=None,
                        help="API URL for GPT-4o")

    # Expert descriptions toggle
    parser.add_argument("--no-expert-desc", action="store_true",
                        help="Disable expert-defined descriptions (use simple prompts)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Interactive mode — test from DOTA/DIOR dataset
# ---------------------------------------------------------------------------

def eval_dior_category(category: str, max_images: int = 5):
    """
    Evaluation mode using DIOR dataset structure.
    Expects:
      D:/DIOR/object/{category}/  — images
      D:/DIOR/hbbtxt/{category}/  — ground-truth labels
    """
    image_folder = f"D:/DIOR/object/{category}"
    gt_folder = f"D:/DIOR/hbbtxt/{category}"
    origin_folder = image_folder

    if not os.path.exists(image_folder):
        print(f"[ERROR] DIOR image folder not found: {image_folder}")
        # Fall back: use project images from original project
        orig = os.environ.get('ORIG_PROJECT', r'd:/KGCS/SAM-fine-tune-main (2)')
        image_folder = orig if os.path.exists(orig) else "."
        gt_folder = None
        origin_folder = None

    pipeline = KGCS_Pipeline(
        target_category=category,
        output_base="E:/test/KGCS_output",
    )

    results = pipeline.detect_batch(
        image_folder=image_folder,
        gt_folder=gt_folder,
        origin_image_folder=origin_folder,
        max_images=max_images,
    )
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Override output base if specified
    output_base = args.output

    # Create pipeline
    pipeline = KGCS_Pipeline(
        target_category=args.category,
        sam_mode=args.sam_mode,
        output_base=output_base,
        use_expert_descriptions=not args.no_expert_desc,
        llm_api_key=args.llm_api_key,
        llm_api_url=args.llm_api_url,
    )

    # --- Single image mode ---
    if args.image:
        if not os.path.exists(args.image):
            print(f"[ERROR] Image not found: {args.image}")
            sys.exit(1)
        result = pipeline.detect_image(
            image_path=args.image,
            gt_folder=args.gt,
            origin_image_folder=args.origin,
            output_subdir=args.category,
        )
        print(f"\nDetection results: {result.get('n_final', 0)} objects found")

    # --- Batch mode ---
    elif args.folder:
        if not os.path.exists(args.folder):
            print(f"[ERROR] Folder not found: {args.folder}")
            sys.exit(1)
        results = pipeline.detect_batch(
            image_folder=args.folder,
            gt_folder=args.gt,
            origin_image_folder=args.origin,
            max_images=args.max,
        )

    # --- Eval mode (DIOR/DOTA) ---
    elif args.eval:
        eval_dior_category(
            category=args.category,
            max_images=args.max if args.max > 0 else 5,
        )

    print("\nDone. Results saved to:", output_base)


if __name__ == "__main__":
    main()
