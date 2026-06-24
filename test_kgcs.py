#!/usr/bin/env python
"""KGCS Test Script — verify modular components."""

import os, sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
ORIG = r"d:/KGCS/SAM-fine-tune-main (2)"
if os.path.exists(ORIG): sys.path.insert(0, ORIG)

def test_sdm():
    print("\n" + "="*60)
    print("Test 1: SDM Dictionary Construction")
    print("="*60)
    from config.settings import TARGET_DESCRIPTIONS, DISTRACTOR_DESCRIPTIONS
    from core.sdm import SceneDescriptionModule
    sdm = SceneDescriptionModule(TARGET_DESCRIPTIONS, DISTRACTOR_DESCRIPTIONS)
    for cat in ["ship", "airplane", "bridge"]:
        d = sdm.build_dictionary(target_category=cat, max_entries=5)
        print(f"  [{cat}] {len(d)} entries")
        for k, v in d.items():
            print(f"    {k}: {v[:60]}...")
    print("  SDM test PASSED")

def test_ism():
    print("\n" + "="*60)
    print("Test 2: ISM Module & CLIP Encoding")
    print("="*60)
    from config.settings import TARGET_DESCRIPTIONS, DISTRACTOR_DESCRIPTIONS
    from core.ism import ImageTextSimilarityModule
    ism = ImageTextSimilarityModule()
    ism.load_reference_descriptions(
        {"ship": TARGET_DESCRIPTIONS["ship"]},
        {"building": DISTRACTOR_DESCRIPTIONS["building"]})
    assert ism.text_features is not None
    print(f"  Text features shape: {ism.text_features.shape}")
    print("  ISM test PASSED")

def get_sample(category="ship"):
    td = os.path.join(PROJECT_ROOT, "test_images")
    if os.path.exists(td) and os.listdir(td):
        return [os.path.join(td, f) for f in os.listdir(td)
                if f.lower().endswith((".jpg", ".png"))][:1]
    cand = [os.path.join(ORIG, f) for f in
            ["11907.jpg","11757.jpg","12895.jpg","test.png","test.jpg"]]
    return [p for p in cand if os.path.exists(p)][:1]

def test_pipeline(category="ship"):
    print("\n" + "="*60)
    print(f"Test 3: Full KGCS Pipeline for [{category}]")
    print("="*60)
    from core.pipeline import KGCS_Pipeline
    samples = get_sample(category)
    if not samples:
        print("  No sample images found"); return
    pipe = KGCS_Pipeline(target_category=category,
                         output_base="E:/test/KGCS_output/test_run")
    r = pipe.detect_image(samples[0], output_subdir=f"test_{category}")
    if "error" in r:
        print(f"  [ERROR] {r['error']}")
    else:
        print(f"  Detections: {r.get('n_final',0)}, "
              f"Timing: {r.get('timings',{})}")
    return r

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--category","-c",default="ship")
    ap.add_argument("--test","-t",default="all",choices=["all","sdm","ism","pipeline"])
    args = ap.parse_args()
    print("#"*60 + "\n# KGCS Modular Verification Tests\n" + "#"*60)
    if args.test in ("all","sdm"): test_sdm()
    if args.test in ("all","ism"): test_ism()
    if args.test in ("all","pipeline"): test_pipeline(args.category)
    print("\n#"*60 + "\n# Tests complete\n" + "#"*60)
