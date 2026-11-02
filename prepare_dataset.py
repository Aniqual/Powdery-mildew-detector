"""
Merge the four Kaggle datasets into one train/val/test folder tree.

Output labels look like  <species>_<condition>, e.g. cherry_powdery_mildew, apple_healthy.
Species and condition are inferred from folder names *relative to each dataset root*,
with per-dataset defaults as fallback (e.g. everything in the apple-powdery-mildew set).

Usage:
  python prepare_dataset.py \
      --cherry ./raw/cherry-leaves \
      --plantvillage ./raw/plantvillage-dataset \
      --cucumber ./raw/cucumber-leaf-disease-dataset \
      --apple ./raw/apple-leaf-disease-powdery-mildew \
      --out ./data
"""
import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPECIES = ["apple", "cherry", "squash", "cucumber"]
SKIP_PARTS = ("grayscale", "segmented")  # PlantVillage has 3 variants; keep colour only

# condition keyword -> canonical name (checked in this order)
CONDITIONS = [("powdery", "powdery_mildew"), ("downy", "downy_mildew"), ("healthy", "healthy")]


def find_keyword(text, table):
    for key, value in table:
        if key in text:
            return value
    return None


def infer_species(text, default):
    for s in SPECIES:
        if s in text:
            return s
    return default


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def collect(sources, verbose):
    seen, items, skipped = set(), defaultdict(list), Counter()
    for name, (root, def_species, def_cond) in sources.items():
        root = Path(root)
        if not root.exists():
            print(f"[warn] {name}: {root} not found, skipping")
            continue
        for p in root.rglob("*"):
            if p.suffix.lower() not in IMG_EXT or not p.is_file():
                continue
            rel = "/".join(p.relative_to(root).parts[:-1]).lower()
            if any(s in rel for s in SKIP_PARTS):
                continue
            species = infer_species(rel, def_species)
            cond = find_keyword(rel, CONDITIONS) or def_cond
            if species is None or cond is None:
                skipped[f"{name}:{rel.split('/')[-1] or '.'}"] += 1
                continue
            h = md5(p)
            if h in seen:
                skipped["duplicate"] += 1
                continue
            seen.add(h)
            items[f"{species}_{cond}"].append(p)
    if verbose:
        print("\nSkipped / ignored folders (top 15):")
        for k, v in skipped.most_common(15):
            print(f"  {k}: {v}")
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cherry")
    ap.add_argument("--plantvillage")
    ap.add_argument("--cucumber")
    ap.add_argument("--apple")
    ap.add_argument("--out", default="./data")
    ap.add_argument("--max-per-class", type=int, default=2000, help="cap to limit imbalance")
    ap.add_argument("--min-per-class", type=int, default=30, help="drop classes smaller than this")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    sources = {}
    if a.cherry:       sources["cherry_leaves"] = (a.cherry, "cherry", None)
    if a.plantvillage: sources["plantvillage"] = (a.plantvillage, None, None)
    if a.cucumber:     sources["cucumber"] = (a.cucumber, "cucumber", None)
    if a.apple:        sources["apple_pm"] = (a.apple, "apple", "powdery_mildew")
    if not sources:
        ap.error("give at least one dataset path")

    items = collect(sources, a.verbose)
    rng = random.Random(a.seed)
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)

    labels, report = [], {}
    for label in sorted(items):
        files = items[label]
        if len(files) < a.min_per_class:
            print(f"[drop] {label}: only {len(files)} images")
            continue
        rng.shuffle(files)
        files = files[: a.max_per_class]
        n = len(files)
        n_val, n_test = int(n * 0.1), int(n * 0.1)
        splits = {
            "test": files[:n_test],
            "val": files[n_test:n_test + n_val],
            "train": files[n_test + n_val:],
        }
        for split, fl in splits.items():
            d = out / split / label
            d.mkdir(parents=True, exist_ok=True)
            for i, src in enumerate(fl):
                shutil.copy2(src, d / f"{i:05d}{src.suffix.lower()}")
        labels.append(label)
        report[label] = {k: len(v) for k, v in splits.items()}

    (out / "labels.json").write_text(json.dumps(labels, indent=2))
    print(f"\n{'class':32s} train  val  test")
    for k, v in report.items():
        print(f"{k:32s} {v['train']:5d} {v['val']:4d} {v['test']:5d}")
    print(f"\n{len(labels)} classes written to {out}")


if __name__ == "__main__":
    main()
