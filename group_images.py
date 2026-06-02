#!/usr/bin/env python3
"""
Group visually similar images using perceptual hashing + optional embedding clustering.

Usage:
    python group_images.py /path/to/images [options]

Quick start (fast, hash-based):
    python group_images.py ~/recovered_photos --threshold 10

Better quality (slower, uses image embeddings):
    python group_images.py ~/recovered_photos --method embeddings --clusters 200
"""

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image
from tqdm import tqdm


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff", ".tif"}


def load_image(path: Path) -> Image.Image | None:
    try:
        img = Image.open(path)
        img.load()
        return img.convert("RGB")
    except Exception:
        return None


# ── Method 1: Perceptual hashing ─────────────────────────────────────────────

def compute_hashes(image_paths: list[Path], hash_fn) -> dict[Path, imagehash.ImageHash]:
    hashes = {}
    for p in tqdm(image_paths, desc="Hashing images"):
        img = load_image(p)
        if img is not None:
            hashes[p] = hash_fn(img)
    return hashes


def group_by_hash(hashes: dict[Path, imagehash.ImageHash], threshold: int) -> list[list[Path]]:
    """Union-Find clustering by hash distance."""
    paths = list(hashes.keys())
    parent = list(range(len(paths)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    hash_list = [hashes[p] for p in paths]
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if hash_list[i] - hash_list[j] <= threshold:
                union(i, j)

    clusters: dict[int, list[Path]] = defaultdict(list)
    for i, p in enumerate(paths):
        clusters[find(i)].append(p)

    return list(clusters.values())


# ── Method 2: Embedding-based clustering ─────────────────────────────────────

def compute_embeddings(image_paths: list[Path]) -> tuple[list[Path], np.ndarray]:
    """Compute simple CNN-style embeddings using PIL (no heavy deps)."""
    # Use a compact color histogram + spatial grid as a lightweight embedding.
    # For much better results install torch+torchvision and swap this out.
    valid_paths = []
    vectors = []

    for p in tqdm(image_paths, desc="Computing embeddings"):
        img = load_image(p)
        if img is None:
            continue
        img = img.resize((64, 64))
        arr = np.array(img, dtype=np.float32)

        # 4×4 spatial grid, each cell: mean RGB → 48-dim vector
        cell = 16
        features = []
        for row in range(4):
            for col in range(4):
                patch = arr[row*cell:(row+1)*cell, col*cell:(col+1)*cell]
                features.extend(patch.mean(axis=(0, 1)).tolist())

        # Global color histogram (8 bins per channel) → 24-dim
        for ch in range(3):
            hist, _ = np.histogram(arr[:, :, ch], bins=8, range=(0, 256))
            features.extend((hist / hist.sum()).tolist())

        vec = np.array(features, dtype=np.float32)
        norm = np.linalg.norm(vec)
        vectors.append(vec / norm if norm > 0 else vec)
        valid_paths.append(p)

    return valid_paths, np.array(vectors)


def group_by_embeddings(image_paths: list[Path], n_clusters: int) -> list[list[Path]]:
    from sklearn.cluster import MiniBatchKMeans

    paths, embeddings = compute_embeddings(image_paths)
    if len(paths) == 0:
        return []

    k = min(n_clusters, len(paths))
    print(f"Clustering {len(paths)} images into {k} groups...")
    km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1024, n_init=3)
    labels = km.fit_predict(embeddings)

    clusters: dict[int, list[Path]] = defaultdict(list)
    for p, label in zip(paths, labels):
        clusters[int(label)].append(p)

    return list(clusters.values())


# ── Output helpers ────────────────────────────────────────────────────────────

def copy_to_output(clusters: list[list[Path]], output_dir: Path, singletons: bool):
    output_dir.mkdir(parents=True, exist_ok=True)
    skipped = 0
    written = 0

    for i, group in enumerate(tqdm(clusters, desc="Copying files")):
        if len(group) == 1 and not singletons:
            skipped += 1
            continue
        folder = output_dir / f"group_{i:04d}"
        folder.mkdir(exist_ok=True)
        for src in group:
            dest = folder / src.name
            if dest.exists():
                dest = folder / f"{src.stem}_{src.stat().st_size}{src.suffix}"
            shutil.copy2(src, dest)
            written += 1

    print(f"Wrote {written} files across {len(clusters) - skipped} groups "
          f"(skipped {skipped} singleton groups — pass --singletons to include them).")


def write_manifest(clusters: list[list[Path]], output_path: Path):
    data = [
        {"group": i, "size": len(g), "files": [str(p) for p in g]}
        for i, g in enumerate(clusters)
    ]
    output_path.write_text(json.dumps(data, indent=2))
    print(f"Manifest written to {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def collect_images(root: Path) -> list[Path]:
    return [
        p for p in root.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Group visually similar images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing images")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output directory for grouped copies (default: <input_dir>_grouped)")
    parser.add_argument("--method", choices=["hash", "embeddings"], default="hash",
                        help="Grouping method (default: hash)")

    # Hash options
    parser.add_argument("--threshold", "-t", type=int, default=10,
                        help="Max hash distance to consider images similar (0=identical, default=10)")
    parser.add_argument("--hash-type", choices=["phash", "dhash", "whash", "average"],
                        default="phash", help="Perceptual hash algorithm (default: phash)")

    # Embedding options
    parser.add_argument("--clusters", "-k", type=int, default=100,
                        help="Number of clusters for embedding method (default: 100)")

    # Output options
    parser.add_argument("--manifest-only", action="store_true",
                        help="Write a JSON manifest instead of copying files")
    parser.add_argument("--singletons", action="store_true",
                        help="Include singleton groups in output (images with no similar match)")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Also write a JSON manifest to this path")

    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"Error: {args.input_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output or args.input_dir.parent / (args.input_dir.name + "_grouped")

    print(f"Scanning {args.input_dir} for images...")
    image_paths = collect_images(args.input_dir)
    print(f"Found {len(image_paths)} images.")

    if not image_paths:
        print("No images found. Exiting.")
        sys.exit(0)

    if args.method == "hash":
        hash_fns = {
            "phash": imagehash.phash,
            "dhash": imagehash.dhash,
            "whash": imagehash.whash,
            "average": imagehash.average_hash,
        }
        hash_fn = hash_fns[args.hash_type]
        hashes = compute_hashes(image_paths, hash_fn)
        print(f"Clustering with threshold={args.threshold}...")
        clusters = group_by_hash(hashes, args.threshold)
    else:
        clusters = group_by_embeddings(image_paths, args.clusters)

    # Sort each group and sort groups by size (largest first)
    clusters = [sorted(g) for g in clusters]
    clusters.sort(key=len, reverse=True)

    sizes = [len(c) for c in clusters]
    print(f"\nResults: {len(clusters)} groups | "
          f"largest={max(sizes)} | median={int(np.median(sizes))} | "
          f"singletons={sum(1 for s in sizes if s == 1)}")

    if args.manifest:
        write_manifest(clusters, args.manifest)

    if args.manifest_only:
        if not args.manifest:
            write_manifest(clusters, output_dir.with_suffix(".json"))
    else:
        copy_to_output(clusters, output_dir, singletons=args.singletons)

    if not args.manifest_only:
        print(f"\nDone. Grouped images are in: {output_dir}")


if __name__ == "__main__":
    main()
