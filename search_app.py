#!/usr/bin/env python3
"""
Visual similarity search + group-type organizer.

Setup (one-time):
    python -m pip install streamlit Pillow numpy

Run:
    python -m streamlit run search_app.py

What it does
------------
1. Builds a one-time embedding index of a photos folder.
2. "Search by image" — find images that look like one (or several) examples.
3. "Group types" — define named categories (e.g. "memes", "receipts",
   "selfies") using MULTIPLE example images each. A group type is the averaged
   fingerprint of its examples, so it captures a *concept* far better than any
   single image. Search the whole library by a group type at any time.
4. "Classify library" — assign every image to its nearest group type and
   export the results into one folder per type.

Note: the embedding here is identical to group_images.py, so an index built by
either tool is compatible.
"""

import io
import pickle
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif", ".gif"}
INDEX_FILENAME = ".image_index.pkl"
GROUP_TYPES_FILENAME = ".group_types.pkl"
THUMB_SIZE = 96


# ── Embedding (identical to group_images.py) ─────────────────────────────────

def embed(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((64, 64))
    arr = np.array(img, dtype=np.float32)

    features = []
    cell = 16
    for row in range(4):
        for col in range(4):
            patch = arr[row * cell:(row + 1) * cell, col * cell:(col + 1) * cell]
            features.extend(patch.mean(axis=(0, 1)).tolist())

    for ch in range(3):
        hist, _ = np.histogram(arr[:, :, ch], bins=8, range=(0, 256))
        features.extend((hist / hist.sum()).tolist())

    vec = np.array(features, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def normalize(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


# ── Image loading helpers ─────────────────────────────────────────────────────

def collect_images(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED]


def load_image_safe(src) -> Image.Image | None:
    """Open a path or an uploaded file object into a fully-loaded PIL image."""
    try:
        img = Image.open(src)
        img.load()
        return img
    except Exception:
        return None


def make_thumb(img: Image.Image) -> bytes:
    thumb = img.convert("RGB").copy()
    thumb.thumbnail((THUMB_SIZE, THUMB_SIZE))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


# ── Index build / load ────────────────────────────────────────────────────────

def build_index(photos_dir: Path) -> dict:
    paths = collect_images(photos_dir)
    if not paths:
        st.error("No images found in that directory.")
        return {}

    bar = st.progress(0.0, text="Building index…")
    valid_paths, vectors = [], []
    for i, p in enumerate(paths):
        bar.progress((i + 1) / len(paths), text=f"Indexing {i + 1}/{len(paths)}: {p.name}")
        img = load_image_safe(p)
        if img is None:
            continue
        vectors.append(embed(img))
        valid_paths.append(str(p))
    bar.empty()

    index = {"paths": valid_paths, "embeddings": np.array(vectors, dtype=np.float32)}
    (photos_dir / INDEX_FILENAME).write_bytes(pickle.dumps(index))
    st.success(f"Index built: {len(valid_paths):,} images.")
    return index


def load_index(photos_dir: Path) -> dict | None:
    fp = photos_dir / INDEX_FILENAME
    if not fp.exists():
        return None
    try:
        return pickle.loads(fp.read_bytes())
    except Exception:
        return None


# ── Group types ───────────────────────────────────────────────────────────────
# Structure:  { name: {"names": [str], "paths": [str|None],
#                       "thumbs": [bytes], "embeddings": np.ndarray (k, D)} }

def load_group_types(photos_dir: Path) -> dict:
    fp = photos_dir / GROUP_TYPES_FILENAME
    if not fp.exists():
        return {}
    try:
        return pickle.loads(fp.read_bytes())
    except Exception:
        return {}


def save_group_types(photos_dir: Path, gts: dict) -> None:
    (photos_dir / GROUP_TYPES_FILENAME).write_bytes(pickle.dumps(gts))


def prototype(gt: dict) -> np.ndarray:
    """The averaged, re-normalized fingerprint of a group type's examples."""
    return normalize(gt["embeddings"].mean(axis=0))


def add_references(gts: dict, name: str, items: list[tuple]) -> None:
    """items: list of (label, embedding, thumb_bytes, source_path_or_None)."""
    labels = [it[0] for it in items]
    embs = np.array([it[1] for it in items], dtype=np.float32)
    thumbs = [it[2] for it in items]
    paths = [it[3] for it in items]

    gt = gts.get(name)
    if gt is None:
        gts[name] = {"names": labels, "embeddings": embs, "thumbs": thumbs, "paths": paths}
    else:
        gt["names"].extend(labels)
        gt["embeddings"] = np.vstack([gt["embeddings"], embs])
        gt["thumbs"].extend(thumbs)
        gt["paths"].extend(paths)


# ── Search & classification ───────────────────────────────────────────────────

def rank(query_vec: np.ndarray, index: dict, top_k: int, skip: set | None = None):
    scores = index["embeddings"] @ query_vec
    order = np.argsort(scores)[::-1]
    skip = skip or set()
    out = []
    for i in order:
        p = index["paths"][i]
        try:
            resolved = str(Path(p).resolve())
        except Exception:
            resolved = p
        if resolved in skip:
            continue
        out.append((p, float(scores[i])))
        if len(out) >= top_k:
            break
    return out


def classify_all(index: dict, gts: dict, threshold: float):
    names = list(gts.keys())
    protos = np.array([prototype(gts[n]) for n in names], dtype=np.float32)  # (T, D)
    sims = index["embeddings"] @ protos.T                                     # (N, T)
    best = sims.argmax(axis=1)
    best_score = sims.max(axis=1)
    labels = [names[best[i]] if best_score[i] >= threshold else None
              for i in range(len(index["paths"]))]
    return labels, best_score


def export_classification(index: dict, labels: list, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    bar = st.progress(0.0, text="Exporting…")
    total = len(index["paths"])
    for i, (path, label) in enumerate(zip(index["paths"], labels)):
        bar.progress((i + 1) / total, text=f"Exporting {i + 1}/{total}")
        folder = out_dir / (label if label else "_unsorted")
        folder.mkdir(exist_ok=True)
        src = Path(path)
        dest = folder / src.name
        if dest.exists():
            dest = folder / f"{src.stem}_{src.stat().st_size}{src.suffix}"
        try:
            shutil.copy2(src, dest)
            counts[label or "_unsorted"] += 1
        except Exception:
            pass
    bar.empty()
    return dict(counts)


# ── UI helpers ────────────────────────────────────────────────────────────────

def show_results(results: list, cols_count: int):
    if not results:
        st.warning("No matches.")
        return
    st.markdown(f"**{len(results)} matches**")
    grid = st.columns(cols_count)
    for i, (path, score) in enumerate(results):
        img = load_image_safe(path)
        if img is None:
            continue
        with grid[i % cols_count]:
            st.image(img, caption=f"{score:.2f} · {Path(path).name}",
                     use_container_width=True)


def gather_examples(uploaded, folder_str: str) -> list[tuple]:
    """Embed uploaded files and/or every image in a folder.
    Returns list of (label, embedding, thumb, source_path_or_None)."""
    items = []
    for f in uploaded or []:
        img = load_image_safe(f)
        if img is not None:
            items.append((f.name, embed(img), make_thumb(img), None))
    if folder_str and Path(folder_str).is_dir():
        for p in collect_images(Path(folder_str)):
            img = load_image_safe(p)
            if img is not None:
                items.append((p.name, embed(img), make_thumb(img), str(p)))
    return items


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Image Organizer", layout="wide", page_icon="🗂️")
    st.title("🗂️ Image Search & Group-Type Organizer")

    with st.sidebar:
        st.header("Library")
        photos_dir_str = st.text_input(
            "Photos folder", placeholder=r"C:\Users\you\Desktop\Photos",
            help="Folder containing your recovered images.",
        )
        st.divider()
        st.header("Display")
        top_k = st.slider("Results to show", 5, 120, 24, 1)
        cols_count = st.select_slider("Columns", options=[3, 4, 5, 6, 7, 8], value=6)

    if not photos_dir_str:
        st.info("Enter your photos folder in the sidebar to begin.")
        return

    photos_dir = Path(photos_dir_str)
    if not photos_dir.is_dir():
        st.error("That folder doesn't exist.")
        return

    # Reload state when the folder changes.
    if st.session_state.get("folder") != str(photos_dir):
        st.session_state["folder"] = str(photos_dir)
        st.session_state["index"] = load_index(photos_dir)
        st.session_state["group_types"] = load_group_types(photos_dir)
        st.session_state.pop("classification", None)

    index = st.session_state.get("index")
    gts = st.session_state.get("group_types", {})

    # Library status + build control.
    c1, c2 = st.columns([3, 1])
    with c1:
        if index:
            st.success(f"📁 {len(index['paths']):,} images indexed.")
        else:
            st.warning("No index found for this folder yet — build one.")
    with c2:
        if st.button("Build / Rebuild index", use_container_width=True):
            st.session_state["index"] = build_index(photos_dir)
            index = st.session_state["index"]

    tab_search, tab_groups, tab_classify = st.tabs(
        ["🔍 Search by image", "🏷️ Group types", "🗂️ Classify library"]
    )

    # ── Tab 1: search by one or more example images ───────────────────────────
    with tab_search:
        st.caption("Find images similar to one or more examples. Add several "
                   "to search by their combined look.")
        uploaded = st.file_uploader(
            "Example image(s)", type=["jpg", "jpeg", "png", "bmp", "webp"],
            accept_multiple_files=True, key="search_up",
        )
        path_in = st.text_input("…or a file path", key="search_path",
                                placeholder=r"C:\...\Photos\img001.jpg")

        queries = []  # (label, embedding, thumb)
        skip = set()
        for f in uploaded or []:
            img = load_image_safe(f)
            if img is not None:
                queries.append((f.name, embed(img), make_thumb(img)))
        if path_in:
            if Path(path_in).is_file():
                img = load_image_safe(path_in)
                if img is not None:
                    queries.append((Path(path_in).name, embed(img), make_thumb(img)))
                    skip.add(str(Path(path_in).resolve()))
            else:
                st.warning("That file path was not found.")

        if queries:
            st.markdown("**Query**")
            qcols = st.columns(min(len(queries), 8))
            for i, (label, _, thumb) in enumerate(queries):
                qcols[i % len(qcols)].image(thumb, caption=label)

            if index:
                qvec = normalize(np.mean([q[1] for q in queries], axis=0))
                show_results(rank(qvec, index, top_k, skip=skip), cols_count)
            else:
                st.info("Build the index first (button above).")

    # ── Tab 2: define & search group types ────────────────────────────────────
    with tab_groups:
        left, right = st.columns([2, 3])

        with left:
            st.subheader("Define a group type")
            name = st.text_input("Name", placeholder="e.g. memes, receipts, selfies")
            g_up = st.file_uploader(
                "Upload example images", type=["jpg", "jpeg", "png", "bmp", "webp"],
                accept_multiple_files=True, key="gt_up",
            )
            g_folder = st.text_input(
                "…or add every image in a folder", key="gt_folder",
                placeholder=r"C:\...\some_examples",
            )
            if st.button("➕ Add examples to group type", disabled=not name.strip()):
                with st.spinner("Embedding examples…"):
                    items = gather_examples(g_up, g_folder)
                if items:
                    add_references(gts, name.strip(), items)
                    save_group_types(photos_dir, gts)
                    st.success(f"Added {len(items)} example(s) to '{name.strip()}'.")
                    st.rerun()
                else:
                    st.warning("No valid images supplied.")

            st.divider()
            st.subheader("Existing group types")
            if not gts:
                st.caption("None defined yet.")
            for gname in list(gts.keys()):
                gt = gts[gname]
                with st.expander(f"{gname} · {len(gt['embeddings'])} examples"):
                    thumbs = gt.get("thumbs", [])
                    tcols = st.columns(6)
                    for i, th in enumerate(thumbs[:18]):
                        tcols[i % 6].image(th)
                    if len(thumbs) > 18:
                        st.caption(f"…and {len(thumbs) - 18} more.")
                    b1, b2 = st.columns(2)
                    if b1.button("🔎 Search library", key=f"use_{gname}"):
                        st.session_state["active_group"] = gname
                    if b2.button("🗑️ Delete", key=f"del_{gname}"):
                        del gts[gname]
                        save_group_types(photos_dir, gts)
                        st.rerun()

        with right:
            st.subheader("Search library by group type")
            if not gts:
                st.info("Define a group type on the left first.")
            elif not index:
                st.info("Build the index first (button above).")
            else:
                options = list(gts.keys())
                active = st.session_state.get("active_group")
                idx0 = options.index(active) if active in options else 0
                chosen = st.selectbox("Group type", options, index=idx0)
                exclude = st.checkbox("Exclude example images from results")

                skip = set()
                if exclude:
                    for p in gts[chosen].get("paths", []):
                        if p:
                            try:
                                skip.add(str(Path(p).resolve()))
                            except Exception:
                                pass
                show_results(rank(prototype(gts[chosen]), index, top_k, skip=skip),
                             cols_count)

    # ── Tab 3: classify the whole library ─────────────────────────────────────
    with tab_classify:
        st.caption("Assign every image to its single nearest group type.")
        if not gts:
            st.info("Define some group types first.")
        elif not index:
            st.info("Build the index first (button above).")
        else:
            threshold = st.slider(
                "Minimum similarity to assign (below this → _unsorted)",
                0.0, 1.0, 0.0, 0.01,
                help="Raise this to leave weak matches unsorted. Start at 0, "
                     "then increase after inspecting the counts.",
            )
            if st.button("Classify all images"):
                with st.spinner("Classifying…"):
                    st.session_state["classification"] = classify_all(index, gts, threshold)

            if "classification" in st.session_state:
                labels, _ = st.session_state["classification"]
                counts = Counter(l or "_unsorted" for l in labels)
                total = sum(counts.values())
                rows = [
                    {"group type": k, "images": v, "share": f"{v / total:.0%}"}
                    for k, v in sorted(counts.items(), key=lambda kv: -kv[1])
                ]
                st.markdown(f"**Results · {total:,} images**")
                st.dataframe(rows, use_container_width=True, hide_index=True)

                st.divider()
                default_out = str(photos_dir.parent / (photos_dir.name + "_sorted"))
                out_str = st.text_input("Export folder (files are copied, not moved)",
                                        value=default_out)
                if st.button("Export into per-type folders"):
                    result = export_classification(index, labels, Path(out_str))
                    st.success(f"Exported {sum(result.values()):,} files to {out_str}")
                    st.json(result)


if __name__ == "__main__":
    main()
