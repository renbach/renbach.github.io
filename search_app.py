#!/usr/bin/env python3
"""
Visual similarity search app.

Setup (one-time):
    python -m pip install streamlit Pillow numpy

Run:
    streamlit run search_app.py
"""

import io
import math
import pickle
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif", ".gif"}
INDEX_FILENAME = ".image_index.pkl"


# ── Embedding (matches group_images.py) ──────────────────────────────────────

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


# ── Index build / load ────────────────────────────────────────────────────────

def build_index(photos_dir: Path) -> dict:
    paths = [p for p in photos_dir.rglob("*") if p.suffix.lower() in SUPPORTED]
    if not paths:
        st.error("No images found in that directory.")
        return {}

    bar = st.progress(0.0, text="Building index…")
    valid_paths, vectors = [], []

    for i, p in enumerate(paths):
        bar.progress((i + 1) / len(paths), text=f"Indexing {i + 1}/{len(paths)}: {p.name}")
        try:
            img = Image.open(p)
            img.load()
            vectors.append(embed(img))
            valid_paths.append(str(p))
        except Exception:
            continue

    bar.empty()
    index = {"paths": valid_paths, "embeddings": np.array(vectors, dtype=np.float32)}
    index_path = photos_dir / INDEX_FILENAME
    index_path.write_bytes(pickle.dumps(index))
    st.success(f"Index built: {len(valid_paths):,} images saved to {index_path}")
    return index


def load_index(photos_dir: Path) -> dict | None:
    index_path = photos_dir / INDEX_FILENAME
    if not index_path.exists():
        return None
    with open(index_path, "rb") as f:
        return pickle.load(f)


# ── Search ────────────────────────────────────────────────────────────────────

def search(query_vec: np.ndarray, index: dict, top_k: int, skip_path: str | None = None):
    scores = index["embeddings"] @ query_vec
    order = np.argsort(scores)[::-1]
    results = []
    for i in order:
        p = index["paths"][i]
        if skip_path and Path(p).resolve() == Path(skip_path).resolve():
            continue
        results.append((p, float(scores[i])))
        if len(results) >= top_k:
            break
    return results


def load_image_safe(path: str) -> Image.Image | None:
    try:
        img = Image.open(path)
        img.load()
        return img
    except Exception:
        return None


def img_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# ── UI ────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Image Search", layout="wide", page_icon="🔍")
    st.title("Image Similarity Search")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Index")
        photos_dir_str = st.text_input(
            "Photos folder",
            placeholder=r"C:\Users\you\Desktop\Photos",
            help="Folder containing your recovered images.",
        )
        top_k = st.slider("Results to show", min_value=5, max_value=60, value=20, step=5)
        cols_count = st.select_slider("Columns", options=[3, 4, 5, 6], value=5)

        index = st.session_state.get("index")

        if photos_dir_str:
            photos_dir = Path(photos_dir_str)
            if not photos_dir.is_dir():
                st.error("Directory not found.")
            else:
                idx_path = photos_dir / INDEX_FILENAME
                if idx_path.exists() and index is None:
                    with st.spinner("Loading index…"):
                        index = load_index(photos_dir)
                        st.session_state["index"] = index
                    st.success(f"Loaded {len(index['paths']):,} images")

                if st.button("Build / Rebuild Index"):
                    index = build_index(photos_dir)
                    st.session_state["index"] = index

    # ── Query ─────────────────────────────────────────────────────────────────
    st.subheader("Query image")
    query_tab, path_tab = st.tabs(["Upload a file", "Enter a file path"])

    query_img = None
    skip_path = None

    with query_tab:
        uploaded = st.file_uploader(
            "Choose an image", type=["jpg", "jpeg", "png", "bmp", "webp"]
        )
        if uploaded:
            query_img = Image.open(uploaded).convert("RGB")

    with path_tab:
        path_input = st.text_input(
            "Full path to image",
            placeholder=r"C:\Users\you\Desktop\Photos\img001.jpg",
        )
        if path_input and Path(path_input).is_file():
            loaded = load_image_safe(path_input)
            if loaded:
                query_img = loaded.convert("RGB")
                skip_path = path_input
            else:
                st.error("Could not open that file.")
        elif path_input:
            st.warning("File not found.")

    # ── Results ───────────────────────────────────────────────────────────────
    if query_img is not None:
        st.divider()
        left, right = st.columns([1, 4])

        with left:
            st.markdown("**Query**")
            st.image(query_img, use_container_width=True)

        with right:
            if index is None:
                st.info("Set your photos folder and build (or load) an index in the sidebar.")
            else:
                query_vec = embed(query_img)
                results = search(query_vec, index, top_k, skip_path=skip_path)

                st.markdown(f"**Top {len(results)} matches**")
                cols = st.columns(cols_count)
                for idx, (path, score) in enumerate(results):
                    img = load_image_safe(path)
                    if img is None:
                        continue
                    with cols[idx % cols_count]:
                        st.image(
                            img,
                            caption=f"{score:.2f}  {Path(path).name}",
                            use_container_width=True,
                        )

    elif index is None:
        st.info(
            "1. Enter your photos folder path in the sidebar.\n"
            "2. Click **Build Index** (one-time, ~2–5 min for 20K images).\n"
            "3. Upload or enter the path of any image to find similar ones."
        )


if __name__ == "__main__":
    main()
