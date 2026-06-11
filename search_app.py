#!/usr/bin/env python3
"""
Visual similarity search + group-type organizer (images AND videos).

Setup (one-time):
    python -m pip install streamlit Pillow numpy
    python -m pip install opencv-python        # optional, enables video support

Run:
    python -m streamlit run search_app.py

What it does
------------
1. Builds a one-time embedding index of a media folder. Videos are
   fingerprinted by sampling several frames and averaging their embeddings.
2. "Search by image" — find media that look like one (or several) example
   images or videos.
3. "Group types" — define named categories (e.g. "memes", "receipts",
   "selfies") using MULTIPLE example images/videos each. A group type is the
   averaged fingerprint of its examples, so it captures a *concept* far
   better than any single example. Search the whole library by a group type.
4. "Classify library" — assign every file to its nearest group type and
   export the results into one folder per type.

Note: the image embedding is identical to group_images.py, so an image index
built by either tool is compatible (rebuild here to add videos).
"""

import io
import os
import pickle
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif", ".gif"}
VIDEO_SUPPORTED = {".mp4", ".mov", ".avi", ".3gp", ".mkv", ".webm", ".m4v",
                   ".wmv", ".mpg", ".mpeg"}
INDEX_FILENAME = ".image_index.pkl"
GROUP_TYPES_FILENAME = ".group_types.pkl"
THUMB_SIZE = 96
POSTER_SIZE = 384
N_VIDEO_FRAMES = 5


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


def collect_media(root: Path) -> list[Path]:
    exts = SUPPORTED | VIDEO_SUPPORTED
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]


def load_image_safe(src) -> Image.Image | None:
    """Open a path or an uploaded file object, applying EXIF rotation if present."""
    try:
        img = Image.open(src)
        img.load()
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
        return img
    except Exception:
        return None


def make_thumb(img: Image.Image, size: int = THUMB_SIZE) -> bytes:
    thumb = img.convert("RGB").copy()
    thumb.thumbnail((size, size))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


# ── Video helpers ─────────────────────────────────────────────────────────────

def is_video(path) -> bool:
    return Path(path).suffix.lower() in VIDEO_SUPPORTED


_ROTATE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
           270: cv2.ROTATE_90_COUNTERCLOCKWISE} if HAS_CV2 else {}


def _apply_video_rotation(frame: np.ndarray, cap) -> np.ndarray:
    """Rotate a raw BGR frame to match the video's stored orientation metadata."""
    try:
        angle = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
        code = _ROTATE.get(angle % 360)
        if code is not None:
            return cv2.rotate(frame, code)
    except Exception:
        pass
    return frame


def sample_video_frames(path, n: int = N_VIDEO_FRAMES) -> list[Image.Image]:
    """Grab n frames spread evenly through the video as PIL images."""
    if not HAS_CV2:
        return []
    frames = []
    cap = cv2.VideoCapture(str(path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 0:
            for i in range(n):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 0.5) / n))
                ok, frame = cap.read()
                if ok:
                    frame = _apply_video_rotation(frame, cap)
                    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        else:
            # Some containers don't report a frame count — read sequentially.
            raw = []
            while len(raw) < 150:
                ok, frame = cap.read()
                if not ok:
                    break
                raw.append(frame)
            step = max(1, len(raw) // n)
            for frame in raw[::step][:n]:
                frame = _apply_video_rotation(frame, cap)
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    except Exception:
        pass
    finally:
        cap.release()
    return frames


def embed_video(path) -> tuple[np.ndarray | None, Image.Image | None]:
    """Average the embeddings of sampled frames into one video fingerprint.
    Returns (embedding, poster_frame) — (None, None) if unreadable."""
    frames = sample_video_frames(path)
    if not frames:
        return None, None
    vec = normalize(np.mean([embed(f) for f in frames], axis=0))
    poster = frames[len(frames) // 2]
    return vec, poster


def embed_uploaded_video(uploaded) -> tuple[np.ndarray | None, Image.Image | None]:
    """OpenCV needs a real file, so spool an uploaded video to a temp path."""
    suffix = Path(uploaded.name).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(uploaded.getbuffer())
        tmp.close()
        return embed_video(tmp.name)
    finally:
        os.unlink(tmp.name)


# ── Index build / load ────────────────────────────────────────────────────────

def build_index(photos_dir: Path) -> dict:
    paths = collect_media(photos_dir)
    if not paths:
        st.error("No images or videos found in that directory.")
        return {}

    bar = st.progress(0.0, text="Building index…")
    valid_paths, vectors = [], []
    video_thumbs: dict[str, bytes] = {}
    skipped_videos = 0
    for i, p in enumerate(paths):
        bar.progress((i + 1) / len(paths), text=f"Indexing {i + 1}/{len(paths)}: {p.name}")
        if is_video(p):
            if not HAS_CV2:
                skipped_videos += 1
                continue
            vec, poster = embed_video(p)
            if vec is None:
                continue
            video_thumbs[str(p)] = make_thumb(poster, POSTER_SIZE)
            vectors.append(vec)
            valid_paths.append(str(p))
        else:
            img = load_image_safe(p)
            if img is None:
                continue
            vectors.append(embed(img))
            valid_paths.append(str(p))
    bar.empty()

    index = {
        "paths": valid_paths,
        "embeddings": np.array(vectors, dtype=np.float32),
        "video_thumbs": video_thumbs,
    }
    (photos_dir / INDEX_FILENAME).write_bytes(pickle.dumps(index))
    n_vid = len(video_thumbs)
    st.success(f"Index built: {len(valid_paths) - n_vid:,} images"
               + (f" + {n_vid:,} videos" if n_vid else "") + ".")
    if skipped_videos:
        st.warning(f"{skipped_videos} video(s) skipped — run "
                   "`python -m pip install opencv-python`, restart, and rebuild "
                   "to include them.")
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


def remove_references(gts: dict, name: str, indices: list[int]) -> None:
    """Remove examples at the given indices; deletes the type if it becomes empty."""
    gt = gts[name]
    for i in sorted(set(indices), reverse=True):
        gt["names"].pop(i)
        gt["thumbs"].pop(i)
        gt["paths"].pop(i)
        gt["embeddings"] = np.delete(gt["embeddings"], i, axis=0)
    if gt["embeddings"].shape[0] == 0:
        del gts[name]


def outlier_indices(gt: dict, keep_pct: float) -> list[int]:
    """Indices of the bottom (1-keep_pct) examples ranked by similarity to prototype."""
    sims = gt["embeddings"] @ prototype(gt)
    n_remove = max(0, int(round(len(sims) * (1.0 - keep_pct))))
    return list(np.argsort(sims)[:n_remove].tolist())


# ── Search & classification ───────────────────────────────────────────────────

def rank(query_vec: np.ndarray, index: dict, top_k: int,
         skip: set | None = None, media: str = "All"):
    scores = index["embeddings"] @ query_vec
    order = np.argsort(scores)[::-1]
    skip = skip or set()
    out = []
    for i in order:
        p = index["paths"][i]
        if media == "Videos only" and not is_video(p):
            continue
        if media == "Images only" and is_video(p):
            continue
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

def show_results(results: list, cols_count: int, index: dict | None = None):
    if not results:
        st.warning("No matches.")
        return
    video_thumbs = (index or {}).get("video_thumbs", {})
    st.markdown(f"**{len(results)} matches**")
    grid = st.columns(cols_count)
    for i, (path, score) in enumerate(results):
        with grid[i % cols_count]:
            if is_video(path):
                caption = f"🎬 {score:.2f} · {Path(path).name}"
                thumb = video_thumbs.get(path)
                if thumb is None and HAS_CV2:
                    frames = sample_video_frames(path, 1)
                    thumb = make_thumb(frames[0], POSTER_SIZE) if frames else None
                if thumb is not None:
                    st.image(thumb, caption=caption, width="stretch")
                else:
                    st.caption(caption)
                with st.expander("▶ Play"):
                    st.video(path)
            else:
                img = load_image_safe(path)
                if img is None:
                    continue
                st.image(img, caption=f"{score:.2f} · {Path(path).name}",
                         width="stretch")


def gather_examples(uploaded, folder_str: str) -> list[tuple]:
    """Embed uploaded files and/or every image/video in a folder.
    Returns list of (label, embedding, thumb, source_path_or_None)."""
    items = []
    for f in uploaded or []:
        if is_video(f.name):
            vec, poster = embed_uploaded_video(f)
            if vec is not None:
                items.append((f.name, vec, make_thumb(poster), None))
        else:
            img = load_image_safe(f)
            if img is not None:
                items.append((f.name, embed(img), make_thumb(img), None))
    if folder_str and Path(folder_str).is_dir():
        for p in collect_media(Path(folder_str)):
            if is_video(p):
                vec, poster = embed_video(p)
                if vec is not None:
                    items.append((p.name, vec, make_thumb(poster), str(p)))
            else:
                img = load_image_safe(p)
                if img is not None:
                    items.append((p.name, embed(img), make_thumb(img), str(p)))
    return items


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Media Organizer", layout="wide", page_icon="🗂️")
    st.title("🗂️ Media Search & Group-Type Organizer")

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
        media_filter = st.radio("Result type", ["All", "Images only", "Videos only"],
                                horizontal=True)
        if not HAS_CV2:
            st.divider()
            st.caption("🎬 Video support is off — run "
                       "`python -m pip install opencv-python`, restart, and "
                       "rebuild the index to include videos.")

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
            n_vid = sum(1 for p in index["paths"] if is_video(p))
            n_img = len(index["paths"]) - n_vid
            st.success(f"📁 {n_img:,} images"
                       + (f" + {n_vid:,} videos" if n_vid else "") + " indexed.")
        else:
            st.warning("No index found for this folder yet — build one.")
    with c2:
        if st.button("Build / Rebuild index", width="stretch"):
            st.session_state["index"] = build_index(photos_dir)
            index = st.session_state["index"]

    tab_search, tab_groups, tab_classify = st.tabs(
        ["🔍 Search by image", "🏷️ Group types", "🗂️ Classify library"]
    )

    # ── Tab 1: search by one or more example images/videos ───────────────────
    with tab_search:
        st.caption("Find media similar to one or more examples (images or "
                   "videos). Add several to search by their combined look.")
        up_types = ["jpg", "jpeg", "png", "bmp", "webp"]
        if HAS_CV2:
            up_types += ["mp4", "mov", "avi", "3gp", "mkv", "webm", "m4v"]
        uploaded = st.file_uploader(
            "Example image(s) or video(s)", type=up_types,
            accept_multiple_files=True, key="search_up",
        )
        path_in = st.text_input("…or a file path", key="search_path",
                                placeholder=r"C:\...\Photos\img001.jpg")

        queries = []  # (label, embedding, thumb)
        skip = set()
        for f in uploaded or []:
            if is_video(f.name):
                vec, poster = embed_uploaded_video(f)
                if vec is not None:
                    queries.append((f"🎬 {f.name}", vec, make_thumb(poster)))
            else:
                img = load_image_safe(f)
                if img is not None:
                    queries.append((f.name, embed(img), make_thumb(img)))
        if path_in:
            if Path(path_in).is_file():
                if is_video(path_in):
                    if not HAS_CV2:
                        st.warning("Install opencv-python to query by video.")
                    else:
                        vec, poster = embed_video(path_in)
                        if vec is not None:
                            queries.append((f"🎬 {Path(path_in).name}", vec,
                                            make_thumb(poster)))
                            skip.add(str(Path(path_in).resolve()))
                else:
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
                show_results(rank(qvec, index, top_k, skip=skip, media=media_filter),
                             cols_count, index)
            else:
                st.info("Build the index first (button above).")

    # ── Tab 2: define & search group types ────────────────────────────────────
    with tab_groups:
        left, right = st.columns([2, 3])

        with left:
            st.subheader("Define a group type")
            name = st.text_input("Name", placeholder="e.g. memes, receipts, selfies")
            g_up = st.file_uploader(
                "Upload example images/videos", type=up_types,
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
                n_ex = len(gt["embeddings"])
                with st.expander(f"{gname} · {n_ex} examples"):
                    thumbs = gt.get("thumbs", [])
                    names_list = gt.get("names", [])

                    # Thumbnail grid (numbered so users can cross-reference).
                    preview_n = min(len(thumbs), 30)
                    tcols = st.columns(6)
                    for i in range(preview_n):
                        tcols[i % 6].image(thumbs[i], caption=f"#{i}")
                    if len(thumbs) > preview_n:
                        st.caption(f"…and {len(thumbs) - preview_n} more.")

                    st.divider()

                    # ── Manual removal ────────────────────────────────────
                    options = [f"#{i} · {nm}" for i, nm in enumerate(names_list)]
                    to_remove = st.multiselect(
                        "Select examples to remove", options, key=f"rm_{gname}",
                        help="Pick by number — the #s match the thumbnail captions above.",
                    )
                    manual_indices = [int(o.split("·")[0].strip("# ")) for o in to_remove]

                    # ── Auto-prune ────────────────────────────────────────
                    keep_pct = st.slider(
                        "Auto-prune: keep top % by closeness to centroid",
                        10, 100, 80, 5, key=f"prune_{gname}", format="%d%%",
                        help="Lower = remove more outliers. 80% keeps the 80% most "
                             "representative examples and flags the rest.",
                    )
                    auto_idx = outlier_indices(gt, keep_pct / 100.0)
                    if auto_idx:
                        st.caption(
                            f"{len(auto_idx)} outlier(s) flagged "
                            f"(bottom {100 - keep_pct}% by centroid similarity): "
                            + ", ".join(f"#{i}" for i in sorted(auto_idx)[:10])
                            + ("…" if len(auto_idx) > 10 else "")
                        )

                    # ── Action buttons ────────────────────────────────────
                    b1, b2, b3, b4 = st.columns(4)
                    if b1.button("🔎 Search library", key=f"use_{gname}"):
                        st.session_state["active_group"] = gname

                    if b2.button(
                        f"✂️ Remove {len(to_remove)} selected",
                        key=f"rm_sel_{gname}",
                        disabled=not to_remove,
                    ):
                        remove_references(gts, gname, manual_indices)
                        save_group_types(photos_dir, gts)
                        st.rerun()

                    if b3.button(
                        f"🧹 Auto-prune {len(auto_idx)}",
                        key=f"prune_btn_{gname}",
                        disabled=not auto_idx,
                    ):
                        remove_references(gts, gname, auto_idx)
                        save_group_types(photos_dir, gts)
                        st.rerun()

                    if b4.button("🗑️ Delete type", key=f"del_{gname}"):
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
                show_results(rank(prototype(gts[chosen]), index, top_k, skip=skip,
                                  media=media_filter),
                             cols_count, index)

    # ── Tab 3: classify the whole library ─────────────────────────────────────
    with tab_classify:
        st.caption("Assign every image and video to its single nearest group type.")
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
                    {"group type": k, "files": v, "share": f"{v / total:.0%}"}
                    for k, v in sorted(counts.items(), key=lambda kv: -kv[1])
                ]
                st.markdown(f"**Results · {total:,} files**")
                st.dataframe(rows, width="stretch", hide_index=True)

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
