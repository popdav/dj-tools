#!/usr/bin/env python3
"""
song-sort-by-bpm-and-keys.py — Sorts songs by Genre into a destination folder.
Also computes/sets BPM and musical Key (if missing) and writes a CSV report.

Features:
- Reads existing tags (genre/BPM/key). Treats "Other" and "Music" as missing.
- Estimates BPM and musical key from audio when missing.
- Infers Genre using BOTH BPM and harmonic key (Camelot) if --infer-genre is on.
- Writes BPM/Key/Genre back to tags when --write-tags is set.
- Copies tracks into dest/<Genre>/, with collision-safe filenames.
- Emits a CSV report of everything processed.

Supports: .mp3, .m4a/.mp4, .flac, .wav, .ogg, .opus, .aac
Requires: mutagen, librosa, soundfile, numpy, tqdm, pandas

Usage:
  python song-sort-by-bpm-and-keys.py "/path/to/Music" "/path/to/Sorted" \
    --write-tags --infer-genre --report "/path/to/report.csv"

Windows tip: avoid a trailing backslash on quoted paths (or use double backslash).
"""

import argparse
import csv
import os
import sys
import shutil
import math
from pathlib import Path
from typing import Optional, Tuple, Dict, Iterable

import numpy as np

# Third-party deps
try:
    import librosa
    from mutagen import File as MFile
    from mutagen.easyid3 import EasyID3
    from tqdm import tqdm
    import pandas as pd
except Exception as e:
    print("Missing dependencies. Install with:\n  pip install mutagen librosa soundfile numpy tqdm pandas", file=sys.stderr)
    raise

AUDIO_EXTS = {".mp3", ".m4a", ".mp4", ".flac", ".wav", ".ogg", ".opus", ".aac"}

# --- Key detection (Krumhansl/Schmuckler style via chroma) ---

KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=float)
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=float)
PITCH_CLASSES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
CAMELOT_MAJOR = ["8B","3B","10B","5B","12B","7B","2B","9B","4B","11B","6B","1B"]
CAMELOT_MINOR = ["5A","12A","7A","2A","9A","4A","11A","6A","1A","8A","3A","10A"]

def detect_key(y: np.ndarray, sr: int) -> Tuple[Optional[str], Optional[str]]:
    """Return (musical_key like 'Am' or 'A', camelot like '8A'/'8B')."""
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        profile_major = KK_MAJOR / KK_MAJOR.sum()
        profile_minor = KK_MINOR / KK_MINOR.sum()
        chroma_mean = np.mean(chroma, axis=1)
        norm = np.linalg.norm(chroma_mean) + 1e-9
        chroma_norm = chroma_mean / norm

        best_key, best_mode, best_corr = None, None, -np.inf
        for i in range(12):
            corr_maj = float(np.dot(chroma_norm, np.roll(profile_major, i)))
            if corr_maj > best_corr:
                best_corr = corr_maj
                best_key, best_mode = (PITCH_CLASSES[i], "major")
            corr_min = float(np.dot(chroma_norm, np.roll(profile_minor, i)))
            if corr_min > best_corr:
                best_corr = corr_min
                best_key, best_mode = (PITCH_CLASSES[i], "minor")

        if best_key is None:
            return None, None

        if best_mode == "major":
            camelot = CAMELOT_MAJOR[PITCH_CLASSES.index(best_key)]
            key_name = f"{best_key}"
        else:
            camelot = CAMELOT_MINOR[PITCH_CLASSES.index(best_key)]
            key_name = f"{best_key}m"  # e.g., Am
        return key_name, camelot
    except Exception:
        return None, None

# --- BPM detection ---

def detect_bpm(y: np.ndarray, sr: int) -> Optional[int]:
    """Return normalized BPM in an 60..200 window (int), or None."""
    try:
        t_candidates = librosa.beat.tempo(y=y, sr=sr, start_bpm=124.0, max_tempo=300.0, aggregate=None)
        if t_candidates is None or len(t_candidates) == 0:
            t = librosa.beat.tempo(y=y, sr=sr)
            bpm = float(t[0]) if np.ndim(t) else float(t)
        else:
            bpm = float(np.median(t_candidates))
        bpm = normalize_bpm(bpm)
        return int(round(bpm))
    except Exception:
        return None

def normalize_bpm(bpm: float) -> float:
    """Bring BPM into 60..200 window by halving/doubling."""
    while bpm < 60:
        bpm *= 2
    while bpm > 200:
        bpm /= 2
    return bpm

# --- Key utils for Camelot conversion and distances ---

def key_to_camelot(key_str: Optional[str]) -> Optional[str]:
    """Convert 'Am'/'A' style to Camelot '8A/8B'. Returns None if unknown."""
    if not key_str:
        return None
    k = key_str.strip().upper().replace("MIN", "M").replace("MAJ", "")
    k = k.replace(" MINOR", "M").replace(" MAJOR", "")
    k = k.replace(" ", "")
    pc_map = {"C":0,"C#":1,"DB":1,"D":2,"D#":3,"EB":3,"E":4,"F":5,"F#":6,"GB":6,
              "G":7,"G#":8,"AB":8,"A":9,"A#":10,"BB":10,"B":11,"HB":11}
    is_minor = k.endswith("M")
    note = k[:-1] if is_minor else k
    if note not in pc_map:
        return None
    idx = pc_map[note]
    return CAMELOT_MINOR[idx] if is_minor else CAMELOT_MAJOR[idx]

def camelot_ring_dist(a: str, b: str) -> int:
    """Distance on Camelot wheel 1..11 (0 if same)."""
    try:
        n1, s1 = int(a[:-1]), a[-1].upper()
        n2, s2 = int(b[:-1]), b[-1].upper()
    except Exception:
        return 99
    return min((n1 - n2) % 12, (n2 - n1) % 12)

# --- Genre profiles + fused inference (BPM + Key) ---

GENRE_PROFILES = {
    # House family
    "Deep House": {
        "bpm": (118, 122),
        "camelot": {"8A","9A","10A","11A","8B","9B"},
        "mode_bias": "A"
    },
    "Progressive House": {
        "bpm": (122, 126),
        "camelot": {"8A","9A","10A","2B","3B"},
        "mode_bias": None
    },
    "Tech House": {
        "bpm": (124, 128),
        "camelot": {"7A","8A","9A","10A","7B","8B"},
        "mode_bias": "A"
    },
    "Funky House": {
        "bpm": (124, 128),
        "camelot": {"4B","5B","9B","10B"},
        "mode_bias": "B"
    },
    "Classic House": {
        "bpm": (125, 129),
        "camelot": {"8A","9A","1A","2A","8B","9B"},
        "mode_bias": None
    },
    "Electro House": {
        "bpm": (128, 132),
        "camelot": {"1A","2A","3A","4A","11B","12B"},
        "mode_bias": None
    },

    # Other styles
    "Disco": {
        "bpm": (110, 118),
        "camelot": {"2B","3B","4B","9B","10B"},
        "mode_bias": "B"
    },
    "Techno": {
        "bpm": (132, 140),
        "camelot": {"1A","2A","3A","4A","11A","12A"},
        "mode_bias": "A"
    },
    "UK Garage": {
        "bpm": (140, 160),
        "camelot": {"2A","3A","4A","5A","6A"},
        "mode_bias": "A"
    },
    "Jungle": {
        "bpm": (160, 175),
        "camelot": {"6A","7A","8A","9A"},
        "mode_bias": "A"
    },
    "Drum & Bass": {
        "bpm": (175, 190),
        "camelot": {"7A","8A","9A","10A"},
        "mode_bias": "A"
    },
    "Hip-Hop": {
        "bpm": (80, 110),
        "camelot": {"8A","9A","1A","2A","8B","9B"},
        "mode_bias": None
    },
    "Downtempo": {
        "bpm": (60,  90),
        "camelot": {"9A","10A","11A","12A","1A"},
        "mode_bias": "A"
    },
    "Hardcore": {
        "bpm": (190, 250),
        "camelot": {"9A","10A","11A"},
        "mode_bias": "A"
    },
}


def infer_genre_from_features(bpm: Optional[int], key: Optional[str], camelot: Optional[str]) -> Optional[str]:
    """Return a genre guess using BPM + Camelot, or None if low confidence."""
    if bpm is None and camelot is None and key is None:
        return None
    if camelot is None and key:
        camelot = key_to_camelot(key)

    best_genre, best_score = None, -1e9
    for g, prof in GENRE_PROFILES.items():
        b_lo, b_hi = prof["bpm"]

        # BPM score: 1.0 in-range; smooth decay outside
        if bpm is None:
            bpm_score = 0.0
        else:
            if b_lo <= bpm <= b_hi:
                bpm_score = 1.0
            else:
                half = max(1.0, (b_hi - b_lo) / 2.0)
                dist = min(abs(bpm - b_lo), abs(bpm - b_hi))
                bpm_score = 1.0 / (1.0 + (dist / half) ** 2)

        # Key score: exact preferred > neighbors
        key_score = 0.0
        if camelot:
            pref = prof["camelot"]
            if camelot in pref:
                key_score = 1.0
            else:
                dmin = min(camelot_ring_dist(camelot, c) for c in pref) if pref else 6
                ladder = {0:1.0, 1:0.8, 2:0.5, 3:0.2}
                key_score = ladder.get(dmin, 0.05)
            bias = prof.get("mode_bias")
            if bias and camelot.endswith(bias):
                key_score += 0.1

        score = 0.65 * bpm_score + 0.35 * key_score
        if score > best_score:
            best_score, best_genre = score, g

    return best_genre if best_score >= 0.35 else None

# --- Tag helpers ---

def read_tags(path: Path) -> Dict[str, Optional[str]]:
    tags = {"genre": None, "bpm": None, "key": None, "camelot": None, "title": None, "artist": None}
    try:
        mf = MFile(str(path))
        if mf is None:
            return tags
        if path.suffix.lower() == ".mp3":
            try:
                id3 = EasyID3(str(path))
                tags["genre"] = first_or_none(id3.get("genre"))
                tags["bpm"] = first_or_none(id3.get("bpm"))
                tags["title"] = first_or_none(id3.get("title"))
                tags["artist"] = first_or_none(id3.get("artist"))
                # TKEY for key
                from mutagen.id3 import ID3
                try:
                    raw = ID3(str(path))
                    if "TKEY" in raw and raw["TKEY"].text:
                        tags["key"] = raw["TKEY"].text[0]
                except Exception:
                    pass
            except Exception:
                pass
        else:
            if mf.tags:
                # Common frames: 'genre', 'bpm', 'initialkey'
                for k in ("genre", "bpm", "initialkey", "title", "artist"):
                    if k in mf.tags and mf.tags[k]:
                        val = mf.tags[k]
                        if isinstance(val, list) and val:
                            val = val[0]
                        if k == "initialkey":
                            tags["key"] = str(val)
                        else:
                            tags[k] = str(val)
    except Exception:
        pass
    return tags

def write_tags(path: Path, bpm: Optional[int], key: Optional[str], camelot: Optional[str], genre: Optional[str]) -> None:
    try:
        mf = MFile(str(path))
        if mf is None:
            return
        # BPM
        if bpm is not None:
            try:
                if path.suffix.lower() == ".mp3":
                    id3 = EasyID3(str(path))
                    id3["bpm"] = [str(int(bpm))]
                    id3.save()
                else:
                    if mf.tags is None:
                        mf.add_tags()
                    mf.tags["bpm"] = [str(int(bpm))]
                    mf.save()
            except Exception:
                pass
        # KEY (TKEY / initialkey)
        if key is not None:
            try:
                if path.suffix.lower() == ".mp3":
                    from mutagen.id3 import ID3, TKEY
                    id3 = ID3(str(path))
                    id3.delall("TKEY")
                    id3.add(TKEY(encoding=3, text=[key]))
                    id3.save()
                else:
                    if mf.tags is None:
                        mf.add_tags()
                    mf.tags["initialkey"] = [key]
                    mf.save()
            except Exception:
                pass
        # GENRE
        if genre is not None:
            try:
                if path.suffix.lower() == ".mp3":
                    id3 = EasyID3(str(path))
                    id3["genre"] = [genre]
                    id3.save()
                else:
                    if mf.tags is None:
                        mf.add_tags()
                    mf.tags["genre"] = [genre]
                    mf.save()
            except Exception:
                pass
        # Optionally store Camelot as comment for non-MP3
        if camelot is not None and path.suffix.lower() != ".mp3":
            try:
                if mf.tags is None:
                    mf.add_tags()
                mf.tags["comment"] = [f"Camelot={camelot}"]
                mf.save()
            except Exception:
                pass
    except Exception:
        pass

def first_or_none(v):
    if v is None:
        return None
    if isinstance(v, list):
        return v[0] if v else None
    return v

# --- File ops ---

def safe_copy(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    target = dst_dir / src.name
    if not target.exists():
        shutil.copy2(src, target)
        return target
    stem = target.stem
    ext = target.suffix
    i = 1
    while True:
        candidate = dst_dir / f"{stem} ({i}){ext}"
        if not candidate.exists():
            shutil.copy2(src, candidate)
            return candidate
        i += 1

# --- Helpers ---

def parse_bpm(bpm_str: Optional[str]) -> Optional[int]:
    if not bpm_str:
        return None
    try:
        # Accept '124', '124.0', '124.3'
        return int(round(float(str(bpm_str).strip())))
    except Exception:
        return None

def sanitize_dirname(name: Optional[str]) -> str:
    invalid = '<>:"/\\|?*'
    if not name:
        return "Unknown"
    out = "".join(c for c in name if c not in invalid)
    return out.strip() or "Unknown"

def find_audio_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            yield p

def normalize_path(p: str) -> Path:
    """Handle trailing backslashes on Windows and expand ~."""
    # normpath removes trailing slashes/backslashes safely
    return Path(os.path.normpath(os.path.expanduser(p))).resolve()

# --- Main pipeline ---

def process_file(path: Path, args) -> Dict[str, object]:
    info = read_tags(path)
    genre = info.get("genre")
    bpm_str = info.get("bpm")
    key = info.get("key")

    # Treat 'Other' / 'Music' as missing genres
    if genre and genre.strip().lower() in {"other", "music"}:
        genre = None

    bpm = parse_bpm(bpm_str)
    camelot = key_to_camelot(key) if key else None
    analyzed = {"bpm": False, "key": False, "genre_inferred": False}

    # Load audio only if needed
    y = None
    sr = None
    need_audio = (bpm is None) or (key is None) or (args.infer_genre and not genre)
    if need_audio:
        try:
            y, sr = librosa.load(str(path), sr=None, mono=True)
            y, _ = librosa.effects.trim(y, top_db=40)
        except Exception:
            y, sr = None, None

    if bpm is None and y is not None and sr:
        new_bpm = detect_bpm(y, sr)
        if new_bpm is not None:
            bpm = new_bpm
            analyzed["bpm"] = True

    if key is None and y is not None and sr:
        k, cam = detect_key(y, sr)
        if k:
            key = k
            camelot = cam
            analyzed["key"] = True

    if args.infer_genre and not genre:
        # ensure camelot
        if camelot is None and key:
            camelot = key_to_camelot(key)
        g = infer_genre_from_features(bpm, key, camelot)
        if g:
            genre = g
            analyzed["genre_inferred"] = True

    # Optionally write tags back (including replacing 'Other'/'Music' with inferred)
    if args.write_tags:
        try:
            write_tags(path, bpm=bpm, key=key, camelot=camelot, genre=genre if args.infer_genre else info.get("genre"))
        except Exception:
            pass

    # Destination dir by Genre (fallback to Unknown)
    genre_dir = sanitize_dirname(genre) if genre else "Unknown"
    dest_dir = Path(args.dest) / genre_dir
    copied_to = safe_copy(path, dest_dir)

    return {
        "source": str(path),
        "copied_to": str(copied_to),
        "title": info.get("title"),
        "artist": info.get("artist"),
        "genre": genre or "Unknown",
        "bpm": bpm,
        "key": key,
        "camelot": camelot,
        "bpm_estimated": analyzed["bpm"],
        "key_estimated": analyzed["key"],
        "genre_inferred": analyzed["genre_inferred"],
    }

def main():
    ap = argparse.ArgumentParser(description="Sort songs by Genre, compute BPM/Key, and copy to destination.")
    ap.add_argument("src", type=str, help="Path to source music folder")
    ap.add_argument("dest", type=str, help="Path to destination folder (created if missing)")
    ap.add_argument("--infer-genre", action="store_true",
                    help="Infer genre using BPM + harmonic key when missing (smart)")
    ap.add_argument("--write-tags", action="store_true",
                    help="Write detected BPM/Key/Genre back into file tags (when possible)")
    ap.add_argument("--report", type=str, default=None,
                    help="CSV path to write a processing report")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit of files to process")
    args = ap.parse_args()

    # Normalize paths to avoid trailing backslash/quote issues on Windows
    src = normalize_path(args.src)
    dest = normalize_path(args.dest)

    if not src.exists():
        print(f"Source folder not found: {src}", file=sys.stderr)
        sys.exit(1)
    dest.mkdir(parents=True, exist_ok=True)

    files = list(find_audio_files(src))
    if args.limit:
        files = files[: args.limit]

    rows = []
    for f in tqdm(files, desc="Processing tracks"):
        try:
            rows.append(process_file(f, args))
        except Exception:
            rows.append({
                "source": str(f),
                "copied_to": None,
                "title": None, "artist": None,
                "genre": "ERROR",
                "bpm": None, "key": None, "camelot": None,
                "bpm_estimated": False, "key_estimated": False, "genre_inferred": False,
            })

    if args.report:
        try:
            df = pd.DataFrame(rows)
            df.to_csv(args.report, index=False)
            print(f"Report written to: {args.report}")
        except Exception as e:
            print(f"Failed to write report: {e}", file=sys.stderr)

    total = len(rows)
    est_bpm = sum(1 for r in rows if r["bpm_estimated"])
    est_key = sum(1 for r in rows if r["key_estimated"])
    inf_genre = sum(1 for r in rows if r["genre_inferred"])
    print(f"\nDone. Tracks: {total} | BPM estimated: {est_bpm} | Key estimated: {est_key} | Genre inferred: {inf_genre}")
    print(f"Destination: {dest}")

if __name__ == "__main__":
    main()
