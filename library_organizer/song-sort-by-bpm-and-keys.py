#!/usr/bin/env python3
"""
song-sort-by-bpm-and-keys.py — Sort songs by your own inferred Genre into a destination folder.
Also computes/sets BPM and musical Key (if missing) and writes a CSV report.

Serato-focused:
- MP3 tags written using stable ID3 frames:
  - BPM -> TBPM
  - Musical key -> TKEY
  - Genre -> TCON (only if --overwrite-genre)
  - Optional Camelot -> TXXX:CAMELOT (custom, safe; Serato can show Camelot from TKEY)
- MP3 TXXX handling preserves other TXXX frames; only replaces CAMELOT TXXX.
- Faster analysis: loads only a window of audio (offset/duration), not whole track.
- Optional folder sub-structure: --subdir camelot / bpm10 / bpm5 / none
- Report includes genre_score (confidence)

Key behavior change:
- Existing Genre tags are IGNORED for sorting/inference (to avoid "crap" genres).
- Your inferred genre becomes the folder name, and optionally overwrites tags.

Requires: mutagen, librosa, soundfile, numpy, tqdm, pandas
"""

import argparse
import os
import sys
import shutil
from pathlib import Path
from typing import Optional, Tuple, Dict, Iterable, Any

import numpy as np

try:
    import librosa
    from mutagen import File as MFile
    from mutagen.easyid3 import EasyID3
    from tqdm import tqdm
    import pandas as pd
except Exception:
    print("Missing dependencies. Install with:\n  pip install mutagen librosa soundfile numpy tqdm pandas",
          file=sys.stderr)
    raise

AUDIO_EXTS = {".mp3", ".m4a", ".mp4", ".flac", ".wav", ".ogg", ".opus", ".aac"}

# --- Key detection (Krumhansl/Schmuckler style via chroma) ---

KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=float)
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=float)
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# aligned to PITCH_CLASSES
CAMELOT_MAJOR = ["8B", "3B", "10B", "5B", "12B", "7B", "2B", "9B", "4B", "11B", "6B", "1B"]
CAMELOT_MINOR = ["5A", "12A", "7A", "2A", "9A", "4A", "11A", "6A", "1A", "8A", "3A", "10A"]


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
            key_name = f"{best_key}m"
        return key_name, camelot
    except Exception:
        return None, None


# --- BPM detection ---

def detect_bpm(y: np.ndarray, sr: int) -> Optional[int]:
    """Return normalized BPM in a 60..250 window (int), or None."""
    try:
        t_candidates = librosa.beat.tempo(
            y=y, sr=sr, start_bpm=124.0, max_tempo=320.0, aggregate=None
        )
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
    """Bring BPM into 60..250 window by halving/doubling."""
    while bpm < 60:
        bpm *= 2
    while bpm > 250:
        bpm /= 2
    return bpm


# --- Key utils ---

def key_to_camelot(key_str: Optional[str]) -> Optional[str]:
    """Convert key strings to Camelot (e.g., 'Am', 'A minor', 'Amin', 'C#', 'Bb major') -> '8A/8B'."""
    if not key_str:
        return None

    k = key_str.strip()
    k = k.replace("♭", "b").replace("♯", "#")
    k = k.replace("Major", "").replace("MAJOR", "").replace("maj", "").replace("Maj", "")
    k = k.replace("Minor", "m").replace("MINOR", "m").replace("min", "m").replace("Min", "m")
    k = k.replace(" ", "")

    if k.lower().endswith("amin"):
        k = k[:-3] + "m"

    is_minor = k.endswith("m") or k.endswith("M")
    note = k[:-1] if is_minor else k

    pc_map = {
        "C": 0, "C#": 1, "DB": 1,
        "D": 2, "D#": 3, "EB": 3,
        "E": 4,
        "F": 5, "F#": 6, "GB": 6,
        "G": 7, "G#": 8, "AB": 8,
        "A": 9, "A#": 10, "BB": 10,
        "B": 11
    }

    n = note.upper()
    if n not in pc_map:
        return None

    idx = pc_map[n]
    return CAMELOT_MINOR[idx] if is_minor else CAMELOT_MAJOR[idx]


def camelot_ring_dist(a: str, b: str) -> int:
    """Distance on Camelot wheel 1..11 (0 if same)."""
    try:
        n1 = int(a[:-1])
        n2 = int(b[:-1])
    except Exception:
        return 99
    return min((n1 - n2) % 12, (n2 - n1) % 12)


# --- Genre inference (your crate genres) ---

GENRE_PROFILES = {
    "Deep House": {"bpm": (118, 122), "camelot": {"8A", "9A", "10A", "11A", "8B", "9B"}, "mode_bias": "A"},
    "Progressive House": {"bpm": (122, 126), "camelot": {"8A", "9A", "10A", "2B", "3B"}, "mode_bias": None},
    "Tech House": {"bpm": (124, 128), "camelot": {"7A", "8A", "9A", "10A", "7B", "8B"}, "mode_bias": "A"},
    "Funky House": {"bpm": (124, 128), "camelot": {"4B", "5B", "9B", "10B"}, "mode_bias": "B"},
    "Classic House": {"bpm": (125, 129), "camelot": {"8A", "9A", "1A", "2A", "8B", "9B"}, "mode_bias": None},
    "Electro House": {"bpm": (128, 132), "camelot": {"1A", "2A", "3A", "4A", "11B", "12B"}, "mode_bias": None},

    "Disco": {"bpm": (110, 118), "camelot": {"2B", "3B", "4B", "9B", "10B"}, "mode_bias": "B"},
    "Techno": {"bpm": (132, 140), "camelot": {"1A", "2A", "3A", "4A", "11A", "12A"}, "mode_bias": "A"},
    "UK Garage": {"bpm": (140, 160), "camelot": {"2A", "3A", "4A", "5A", "6A"}, "mode_bias": "A"},
    "Jungle": {"bpm": (160, 175), "camelot": {"6A", "7A", "8A", "9A"}, "mode_bias": "A"},
    "Drum & Bass": {"bpm": (175, 190), "camelot": {"7A", "8A", "9A", "10A"}, "mode_bias": "A"},
    "Hip-Hop": {"bpm": (80, 110), "camelot": {"8A", "9A", "1A", "2A", "8B", "9B"}, "mode_bias": None},
    "Downtempo": {"bpm": (60, 90), "camelot": {"9A", "10A", "11A", "12A", "1A"}, "mode_bias": "A"},
    "Hardcore": {"bpm": (190, 250), "camelot": {"9A", "10A", "11A"}, "mode_bias": "A"},
}


def infer_genre_from_features(
    bpm: Optional[int], key: Optional[str], camelot: Optional[str]
) -> Tuple[Optional[str], float]:
    """Return (genre_guess, score) using BPM + Camelot."""
    if bpm is None and camelot is None and key is None:
        return None, 0.0
    if camelot is None and key:
        camelot = key_to_camelot(key)

    best_genre, best_score = None, -1e9
    for g, prof in GENRE_PROFILES.items():
        b_lo, b_hi = prof["bpm"]

        if bpm is None:
            bpm_score = 0.0
        else:
            if b_lo <= bpm <= b_hi:
                bpm_score = 1.0
            else:
                half = max(1.0, (b_hi - b_lo) / 2.0)
                dist = min(abs(bpm - b_lo), abs(bpm - b_hi))
                bpm_score = 1.0 / (1.0 + (dist / half) ** 2)

        key_score = 0.0
        if camelot:
            pref = prof["camelot"]
            if camelot in pref:
                key_score = 1.0
            else:
                dmin = min(camelot_ring_dist(camelot, c) for c in pref) if pref else 6
                ladder = {0: 1.0, 1: 0.8, 2: 0.5, 3: 0.2}
                key_score = ladder.get(dmin, 0.05)

            bias = prof.get("mode_bias")
            if bias and camelot.endswith(bias):
                key_score += 0.1

        score = 0.65 * bpm_score + 0.35 * key_score
        if score > best_score:
            best_score, best_genre = score, g

    if best_score < 0.35:
        return None, float(best_score)
    return best_genre, float(best_score)


# --- Tag reading (we ignore existing genre for decisions) ---

def first_or_none(v):
    if v is None:
        return None
    if isinstance(v, list):
        return v[0] if v else None
    return v


def read_tags(path: Path) -> Dict[str, Optional[str]]:
    """
    Read only what's useful for analysis/report:
    - title, artist, bpm, key, camelot (optional)
    We intentionally DO NOT rely on genre for anything.
    """
    tags = {"genre": None, "bpm": None, "key": None, "camelot": None, "title": None, "artist": None}
    try:
        mf = MFile(str(path))
        if mf is None:
            return tags

        if path.suffix.lower() == ".mp3":
            try:
                id3 = EasyID3(str(path))
                tags["title"] = first_or_none(id3.get("title"))
                tags["artist"] = first_or_none(id3.get("artist"))
                tags["bpm"] = first_or_none(id3.get("bpm"))
                tags["genre"] = first_or_none(id3.get("genre"))  # read for report only
            except Exception:
                pass

            try:
                from mutagen.id3 import ID3
                raw = ID3(str(path))
                if "TKEY" in raw and raw["TKEY"].text:
                    tags["key"] = raw["TKEY"].text[0]
                if "TBPM" in raw and raw["TBPM"].text:
                    tags["bpm"] = raw["TBPM"].text[0]
                for fr in raw.getall("TXXX"):
                    if getattr(fr, "desc", "").strip().upper() == "CAMELOT" and fr.text:
                        tags["camelot"] = fr.text[0]
                        break
            except Exception:
                pass
        else:
            if mf.tags:
                for k in ("title", "artist", "bpm", "initialkey", "genre", "comment"):
                    if k in mf.tags and mf.tags[k]:
                        val = mf.tags[k]
                        if isinstance(val, list) and val:
                            val = val[0]
                        if k == "initialkey":
                            tags["key"] = str(val)
                        elif k == "comment":
                            s = str(val)
                            if "camelot=" in s.lower():
                                tags["camelot"] = s.split("=", 1)[-1].strip()
                        else:
                            tags[k] = str(val)
    except Exception:
        pass
    return tags


# --- Tag writing ---

def write_tags_mp3(
    path: Path,
    bpm: Optional[int],
    key: Optional[str],
    camelot: Optional[str],
    genre: Optional[str],
    write_camelot: bool = True,
    overwrite_genre: bool = False,
) -> None:
    """Write MP3 tags using stable ID3 frames; preserve unrelated frames (including Serato data)."""
    from mutagen.id3 import ID3, TBPM, TKEY, TCON, TXXX

    id3 = ID3(str(path))

    if bpm is not None:
        id3.delall("TBPM")
        id3.add(TBPM(encoding=3, text=[str(int(bpm))]))

    if key is not None:
        id3.delall("TKEY")
        id3.add(TKEY(encoding=3, text=[key]))

    if overwrite_genre and genre is not None:
        id3.delall("TCON")
        id3.add(TCON(encoding=3, text=[genre]))

    if write_camelot and camelot is not None:
        kept = []
        for fr in id3.getall("TXXX"):
            if getattr(fr, "desc", "").strip().upper() != "CAMELOT":
                kept.append(fr)
        id3.delall("TXXX")
        for fr in kept:
            id3.add(fr)
        id3.add(TXXX(encoding=3, desc="CAMELOT", text=[camelot]))

    id3.save()


def write_tags(
    path: Path,
    bpm: Optional[int],
    key: Optional[str],
    camelot: Optional[str],
    genre: Optional[str],
    write_camelot: bool = True,
    overwrite_genre: bool = False,
) -> None:
    try:
        if path.suffix.lower() == ".mp3":
            write_tags_mp3(
                path, bpm=bpm, key=key, camelot=camelot, genre=genre,
                write_camelot=write_camelot, overwrite_genre=overwrite_genre
            )
            return

        mf = MFile(str(path))
        if mf is None:
            return
        if mf.tags is None:
            try:
                mf.add_tags()
            except Exception:
                pass

        if bpm is not None:
            try:
                mf.tags["bpm"] = [str(int(bpm))]
                mf.save()
            except Exception:
                pass

        if key is not None:
            try:
                mf.tags["initialkey"] = [key]
                mf.save()
            except Exception:
                pass

        if overwrite_genre and genre is not None:
            try:
                mf.tags["genre"] = [genre]
                mf.save()
            except Exception:
                pass

        if write_camelot and camelot is not None:
            try:
                existing = mf.tags.get("comment", [])
                if isinstance(existing, str):
                    existing = [existing]
                if not isinstance(existing, list):
                    existing = [str(existing)]
                new = [str(c) for c in existing if "camelot=" not in str(c).lower()]
                new.append(f"Camelot={camelot}")
                mf.tags["comment"] = new
                mf.save()
            except Exception:
                pass
    except Exception:
        pass


# --- File ops & helpers ---

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


def parse_bpm(bpm_str: Optional[str]) -> Optional[int]:
    if not bpm_str:
        return None
    try:
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
        try:
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                yield p
        except Exception:
            continue


def normalize_path(p: str) -> Path:
    return Path(os.path.normpath(os.path.expanduser(p))).resolve()


def subdir_for_track(subdir_mode: str, bpm: Optional[int], camelot: Optional[str]) -> Path:
    if subdir_mode == "none":
        return Path()
    if subdir_mode == "camelot":
        return Path(sanitize_dirname(camelot) if camelot else "Camelot_Unknown")
    if subdir_mode in ("bpm10", "bpm5"):
        if bpm is None:
            return Path("BPM_Unknown")
        step = 10 if subdir_mode == "bpm10" else 5
        lo = (bpm // step) * step
        hi = lo + (step - 1)
        return Path(f"BPM_{lo}-{hi}")
    return Path()


def load_audio_window(path: Path, offset: float, duration: float) -> Tuple[Optional[np.ndarray], Optional[int]]:
    try:
        y, sr = librosa.load(str(path), sr=None, mono=True, offset=offset, duration=duration)
        y, _ = librosa.effects.trim(y, top_db=40)
        return y, sr
    except Exception:
        return None, None


# --- Main pipeline ---

def process_file(path: Path, args) -> Dict[str, Any]:
    info = read_tags(path)

    # We ignore existing genre for decisions
    existing_genre = info.get("genre")

    bpm = parse_bpm(info.get("bpm"))
    key = info.get("key")
    camelot = info.get("camelot") or (key_to_camelot(key) if key else None)

    analyzed = {"bpm": False, "key": False, "genre_inferred": False}

    # Decide if we need audio
    need_audio = (bpm is None) or (key is None) or (args.genre_mode == "infer")
    y = sr = None
    if need_audio:
        y, sr = load_audio_window(path, offset=args.audio_offset, duration=args.audio_duration)

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

    # Your genre outcome (always your own)
    genre = "Unknown"
    genre_score = 0.0

    if args.genre_mode == "infer":
        g, score = infer_genre_from_features(bpm, key, camelot)
        genre_score = score
        if g:
            genre = g
            analyzed["genre_inferred"] = True

    # Write tags back (BPM/Key always if present; Genre only if --overwrite-genre)
    if args.write_tags:
        write_tags(
            path,
            bpm=bpm,
            key=key,
            camelot=camelot,
            genre=genre,
            write_camelot=(not args.no_write_camelot),
            overwrite_genre=args.overwrite_genre,
        )

    # Folder naming always uses your computed genre
    genre_dir = sanitize_dirname(genre)
    extra = subdir_for_track(args.subdir, bpm=bpm, camelot=camelot)
    dest_dir = Path(args.dest) / genre_dir / extra
    copied_to = safe_copy(path, dest_dir)

    return {
        "source": str(path),
        "copied_to": str(copied_to),
        "title": info.get("title"),
        "artist": info.get("artist"),
        "existing_genre": existing_genre,
        "crate_genre": genre,
        "bpm": bpm,
        "key": key,
        "camelot": camelot,
        "bpm_estimated": analyzed["bpm"],
        "key_estimated": analyzed["key"],
        "genre_inferred": analyzed["genre_inferred"],
        "genre_score": float(genre_score),
    }


def main():
    ap = argparse.ArgumentParser(description="Sort songs by your own Genre, compute BPM/Key, and copy to destination.")
    ap.add_argument("src", type=str, help="Path to source music folder")
    ap.add_argument("dest", type=str, help="Path to destination folder (created if missing)")

    ap.add_argument("--write-tags", action="store_true",
                    help="Write detected BPM/Key back into file tags (when possible).")
    ap.add_argument("--overwrite-genre", action="store_true",
                    help="Overwrite existing Genre tags with your inferred crate genre.")
    ap.add_argument("--genre-mode", choices=["infer", "unknown"], default="infer",
                    help="infer: always infer crate genre from BPM+Key (default). unknown: always use Unknown.")
    ap.add_argument("--no-write-camelot", action="store_true",
                    help="Do NOT write Camelot tag (TXXX:CAMELOT for MP3 / comment for others).")
    ap.add_argument("--report", type=str, default=None,
                    help="CSV path to write a processing report")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit of files to process")

    ap.add_argument("--subdir", choices=["none", "camelot", "bpm10", "bpm5"], default="none",
                    help="Optional subfolder structure under Genre (default: none)")
    ap.add_argument("--audio-offset", type=float, default=30.0,
                    help="Seconds into track to start analysis window (default: 30)")
    ap.add_argument("--audio-duration", type=float, default=120.0,
                    help="Seconds of audio to analyze (default: 120)")

    args = ap.parse_args()

    src = normalize_path(args.src)
    dest = normalize_path(args.dest)
    args.dest = str(dest)

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
                "existing_genre": None,
                "crate_genre": "ERROR",
                "bpm": None, "key": None, "camelot": None,
                "bpm_estimated": False, "key_estimated": False, "genre_inferred": False,
                "genre_score": 0.0,
            })

    if args.report:
        try:
            pd.DataFrame(rows).to_csv(args.report, index=False)
            print(f"Report written to: {args.report}")
        except Exception as e:
            print(f"Failed to write report: {e}", file=sys.stderr)

    total = len(rows)
    est_bpm = sum(1 for r in rows if r.get("bpm_estimated"))
    est_key = sum(1 for r in rows if r.get("key_estimated"))
    inf_genre = sum(1 for r in rows if r.get("genre_inferred"))
    print(f"\nDone. Tracks: {total} | BPM estimated: {est_bpm} | Key estimated: {est_key} | Genre inferred: {inf_genre}")
    print(f"Destination: {dest}")


if __name__ == "__main__":
    main()
