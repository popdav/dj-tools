#!/usr/bin/env python3
"""
deezer-playlist-to-csv.py

Export Deezer data to CSV:
  - Playlist tracks (public): https://www.deezer.com/.../playlist/<id>  or <id> (with --mode playlist)
  - Loved tracks (public):    https://www.deezer.com/.../profile/<uid>/loved or <uid> (default mode)

Examples:
  pip install requests

  # Loved tracks (auto-detected from URL)
  python deezer-playlist-to-csv.py "https://www.deezer.com/sr/profile/1811096822/loved" loved.csv

  # Playlist (auto-detected from URL)
  python deezer-playlist-to-csv.py "https://www.deezer.com/en/playlist/908622995" playlist.csv

  # Numeric ID only: specify mode
  python deezer-playlist-to-csv.py 1811096822 loved.csv --mode loved
  python deezer-playlist-to-csv.py 908622995 playlist.csv --mode playlist
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests

UA = "deezer-playlist-to-csv/2.0"
API_PLAYLIST = "https://api.deezer.com/playlist/{playlist_id}"
API_USER_TRACKS = "https://api.deezer.com/user/{user_id}/tracks"


# -------------------- Helpers --------------------

def norm(x: Any) -> str:
    return str(x or "").strip()


def safe_get(d: Dict[str, Any], path: str, default: Any = "") -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur if cur is not None else default


def deezer_get_json(url: str, *, timeout: int = 20, retries: int = 3) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
            r.raise_for_status()
            data = r.json()

            if isinstance(data, dict) and "error" in data:
                raise RuntimeError(f"Deezer API error: {data['error']}")

            if not isinstance(data, dict):
                raise RuntimeError(f"Unexpected response type: {type(data)}")

            return data
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(0.8 * attempt)
            else:
                raise RuntimeError(f"Failed to fetch {url}: {last_err}") from last_err
    raise RuntimeError("unreachable")


def fetch_paginated(first_page: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    data = first_page.get("data", [])
    if isinstance(data, list):
        items.extend(data)

    next_url = first_page.get("next")
    while next_url:
        page = deezer_get_json(next_url)
        data = page.get("data", [])
        if isinstance(data, list):
            items.extend(data)
        next_url = page.get("next")

    return items


# -------------------- Input detection --------------------

def detect_source(source: str) -> Tuple[str, str]:
    """
    Returns (mode, id) where mode is 'playlist' or 'loved'.

    If `source` is a Deezer URL:
      - contains /playlist/<id> => playlist
      - contains /profile/<id>/loved (or /profile/<id>) => loved (user tracks)

    If `source` is numeric => mode is unknown here; caller decides via --mode.
    """
    s = source.strip()

    if re.fullmatch(r"\d+", s):
        return ("numeric", s)

    parsed = urlparse(s)
    path = parsed.path

    m_pl = re.search(r"/playlist/(\d+)", path)
    if m_pl:
        return ("playlist", m_pl.group(1))

    m_prof = re.search(r"/profile/(\d+)", path) or re.search(r"/user/(\d+)", path)
    if m_prof:
        # If they share /loved specifically, it's still user tracks; treat as loved.
        return ("loved", m_prof.group(1))

    raise ValueError(f"Unrecognized Deezer URL format: {source!r}")


# -------------------- Fetchers --------------------

def fetch_playlist(playlist_id: str) -> Tuple[str, List[Dict[str, Any]]]:
    playlist = deezer_get_json(API_PLAYLIST.format(playlist_id=playlist_id))
    title = norm(playlist.get("title")) or f"playlist:{playlist_id}"

    tracks_page = playlist.get("tracks", {})
    if not isinstance(tracks_page, dict):
        return title, []

    tracks = fetch_paginated(tracks_page)
    return title, tracks


def fetch_loved(user_id: str) -> Tuple[str, List[Dict[str, Any]]]:
    first_page = deezer_get_json(API_USER_TRACKS.format(user_id=user_id))
    title = f"loved:user:{user_id}"
    tracks = fetch_paginated(first_page)
    return title, tracks


# -------------------- CSV export --------------------

def export_csv(
    *,
    out_path: str,
    source_type: str,   # playlist | loved
    source_id: str,
    source_title: str,
    tracks: List[Dict[str, Any]],
) -> None:
    fieldnames = [
        "source_type",
        "source_id",
        "source_title",
        "index",
        "artist",
        "title",
        "artist_title",      # Artist - Title
        "album",
        "duration_sec",
        "bpm",
        "release_date",
        "isrc",
        "deezer_track_id",
        "deezer_track_link",
        "preview_url",
        "explicit_lyrics",
        "rank",
        # For playlists only (may be blank for loved)
        "added_at_epoch",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for i, t in enumerate(tracks, start=1):
            artist = norm(safe_get(t, "artist.name"))
            title = norm(t.get("title"))
            album = norm(safe_get(t, "album.title"))

            w.writerow(
                {
                    "source_type": source_type,
                    "source_id": source_id,
                    "source_title": source_title,
                    "index": i,
                    "artist": artist,
                    "title": title,
                    "artist_title": f"{artist} - {title}".strip(" -"),
                    "album": album,
                    "duration_sec": t.get("duration", "") or "",
                    "bpm": t.get("bpm", "") or "",
                    "release_date": t.get("release_date", "") or "",
                    "isrc": t.get("isrc", "") or "",
                    "deezer_track_id": t.get("id", "") or "",
                    "deezer_track_link": t.get("link", "") or "",
                    "preview_url": t.get("preview", "") or "",
                    "explicit_lyrics": t.get("explicit_lyrics", "") or "",
                    "rank": t.get("rank", "") or "",
                    "added_at_epoch": t.get("time_add", "") or "",
                }
            )


# -------------------- Main --------------------

def main() -> int:
    p = argparse.ArgumentParser(description="Export Deezer playlist or loved tracks to CSV.")
    p.add_argument("source", help="Deezer playlist URL/ID, or profile loved URL/user_id")
    p.add_argument("out_csv", help="Output CSV path, e.g. out.csv")
    p.add_argument(
        "--mode",
        choices=["auto", "playlist", "loved"],
        default="auto",
        help="How to interpret numeric IDs. Default: auto (URLs are auto-detected; numeric defaults to loved).",
    )

    args = p.parse_args()

    detected_mode, id_value = detect_source(args.source)

    mode = detected_mode
    if detected_mode == "numeric":
        # Numeric only: either user specifies --mode, or default to loved.
        mode = args.mode if args.mode != "auto" else "loved"
    else:
        # URL: ignore --mode if auto-detected
        mode = detected_mode

    if mode == "playlist":
        title, tracks = fetch_playlist(id_value)
        export_csv(
            out_path=args.out_csv,
            source_type="playlist",
            source_id=id_value,
            source_title=title,
            tracks=tracks,
        )
        print(f"Exported {len(tracks)} tracks from playlist '{title}' (ID {id_value}) -> {args.out_csv}")
        return 0

    if mode == "loved":
        title, tracks = fetch_loved(id_value)
        export_csv(
            out_path=args.out_csv,
            source_type="loved",
            source_id=id_value,
            source_title=title,
            tracks=tracks,
        )
        print(f"Exported {len(tracks)} loved tracks (user {id_value}) -> {args.out_csv}")
        return 0

    raise SystemExit(f"Unexpected mode: {mode}")


if __name__ == "__main__":
    raise SystemExit(main())
