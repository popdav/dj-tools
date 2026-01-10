#!/usr/bin/env python3
"""
spotify-playlist-to-csv.py

Export a PUBLIC Spotify playlist to CSV using Client Credentials (no user login).

Setup (one time):
  1) Create an app in Spotify Developer Dashboard.
  2) Set env vars:
     SPOTIFY_CLIENT_ID
     SPOTIFY_CLIENT_SECRET

Usage:
  pip install requests
  python spotify-playlist-to-csv.py "https://open.spotify.com/playlist/226cXuqli6KTIeyVfW9Rcr?si=..." out.csv
  python spotify-playlist-to-csv.py 226cXuqli6KTIeyVfW9Rcr out.csv
"""

from __future__ import annotations

import base64
import csv
import os
import re
import sys
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests


TOKEN_URL = "https://accounts.spotify.com/api/token"
API_BASE = "https://api.spotify.com/v1"
UA = "spotify-playlist-to-csv/1.0"


def norm(x: Any) -> str:
    return str(x or "").strip()


def parse_playlist_id(value: str) -> str:
    value = value.strip()

    # Raw Spotify playlist IDs are base62-ish, commonly 22 chars, but be flexible.
    if re.fullmatch(r"[A-Za-z0-9]{10,64}", value):
        return value

    parsed = urlparse(value)
    if "open.spotify.com" in parsed.netloc:
        parts = [p for p in parsed.path.split("/") if p]  # ["playlist", "<id>"]
        if len(parts) >= 2 and parts[0] == "playlist":
            return parts[1]

    raise ValueError(f"Could not parse Spotify playlist id from: {value!r}")


def get_app_token(client_id: str, client_secret: str) -> str:
    auth = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("ascii")
    r = requests.post(
        TOKEN_URL,
        headers={"Authorization": f"Basic {auth}", "User-Agent": UA},
        data={"grant_type": "client_credentials"},
        timeout=20,
    )
    r.raise_for_status()
    data = r.json()
    token = data.get("access_token")
    if not token:
        raise RuntimeError(f"No access_token in response: {data}")
    return token


def api_get(url: str, token: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    r = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}", "User-Agent": UA},
        params=params,
        timeout=25,
    )
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected response type: {type(data)}")
    return data


def join_artists(track: Dict[str, Any]) -> str:
    artists = track.get("artists") or []
    names = [a.get("name") for a in artists if isinstance(a, dict) and a.get("name")]
    return ", ".join(names)


def fetch_playlist_and_tracks(playlist_id: str, token: str) -> tuple[str, List[Dict[str, Any]]]:
    # Playlist metadata
    pl = api_get(f"{API_BASE}/playlists/{playlist_id}", token)
    title = norm(pl.get("name")) or f"playlist:{playlist_id}"

    # Tracks pagination: limit=100, offset increments
    tracks: List[Dict[str, Any]] = []
    offset = 0
    limit = 100

    while True:
        page = api_get(
            f"{API_BASE}/playlists/{playlist_id}/tracks",
            token,
            params={
                "limit": limit,
                "offset": offset,
                # keep response light but useful
                "fields": "items(added_at,track(id,name,artists(name),album(name,release_date),duration_ms,explicit,popularity,external_urls,external_ids(isrc))),total,next",
            },
        )

        items = page.get("items", [])
        if not isinstance(items, list):
            break

        for it in items:
            t = (it or {}).get("track") or {}
            if not t or t.get("id") is None:
                continue  # can be null if unavailable
            tracks.append({"added_at": it.get("added_at"), "track": t})

        # stop condition
        got = len(items)
        if got < limit:
            break
        offset += limit

    return title, tracks


def export_csv(playlist_id: str, playlist_title: str, items: List[Dict[str, Any]], out_path: str) -> None:
    fieldnames = [
        "source_type",
        "source_id",
        "source_title",
        "index",
        "artist",
        "title",
        "artist_title",
        "album",
        "release_date",
        "duration_ms",
        "isrc",
        "spotify_track_id",
        "spotify_track_url",
        "added_at",
        "explicit",
        "popularity",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for i, item in enumerate(items, start=1):
            t = item["track"]
            artist = norm(join_artists(t))
            title = norm(t.get("name"))
            album = norm((t.get("album") or {}).get("name"))
            release_date = norm((t.get("album") or {}).get("release_date"))
            isrc = norm((t.get("external_ids") or {}).get("isrc"))
            url = norm((t.get("external_urls") or {}).get("spotify"))

            w.writerow(
                {
                    "source_type": "playlist",
                    "source_id": playlist_id,
                    "source_title": playlist_title,
                    "index": i,
                    "artist": artist,
                    "title": title,
                    "artist_title": f"{artist} - {title}".strip(" -"),
                    "album": album,
                    "release_date": release_date,
                    "duration_ms": t.get("duration_ms", "") or "",
                    "isrc": isrc,
                    "spotify_track_id": t.get("id", "") or "",
                    "spotify_track_url": url,
                    "added_at": norm(item.get("added_at")),
                    "explicit": t.get("explicit", "") if "explicit" in t else "",
                    "popularity": t.get("popularity", "") if "popularity" in t else "",
                }
            )


def main(argv: List[str]) -> int:
    if len(argv) < 3:
        print(__doc__.strip())
        return 2

    source = argv[1]
    out_csv = argv[2]

    client_id = os.environ.get("SPOTIFY_CLIENT_ID", "").strip()
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        raise SystemExit(
            "Missing env vars. Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET (Spotify Developer Dashboard app)."
        )

    playlist_id = parse_playlist_id(source)
    token = get_app_token(client_id, client_secret)
    title, items = fetch_playlist_and_tracks(playlist_id, token)
    export_csv(playlist_id, title, items, out_csv)

    print(f"Exported {len(items)} tracks from playlist '{title}' (ID {playlist_id}) -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
