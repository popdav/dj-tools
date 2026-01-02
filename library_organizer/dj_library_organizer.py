# dj_library_organizer.py

import os
from mutagen import File
from mutagen.mp3 import MP3
from mutagen.flac import FLAC
from mutagen.mp4 import MP4
from mutagen.wave import WAVE
from pathlib import Path
import requests
import musicbrainzngs

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

import base64

import beatport
import re

import librosa
import essentia.standard as es

def detect_bpm(file_path):
    try:
        y, sr = librosa.load(file_path, mono=True)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return int(round(tempo))
    except Exception as e:
        print(f"Error detecting BPM: {e}")
        return None

def detect_key(file_path):
    try:
        loader = es.MonoLoader(filename=file_path)
        audio = loader()
        key, scale, strength = es.KeyExtractor()(audio)
        # Convert to Camelot notation
        camelot = key_to_camelot(key, scale)
        return camelot
    except Exception as e:
        print(f"Error detecting key: {e}")
        return None

SPOTIFY_CLIENT_ID = "31b28dc5632c414b99b709a3d2cf47c9"
SPOTIFY_CLIENT_SECRET = "063f22edff554f4cb66486366f4dd2c1"

# Initialize client
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET
))

def get_genre_from_spotify(artist, title):
    try:
        results = sp.search(q=f"track:{title} artist:{artist}", type="track", limit=1)
        print(f"Spotify search results for {artist} - {title}: {results}")
        items = results.get("tracks", {}).get("items", [])
        if items:
            track = items[0]

            # Try artist genre first
            artist_id = track["artists"][0]["id"]
            artist_data = sp.artist(artist_id)
            artist_genres = artist_data.get("genres", [])

            if artist_genres:
                return artist_genres[0].title()

            # If artist has no genre, try album
            album_id = track["album"]["id"]
            album_data = sp.album(album_id)
            print(f"Album data for {artist} - {title}: {album_data}")
            album_genres = album_data.get("genres", [])

            if album_genres:
                return album_genres[0].title()
    except Exception as e:
        print(f"Spotify error for {artist} - {title}: {e}")
    return "Unknown"

def search_beatport_genre(artist, title):
    url = "https://api.beatport.com/v4/catalog/search"

    params = {
        "q": f"{artist} {title}",
        "type": "track",
        "perPage": 1
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # Navigate to genre info
        if data.get("results", {}).get("track"):
            track = data["results"]["track"][0]
            genre = track.get("genre", {}).get("name")
            return genre or "Unknown"
        else:
            return "Unknown"

    except Exception as e:
        print(f"Beatport API error: {e}")
        return "Unknown"


# Set a user-agent (MusicBrainz requires this)
musicbrainzngs.set_useragent("PersonalDJToolkit", "0.1", "popdav@outlook.com")

def get_genre_from_musicbrainz(artist, title):
    try:
        result = musicbrainzngs.search_recordings(artist=artist, recording=title, limit=1)
        recordings = result.get("recording-list", [])
        if recordings:
            recording = recordings[0]
            if "tag-list" in recording and recording["tag-list"]:
                # Use the first genre tag (capitalize for style)
                return recording["tag-list"][0]["name"].capitalize()
    except Exception as e:
        print(f"MusicBrainz error for {artist} - {title}: {e}")

    return "Unknown"

LASTFM_API_KEY = "e1783d5edefbd75fbb3e412fc9d6355d"

def get_genre_from_lastfm(artist, title):
    try:
        url = "http://ws.audioscrobbler.com/2.0/"
        params = {
            "method": "track.gettoptags",
            "artist": artist,
            "track": title,
            "api_key": LASTFM_API_KEY,
            "format": "json",
        }
        resp = requests.get(url, params=params)
        data = resp.json()
        # Extract the first meaningful tag
        if "toptags" in data and "tag" in data["toptags"]:
            tags = data["toptags"]["tag"]
            if tags:
                return tags[0]["name"].capitalize()
    except Exception as e:
        print(f"Last.fm error for {artist} - {title}: {e}")

    return "Unknown"

def search_deezer(query):
    search_url = "https://api.deezer.com/search"
    params = {"q": query}
    try:
        resp = requests.get(search_url, params=params)
        data = resp.json()
        if data["data"]:
            top = data["data"][0]
            track_id = top["id"]
            artist = top["artist"]["name"]
            title = top["title"]
            album = top["album"]["title"]

            # Fetch full track details to get genre
            track_url = f"https://api.deezer.com/track/{track_id}"
            track_resp = requests.get(track_url)
            track_data = track_resp.json()
            print(f"Track data: {track_data}")
            # Get genre from album data (via genre_id)
            genre = "Unknown"
            if "genre_id" in track_data:
                genre_id = track_data["genre_id"]
                if genre_id:
                    genre_lookup = requests.get(f"https://api.deezer.com/genre/{genre_id}")
                    genre_data = genre_lookup.json()
                    genre = genre_data.get("name", "Unknown")

            return {
                "title": title,
                "artist": artist,
                "album": album,
                "genre": genre,
            }

    except Exception as e:
        print(f"API error for '{query}': {e}")
    return None


SUPPORTED_EXTS = [".mp3", ".flac", ".wav", ".m4a"]

# 🎯 Main extractor function
def extract_metadata(file_path):
    try:
        audio = File(file_path, easy=True)
        title = audio.get("title", [None])[0]
        artist = audio.get("artist", [None])[0]

        # Sanitize: If title contains ' - ', split into artist and title if artist is missing or generic
        if title and "-" in title:
            parts = title.split("-", 1)
            if not artist or artist.lower() in ["unknown", "various artists", "artist"]:
                artist, title = parts[0].strip(), parts[1].strip()

        if title and artist:
            genre = audio.get("genre", [None])[0] or "Unknown"
            if genre == 'Music' or genre == 'Other':
                genre = "Unknown"

            if genre == "Unknown":
                genre = get_genre_from_spotify(artist, title)
            if genre.lower() == "unknown":
                genre = get_genre_from_lastfm(artist, title)
            if genre == "Unknown":
                genre = get_genre_from_musicbrainz(artist, title)

            metadata = {
                "file": file_path.name,
                "title": title,
                "artist": artist,
                "album": audio.get("album", ["Unknown"])[0],
                "genre": genre,
                "source": "local+fallback",
            }
        else:
            # Fallback to API search
            guess = file_path.stem.replace("_", " ").replace("-", " - ")
            metadata_from_api = search_deezer(guess)
            if metadata_from_api:
                genre = metadata_from_api["genre"]
                if genre == "Unknown":
                    genre = get_genre_from_spotify(metadata_from_api["artist"], metadata_from_api["title"])
                if genre == "Unknown":
                    genre = get_genre_from_lastfm(metadata_from_api["artist"], metadata_from_api["title"])
                if genre == "Unknown":
                    genre = get_genre_from_musicbrainz(metadata_from_api["artist"], metadata_from_api["title"])
                

                metadata_from_api["genre"] = genre

                metadata = {
                    "file": file_path.name,
                    **metadata_from_api,
                    "source": "deezer+fallback",
                }
            else:
                metadata = {
                    "file": file_path.name,
                    "title": title or "Unknown",
                    "artist": artist or "Unknown",
                    "album": "Unknown",
                    "genre": "Unknown",
                    "source": "fallback",
                }

        return metadata
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def scan_folder(folder_path):
    folder_path = Path(folder_path)
    print(f"\n🎧 Scanning: {folder_path}\n")

    counterWithMetaData = 0
    counter = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in SUPPORTED_EXTS):
                counter += 1
                file_path = Path(root) / file
                meta = extract_metadata(file_path)
                if meta:
                    counterWithMetaData += 1
                    print(f"{meta['artist']} - {meta['title']} ({meta['genre']})")
                    
                    # Sanitize genre folder name
                    safe_genre = re.sub(r'[\\/:*?"<>|]', '_', meta["genre"]).strip()
                    genre_folder = folder_path / safe_genre
                    genre_folder.mkdir(parents=True, exist_ok=True)
                    new_path = genre_folder / file_path.name
                    try:
                        if file_path.resolve() != new_path.resolve():
                            file_path.rename(new_path)
                    except Exception as e:
                        print(f"Could not move {file_path} to {genre_folder}: {e}")

    print(f"\n🎶 Found {counterWithMetaData} supported audio files.")
    print(f"🎶 Total files scanned: {counter}")


def bpm_to_genre(bpm):
    if bpm is None:
        return "Unknown"
    if bpm < 85:
        return "Chill / Hip-Hop"
    elif 85 <= bpm < 110:
        return "Reggaeton / Dancehall"
    elif 110 <= bpm < 120:
        return "Techno / Indie Dance"
    elif 120 <= bpm < 124:
        return "House"
    elif 124 <= bpm < 128:
        return "Tech House"
    elif 128 <= bpm < 132:
        return "Progressive / Electro"
    elif 132 <= bpm < 138:
        return "Trance"
    elif 138 <= bpm < 150:
        return "Hard Techno"
    elif 160 <= bpm < 180:
        return "Drum & Bass"
    else:
        return "Other"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DJ Library Organizer (basic scanner)")
    parser.add_argument("path", help="Path to your music folder")

    args = parser.parse_args()
    scan_folder(args.path)
    print("\n🎶 Scan complete!")