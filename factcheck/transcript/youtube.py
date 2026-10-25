"""
YouTube handling:
- Validating YouTube URLs
- Extracting the video ID
- Fetching official captions via youtube-transcript-api
"""

import re

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

from factcheck.transcript.segments import make_segment


YOUTUBE_URL_PATTERNS = [
    r"(?:youtube\.com/watch\?v=)([a-zA-Z0-9_-]{11})",
    r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
    r"(?:youtube\.com/shorts/)([a-zA-Z0-9_-]{11})",
]


def extract_video_id(url: str) -> str | None:
    """Return the 11-character YouTube video ID, or None if invalid."""
    for pattern in YOUTUBE_URL_PATTERNS:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def is_valid_youtube_url(url: str) -> bool:
    return extract_video_id(url) is not None


def get_official_transcript(video_id: str) -> list[dict] | None:
    """
    Fetch YouTube captions using youtube-transcript-api 1.x.

    Returns:
        List of transcript segments, or None if captions are unavailable.
    """

    try:
        # youtube-transcript-api >= 1.2 uses the instance-based API
        api = YouTubeTranscriptApi()

        # Prefer English, but fall back to other available captions.
        transcript = api.fetch(
            video_id,
            languages=["en"]
        )

        raw = transcript.to_raw_data()

    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
        return None

    except Exception as exc:
        print(f"Caption retrieval failed: {exc}")
        return None

    return [
        make_segment(
            entry["start"],
            entry["start"] + entry.get("duration", 0),
            entry["text"],
        )
        for entry in raw
    ]