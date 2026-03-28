from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
GAMES_PATH = BASE_DIR / "games.json"
MAX_GAMES = 5000
DEFAULT_MATCH_COUNT = 5


def create_search_engine() -> "GameSearchEngine":
    return GameSearchEngine(GAMES_PATH)


@dataclass
class GameRecord:
    app_id: str
    raw: dict[str, Any]

    @property
    def name(self) -> str:
        return self.raw.get("name", "Unknown title")

    @property
    def short_description(self) -> str:
        return self.raw.get("short_description", "")

    def to_result(self, score: float) -> dict[str, Any]:
        return {
            "app_id": self.app_id,
            "name": self.name,
            "score": round(score, 4),
            "short_description": self.short_description,
            "genres": self.raw.get("genres", []),
            "tags": self._normalize_tags(self.raw.get("tags")),
            "price": self.raw.get("price"),
            "release_date": self.raw.get("release_date"),
            "header_image": self.raw.get("header_image"),
            "store_page": f"https://store.steampowered.com/app/{self.app_id}",
            "platforms": {
                "windows": bool(self.raw.get("windows")),
                "mac": bool(self.raw.get("mac")),
                "linux": bool(self.raw.get("linux")),
            },
        }

    @staticmethod
    def _normalize_tags(tags: Any) -> list[str]:
        if isinstance(tags, dict):
            return list(tags.keys())[:8]
        if isinstance(tags, list):
            return tags[:8]
        return []


class GameSearchEngine:
    """
    This implementation is intentionally crude:
    - it loads a subset of games
    - it ignores the query for ranking
    - it returns random games as "matches"
    - it writes a simple canned answer instead of calling an LLM

    Suggested improvements:
    1. Replace `retrieve_candidates()` with keyword search, BM25, embeddings, or vector search.
    2. Replace `rank_candidates()` with a real ranking function.
    3. Replace `generate_answer()` with an LLM prompt over retrieved context.

    Keep the public `search()` return shape stable so the Flask app and frontend keep working.
    """

    def __init__(self, games_path: Path) -> None:
        self.games_path = games_path
        self.records = self.load_records()

    def load_records(self) -> list[GameRecord]:
        payload = json.loads(self.games_path.read_text())
        records: list[GameRecord] = []

        for app_id, raw in payload.items():
            if not raw.get("name"):
                continue
            records.append(GameRecord(app_id=app_id, raw=raw))
            if len(records) >= MAX_GAMES:
                break

        return records

    def search(self, query: str) -> dict[str, Any]:
        candidates = self.retrieve_candidates(query)
        ranked_matches = self.rank_candidates(query, candidates)
        results = [record.to_result(score) for record, score in ranked_matches]

        return {
            "matches": results,
            "answer": self.generate_answer(query, ranked_matches),
            "meta": {
                "indexed_games": len(self.records),
                "retrieval_mode": "random-demo",
                "note": "Replace the scaffold in recommender.py with your own retrieval, ranking, and LLM logic.",
            },
        }

    def retrieve_candidates(self, query: str) -> list[GameRecord]:
        """
        Very weak baseline retrieval.

        Right now this ignores `query` and returns a random sample.
        """
        if not self.records:
            return []
        sample_size = min(DEFAULT_MATCH_COUNT, len(self.records))
        return random.sample(self.records, sample_size)

    def rank_candidates(
        self, query: str, candidates: list[GameRecord]
    ) -> list[tuple[GameRecord, float]]:
        """
        Very weak baseline ranking.

        Right now every candidate gets a random score.
        """
        ranked = [(record, random.random()) for record in candidates]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    def generate_answer(self, query: str, matches: list[tuple[GameRecord, float]]) -> str:
        """
        Very weak baseline response generation.
        """
        if not matches:
            return "No games were available to recommend."

        names = ", ".join(record.name for record, _ in matches[:3])
        return (
            f'This scaffold ignores most of the query, but for "{query}" it picked: {names}. '
            "Open recommender.py and replace the random retrieval, ranking, and answer generation steps."
        )
