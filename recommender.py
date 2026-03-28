from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ollama
from rank_bm25 import BM25Okapi

BASE_DIR = Path(__file__).resolve().parent
GAMES_PATH = BASE_DIR / "games.json"
MAX_GAMES = 5000
DEFAULT_MATCH_COUNT = 5

# Ollama model to use — swap for any model you have pulled locally
OLLAMA_MODEL = "phi3.5"


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
    def __init__(self, games_path: Path) -> None:
        self.games_path = games_path
        self.records = self.load_records()

        # Build BM25 index once at startup over name + description + tags
        corpus = []
        for r in self.records:
            tags = r._normalize_tags(r.raw.get("tags"))
            genres = r.raw.get("genres", [])
            text = f"{r.name} {r.short_description} {' '.join(tags)} {' '.join(genres)}"
            corpus.append(text.lower().split())
        self.bm25 = BM25Okapi(corpus)

    def load_records(self) -> list[GameRecord]:
        payload = json.loads(self.games_path.read_text(encoding="utf-8"))
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
                "retrieval_mode": "bm25+ollama",
                "note": f"BM25 retrieval over {len(self.records)} games, ranked and answered via {OLLAMA_MODEL}.",
            },
        }

    def retrieve_candidates(self, query: str) -> list[GameRecord]:
        """
        BM25 retrieval over the full game corpus.
        Returns the top-20 candidates by BM25 score for the given query.
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Pair each record with its BM25 score and take the top 20
        scored = sorted(
            zip(self.records, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        top_candidates = [record for record, score in scored[:20] if score > 0]

        # Fall back to random if BM25 finds nothing (very generic query)
        if not top_candidates:
            top_candidates = random.sample(self.records, min(DEFAULT_MATCH_COUNT, len(self.records)))

        return top_candidates

    def rank_candidates(
        self, query: str, candidates: list[GameRecord]
    ) -> list[tuple[GameRecord, float]]:
        """
        Re-score the BM25 candidates and return the top DEFAULT_MATCH_COUNT.
        BM25 scores are already meaningful, so we just re-compute them cleanly
        and slice to the final result count.
        """
        tokenized_query = query.lower().split()
        # Map app_id -> index for fast lookup
        id_to_idx = {r.app_id: i for i, r in enumerate(self.records)}
        # Compute all BM25 scores in one call instead of once per candidate
        all_scores = self.bm25.get_scores(tokenized_query)

        scored = []
        for record in candidates:
            idx = id_to_idx.get(record.app_id)
            if idx is None:
                continue
            scored.append((record, float(all_scores[idx])))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:DEFAULT_MATCH_COUNT]

    def generate_answer(self, query: str, matches: list[tuple[GameRecord, float]]) -> str:
        """
        Calls a local Ollama model with the top retrieved games as context
        to produce a short, helpful recommendation summary.
        """
        if not matches:
            return "No games were found matching your description."

        # Build a compact context block for the LLM
        context_lines = []
        for i, (record, score) in enumerate(matches, 1):
            tags = ", ".join(record._normalize_tags(record.raw.get("tags")))
            context_lines.append(
                f"{i}. {record.name} — {record.short_description or 'No description.'}"
                + (f" Tags: {tags}." if tags else "")
            )
        context = "\n".join(context_lines)

        prompt = (
            f"A user is looking for Steam games matching this description:\n\"{query}\"\n\n"
            f"Here are the top retrieved candidates:\n{context}\n\n"
            "In 2-3 sentences, explain why these games are good matches for the user's request. "
            "Be specific and mention game names. Do not add games not in the list."
        )

        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.message.content.strip()
        except Exception as exc:
            # Graceful fallback if Ollama is unavailable
            names = ", ".join(r.name for r, _ in matches[:3])
            return f"Top matches for \"{query}\": {names}. (LLM unavailable: {exc})"