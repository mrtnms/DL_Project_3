from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ollama
from rank_bm25 import BM25Okapi
from rerankers import Reranker

BASE_DIR = Path(__file__).resolve().parent
GAMES_PATH = BASE_DIR / "games.json"
MAX_GAMES = 5000
DEFAULT_MATCH_COUNT = 5

# Ollama model to use — swap for any model you have pulled locally
OLLAMA_MODEL = "phi3.5"

# Toggle to compare reranker vs BM25-only results
# Set to False, restart Flask, run your query, then set back to True and restart
USE_RERANKER = True


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

        # Reranker — repo default model, loaded once at startup
        try:
            self.reranker = Reranker("mixedbread-ai/mxbai-rerank-large-v1", model_type="cross-encoder")
            print("[reranker] Loaded successfully.")
        except Exception as exc:
            print(f"[reranker] Failed to load: {exc}")
            self.reranker = None

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
        """Fast path: BM25 retrieval + ranking only, no LLM."""
        candidates = self.retrieve_candidates(query)
        ranked_matches = self.rank_candidates(query, candidates)
        results = [record.to_result(score) for record, score in ranked_matches]

        return {
            "matches": results,
            "meta": {
                "indexed_games": len(self.records),
                "retrieval_mode": "bm25+reranker" if USE_RERANKER else "bm25",
                "note": f"BM25 retrieval over {len(self.records)} games" + (", reranked via mxbai-rerank-large-v1." if USE_RERANKER else ", no reranker."),
            },
        }

    def explain(self, query: str) -> dict[str, Any]:
        """Slow path: runs BM25 again then calls the LLM."""
        candidates = self.retrieve_candidates(query)
        ranked_matches = self.rank_candidates(query, candidates)

        return {
            "answer": self.generate_answer(query, ranked_matches),
        }

    def retrieve_candidates(self, query: str) -> list[GameRecord]:
        """
        BM25 retrieval over the full game corpus.
        Returns the top-20 candidates by BM25 score for the given query.
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        scored = sorted(
            zip(self.records, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        top_candidates = [record for record, score in scored[:20]]

        # Fall back to random if corpus is empty
        if not top_candidates:
            top_candidates = random.sample(self.records, min(DEFAULT_MATCH_COUNT, len(self.records)))

        return top_candidates

    def rank_candidates(
        self, query: str, candidates: list[GameRecord]
    ) -> list[tuple[GameRecord, float]]:
        """
        Re-ranks BM25 candidates using the rerankers library default cross-encoder
        (mixedbread-ai/mxbai-rerank-large-v1). Falls back to BM25 scores if reranker fails.
        """
        if not candidates:
            return []

        docs = []
        for record in candidates:
            tags = ", ".join(record._normalize_tags(record.raw.get("tags")))
            text = f"{record.name}. {record.short_description}"
            if tags:
                text += f" Tags: {tags}."
            docs.append(text)

        try:
            if self.reranker is None or not USE_RERANKER:
                raise RuntimeError("Reranker disabled." if not USE_RERANKER else "Reranker not loaded.")
            results = self.reranker.rank(
                query=query,
                docs=docs,
                doc_ids=list(range(len(candidates))),
            )
            score_by_idx = {result.doc_id: float(result.score) for result in results}
            scored = [
                (candidates[i], score_by_idx[i])
                for i in range(len(candidates))
                if i in score_by_idx
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            print(f"[reranker] SUCCESS — top game: {scored[0][0].name}")
            return scored[:DEFAULT_MATCH_COUNT]

        except Exception as exc:
            print(f"[reranker] FAILED — falling back to BM25. Error: {exc}")
            tokenized_query = query.lower().split()
            all_scores = self.bm25.get_scores(tokenized_query)
            id_to_idx = {r.app_id: i for i, r in enumerate(self.records)}
            scored = [
                (record, float(all_scores[id_to_idx[record.app_id]]))
                for record in candidates
                if record.app_id in id_to_idx
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:DEFAULT_MATCH_COUNT]

    def generate_answer(self, query: str, matches: list[tuple[GameRecord, float]]) -> str:
        """
        Calls a local Ollama model with the top retrieved games as context
        to produce a short, helpful recommendation summary.
        """
        if not matches:
            return "No games were found matching your description."

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
            names = ", ".join(r.name for r, _ in matches[:3])
            return f"Top matches for \"{query}\": {names}. (LLM unavailable: {exc})"