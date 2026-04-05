const form = document.querySelector("#search-form");
const queryField = document.querySelector("#query");
const submitButton = document.querySelector("#submit-button");
const statusNode = document.querySelector("#status");
const answerNode = document.querySelector("#answer");
const matchesNode = document.querySelector("#matches");
const matchCountNode = document.querySelector("#match-count");
const matchTemplate = document.querySelector("#match-template");

function setStatus(message) {
  statusNode.textContent = message;
}

function renderMatches(matches) {
  matchesNode.innerHTML = "";
  matchCountNode.textContent = String(matches.length);

  if (!matches.length) {
    matchesNode.innerHTML = "<p class='empty'>No matches returned.</p>";
    return;
  }

  for (const match of matches) {
    const fragment = matchTemplate.content.cloneNode(true);
    const image = fragment.querySelector(".match-image");
    const title = fragment.querySelector(".match-title");
    const score = fragment.querySelector(".match-score");
    const description = fragment.querySelector(".match-description");
    const meta = fragment.querySelector(".match-meta");
    const tags = fragment.querySelector(".match-tags");
    const link = fragment.querySelector(".match-link");

    image.src = match.header_image || "";
    image.alt = match.name;
    image.loading = "lazy";
    title.textContent = match.name;
    score.textContent = `score ${match.score}`;
    description.textContent = match.short_description || "No short description available.";

    const genreText = (match.genres || []).slice(0, 3).join(", ") || "Unknown genre";
    const platformText = Object.entries(match.platforms || {})
      .filter(([, enabled]) => enabled)
      .map(([platform]) => platform)
      .join(", ") || "No platform data";
    meta.textContent = `${genreText} | ${match.release_date || "Unknown release date"} | ${platformText}`;

    for (const tag of (match.tags || []).slice(0, 6)) {
      const pill = document.createElement("span");
      pill.className = "tag";
      pill.textContent = tag;
      tags.appendChild(pill);
    }

    link.href = match.store_page;
    matchesNode.appendChild(fragment);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const query = queryField.value.trim();
  if (!query) {
    setStatus("Enter a description first.");
    return;
  }

  submitButton.disabled = true;
  submitButton.textContent = "Searching...";
  answerNode.textContent = "Your personal recommendation assistant is thinking hard... results incoming shortly.";
  matchesNode.innerHTML = "";
  matchCountNode.textContent = "0";

  const body = JSON.stringify({ query });
  const headers = { "Content-Type": "application/json" };

  // Fire both requests in parallel — matches are fast, LLM is slow
  const matchesPromise = fetch("/api/search", { method: "POST", headers, body });
  const answerPromise  = fetch("/api/explain", { method: "POST", headers, body });

  // Show matches as soon as BM25 returns
  try {
    const res = await matchesPromise;
    const payload = await res.json();
    if (!res.ok) throw new Error(payload.error || "Search failed");
    renderMatches(payload.matches || []);
    submitButton.disabled = false;
    submitButton.textContent = "Find matches";
    setStatus(`Done. Indexed ${payload.meta.indexed_games} games — LLM still thinking...`);
  } catch (err) {
    renderMatches([]);
    submitButton.disabled = false;
    submitButton.textContent = "Find matches";
    setStatus("Search failed.");
    answerNode.textContent = err.message;
  }

  // Fill in the LLM answer whenever it arrives (could be ~2 min later)
  try {
    const res = await answerPromise;
    const payload = await res.json();
    if (!res.ok) throw new Error(payload.error || "Explanation failed");
    answerNode.textContent = payload.answer;
    setStatus("Done.");
  } catch (err) {
    answerNode.textContent = `LLM unavailable: ${err.message}`;
    setStatus("Done (LLM failed).");
  }
});
