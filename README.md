# Assignment 3:Building an LLM-powered Game Recommendation System

In this assignment, you will implement an RAG / LLM driven recommendation engine to recommend Steam games.

## Dataset

A data set is provided, `games.json`, containing the name and description of a large amount of games. You can use the other fields as well if you want to.

## Objective

The file `recommender.py` contains a scaffold implementation which at the moment returns random games. Your task is to extend this scaffold and have an LLM-driven recommender engine that hopefully makes better predictions.

- You will most likely need an embedding model to embed the game information in vectors
- You might want to store the games in a vector database for faster lookup
- You will need to craft a prompt to send to an LLM with the retrieved context

You don't need to use a cloud provider, e.g. you can call models running in a local Ollama installation, and small models such as Phi 3.5 will probably work well enough for this task.

Open `recommender.py` and replace:

- `retrieve_candidates()`
- `rank_candidates()`
- `generate_answer()`

Keep the JSON shape returned by `search()` the same so the Flask API and frontend continue to work.

You don't need to change `app.py`, but you can make changes to the CSS sheet and HTML template to implement your own design should you want to.

Think about how you can go further than just embedding the query and comparing it with the embedded games you have, e.g.:

- Which vector database should you use (if any) and which distance metric
- Check out [`rerankers`](https://github.com/AnswerDotAI/rerankers) and see if it improves your retrieved items
- How would you handle users asking for specific release years or price ranges without adding sliders or drop downs to the web page (that would be the easy way)

## Running

First sync the uv environment:

```bash
uv sync
```

Then run:

```bash
uv run flask --app app run --debug
```

Then open `http://127.0.0.1:5000`.

