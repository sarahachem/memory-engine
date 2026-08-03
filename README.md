# Memory Engine

A memory subsystem for conversational AI: deciding what's worth remembering about a user, retrieving it precisely, and changing it safely — with an evaluation harness that backs every one of those claims.

See [docs/architecture.md](docs/architecture.md) for the full design rationale, the model-comparison evidence, and what's still open.

## What this is

Most chatbot "memory" is store-what-was-said-and-repeat-it-back. This module treats memory as two separate problems with different failure modes:

- **Read path** — retrieve relevant memories for a response. Optimizes for *precision*: an irrelevant memory can distort a response even if it's topically similar.
- **Write path** — decide whether a message should change what's remembered. Optimizes for *safety*: extract atomic facts, reconcile them against existing memory (create/update/invalidate/no-op), and require a second, adversarial model pass before any destructive change is allowed to persist.

Every memory event is appended, never overwritten — the "active" state is a projection over that log, which is what makes correction and deletion honest rather than silent rewrites.

## Running it

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env   # fill in OPENAI_API_KEY
.venv/bin/uvicorn memory_engine.main:app --reload
```

Requires a local [Ollama](https://ollama.com) instance running the embedding model (`embeddinggemma` by default) for retrieval — extraction, reconciliation, and validation call the configured hosted OpenAI models.

### Try it

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "demo", "message": "I have a peanut allergy, just so you know."}'
```

This runs the full write path (extract → reconcile → validate → store) and the full read path (retrieve → rerank) against the same message in one call, so you can see both sides of the module working end to end.

```bash
curl http://localhost:8000/memory/demo                     # what's remembered, with provenance
curl -X DELETE http://localhost:8000/memory/demo/<id>       # true erasure, not a tombstone
```

## Testing

```bash
.venv/bin/pytest
```

209 tests covering extraction, reconciliation, mutation validation, retrieval, reranking, persistence, and true erasure.

## Evaluating

Every boundary has its own dataset and evaluator under `evaluation/`, following the same discipline throughout: prompts are tuned only against development data; a frozen holdout is run exactly once and never reused.

```bash
.venv/bin/python -m evaluation.evaluators.memory_extraction_evaluator --model gpt-5.4-mini
.venv/bin/python -m evaluation.evaluators.memory_reconciliation_evaluator --model gpt-5.6-terra
.venv/bin/python -m evaluation.evaluators.memory_mutation_validator_evaluator --model gpt-5.6-terra
.venv/bin/python -m evaluation.evaluators.memory_retrieval_evaluator
.venv/bin/python -m evaluation.evaluators.context_memory_evaluator
```

## Layout

```
memory_engine/
  memory/            the module itself — extraction, reconciliation,
                      validation, retrieval, reranking, persistence
  llm.py             provider-neutral LLM client (OpenAI + Ollama)
  embeddings.py      provider-neutral embedding client (Ollama)
  models.py          the small set of shared domain types
  api.py             a minimal FastAPI surface so the module is runnable,
                      not just importable
evaluation/
  datasets/          one JSON dataset per boundary, dev + frozen holdouts
  evaluators/        one evaluator per boundary
tests/               209 tests
```
