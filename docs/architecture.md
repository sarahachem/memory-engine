# Architecture

## Two paths, two objectives

The read path optimizes for **precision** — an irrelevant memory can distort a response even if it's topically similar. The write path optimizes for **safety** — a wrong mutation persists until something else corrects it, so every destructive change gets a second, adversarial look before it's allowed to happen.

```
Read path (response personalization, never mutates)

  User message
    -> active-memory projection
    -> semantic candidate generation
    -> precision reranker (ID-select only)
    -> relevant memories, or none
```

```
Write path (memory capture, append-only, fails closed)

  User message
    -> eligible for memory capture?
         no  -> no processing
         yes -> atomic fact extraction
                 -> candidates per fact
                 -> batched reconciliation
                    (CREATE / UPDATE / INVALIDATE / NOOP)
                 -> destructive?
                      yes -> adversarial mutation validator
                              -> approved -> append-only event store
                              -> rejected -> no mutation
                      no  -> append-only event store
```

Every memory event — created, superseded, invalidated, deleted — is appended, never overwritten. The "active" state a caller sees is a projection over that log, which is what makes correction and deletion honest: nothing is quietly rewritten underneath a user's back.

## ReRanking

Cosine similarity scores for genuinely relevant and genuinely irrelevant memories **overlap**. There's no single threshold that's simultaneously safe and useful. A memory about a user's Spanish class can outscore a memory that's actually about their Japanese trip, because the embedding model is measuring topical proximity, not truth.

So embeddings only generate a high-recall candidate shortlist. A separate reranker makes the actual relevance call, and it's deliberately constrained: it receives candidates and returns *only the IDs it thinks are relevant* — never rewritten content, never new text. If its output doesn't validate (unknown ID, duplicate ID, malformed schema), the system fails closed to *no personalization* rather than guessing.

```
Reranker call
  -> output valid, IDs relevant   -> use those memories
  -> output valid, empty list      -> correctly show none
  -> malformed / unknown ID        -> fail closed: no memory context
```

A valid "nothing relevant" and a broken call are handled differently, and both are safe.

## Mutation gets a second model pass

Every `UPDATE` or `INVALIDATE` is reviewed by a second, adversarial prompt before it's allowed to persist — its only job is to try to falsify the proposed change. This exists because of measured evidence, not caution for its own sake: several models confidently proposed **destructive** memory changes that were wrong — treating "wants to save for a house" and "has decided to save for a house" as the same fact needing an update, when the user's actual commitment level had materially changed and both were worth keeping distinct.

That second pass is not "real" independent verification in a statistical sense — it's often the same underlying model family reviewing itself, so it can share blind spots with the first call. Its value comes from a stricter schema, a narrower input, and a different, skeptical role, not from genuine model diversity.

It's also deliberately the *only* place this happens. The general rule elsewhere is: don't bolt a second model onto a response just to double-check the first one — it adds latency and cost without an independently observable truth signal. Mutation earned the one exception because a wrong mutation is different from a wrong response: a bad answer is forgotten after the turn, but a bad mutation persists as a permanently wrong belief about the user until something else corrects it.

> Use models for semantic interpretation and bounded judgment. Use software for contracts, permissions, safety, provenance, and deterministic execution.

## Evaluation discipline

Every boundary — extraction, reconciliation, mutation validation, retrieval, reranking — has its own dataset, its own evaluator, and a model comparison before anything is wired into production use. Cheaper models are rejected for specific, named failures, not generic "worse" scores.

| Boundary | Selected model | Development | Fresh holdout | Status |
|---|---|---|---|---|
| Fact extraction | GPT-5.4 Mini | 27/27 | 10/10 | passed |
| Reconciliation (CREATE/UPDATE/INVALIDATE) | GPT-5.6 Terra | 18/18 | 12/12 | passed |
| Mutation validation | GPT-5.6 Terra | 14/14 | 12/12, 0 unsafe approvals | passed |
| Cheap-model reconciliation | Mistral 24B (local) | 7/15 | — | rejected |
| Nano-tier mutation judgment | GPT-5.4 Nano | unsafe approval found | — | rejected |
| Extraction "auditor" second pass | experimental | 87.5% vs. 100% baseline | — | not activated |

The auditor row is deliberate: adding a review step to extraction was tried, measured honestly, and it made a perfect baseline *worse* — so it isn't running. A negative result, kept and documented, is as valuable as anything that shipped.

**Discipline that made this trustworthy:**
- Prompts are tuned only against development data.
- A frozen configuration is checked once on a fresh holdout.
- If a holdout ever influences a change, it becomes development evidence and a *new* holdout must be created — a consumed holdout is never reused as if it were still unseen.
- Semantic judges are used for evaluation only, never added to production interaction.

## What's explicitly not solved yet

- **Scale.** All retrieval evaluation to date runs on small, hand-authored corpora. A separate hard-negative dataset (`evaluation/datasets/memory_retrieval_hard_negatives.json`) pushes this further — realistic corpora of 15-20 candidate memories with genuine near-duplicates, invalidated lexical decoys, and cross-lingual pairs — but production-scale recall across hundreds or thousands of memories per user is still unproven.
- **True independent validation.** The mutation validator currently shares a model family with the extractor and reconciler it's checking. It passes every measured case, but that isn't the same guarantee as a genuinely independent reviewer.
- **Human-in-the-loop confirmation.** Validated mutations apply automatically once approved. Which changes should stay silent, which should be explainable after the fact, and which actually warrant asking before applying at all, is still an open product question.
- **Lifecycle.** Not every durable memory should live forever by default. Which kinds should expire, get periodically reconfirmed, or persist until explicitly changed is undecided.
