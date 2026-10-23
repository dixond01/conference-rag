"""
Experiment: LLM-as-Judge for Retrieval Strategies
==================================================
Instead of hand-labeling every talk, asks an LLM on Groq to rate how
relevant each retrieved talk is to the question (1-5), then scores each
strategy from compare_strategies.py by those ratings.

The judge sees each talk's title plus its best-matching passages (the
evidence retrieval actually found), not the whole talk, to stay inside
the free tier's tokens-per-minute limit. One request per question; talk
order is shuffled so the judge can't favor whichever strategy is listed first.

Ratings are cached in scripts/output/judge_cache.json after every question,
so an interrupted run (e.g. daily rate limit) resumes where it stopped.

Usage:
    python scripts/judge_strategies.py              # first 3 pilot questions
    python scripts/judge_strategies.py 6            # first 6 pilot questions
    python scripts/judge_strategies.py eval         # 60 questions in eval_questions.json

Output:
    Printed report, also saved to scripts/output/judge_results.md
"""

import hashlib
import json
import os
import random
import re
import sys
import time

import numpy as np
import requests
import torch
from scipy.stats import wilcoxon
from sentence_transformers import SentenceTransformer

from compare_strategies import (
    CHUNK_FILES, MODEL_NAME, ON_TOPIC, OUTPUT_DIR, POOL_SIZE, STRATEGIES,
    load_chunks, rank_talks,
)


GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions'
# A different model than the app's gpt-oss-120b, so judging uses its own rate-limit bucket
JUDGE_MODEL = 'openai/gpt-oss-20b'
PASSAGES_PER_TALK = 2
MAX_PASSAGE_CHARS = 300
EVAL_QUESTIONS_FILE = os.path.join('scripts', 'eval_questions.json')
CACHE_FILE = os.path.join(OUTPUT_DIR, 'judge_cache.json')
REPORT_FILE = os.path.join(OUTPUT_DIR, 'judge_results.md')
BOOTSTRAP_SAMPLES = 10000
# Each question is judged this many times with a different talk order; scores are averaged.
# The pilot showed reordering alone can move a talk's score by 2 points.
REPEATS = 2

# Paired comparisons to test: (strategy A, strategy B, what it isolates)
COMPARISONS = [
    (('sentence', 'weighted'), ('sentence', 'average'), 'weighting, sentence chunks'),
    (('paragraph', 'weighted'), ('paragraph', 'average'), 'weighting, paragraph chunks'),
    (('paragraph', 'average'), ('sentence', 'average'), 'chunking, average ranking'),
    (('paragraph', 'weighted'), ('sentence', 'weighted'), 'chunking, weighted ranking'),
]

JUDGE_PROMPT = (
    "You are grading a search engine over General Conference talks. "
    "For each talk, rate how well it addresses the user's question, based on its title and the "
    "passages the search engine matched:\n"
    "5 = directly and substantially answers the question\n"
    "4 = clearly relevant, answers part of it\n"
    "3 = related topic, but does not really answer it\n"
    "2 = shares words or a theme, mostly off-question\n"
    "1 = unrelated\n"
    'Respond with JSON only: {"ratings": [{"id": <talk id>, "score": <1-5>}, ...]}'
)


def call_groq(api_key, messages):
    """POST to Groq, waiting out rate limits using its retry-after header."""
    for _ in range(8):
        res = requests.post(GROQ_URL, timeout=90, headers={'Authorization': f'Bearer {api_key}'}, json={
            'model': JUDGE_MODEL,
            'messages': messages,
            'temperature': 0,          # grading should be as repeatable as possible
            'reasoning_effort': 'low',
            'max_completion_tokens': 800,
            'response_format': {'type': 'json_object'},
        })
        if res.status_code != 429:
            res.raise_for_status()
            return res.json()['choices'][0]['message']['content']
        wait = float(res.headers.get('retry-after', 20))
        print(f"   ⏳ rate limited, waiting {wait:.0f}s", flush=True)
        time.sleep(wait + 1)
    raise RuntimeError('Still rate limited after 8 attempts (daily limit?). Re-run later to resume.')


def judge_question(api_key, question, candidates, repeat):
    """candidates: {title: [passages]} -> {title: score}"""
    titles = sorted(candidates)
    # Deterministic shuffle per (question, repeat): reproducible, but strategy order is hidden
    seed = int(hashlib.sha256(f'{question}#{repeat}'.encode()).hexdigest(), 16) % 2**32
    random.Random(seed).shuffle(titles)

    blocks = []
    for i, title in enumerate(titles, 1):
        passages = '\n'.join(f'  - "{p[:MAX_PASSAGE_CHARS]}"' for p in candidates[title][:PASSAGES_PER_TALK])
        blocks.append(f'[{i}] "{title}"\n{passages}')
    content = call_groq(api_key, [
        {'role': 'system', 'content': JUDGE_PROMPT},
        {'role': 'user', 'content': f'Question: {question}\n\nTalks:\n\n' + '\n\n'.join(blocks)},
    ])
    match = re.search(r'\{.*\}', content, re.DOTALL)  # tolerate stray text around the JSON
    ratings = json.loads(match.group(0))['ratings']
    return {titles[r['id'] - 1]: int(r['score']) for r in ratings if 1 <= int(r['id']) <= len(titles)}


def bootstrap_ci(values, rng):
    """95% CI for the mean by resampling questions."""
    values = np.asarray(values)
    means = rng.choice(values, size=(BOOTSTRAP_SAMPLES, len(values)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def holm(p_values):
    """Holm-Bonferroni adjusted p-values (controls false positives across several tests)."""
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else '3'
    if arg == 'eval':
        with open(EVAL_QUESTIONS_FILE, 'r', encoding='utf-8') as f:
            questions = json.load(f)
    else:
        questions = ON_TOPIC[:int(arg)]

    with open('config.secret.json', 'r') as f:
        api_key = json.load(f)['GROQ_API_KEY']

    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            cache = json.load(f)

    print("Loading embeddings and model...", flush=True)
    corpora = {name: load_chunks(path) for name, path in CHUNK_FILES.items()}
    model = SentenceTransformer(MODEL_NAME, model_kwargs={"dtype": torch.float32})
    q_vectors = model.encode(questions, normalize_embeddings=True)

    results, ratings = {}, {}
    repeat_pairs = []  # (run 1 score, run 2 score) for the same talk, to measure judge consistency
    for n, (question, q) in enumerate(zip(questions, q_vectors), 1):
        for chunking, (vectors, meta) in corpora.items():
            scores = vectors @ q
            for ranking in ('average', 'weighted'):
                results[(question, (chunking, ranking))] = rank_talks(
                    scores, meta, POOL_SIZE[chunking], ranking)

        # Every talk any strategy returned, with passages pooled from all strategies
        candidates = {}
        for strategy in STRATEGIES:
            for talk in results[(question, strategy)]:
                passages = candidates.setdefault(talk['title'], [])
                passages += [t for t in talk['texts'] if t not in passages]

        runs = []
        for repeat in range(REPEATS):
            key = f'{question}#{repeat}'
            if not all(title in cache.get(key, {}) for title in candidates):
                print(f"⚖️  [{n}/{len(questions)}] run {repeat + 1}/{REPEATS}, "
                      f"{len(candidates)} talks: {question}", flush=True)
                cache[key] = judge_question(api_key, question, candidates, repeat)
                with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                    json.dump(cache, f, indent=2, ensure_ascii=False)
            runs.append(cache[key])
        repeat_pairs.extend((runs[0][t], runs[1][t]) for t in candidates
                            if REPEATS > 1 and t in runs[0] and t in runs[1])
        ratings[question] = {
            t: float(np.mean([r[t] for r in runs if t in r]))
            for t in candidates if any(t in r for r in runs)
        }

    # Per-question score for each strategy = mean rating of its top 3 talks
    per_question = {s: [] for s in STRATEGIES}
    per_question_precision = {s: [] for s in STRATEGIES}
    unrated = 0
    for question in questions:
        for strategy in STRATEGIES:
            scores = [ratings[question].get(t['title']) for t in results[(question, strategy)]]
            unrated += scores.count(None)
            scores = [s for s in scores if s is not None]
            per_question[strategy].append(np.mean(scores) if scores else np.nan)
            per_question_precision[strategy].append(np.mean([s >= 4 for s in scores]) if scores else np.nan)
    # Keep only questions every strategy has a score for, so comparisons stay paired
    keep = ~np.any([np.isnan(per_question[s]) for s in STRATEGIES], axis=0)
    per_question = {s: np.asarray(v)[keep] for s, v in per_question.items()}
    per_question_precision = {s: np.asarray(v)[keep] for s, v in per_question_precision.items()}
    n_used = int(keep.sum())
    rng = np.random.default_rng(0)

    lines = ["# LLM-as-judge results", "",
             f"Judge: {JUDGE_MODEL} on Groq (temperature 0, shuffled talk order), rating 1-5 from title + "
             f"best {PASSAGES_PER_TALK} matched passages. {n_used} questions"
             + (f", {unrated} unrated talk slots skipped" if unrated else "")
             + f", each judged {REPEATS}x in different orders and averaged. Relevant = average score ≥ 4.", ""]
    if repeat_pairs:
        a, b = np.array(repeat_pairs).T
        lines += ["## Judge consistency (same talk, different order)", "",
                  f"- Ratings compared: {len(a)}",
                  f"- Exact agreement: {np.mean(a == b):.0%}",
                  f"- Within 1 point: {np.mean(np.abs(a - b) <= 1):.0%}",
                  f"- Mean absolute difference: {np.mean(np.abs(a - b)):.2f} points", ""]
    lines += ["## Strategy scores", "",
             "| Strategy | Mean score of top 3 | 95% CI | Precision@3 (score ≥ 4) |", "|---|---|---|---|"]
    for strategy in STRATEGIES:
        lo, hi = bootstrap_ci(per_question[strategy], rng)
        lines.append(f"| {strategy[0]} + {strategy[1]} | {per_question[strategy].mean():.2f} | "
                     f"{lo:.2f} – {hi:.2f} | {per_question_precision[strategy].mean():.0%} |")

    if n_used >= 10:
        rows, p_values = [], []
        for a, b, label in COMPARISONS:
            diff = per_question[a] - per_question[b]
            lo, hi = bootstrap_ci(diff, rng)
            nonzero = int(np.count_nonzero(diff))
            p = wilcoxon(per_question[a], per_question[b]).pvalue if nonzero else 1.0
            p_values.append(p)
            rows.append((a, b, label, diff.mean(), lo, hi, nonzero))
        adjusted = holm(np.array(p_values))
        lines += ["", "## Paired comparisons (same questions, A − B)", "",
                  "Wilcoxon signed-rank test per question; p-values Holm-adjusted across the 4 tests.", "",
                  "| A | B | Isolates | Mean difference | 95% CI | Questions that differ | Adjusted p | Significant (p < 0.05) |",
                  "|---|---|---|---|---|---|---|---|"]
        for (a, b, label, mean, lo, hi, nonzero), p in zip(rows, adjusted):
            lines.append(f"| {a[0]} + {a[1]} | {b[0]} + {b[1]} | {label} | {mean:+.2f} | "
                         f"{lo:+.2f} – {hi:+.2f} | {nonzero}/{n_used} | {p:.4f} | {'yes' if p < 0.05 else 'no'} |")
    else:
        lines += ["", "(Paired significance tests need at least 10 questions; run with `eval`.)"]

    lines += ["", "## Ratings per question", ""]
    for question in questions:
        rated = sorted(ratings[question].items(), key=lambda kv: -kv[1])
        lines.append(f"- **{question}**: " + "; ".join(f"{t} ({s})" for t, s in rated))

    report = "\n".join(lines)
    print("\n" + report)
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report + "\n")
    print(f"\n💾 Saved to {REPORT_FILE}")


if __name__ == '__main__':
    main()
