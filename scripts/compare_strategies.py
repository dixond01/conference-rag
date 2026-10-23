"""
Experiment: Compare Retrieval Strategies
=========================================
Runs the same questions through a 2x2 grid of strategies and reports how
the top talks differ:

    chunking:  sentence  vs  paragraph
    ranking:   average   vs  weighted (average x ln(1 + matches))

"sentence + average" is the app's original behavior (the baseline).

Everything runs locally against the saved embedding files, so nothing in
Supabase changes and no API quota is used.

Usage:
    python scripts/compare_strategies.py

Output:
    Printed report, also saved to scripts/output/comparison.md
    scripts/output/relevance_labels.json  (a template on the first run)

Optional precision@3:
    Edit relevance_labels.json, set each talk to 1 (relevant) or 0 (not),
    then re-run. The report then includes precision@3 per strategy.
"""

import collections
import json
import math
import os

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


OUTPUT_DIR = os.path.join('scripts', 'output')
CHUNK_FILES = {
    'sentence': os.path.join(OUTPUT_DIR, 'sentences_with_embeddings.json'),
    'paragraph': os.path.join(OUTPUT_DIR, 'paragraphs_with_embeddings.json'),
}
REPORT_FILE = os.path.join(OUTPUT_DIR, 'comparison.md')
LABELS_FILE = os.path.join(OUTPUT_DIR, 'relevance_labels.json')
MODEL_NAME = 'thenlper/gte-small'

# Retrieve the same share of each corpus (~0.2%) so both chunkings see a
# similar amount of text: 50 of ~25.7k sentences, 25 of ~12.5k paragraphs
POOL_SIZE = {'sentence': 50, 'paragraph': 25}
TOP_TALKS = 3

ON_TOPIC = [
    "What have church leaders taught about prayer?",
    "How can I find peace during hard times?",
    "Why is the temple important?",
    "How do I repent and be forgiven?",
    "How can I strengthen my family?",
    "How can I help someone who is struggling with their faith?",
]
OFF_TOPIC = [
    "How do I change the oil in my car?",
    "Best recipe for chocolate chip cookies",
    "Who won the 1998 World Cup?",
]
BASELINE = ('sentence', 'average')
STRATEGIES = [(c, r) for c in ('sentence', 'paragraph') for r in ('average', 'weighted')]


def load_chunks(path):
    with open(path, 'r', encoding='utf-8') as f:
        records = json.load(f)
    vectors = np.array([r['embedding'] for r in records], dtype=np.float32)
    meta = [(r['talk_id'], r['title'], r['speaker'], r['text']) for r in records]
    return vectors, meta


def rank_talks(scores, meta, pool_size, ranking):
    """Group the top pool_size chunks by talk, then rank talks."""
    top = np.argsort(-scores)[:pool_size]
    by_talk = collections.defaultdict(list)
    texts = collections.defaultdict(list)  # matched chunks, best first
    for i in top:
        by_talk[meta[i][0]].append(float(scores[i]))
        texts[meta[i][0]].append(meta[i][3])

    def score(sims):
        avg = sum(sims) / len(sims)
        return avg * math.log(1 + len(sims)) if ranking == 'weighted' else avg

    titles = {m[0]: m[1] for m in meta}
    ranked = sorted(by_talk.items(), key=lambda kv: -score(kv[1]))[:TOP_TALKS]
    return [
        {'title': titles[t], 'matches': len(s), 'avg': sum(s) / len(s), 'texts': texts[t]}
        for t, s in ranked
    ]


def overlap(a, b):
    """Share of top talks two strategies agree on (order ignored)."""
    return len({t['title'] for t in a} & {t['title'] for t in b}) / TOP_TALKS


def main():
    print("Loading embeddings and model...")
    corpora = {name: load_chunks(path) for name, path in CHUNK_FILES.items()}
    model = SentenceTransformer(MODEL_NAME, model_kwargs={"dtype": torch.float32})

    questions = ON_TOPIC + OFF_TOPIC
    q_vectors = model.encode(questions, normalize_embeddings=True)

    results = {}   # (question, strategy) -> top talks
    top1 = collections.defaultdict(dict)  # chunking -> question -> best chunk score
    for question, q in zip(questions, q_vectors):
        for chunking, (vectors, meta) in corpora.items():
            scores = vectors @ q
            top1[chunking][question] = float(scores.max())
            for ranking in ('average', 'weighted'):
                results[(question, (chunking, ranking))] = rank_talks(
                    scores, meta, POOL_SIZE[chunking], ranking)

    labels = {}
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, 'r', encoding='utf-8') as f:
            labels = json.load(f)

    lines = ["# Retrieval strategy comparison", ""]
    lines.append(f"Baseline = {BASELINE[0]} + {BASELINE[1]} (the app's original behavior). "
                 f"Pool: top {POOL_SIZE['sentence']} sentences / top {POOL_SIZE['paragraph']} paragraphs. "
                 f"Each talk shows (matching chunks, average similarity).")

    # 1. Side-by-side top talks
    for question in ON_TOPIC:
        lines += ["", f"## {question}", "",
                  "| Strategy | #1 | #2 | #3 | Overlap with baseline |",
                  "|---|---|---|---|---|"]
        base = results[(question, BASELINE)]
        for strategy in STRATEGIES:
            talks = results[(question, strategy)]
            cells = [f"{t['title']} ({t['matches']}, {t['avg']:.3f})" for t in talks]
            cells += [''] * (TOP_TALKS - len(cells))
            lines.append(f"| {strategy[0]} + {strategy[1]} | " + " | ".join(cells)
                         + f" | {overlap(talks, base):.0%} |")

    # 2. Average overlap with the baseline
    lines += ["", "## Average overlap with baseline (on-topic questions)", "",
              "| Strategy | Overlap@3 |", "|---|---|"]
    for strategy in STRATEGIES:
        avg = np.mean([overlap(results[(q, strategy)], results[(q, BASELINE)]) for q in ON_TOPIC])
        lines.append(f"| {strategy[0]} + {strategy[1]} | {avg:.0%} |")

    # 3. Score separation: does the best chunk score tell on-topic from off-topic?
    lines += ["", "## Score separation (best single chunk score)", "",
              "Raw scores aren't comparable across chunk sizes; the gap is.", "",
              "| Chunking | On-topic min | Off-topic max | Gap |", "|---|---|---|---|"]
    for chunking in CHUNK_FILES:
        on_min = min(top1[chunking][q] for q in ON_TOPIC)
        off_max = max(top1[chunking][q] for q in OFF_TOPIC)
        lines.append(f"| {chunking} | {on_min:.3f} | {off_max:.3f} | {on_min - off_max:+.3f} |")

    # 4. Precision@3 from hand labels (optional)
    if labels:
        lines += ["", "## Precision@3 (from your relevance labels)", "",
                  "| Strategy | Precision@3 | Unlabeled |", "|---|---|---|"]
        for strategy in STRATEGIES:
            hits, total, missing = 0, 0, 0
            for q in ON_TOPIC:
                for t in results[(q, strategy)]:
                    value = labels.get(q, {}).get(t['title'])
                    if value in (0, 1):
                        hits += value
                        total += 1
                    else:
                        missing += 1
            p = f"{hits / total:.0%}" if total else "n/a"
            lines.append(f"| {strategy[0]} + {strategy[1]} | {p} | {missing} |")
    else:
        # First run: write a template listing every talk any strategy returned
        template = {
            q: {t['title']: None for s in STRATEGIES for t in results[(q, s)]}
            for q in ON_TOPIC
        }
        with open(LABELS_FILE, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2, ensure_ascii=False)
        lines += ["", f"Label relevance in {LABELS_FILE} (1 or 0) and re-run for precision@3."]

    report = "\n".join(lines)
    print("\n" + report)
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report + "\n")
    print(f"\n💾 Saved to {REPORT_FILE}")


if __name__ == '__main__':
    main()
