"""
Experiment: Paragraph Embeddings
=================================
Builds one record per paragraph from scripts/output/talks.json (using the
"paragraphs" field the scraper saves) and embeds them with the same model
and settings as 04_embed_data.py, so sentence and paragraph chunking can be
compared on equal terms.

talk_ids are copied from sentences.json (matched by url) so the same talk
has the same id in both chunkings.

Usage:
    python scripts/06_embed_paragraphs.py

Input:
    scripts/output/talks.json, scripts/output/sentences.json

Output:
    scripts/output/paragraphs_with_embeddings.json
"""

import json
import os
import sys

import torch
from sentence_transformers import SentenceTransformer


TALKS_FILE = os.path.join('scripts', 'output', 'talks.json')
SENTENCES_FILE = os.path.join('scripts', 'output', 'sentences.json')
OUTPUT_FILE = os.path.join('scripts', 'output', 'paragraphs_with_embeddings.json')
MODEL_NAME = 'thenlper/gte-small'
BATCH_SIZE = 64  # paragraphs are ~2.5x longer than sentences
MIN_CHARS = 20   # same noise filter as split_into_sentences()


def main():
    for path in (TALKS_FILE, SENTENCES_FILE):
        if not os.path.exists(path):
            print(f"❌ {path} not found. Run the scrape and import steps first.")
            sys.exit(1)

    with open(TALKS_FILE, 'r', encoding='utf-8') as f:
        talks = json.load(f)
    with open(SENTENCES_FILE, 'r', encoding='utf-8') as f:
        talk_id_by_url = {s['url']: s['talk_id'] for s in json.load(f)}

    if 'paragraphs' not in talks[0]:
        print("❌ talks.json has no 'paragraphs' field. Re-run 02_scrape_data.py.")
        sys.exit(1)

    records = []
    for talk in talks:
        talk_id = talk_id_by_url.get(talk['url'])
        if not talk_id:
            continue  # talk had no sentences imported
        paragraphs = [p for p in talk['paragraphs'] if len(p) > MIN_CHARS]
        for i, paragraph in enumerate(paragraphs, 1):
            records.append({
                'talk_id': talk_id,
                'title': talk['title'],
                'speaker': talk['speaker'],
                'calling': talk['calling'],
                'year': int(talk['year']) if talk['year'] else None,
                'season': talk['season'],
                'url': talk['url'],
                'paragraph_num': i,
                'text': paragraph,
            })

    print("=" * 60)
    print(f"Embedding {len(records):,} paragraphs from {len(talks)} talks")
    print("=" * 60)
    print(f"   Model: gte-small (384 dimensions), batch size {BATCH_SIZE}\n")

    # float32: on CPU, the default float16 load ran ~12x slower
    model = SentenceTransformer(MODEL_NAME, model_kwargs={"dtype": torch.float32})
    vectors = model.encode(
        [r['text'] for r in records],
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    for record, vec in zip(records, vectors):
        record['embedding'] = vec.tolist()

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False)

    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"\n✅ Embedded {len(records):,} paragraphs")
    print(f"💾 Saved to {OUTPUT_FILE} ({file_size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
