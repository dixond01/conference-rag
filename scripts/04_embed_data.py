"""
Step 4: Generate Embeddings
===============================
Reads scripts/output/sentences.json and generates OpenAI embeddings
for each sentence. Saves the result to scripts/output/sentences_with_embeddings.json.

This is the most expensive step (~$0.60 in API costs). The output is
saved to disk so you won't lose your work if the next step fails.

Usage:
    python scripts/04_embed_data.py

Input:
    scripts/output/sentences.json  — from Step 3 (import)

Output:
    scripts/output/sentences_with_embeddings.json  — sentence records with embedding vectors

Prerequisites:
    - config.secret.json with OPENAI_API_KEY
    - scripts/output/sentences.json from the import step
"""

import json
import os
import sys

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


INPUT_FILE = os.path.join('scripts', 'output', 'sentences.json')
OUTPUT_FILE = os.path.join('scripts', 'output', 'sentences_with_embeddings.json')
MODEL_NAME = 'thenlper/gte-small'
BATCH_SIZE = 256


def load_secrets():
    with open('config.secret.json', 'r') as f:
        return json.load(f)


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ {INPUT_FILE} not found. Run scripts/03_import_data.py first.")
        sys.exit(1)

    # Load sentences
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        records = json.load(f)

    print("=" * 60)
    print(f"Generating Embeddings for {len(records):,} Sentences")
    print("=" * 60)
    print(f"   Model: gte-small (384 dimensions)")
    print(f"   Batch size: {BATCH_SIZE}\n")

    # Check if there's a partial result we can resume from
    already_embedded = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        already_embedded = len(existing)
        if already_embedded >= len(records):
            print(f"✅ All {len(records):,} sentences already have embeddings in {OUTPUT_FILE}")
            print(f"   Delete {OUTPUT_FILE} to re-generate.")
            return
        print(f"   ⏩ Resuming from sentence {already_embedded:,} (found partial output)")
        records_to_embed = records[already_embedded:]
    else:
        existing = []
        records_to_embed = records

    # Generate embeddings
    # float32: on CPU, the default float16 load ran ~12x slower
    model = SentenceTransformer(MODEL_NAME, model_kwargs={"dtype": torch.float32})

    embedded_records = list(existing)  # Start from any partial results
    errors = 0

    for i in tqdm(range(0, len(records_to_embed), BATCH_SIZE), desc="Embedding"):
        batch = records_to_embed[i:i + BATCH_SIZE]
        batch_texts = [r['text'] for r in batch]

        try:
            vectors = model.encode(batch_texts, normalize_embeddings=True)
            for record, vec in zip(batch, vectors):
                record_with_embedding = dict(record)
                record_with_embedding['embedding'] = vec.tolist()
                embedded_records.append(record_with_embedding)

        except Exception as e:
            print(f"\n   ❌ Batch error: {e}")
            errors += len(batch)

        # Save progress every 10 batches (in case of crash)
        if (i // BATCH_SIZE + 1) % 10 == 0:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(embedded_records, f, ensure_ascii=False)

    # Final save
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(embedded_records, f, ensure_ascii=False)

    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"\n✅ Embedding complete!")
    print(f"   Embedded: {len(embedded_records):,} sentences")
    if errors:
        print(f"   Errors:   {errors:,}")

    print(f"\n💾 Saved to {OUTPUT_FILE} ({file_size_mb:.1f} MB)")
    print(f"   This file is your safety net — embeddings are preserved on disk.")
    print(f"\nNext: python scripts/05_update_embeddings.py")


if __name__ == '__main__':
    main()
