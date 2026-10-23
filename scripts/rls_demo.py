"""
RLS Demo: See Row Level Security in Action
============================================
Compares what the ANON key (public, in the browser) and the SERVICE key
(secret, bypasses RLS) can see and do on each table.

Because sentence_embeddings is still empty, the script first inserts one
test row with the service key, so "0 rows" from the anon key actually
proves RLS is hiding data rather than the table just being empty.
The test row is deleted at the end.

Usage:
    python scripts/rls_demo.py
"""

import json
import uuid

from supabase import create_client

with open('config.public.json', 'r') as f:
    public = json.load(f)
with open('config.secret.json', 'r') as f:
    secrets = json.load(f)

anon = create_client(public['SUPABASE_URL'], public['SUPABASE_ANON_KEY'])
service = create_client(public['SUPABASE_URL'], secrets['SUPABASE_SERVICE_KEY'])

TEST_TITLE = 'RLS demo row (safe to delete)'


def count_rows(client, table):
    result = client.table(table).select('id', count='exact').execute()
    return result.count or 0


def try_insert(client, table, row):
    try:
        client.table(table).insert(row).execute()
        return "✅ allowed"
    except Exception as e:
        return f"❌ blocked ({getattr(e, 'message', e)})"


def main():
    print("=" * 60)
    print("RLS Demo")
    print("=" * 60)

    # Setup: put one row in sentence_embeddings using the service key
    service.table('sentence_embeddings').insert({
        'talk_id': str(uuid.uuid4()),
        'title': TEST_TITLE,
        'text': 'This row exists, but can you see it?',
    }).execute()

    try:
        print("\n1. READ sentence_embeddings  (policy: TO authenticated)")
        print(f"   service key: {count_rows(service, 'sentence_embeddings')} rows  (bypasses RLS)")
        print(f"   anon key:    {count_rows(anon, 'sentence_embeddings')} rows  🔒 hidden by RLS")

        print("\n2. WRITE sentence_embeddings  (no INSERT policy at all)")
        print("   anon key:   ", try_insert(anon, 'sentence_embeddings', {
            'talk_id': str(uuid.uuid4()), 'title': 'hacker', 'text': 'injected',
        }))

        print("\n3. WRITE page_views  (policy: TO anon, authenticated)")
        print("   anon key:   ", try_insert(anon, 'page_views', {
            'page_url': 'rls_demo.py', 'user_agent': 'rls demo script',
        }))

        print("\n4. READ page_views  (policy: TO anon, authenticated)")
        print(f"   anon key:    {count_rows(anon, 'page_views')} rows  🌍 public")
    finally:
        # Cleanup: remove the demo rows
        service.table('sentence_embeddings').delete().eq('title', TEST_TITLE).execute()
        service.table('page_views').delete().eq('page_url', 'rls_demo.py').execute()
        print("\n🧹 Demo rows removed.")


if __name__ == '__main__':
    main()
