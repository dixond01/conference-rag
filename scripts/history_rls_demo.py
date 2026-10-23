"""
RLS Demo: Per-User Access on query_history
===========================================
Shows what the ANON key (public) and SERVICE key (admin, bypasses RLS)
can do with the query_history table. A test row is written under the
first existing user's id with the service key, then removed.

The cross-user case (user A reading user B's rows) needs two real
logins, so it isn't covered here: test it in the browser console.

Usage:
    python scripts/history_rls_demo.py
"""

import json

from supabase import create_client

with open('config.public.json', 'r') as f:
    public = json.load(f)
with open('config.secret.json', 'r') as f:
    secrets = json.load(f)

anon = create_client(public['SUPABASE_URL'], public['SUPABASE_ANON_KEY'])
service = create_client(public['SUPABASE_URL'], secrets['SUPABASE_SERVICE_KEY'])

TEST_QUERY = 'RLS demo row (safe to delete)'


def try_call(fn):
    try:
        result = fn()
        return f"✅ allowed ({len(result.data)} rows)"
    except Exception as e:
        return f"❌ blocked ({getattr(e, 'message', e)})"


def main():
    users = service.auth.admin.list_users()
    if not users:
        print("No users yet: sign in on the site once, then re-run.")
        return
    user_id = users[0].id

    service.table('query_history').insert({
        'user_id': user_id, 'mode': 'rag', 'query': TEST_QUERY,
        'answer': 'Only its owner should ever see this.', 'sources': [],
    }).execute()

    try:
        print("1. READ query_history")
        total = service.table('query_history').select('id', count='exact').execute().count
        print(f"   service key: {total} rows (bypasses RLS: sees every user's history)")
        print(f"   anon key:    {len(anon.table('query_history').select('id').execute().data)} rows 🔒")

        print("\n2. INSERT as anon, claiming a real user's id")
        print("   anon key:   ", try_call(lambda: anon.table('query_history').insert({
            'user_id': user_id, 'mode': 'rag', 'query': 'forged', 'sources': [],
        }).execute()))

        print("\n3. DELETE as anon")
        deleted = anon.table('query_history').delete().eq('query', TEST_QUERY).execute().data
        print(f"   anon key:    deleted {len(deleted)} rows 🔒 (RLS hides the rows, so nothing matches)")
    finally:
        service.table('query_history').delete().eq('query', TEST_QUERY).execute()
        service.table('query_history').delete().eq('query', 'forged').execute()
        print("\n🧹 Demo rows removed.")


if __name__ == '__main__':
    main()
