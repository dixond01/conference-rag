"""
Step 7 Feature: Query History Table
====================================
Creates the query_history table that powers the "My notes" page
(history.html). Each signed-in user can read, add, and delete ONLY
their own rows, enforced by Row Level Security with auth.uid().

Compare with the other tables:
    page_views           anyone can read/write          (TO anon, authenticated USING true)
    sentence_embeddings  any signed-in user can read    (TO authenticated USING true)
    query_history        each user sees only their rows (USING user_id = auth.uid())

Usage:
    python scripts/07_create_history_table.py

Prerequisites:
    - config.secret.json with SUPABASE_ACCESS_TOKEN and SUPABASE_PROJECT_REF
"""

import json
import sys

import requests


with open('config.secret.json', 'r') as f:
    secrets = json.load(f)

SUPABASE_ACCESS_TOKEN = secrets['SUPABASE_ACCESS_TOKEN']
SUPABASE_PROJECT_REF = secrets['SUPABASE_PROJECT_REF']

HISTORY_SQL = """
CREATE TABLE IF NOT EXISTS query_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- Filled in from the caller's JWT, so the browser never sends a user_id
    user_id UUID NOT NULL DEFAULT auth.uid() REFERENCES auth.users ON DELETE CASCADE,
    mode TEXT NOT NULL CHECK (mode IN ('keyword', 'semantic', 'rag')),
    query TEXT NOT NULL,
    answer TEXT,                          -- RAG answer; NULL for keyword/semantic
    sources JSONB NOT NULL DEFAULT '[]',  -- [{title, speaker, url, similarity, matchCount, snippet}]
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- The history page always asks for one user's rows, newest first
CREATE INDEX IF NOT EXISTS query_history_user_created_idx
ON query_history (user_id, created_at DESC);

ALTER TABLE query_history ENABLE ROW LEVEL SECURITY;

-- Own rows only. There is deliberately no UPDATE policy: history can't be edited.
DROP POLICY IF EXISTS "Users read own history" ON query_history;
CREATE POLICY "Users read own history"
ON query_history FOR SELECT
TO authenticated
USING (user_id = auth.uid());

DROP POLICY IF EXISTS "Users add own history" ON query_history;
CREATE POLICY "Users add own history"
ON query_history FOR INSERT
TO authenticated
WITH CHECK (user_id = auth.uid());   -- blocks inserting rows under someone else's id

DROP POLICY IF EXISTS "Users delete own history" ON query_history;
CREATE POLICY "Users delete own history"
ON query_history FOR DELETE
TO authenticated
USING (user_id = auth.uid());

NOTIFY pgrst, 'reload schema';
"""


def main():
    print("=" * 60)
    print("Creating query_history table")
    print("=" * 60)

    url = f"https://api.supabase.com/v1/projects/{SUPABASE_PROJECT_REF}/database/query"
    headers = {
        "Authorization": f"Bearer {SUPABASE_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    resp = requests.post(url, headers=headers, json={"query": HISTORY_SQL})
    if resp.status_code not in (200, 201):
        print(f"❌ Failed: {resp.status_code}")
        print(resp.text[:500])
        sys.exit(1)

    check = requests.post(url, headers=headers, json={"query": """
        SELECT policyname, cmd FROM pg_policies
        WHERE tablename = 'query_history' ORDER BY cmd;
    """})
    print("✅ query_history created with RLS policies:")
    for row in check.json():
        print(f"   {row['cmd']:<7} {row['policyname']}")
    print("\nNext: open the site, run a search, then click 'My notes'.")


if __name__ == '__main__':
    main()
