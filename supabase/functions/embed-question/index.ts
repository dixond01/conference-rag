import 'jsr:@supabase/functions-js/edge-runtime.d.ts';
import { corsHeaders } from '../_shared/cors.ts';
import { getUser } from '../_shared/auth.ts';

// Same model as scripts/04_embed_data.py (thenlper/gte-small, 384 dims).
// Created once per warm instance and reused across requests.
const session = new Supabase.ai.Session('gte-small');

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...corsHeaders, 'Content-Type': 'application/json' },
  });
}

Deno.serve(async (req) => {
  // CORS preflight
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders });
  }
  if (req.method !== 'POST') {
    return json({ error: 'Use POST' }, 405);
  }

  const user = await getUser(req);
  if (!user) {
    return json({ error: 'Not authenticated' }, 401);
  }

  let question: unknown;
  try {
    ({ question } = await req.json());
  } catch {
    return json({ error: 'Body must be JSON' }, 400);
  }
  if (typeof question !== 'string' || !question.trim()) {
    return json({ error: 'Missing "question"' }, 400);
  }

  try {
    // Must match the local run: mean pooling + normalize_embeddings=True
    const embedding = await session.run(question, { mean_pool: true, normalize: true });
    return json({ embedding });
  } catch (err) {
    console.error('Embedding failed:', err);
    return json({ error: 'Failed to generate embedding' }, 500);
  }
});
