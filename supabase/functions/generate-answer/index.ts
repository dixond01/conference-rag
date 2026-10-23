import { corsHeaders } from '../_shared/cors.ts';
import { getUser } from '../_shared/auth.ts';

// Groq exposes an OpenAI-compatible chat completions API
const GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions';
// Override with: supabase secrets set GROQ_MODEL=...
// (llama-3.3-70b-versatile was retired; list current models at GET /openai/v1/models)
const MODEL = Deno.env.get('GROQ_MODEL') ?? 'openai/gpt-oss-120b';
// Free tier allows 8,000 tokens/minute per model, and Groq counts prompt + max output.
// ~1.5k tokens per talk keeps one request around 5k, so ~1 question every 40s.
const MAX_CHARS_PER_TALK = 6000;

interface ContextTalk {
  title: string;
  speaker: string;
  text: string;
}

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...corsHeaders, 'Content-Type': 'application/json' },
  });
}

function buildPrompt(question: string, talks: ContextTalk[]) {
  const context = talks
    .map((t, i) => `[Talk ${i + 1}] "${t.title}" by ${t.speaker}\n${t.text.slice(0, MAX_CHARS_PER_TALK)}`)
    .join('\n\n---\n\n');

  const system =
    'You answer questions about General Conference talks of The Church of Jesus Christ of Latter-day Saints. ' +
    'Use ONLY the talks provided. Cite the talks you draw from by title and speaker, e.g. ' +
    '("The Ninety and Ten", Elder Michael A. Dunn). ' +
    'If the talks do not answer the question, say so instead of guessing. ' +
    // app.js shows the answer as plain text, so markdown would appear as raw ** and |
    'Write plain text in 2-3 short paragraphs. Do not use markdown, tables, bullet points, or bold.';

  const user = `Conference talks:\n\n${context}\n\nQuestion: ${question}`;
  return { system, user };
}

Deno.serve(async (req) => {
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

  let question: unknown, contextTalks: unknown;
  try {
    ({ question, context_talks: contextTalks } = await req.json());
  } catch {
    return json({ error: 'Body must be JSON' }, 400);
  }
  if (typeof question !== 'string' || !question.trim()) {
    return json({ error: 'Missing "question"' }, 400);
  }
  if (!Array.isArray(contextTalks)) {
    return json({ error: '"context_talks" must be an array' }, 400);
  }

  // app.js pings this on page load with no talks; answer without spending Groq quota
  if (contextTalks.length === 0) {
    return json({ answer: 'No conference talks were provided to answer from.' });
  }

  const apiKey = Deno.env.get('GROQ_API_KEY');
  if (!apiKey) {
    console.error('GROQ_API_KEY secret is not set');
    return json({ error: 'Server is missing its API key' }, 500);
  }

  const { system, user: userPrompt } = buildPrompt(question, contextTalks as ContextTalk[]);

  const groqRes = await fetch(GROQ_URL, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: MODEL,
      messages: [
        { role: 'system', content: system },
        { role: 'user', content: userPrompt },
      ],
      temperature: 0.3,
      // gpt-oss is a reasoning model; low effort cuts hidden reasoning tokens
      reasoning_effort: 'low',
      max_completion_tokens: 800,
    }),
  });

  if (!groqRes.ok) {
    // Log Groq's message server-side; don't forward provider details to the browser
    console.error('Groq error', groqRes.status, await groqRes.text());
    if (groqRes.status === 429) {
      // Groq sends how long to wait in the retry-after header (seconds)
      const retryAfter = Math.ceil(Number(groqRes.headers.get('retry-after')) || 60);
      return json({
        error: `Rate limit reached. Try again in ${retryAfter} seconds.`,
        code: 'rate_limited',
        retry_after_seconds: retryAfter,
      }, 429);
    }
    return json({ error: 'The AI service failed to generate an answer.', code: 'upstream_error' }, 502);
  }

  const data = await groqRes.json();
  const answer = data.choices?.[0]?.message?.content ?? '';
  return json({ answer });
});
