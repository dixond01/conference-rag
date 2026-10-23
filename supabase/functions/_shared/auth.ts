import { createClient } from 'npm:@supabase/supabase-js@2';

// Functions are deployed with --no-verify-jwt, so each one checks auth here.
// Returns the logged-in user, or null if the token is missing, invalid,
// or only the anon key (a valid JWT, but not a user).
export async function getUser(req: Request) {
  const token = req.headers.get('Authorization')?.replace('Bearer ', '');
  if (!token) return null;

  // SUPABASE_URL and SUPABASE_ANON_KEY are injected into every Edge Function
  const supabase = createClient(
    Deno.env.get('SUPABASE_URL')!,
    Deno.env.get('SUPABASE_ANON_KEY')!,
  );
  const { data, error } = await supabase.auth.getUser(token);
  return error ? null : data.user;
}
