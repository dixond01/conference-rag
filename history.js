// ============================================
// MY NOTES (QUERY HISTORY) PAGE
// ============================================
// Reads the signed-in user's rows from query_history. There is no
// user_id filter in these queries: RLS (user_id = auth.uid()) makes the
// database return only the caller's own rows.

const MODE_LABELS = { rag: '🤖 Ask a Question', semantic: '🧠 Semantic', keyword: '🔍 Keyword' };
const MAX_ENTRIES = 500;

let supabaseClient = null;
let entries = [];          // newest first
let sessionStart = null;   // this sign-in's time, for the "This session" filter
let scope = 'session';

const historyScreen = document.getElementById('history-screen');
const historyList = document.getElementById('history-list');
const historyCount = document.getElementById('history-count');
const historyMessage = document.getElementById('history-message');
const modeFilter = document.getElementById('mode-filter');
const searchInput = document.getElementById('history-search');
const loading = document.getElementById('loading');

// config.js loads config.public.json asynchronously, then fires 'config-loaded'.
// The fetch can finish before this script runs, so also start right away if
// the real values are already there (same approach as initializeApp in app.js).
window.addEventListener('config-loaded', init);
if (!SUPABASE_CONFIG.url.includes('YOUR_SUPABASE')) init();

async function init() {
    if (supabaseClient) return;  // already started (both paths can fire)
    supabaseClient = window.supabase.createClient(SUPABASE_CONFIG.url, SUPABASE_CONFIG.anonKey);

    // Same login wall as the main page: the session is shared through localStorage
    const { data: { session } } = await supabaseClient.auth.getSession();
    if (!session) {
        window.location.href = 'index.html';
        return;
    }
    const user = session.user;
    sessionStart = new Date(user.last_sign_in_at || 0);
    document.getElementById('user-email').textContent = user.email;

    document.getElementById('logout-btn').addEventListener('click', async () => {
        await supabaseClient.auth.signOut();
        window.location.href = 'index.html';
    });
    document.querySelectorAll('.segment').forEach(btn => btn.addEventListener('click', () => {
        scope = btn.dataset.scope;
        document.querySelectorAll('.segment').forEach(b => b.classList.toggle('active', b === btn));
        render();
    }));
    modeFilter.addEventListener('change', render);
    searchInput.addEventListener('input', render);
    document.getElementById('export-btn').addEventListener('click', exportNotes);
    document.getElementById('clear-btn').addEventListener('click', clearShown);
    historyList.addEventListener('click', onCardAction);

    historyScreen.classList.remove('hidden');
    await loadEntries();
}

async function loadEntries() {
    loading.classList.remove('hidden');
    const { data, error } = await supabaseClient
        .from('query_history')
        .select('id, mode, query, answer, sources, created_at')
        .order('created_at', { ascending: false })
        .limit(MAX_ENTRIES);
    loading.classList.add('hidden');

    if (error) {
        showMessage(`Could not load your notes: ${error.message}`, 'error');
        return;
    }
    entries = data;
    render();
}

// Entries matching the current scope, type, and text filters
function visibleEntries() {
    const mode = modeFilter.value;
    const term = searchInput.value.trim().toLowerCase();
    return entries.filter(e =>
        (scope === 'all' || new Date(e.created_at) >= sessionStart) &&
        (mode === 'all' || e.mode === mode) &&
        (!term || [e.query, e.answer, ...e.sources.map(s => s.title)]
            .some(text => text && text.toLowerCase().includes(term)))
    );
}

function render() {
    const shown = visibleEntries();
    historyCount.textContent = `${shown.length} of ${entries.length} saved ${entries.length === 1 ? 'entry' : 'entries'}`;
    document.getElementById('export-btn').disabled = shown.length === 0;
    document.getElementById('clear-btn').disabled = shown.length === 0;

    if (shown.length === 0) {
        historyList.innerHTML = `<div class="no-results">${entries.length === 0
            ? 'No notes yet. Run a search on the main page and it will appear here.'
            : scope === 'session'
                ? 'Nothing from this session matches. Try "All history".'
                : 'No notes match these filters.'}</div>`;
        return;
    }

    let html = '';
    let currentDay = '';
    for (const entry of shown) {
        const day = new Date(entry.created_at).toLocaleDateString(undefined,
            { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' });
        if (day !== currentDay) {
            html += `<h3 class="history-day">${escapeHtml(day)}</h3>`;
            currentDay = day;
        }
        html += renderEntry(entry);
    }
    historyList.innerHTML = html;
}

function renderEntry(entry) {
    const time = new Date(entry.created_at).toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' });
    const sources = entry.sources.map(s => {
        const details = [
            s.speaker,
            s.matchCount ? `${s.matchCount} matching sentence${s.matchCount === 1 ? '' : 's'}` : '',
            typeof s.similarity === 'number' ? `${(s.similarity * 100).toFixed(0)}% similar` : '',
        ].filter(Boolean).join(' · ');
        return `<li>
            ${linkOrText(s.url, s.title)}
            <span class="result-speaker">${escapeHtml(details)}</span>
            ${s.snippet ? `<div class="history-snippet">“${escapeHtml(s.snippet)}”</div>` : ''}
        </li>`;
    }).join('');

    return `<div class="result-card history-entry${entry.mode === 'rag' ? ' rag-answer' : ''}" data-id="${entry.id}">
        <div class="result-card-header">
            <div>
                <span class="mode-badge mode-${entry.mode}">${MODE_LABELS[entry.mode]}</span>
                <span class="history-time">${escapeHtml(time)}</span>
            </div>
            <div class="history-entry-buttons">
                <button class="icon-btn" data-action="copy" title="Copy as notes">📋 Copy</button>
                <button class="icon-btn" data-action="delete" title="Delete this entry">🗑️</button>
            </div>
        </div>
        <div class="history-query">${escapeHtml(entry.query)}</div>
        ${entry.answer ? `<div class="result-sentences history-answer">${escapeHtml(entry.answer)}</div>` : ''}
        ${sources ? `<div class="result-sources"><strong>Sources:</strong></div><ul class="history-sources">${sources}</ul>` : ''}
    </div>`;
}

async function onCardAction(event) {
    const button = event.target.closest('[data-action]');
    if (!button) return;
    const id = button.closest('.history-entry').dataset.id;
    const entry = entries.find(e => e.id === id);

    if (button.dataset.action === 'copy') {
        try {
            await navigator.clipboard.writeText(entryToMarkdown(entry));
            button.textContent = '✅ Copied';
            setTimeout(() => { button.textContent = '📋 Copy'; }, 1500);
        } catch {
            showMessage('Copy failed: your browser blocked clipboard access.', 'error');
        }
    } else if (button.dataset.action === 'delete') {
        await deleteEntries([id]);
    }
}

async function deleteEntries(ids) {
    // RLS DELETE policy: these only succeed for the caller's own rows
    const { error } = await supabaseClient.from('query_history').delete().in('id', ids);
    if (error) {
        showMessage(`Delete failed: ${error.message}`, 'error');
        return;
    }
    entries = entries.filter(e => !ids.includes(e.id));
    render();
}

async function clearShown() {
    const shown = visibleEntries();
    if (!shown.length) return;
    if (!confirm(`Permanently delete ${shown.length} shown ${shown.length === 1 ? 'entry' : 'entries'}?`)) return;
    await deleteEntries(shown.map(e => e.id));
}

// Plain-text friendly Markdown, so it pastes cleanly into notes or a doc
function entryToMarkdown(entry) {
    const lines = [`## ${entry.query}`, `_${MODE_LABELS[entry.mode]} · ${new Date(entry.created_at).toLocaleString()}_`, ''];
    if (entry.answer) lines.push(entry.answer, '');
    if (entry.sources.length) {
        lines.push('**Sources:**');
        for (const s of entry.sources) {
            const title = s.url ? `[${s.title}](${s.url})` : s.title;
            lines.push(`- ${title}${s.speaker ? `, ${s.speaker}` : ''}`);
            if (s.snippet) lines.push(`  > ${s.snippet}`);
        }
        lines.push('');
    }
    return lines.join('\n');
}

function exportNotes() {
    const shown = visibleEntries();
    const today = new Date().toISOString().slice(0, 10);
    const markdown = `# Conference talk notes (${today})\n\n` + shown.map(entryToMarkdown).join('\n---\n\n');
    const link = document.createElement('a');
    link.href = URL.createObjectURL(new Blob([markdown], { type: 'text/markdown' }));
    link.download = `conference-notes-${today}.md`;
    link.click();
    URL.revokeObjectURL(link.href);
}

function linkOrText(url, title) {
    // Only link to real web URLs (sources come from the database, but never trust stored hrefs)
    if (url && /^https?:\/\//.test(url)) {
        return `<a href="${escapeHtml(url)}" target="_blank" rel="noopener" class="result-title-link">${escapeHtml(title)}</a>`;
    }
    return `<span class="result-title">${escapeHtml(title)}</span>`;
}

function showMessage(text, type) {
    historyMessage.textContent = text;
    historyMessage.className = `message ${type}`;
}

// Also escapes quotes, since some values go inside attributes (href="...")
function escapeHtml(text) {
    return String(text ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}
