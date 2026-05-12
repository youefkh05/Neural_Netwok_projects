"""
Merged Web UI for Search & RAG System
=====================================
Combines Step 2 (Classical + Semantic Search) and Step 3 (RAG System)
into a single Flask application with tabbed interface.

Imported and run by main.py
"""

from flask import Flask, request, jsonify, render_template_string
import traceback
from threading import Thread
import signal
import threading

# HTML Template with merged interfaces
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>نظام البحث والإجابة — الأيام</title>
<link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #22263a;
    --accent: #6c8fff;
    --text: #e8eaf6;
    --muted: #8892b0;
    --border: #2d3250;
    --classical: #3d5a80;
    --semantic: #6a3d7a;
    --classical-light: #5b8fb9;
    --semantic-light: #a06bb5;
    --radius: 14px;
  }
  
  * { box-sizing: border-box; margin: 0; padding: 0; }
  
  body {
    font-family: 'Tajawal', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    direction: rtl;
    font-size: 18px;
  }
  
  header {
    background: linear-gradient(135deg, #1a1d27 0%, #12142a 100%);
    border-bottom: 1px solid var(--border);
    padding: 30px 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 16px;
    flex-wrap: wrap;
  }
  
  .logo { font-size: 3rem; }
  
  header h1 { font-size: 2rem; font-weight: 700; }
  header p { font-size: 1.1rem; color: var(--muted); margin-top: 5px; }
  
  .book-badge {
    background: linear-gradient(135deg, #2d1f4a, #3d2a6a);
    border: 1px solid #6a3d9a;
    border-radius: 10px;
    padding: 8px 16px;
    font-size: 1rem;
    color: #c49ef5;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  
  .tabs {
    display: flex;
    gap: 0;
    border-bottom: 2px solid var(--border);
    padding: 0 20px;
    background: rgba(0, 0, 0, 0.2);
    flex-wrap: wrap;
    justify-content: center;
  }
  
  .tab-btn {
    background: transparent;
    border: none;
    color: var(--muted);
    padding: 16px 28px;
    cursor: pointer;
    font-family: 'Tajawal', sans-serif;
    font-size: 1.2rem;
    font-weight: 500;
    border-bottom: 3px solid transparent;
    transition: all .2s;
  }
  
  .tab-btn:hover { color: var(--accent); }
  
  .tab-btn.active {
    color: var(--accent);
    border-bottom-color: var(--accent);
  }
  
  .search-section {
    max-width: 800px;
    margin: 40px auto;
    padding: 0 24px;
  }
  
  .search-box {
    display: flex;
    gap: 12px;
    align-items: center;
    background: var(--surface);
    border: 2px solid var(--border);
    border-radius: 16px;
    padding: 8px 8px 8px 16px;
    transition: border-color .2s;
  }
  
  .search-box:focus-within { border-color: var(--accent); }
  
  .search-box input {
    flex: 1;
    background: transparent;
    border: none;
    outline: none;
    font-family: 'Tajawal', sans-serif;
    font-size: 1.1rem;
    color: var(--text);
    direction: rtl;
  }
  
  .search-box input::placeholder { color: var(--muted); }
  
  .search-box select {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    color: var(--muted);
    font-family: 'Tajawal', sans-serif;
    font-size: 1rem;
    padding: 10px 12px;
    cursor: pointer;
    outline: none;
  }
  
  .btn-search {
    background: linear-gradient(135deg, var(--accent), #4a6fd4);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 28px;
    font-family: 'Tajawal', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    cursor: pointer;
    transition: opacity .2s, transform .1s;
    white-space: nowrap;
  }
  
  .btn-search:hover { opacity: .9; transform: translateY(-1px); }
  .btn-search:active { transform: translateY(0); }
  
  .categories {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 16px;
    align-items: center;
  }
  
  .cat-label { font-size: 0.78rem; color: var(--muted); white-space: nowrap; }
  .cat-group { display: flex; flex-wrap: wrap; gap: 6px; }
  
  .demo-chip {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.92rem;
    color: var(--muted);
    cursor: pointer;
    transition: all .2s;
    white-space: nowrap;
  }
  
  .demo-chip:hover { border-color: var(--accent); color: var(--accent); }
  .demo-chip.direct:hover { border-color: #52b788; color: #52b788; }
  .demo-chip.indirect:hover { border-color: #ff9f43; color: #ff9f43; }
  .demo-chip.hard:hover { border-color: #ff6b6b; color: #ff6b6b; }
  
  .tab-content {
    display: none;
  }
  
  .tab-content.active {
    display: block;
  }
  
  .loading { 
    text-align: center; 
    padding: 60px; 
    color: var(--muted); 
    display: none;
  }
  
  .spinner {
    width: 36px; 
    height: 36px;
    border: 3px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin .7s linear infinite;
    margin: 0 auto 16px;
  }
  
  @keyframes spin { to { transform: rotate(360deg); } }
  
  .results-section {
    max-width: 1400px;
    margin: 0 auto;
    padding: 0 24px 60px;
    display: none;
  }
  
  .results-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 24px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border);
    flex-wrap: wrap;
  }
  
  .query-badge {
    background: linear-gradient(135deg, #1e2340, #252a45);
    border: 1px solid var(--accent);
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 1rem;
    color: var(--accent);
    font-weight: 600;
  }
  
  .results-header span { color: var(--muted); font-size: 0.85rem; }
  
  .columns {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    align-items: start;
  }
  
  .col-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 14px 18px;
    border-radius: var(--radius) var(--radius) 0 0;
    font-weight: 700;
    font-size: 0.95rem;
  }
  
  .col-classical { background: var(--classical); }
  .col-semantic { background: var(--semantic); }
  
  .result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-top: none;
    padding: 18px 20px;
    transition: background .2s;
  }
  
  .result-card:last-child { border-radius: 0 0 var(--radius) var(--radius); }
  .result-card:hover { background: var(--surface2); }
  
  .result-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }
  
  .rank-badge {
    width: 28px; 
    height: 28px;
    border-radius: 50%;
    display: flex; 
    align-items: center; 
    justify-content: center;
    font-size: 0.8rem; 
    font-weight: 700; 
    flex-shrink: 0;
  }
  
  .classical .rank-badge { background: var(--classical-light); }
  .semantic .rank-badge { background: var(--semantic-light); }
  
  .score-badge {
    font-size: 0.75rem;
    padding: 3px 10px;
    border-radius: 20px;
    font-weight: 600;
  }
  
  .classical .score-badge {
    background: rgba(91,143,185,.15);
    color: var(--classical-light);
    border: 1px solid rgba(91,143,185,.3);
  }
  
  .semantic .score-badge {
    background: rgba(160,107,181,.15);
    color: var(--semantic-light);
    border: 1px solid rgba(160,107,181,.3);
  }
  
  .result-text {
    font-size: 1.1rem;
    line-height: 2.2;
    color: var(--text);
    text-align: justify;
    word-break: break-word;
  }
  
  .rag-section {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    margin-bottom: 20px;
  }
  
  .rag-label {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--accent);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  
  .rag-content {
    font-size: 1.05rem;
    line-height: 2.2;
    color: var(--text);
    text-align: justify;
  }
  
  .context-box {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 15px;
    margin-top: 12px;
  }
  
  .context-item {
    font-size: 1rem;
    line-height: 2;
    color: var(--muted);
    margin-bottom: 12px;
  }
  
  .context-item:last-child { margin-bottom: 0; }
  
  .error-msg {
    background: rgba(255,80,80,.1);
    border: 1px solid rgba(255,80,80,.3);
    border-radius: var(--radius);
    padding: 16px 20px;
    color: #ff8080;
    text-align: center;
    display: none;
    max-width: 900px;
    margin: 0 auto;
  }
  
  @media (max-width: 768px) {
    .columns { grid-template-columns: 1fr; }
    header { padding: 14px 20px; }
    .book-badge { margin-right: 0; }
    .search-section { margin: 24px auto; }
    .tabs { padding: 0 12px; }
    .tab-btn { padding: 12px 14px; font-size: 0.9rem; }
  }
</style>
</head>
<body>

<header>
  <div class="logo">📚</div>
  <div>
    <h1>نظام البحث والإجابة العربي</h1>
    <p>بحث ذكي + نظام أسئلة وأجوبة قائم على الاسترجاع</p>
  </div>
  <div class="book-badge" style="margin-right: auto;">📖 الأيام — طه حسين</div>
</header>

<div class="tabs">
  <button class="tab-btn active" onclick="switchTab('search')">🔍 بحث ذكي</button>
  <button class="tab-btn" onclick="switchTab('rag')">💬 أسئلة وأجوبة</button>
</div>

<!-- TAB 1: SEARCH -->
<div id="search" class="tab-content active">
  <div class="search-section">
    <div class="search-box">
      <input type="text" id="queryInput" placeholder="اكتب سؤالك أو البحث عن موضوع..." />
      <select id="topK">
        <option value="3">3 نتائج</option>
        <option value="5" selected>5 نتائج</option>
        <option value="8">8 نتائج</option>
      </select>
      <button class="btn-search" onclick="doSearch()">🔎 بحث</button>
    </div>

    <div class="categories">
      <span class="cat-label">🟢 مباشر:</span>
      <div class="cat-group" id="directChips"></div>
      <span class="cat-label">🟠 غير مباشر:</span>
      <div class="cat-group" id="indirectChips"></div>
      <span class="cat-label">🔴 صعب:</span>
      <div class="cat-group" id="hardChips"></div>
    </div>
  </div>

  <div class="loading" id="loadingSearch">
    <div class="spinner"></div>
    <p>جاري البحث...</p>
  </div>

  <div class="error-msg" id="errorMsgSearch"></div>

  <div class="results-section" id="resultsSearch">
    <div class="results-header">
      <div class="query-badge" id="queryDisplay"></div>
      <span id="resultCount"></span>
    </div>
    <div class="columns">
      <div id="classicalCol">
        <div class="col-header col-classical">🏛️ Classical Search (BM25 / TF-IDF)</div>
        <div id="classicalCards"></div>
      </div>
      <div id="semanticCol">
        <div class="col-header col-semantic">🧠 Semantic Search (Embeddings)</div>
        <div id="semanticCards"></div>
      </div>
    </div>
  </div>
</div>

<!-- TAB 2: RAG -->
<div id="rag" class="tab-content">
  <div class="search-section">
    <div class="search-box">
      <input type="text" id="queryInputRag" placeholder="اسأل سؤالاً عن كتاب الأيام..." />
      <button class="btn-search" onclick="doRagSearch()">💬 أجب</button>
    </div>

    <div class="categories">
      <span class="cat-label">🟢 مباشر:</span>
      <div class="cat-group" id="directChipsRag"></div>
      <span class="cat-label">🟠 غير مباشر:</span>
      <div class="cat-group" id="indirectChipsRag"></div>
      <span class="cat-label">🔴 صعب:</span>
      <div class="cat-group" id="hardChipsRag"></div>
    </div>
  </div>

  <div class="loading" id="loadingRag">
    <div class="spinner"></div>
    <p>⏳ Processing Retrieval & Generation... Please wait.</p>
  </div>

  <div class="error-msg" id="errorMsgRag"></div>

  <div class="results-section" id="resultsRag">
    <div class="query-badge" id="queryDisplayRag" style="margin-bottom: 20px;"></div>
    
    <div class="rag-section">
      <div class="rag-label">🔹 RAG Answer (Based on Context):</div>
      <div class="rag-content" id="ragAnswer"></div>
    </div>

    <div class="rag-section">
      <div class="rag-label">🔸 LLM Only Answer (General Knowledge):</div>
      <div class="rag-content" id="llmAnswer"></div>
    </div>

    <div class="rag-section">
      <div class="rag-label">📄 Evidence (Retrieved Context):</div>
      <div class="context-box" id="contextBox"></div>
    </div>
  </div>
</div>

<script>
  const QUERIES = {
    direct: [
      "ما الذي كان يفعله الصبي بالنعال حول سيدنا في الكتاب؟",
      "كيف كان سيدنا يمشي في طريقه إلى الكتاب؟",
      "ماذا طلب سيدنا من الأسرة أجراً على ختم الصبي للقرآن؟",
      "لماذا كان الصبي يكره عمه عند المائدة؟",
    ],
    indirect: [
      "وصف الكائنات الخرافية التي تعيش في القناة.",
      "لماذا كان الطفل يخشى الليل والعفاريت في غرفته؟",
      "كيف كانت علاقة الصبي بإخوته وأخواته؟"
    ],
    hard: [
      "تأثير وباء الكوليرا ووفاة شقيق الكاتب طالب الطب.",
      "موقف الصبي بعد نسيان القرآن أمام والده وصديقيه.",
      "الفرق بين علماء الريف الرسميين وأصحاب العلم اللدني."
    ],
  };

  function switchTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    
    // Show selected tab
    document.getElementById(tabName).classList.add('active');
    event.target.classList.add('active');
  }

  function buildChips(queries, type, containerId) {
    const el = document.getElementById(containerId);
    queries.forEach(q => {
      const c = document.createElement('div');
      c.className = `demo-chip ${type}`;
      c.textContent = q;
      c.onclick = () => {
        if (containerId.includes('Rag')) {
          document.getElementById('queryInputRag').value = q;
          doRagSearch();
        } else {
          document.getElementById('queryInput').value = q;
          doSearch();
        }
      };
      el.appendChild(c);
    });
  }

  // Build chips for both tabs
  buildChips(QUERIES.direct,   'direct',   'directChips');
  buildChips(QUERIES.indirect, 'indirect', 'indirectChips');
  buildChips(QUERIES.hard,     'hard',     'hardChips');
  buildChips(QUERIES.direct,   'direct',   'directChipsRag');
  buildChips(QUERIES.indirect, 'indirect', 'indirectChipsRag');
  buildChips(QUERIES.hard,     'hard',     'hardChipsRag');

  // Enter key handlers
  document.getElementById('queryInput').addEventListener('keydown', e => {
    if (e.key === 'Enter') doSearch();
  });

  document.getElementById('queryInputRag').addEventListener('keydown', e => {
    if (e.key === 'Enter') doRagSearch();
  });

  // ────────────────────────────────────────────────────────────────────────────
  // SEARCH TAB
  // ────────────────────────────────────────────────────────────────────────────

  async function doSearch() {
    const query = document.getElementById('queryInput').value.trim();
    const topK = document.getElementById('topK').value;
    if (!query) return;

    document.getElementById('loadingSearch').style.display = 'block';
    document.getElementById('resultsSearch').style.display = 'none';
    document.getElementById('errorMsgSearch').style.display = 'none';

    try {
      const res = await fetch('/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, top_k: parseInt(topK) })
      });
      const data = await res.json();

      document.getElementById('queryDisplay').textContent = `"${query}"`;
      document.getElementById('resultCount').textContent = `${data.classical.length} نتيجة من كل طريقة`;

      const allZero = data.classical.every(r => r.score === 0);
      renderCards('classicalCards', data.classical, 'classical', allZero);
      renderCards('semanticCards', data.semantic, 'semantic', false);

      document.getElementById('loadingSearch').style.display = 'none';
      document.getElementById('resultsSearch').style.display = 'block';
    } catch(err) {
      document.getElementById('loadingSearch').style.display = 'none';
      document.getElementById('errorMsgSearch').style.display = 'block';
      document.getElementById('errorMsgSearch').textContent = 'خطأ: ' + err.message;
    }
  }

  function renderCards(containerId, results, type, showZeroWarning) {
    const el = document.getElementById(containerId);
    el.innerHTML = '';

    if (showZeroWarning) {
      el.innerHTML += `<div style="padding:12px 18px; background:rgba(255,159,67,.1);
        border:1px solid rgba(255,159,67,.3); color:#ff9f43; font-size:0.85rem;
        border-top:none;">
        ⚠️ لم تُوجد كلمات مطابقة — النتائج مرتبة عشوائياً
      </div>`;
    }

    results.forEach(r => {
      const label = type === 'classical' ? 'درجة' : 'تشابه';
      el.innerHTML += `
        <div class="result-card ${type}">
          <div class="result-meta">
            <div class="rank-badge">#${r.rank}</div>
            <div class="score-badge">${label}: ${r.score}</div>
          </div>
          <div class="result-text">${r.text}</div>
        </div>`;
    });
  }

  // ────────────────────────────────────────────────────────────────────────────
  // RAG TAB
  // ────────────────────────────────────────────────────────────────────────────

  async function doRagSearch() {
    const query = document.getElementById('queryInputRag').value.trim();
    if (!query) return;

    document.getElementById('loadingRag').style.display = 'block';
    document.getElementById('resultsRag').style.display = 'none';
    document.getElementById('errorMsgRag').style.display = 'none';
    
    let loadingText = document.querySelector('#loadingRag p');
    const startTime = Date.now();
    let timeoutWarningShown = false;

    // Show timeout warning after 45 seconds
      const warningTimer = setTimeout(() => {
      if (loadingText) {
          loadingText.innerHTML = '⏱️ معالجة طويلة جداً... جاري الانتظار (قد تستغرق أكثر من دقيقة ونصف)<br><small>إذا استمر طويلاً، جرب سؤال أقصر</small>';
        timeoutWarningShown = true;
      }
      }, 90000);

    try {
      const controller = new AbortController();
      const timeoutHandle = setTimeout(() => controller.abort(), 190000); // 190 sec abort

      const res = await fetch('/ask_rag', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
        signal: controller.signal
      });

      clearTimeout(timeoutHandle);
      clearTimeout(warningTimer);

      const data = await res.json();
      document.getElementById('loadingRag').style.display = 'none';

      if (data.error) {
        document.getElementById('errorMsgRag').textContent = `❌ ${data.error}`;
        document.getElementById('errorMsgRag').style.display = 'block';
        return;
      }

      document.getElementById('queryDisplayRag').textContent = `"${query}"`;
      document.getElementById('ragAnswer').innerText = data.rag_answer;
      document.getElementById('llmAnswer').innerText = data.llm_only_answer;
      
      const contextHtml = data.context.map((c, i) => 
        `<div class="context-item"><strong>[نص ${i+1}]</strong><br>${c}</div>`
      ).join('');
      document.getElementById('contextBox').innerHTML = contextHtml;

      document.getElementById('resultsRag').style.display = 'block';
    } catch(e) {
      clearTimeout(warningTimer);
      document.getElementById('loadingRag').style.display = 'none';
      if (e.name === 'AbortError') {
        document.getElementById('errorMsgRag').textContent = '⏱️ انتهت المهلة الزمنية - المعالجة استغرقت وقتاً طويلاً جداً. جرب سؤال أقصر.';
      } else {
        document.getElementById('errorMsgRag').textContent = '❌ خطأ الاتصال: ' + e.message;
      }
      document.getElementById('errorMsgRag').style.display = 'block';
    }
  }
</script>

</body>
</html>
"""


def create_app(loader):
    """Create Flask app with all routes."""
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        return render_template_string(HTML_TEMPLATE)
    
    @app.route('/search', methods=['POST'])
    def search():
        """Combined search: classical + semantic."""
        try:
            data = request.json
            query = data.get('query', '').strip()
            top_k = min(int(data.get('top_k', 5)), 10)
            
            if not query:
                return jsonify({'classical': [], 'semantic': []})
            
            # Import functions from main
            from main import classical_search, semantic_search
            
            classical_results = classical_search(
                query, 
                loader.paragraphs, 
                loader.bm25_data, 
                top_k
            )
            semantic_results = semantic_search(
                query, 
                loader.paragraphs, 
                loader.embed_model, 
                loader.faiss_index, 
                top_k
            )
            
            return jsonify({
                'classical': classical_results,
                'semantic': semantic_results,
            })
        
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
    
    @app.route('/ask_rag', methods=['POST'])
    def ask_rag():
        """RAG endpoint: retrieve + generate with timeout."""
        try:
            query = request.json.get('query', '')
            
            # Import RAG function from main
            from main import rag_answer
            
            result = {'data': None, 'error': None}
            
            def run_rag():
                try:
                    result['data'] = rag_answer(
                        query,
                        loader.paragraphs,
                        loader.embed_model,
                        loader.faiss_index,
                        loader.llm_model,
                        loader.tokenizer,
                        top_k=5
                    )
                except Exception as e:
                    result['error'] = str(e)
            
            # Run in thread with 120-second timeout
            thread = threading.Thread(target=run_rag, daemon=False)
            thread.start()
            thread.join(timeout=180)
            
            if thread.is_alive():
                return jsonify({
                "error": "⏱️ Processing took too long (>3 minutes). عملية المعالجة استغرقت وقتاً طويلاً جداً. يرجى محاولة سؤال أبسط."
                }), 504
            
            if result['error']:
                return jsonify({"error": result['error']}), 500
            
            if result['data']:
                return jsonify(result['data'])
            else:
                return jsonify({"error": "No result returned"}), 500
        
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": f"Server error: {str(e)}"}), 500
    
    return app
