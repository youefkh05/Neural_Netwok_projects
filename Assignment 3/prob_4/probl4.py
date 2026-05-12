"""
Arabic Book Retrieval & RAG System — Unified Main Module
=========================================================
Combines:
  - Step 1: Book preparation, embedding generation, FAISS indexing
  - Step 2: Classical (BM25/TF-IDF) and semantic search
  - Step 3: RAG system with LLM-based answer generation

Run:  python main.py
      Opens web UI at http://localhost:5010
"""

import os, re, sys, pickle, math, torch, subprocess, time
import numpy as np
from pathlib import Path
from collections import Counter

# Setup virtual environment path
venv_site = Path(__file__).resolve().parent / ".venv" / "Lib" / "site-packages"
if venv_site.exists() and str(venv_site) not in sys.path:
    sys.path.insert(0, str(venv_site))

import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════

BOOK_FILE = "./arabic_book.txt"
OUTPUT_DIR = "./output"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
LLM_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ENABLE_LLM = True
WEB_UI_PORT = 5010


# ════════════════════════════════════════════════════════════════════════════════
# STEP 1: BOOK PREPARATION & EMBEDDINGS
# ════════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """Clean and normalize raw text."""
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'^\s*\d+\s*', '', text, flags=re.MULTILINE)
    return text.strip()


def split_into_sentences(text: str) -> list:
    """
    Split text into complete sentences.
    Handles very long sentences by splitting on semicolons.
    """
    raw_sentences = re.split(r'(?<=[.!؟])\s+', text)
    sentences = []
    
    for s in raw_sentences:
        s = s.strip()
        if len(s) > 300:
            sub = re.split(r'(?<=؛)\s+', s)
            for ss in sub:
                ss = ss.strip()
                if len(ss) > 20 and len(re.findall(r'[\u0600-\u06FF]', ss)) >= 5:
                    sentences.append(ss)
        else:
            if len(s) > 20 and len(re.findall(r'[\u0600-\u06FF]', s)) >= 5:
                sentences.append(s)
    
    return sentences


def create_chunks(sentences: list, chunk_size: int = 4, overlap: int = 2) -> list:
    """
    Group sentences into chunks with overlap for context preservation.
    """
    if len(sentences) == 0:
        return []

    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(sentences), step):
        chunk_sents = sentences[i : i + chunk_size]
        
        if len(chunk_sents) < 2:
            break

        chunk_text = ' '.join(chunk_sents)
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', chunk_text))
        
        if arabic_chars < 100:
            continue

        meta_patterns = [
            r'حقوق النشر', r'جميع الحقوق', r'محفوظة لمؤسسة',
            r'تصميم الغلاف', r'الجزء الأول', r'الجزء الثاني',
            r'الفصل الأول', r'هنداوي', r'دار النشر',
        ]
        
        is_meta = any(re.search(p, chunk_text) for p in meta_patterns)
        if not is_meta:
            chunks.append(chunk_text)

    return chunks


def prepare_book_and_embeddings(book_file: str = BOOK_FILE, output_dir: str = OUTPUT_DIR):
    """
    Main function to prepare book data and generate embeddings.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("STEP 1: BOOK PREPARATION & EMBEDDINGS")
    print("=" * 80)
    
    # Load and clean
    print("\n📖 Loading text...")
    raw_text = open(book_file, encoding='utf-8').read()
    print(f"   Raw text: {len(raw_text)} characters")
    
    cleaned_text = clean_text(raw_text)
    print(f"   Cleaned: {len(cleaned_text)} characters")
    
    # Split into sentences
    print("\n✂️  Splitting into sentences...")
    sentences = split_into_sentences(cleaned_text)
    print(f"   Total sentences: {len(sentences)}")
    
    # Create chunks
    print("\n📦 Creating chunks (chunk_size=4, overlap=2)...")
    paragraphs = create_chunks(sentences, chunk_size=4, overlap=2)
    print(f"   Total chunks: {len(paragraphs)}")
    
    if len(paragraphs) == 0:
        print("❌ No chunks generated! Check your book file.")
        return False
    
    # Generate embeddings
    print(f"\n🔤 Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL, device='cpu')
    
    print("   Generating embeddings...")
    embeddings = model.encode(
        paragraphs,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=True
    )
    embeddings = np.array(embeddings, dtype='float32')
    faiss.normalize_L2(embeddings)
    
    # Create FAISS index
    print("\n🗂️  Creating FAISS index...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    
    # Save
    print("\n💾 Saving to output/...")
    faiss.write_index(index, f"{output_dir}/faiss_index.bin")
    pickle.dump(paragraphs, open(f"{output_dir}/paragraphs.pkl", 'wb'))
    pickle.dump({
        'total_paragraphs': len(paragraphs),
        'total_sentences': len(sentences),
        'embedding_model': EMBEDDING_MODEL,
        'embedding_dimension': embeddings.shape[1],
        'index_metric': 'cosine',
        'chunk_size': 4,
        'overlap': 2,
    }, open(f"{output_dir}/metadata.pkl", 'wb'))
    
    print(f"\n✅ Done! {index.ntotal} chunks indexed")
    print(f"   Avg chunk: {sum(len(p) for p in paragraphs) // len(paragraphs)} chars")
    return True


# ════════════════════════════════════════════════════════════════════════════════
# STEP 2: SEARCH FUNCTIONS (Classical + Semantic)
# ════════════════════════════════════════════════════════════════════════════════

def normalize_arabic(text: str) -> str:
    """Normalize Arabic text."""
    text = re.sub(r'[\u0617-\u061A\u064B-\u065F]', '', text)
    text = re.sub(r'[أإآ]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    return text


def tokenize(text: str) -> list:
    """Tokenize and filter Arabic text."""
    normalized = normalize_arabic(text)
    tokens = re.findall(r'[\u0600-\u06FF]{2,}', normalized)
    
    stop_words = {
        'من', 'في', 'على', 'إلى', 'عن', 'مع', 'هذا', 'هذه', 'ذلك', 'تلك',
        'التي', 'الذي', 'الذين', 'اللاتي', 'كان', 'كانت', 'هو', 'هي', 'هم',
        'انه', 'انها', 'وكان', 'وكانت', 'قد', 'لا', 'ما', 'لم', 'لن', 'او',
        'ثم', 'حتى', 'اذا', 'ان', 'إن', 'لو', 'لكن', 'بل', 'كل', 'بعض',
        'وقد', 'وما', 'فقد', 'وهو', 'وهي', 'وان', 'فان', 'فكان', 'ولم',
        'كما', 'مما', 'عند', 'بين', 'حين', 'بعد', 'قبل', 'غير', 'ليس',
    }
    
    return [t for t in tokens if t not in stop_words and len(t) >= 2]


def is_metadata_chunk(text: str) -> bool:
    """Check if chunk is metadata/copyright."""
    meta_patterns = [
        r'حقوق النشر', r'جميع الحقوق', r'محفوظة لمؤسسة',
        r'تصميم الغلاف', r'هنداوي', r'دار النشر',
    ]
    return any(re.search(p, text) for p in meta_patterns)


def setup_search_models(paragraphs: list, embed_model):
    """Setup BM25 or TF-IDF for classical search."""
    corpus_tokens = [tokenize(p) for p in paragraphs]
    non_empty = sum(1 for t in corpus_tokens if len(t) > 0)
    print(f"   Tokenized: {non_empty}/{len(corpus_tokens)} paragraphs")
    
    try:
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi(corpus_tokens)
        print("   BM25 ready ✅")
        return bm25, corpus_tokens, True
    except ImportError:
        print("   Using TF-IDF fallback ⚠️")
        N = len(corpus_tokens)
        df = {}
        for tokens in corpus_tokens:
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1
        idf = {t: math.log((N + 1) / (f + 1)) + 1 for t, f in df.items()}
        return idf, corpus_tokens, False


def classical_search(query: str, paragraphs: list, bm25_data: tuple, top_k: int = 5) -> list:
    """Classical search using BM25 or TF-IDF."""
    bm25, corpus_tokens, use_bm25 = bm25_data
    qt = tokenize(query)
    
    if not qt:
        return []
    
    if use_bm25:
        scores = bm25.get_scores(qt)
    else:
        idf = bm25
        def tfidf_score(qt, dt):
            tf = Counter(dt)
            dl = max(len(dt), 1)
            return sum((tf.get(t, 0) / dl) * idf.get(t, 0.0) for t in qt)
        scores = np.array([tfidf_score(qt, d) for d in corpus_tokens])
    
    top_indices = np.argsort(scores)[::-1][:top_k * 3]
    
    results = []
    for idx in top_indices:
        if is_metadata_chunk(paragraphs[idx]):
            continue
        results.append({
            "rank": len(results) + 1,
            "score": round(float(scores[idx]), 4),
            "text": paragraphs[idx],
        })
        if len(results) == top_k:
            break
    
    return results


def semantic_search(query: str, paragraphs: list, embed_model, faiss_index, top_k: int = 5) -> list:
    """Semantic search using embeddings."""
    query_normalized = normalize_arabic(query)
    qv = embed_model.encode([query_normalized], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(qv)
    fetch_k = min(top_k * 3, faiss_index.ntotal)
    sims, indices = faiss_index.search(qv, fetch_k)
    
    results = []
    for i in range(fetch_k):
        if i >= len(sims[0]):
            break
        idx = indices[0][i]
        if idx < 0 or is_metadata_chunk(paragraphs[idx]):
            continue
        results.append({
            "rank": len(results) + 1,
            "score": round(float(sims[0][i]), 4),
            "text": paragraphs[idx],
        })
        if len(results) == top_k:
            break
    
    return results


# ════════════════════════════════════════════════════════════════════════════════
# STEP 3: RAG FUNCTIONS (LLM-based Answer Generation)
# ════════════════════════════════════════════════════════════════════════════════

def generate_response(messages: list, llm_model, tokenizer, max_tokens: int = 200) -> str:
    """Generate LLM response from messages (faster generation)."""
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt")

        device = next(llm_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = llm_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                num_beams=1,
                early_stopping=True
            )

        # inputs is a dict after moving tensors to device; access keys accordingly
        input_len = inputs['input_ids'].shape[1]
        response = generated_ids[0][input_len:]
        result = tokenizer.decode(response, skip_special_tokens=True).strip()
        
        if not result or len(result) < 5:
            return "لم يتمكن النموذج من توليد إجابة مناسبة. يرجى المحاولة مرة أخرى."
        
        return result
    except Exception as e:
        print(f"Error in generate_response: {e}")
        raise


def build_extractive_answer(query: str, retrieved_context: list) -> str:
    """Build a fast extractive answer when the LLM is disabled or unavailable."""
    if not retrieved_context:
        return "لم أجد في النص ما يجيب عن هذا السؤال مباشرة."

    joined = " ".join(retrieved_context)
    sentences = re.split(r'(?<=[.!؟])\s+', joined)
    selected = [s.strip() for s in sentences if len(s.strip()) > 20][:2]

    if not selected:
        selected = [retrieved_context[0].strip()]

    if len(selected) == 1:
        return f"استنادًا إلى النص: {selected[0]}"

    return f"استنادًا إلى النص: {selected[0]} {selected[1]}"


def rag_answer(query: str, paragraphs: list, embed_model, faiss_index,
               llm_model, tokenizer, top_k: int = 5) -> dict:
    """
    Generate RAG answer: retrieve context + generate response.
    Uses a fast extractive answer when the LLM is disabled.
    """
    try:
        q_vec = embed_model.encode([query]).astype('float32')
        faiss.normalize_L2(q_vec)
        distances, indices = faiss_index.search(q_vec, k=top_k)
        retrieved_context = [paragraphs[i] for i in indices[0] if i >= 0][:5]
        
        if not retrieved_context:
            retrieved_context = [paragraphs[0]]

        if ENABLE_LLM and llm_model is not None and tokenizer is not None:
            # Format context properly
            context_lines = []
            for i, para in enumerate(retrieved_context, 1):
                para = para.strip()
                if len(para) > 10:  # Only include non-empty paragraphs
                    context_lines.append(f"[{i}] {para}")
            
            context_str = "\n\n".join(context_lines)
            
            # RAG Answer with context
            rag_messages = [
                {"role": "system", "content": """أنت مساعد متخصص في كتاب "الأيام" لطه حسين.
تعليمات:
1. أجب بناءً على النصوص المقدمة أدناه فقط.
2. كن محدداً ودقيقاً في إجابتك.
3. استخدم معلومات من النص مباشرة.
4. إذا لم تجد إجابة، قل ذلك بوضوح.
5. اجعل إجابتك قصيرة ومركزة."""},
                {"role": "user", "content": f"النصوص من الكتاب:\n{context_str}\n\n---\n\nالسؤال: {query}\n\nالإجابة:"}
            ]
            
            # LLM-only answer (without context)
            llm_messages = [
                {"role": "system", "content": "أنت مساعد متخصص في الأدب العربي. أجب على السؤال بناءً على معرفتك العامة."},
                {"role": "user", "content": query}
            ]
            
            # Generate both answers with proper token limits
            rag_ans = generate_response(rag_messages, llm_model, tokenizer, max_tokens=150)
            llm_ans = generate_response(llm_messages, llm_model, tokenizer, max_tokens=120)
        else:
            rag_ans = build_extractive_answer(query, retrieved_context)
            llm_ans = "تم تعطيل توليد النموذج اللغوي لتسريع الاستجابة."

        return {
            "rag_answer": rag_ans,
            "llm_only_answer": llm_ans,
            "context": retrieved_context,
            "success": True
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "rag_answer": f"خطأ في معالجة السؤال: {str(e)}",
            "llm_only_answer": "لم يتمكن النموذج من الإجابة",
            "context": [],
            "success": False,
            "error": str(e)
        }


# ════════════════════════════════════════════════════════════════════════════════
# LOADER: Load models and data
# ════════════════════════════════════════════════════════════════════════════════

class SystemLoader:
    """Load all required models and data."""
    
    def __init__(self, output_dir: str = OUTPUT_DIR):
        self.output_dir = output_dir
        self.paragraphs = None
        self.metadata = None
        self.faiss_index = None
        self.embed_model = None
        self.llm_model = None
        self.tokenizer = None
        self.bm25_data = None
    
    def load_all(self):
        """Load all components."""
        print("\n" + "=" * 80)
        print("LOADING SYSTEM COMPONENTS")
        print("=" * 80)
        
        print("\n📚 Loading index and data...")
        self.paragraphs = pickle.load(open(f"{self.output_dir}/paragraphs.pkl", "rb"))
        self.metadata = pickle.load(open(f"{self.output_dir}/metadata.pkl", "rb"))
        self.faiss_index = faiss.read_index(f"{self.output_dir}/faiss_index.bin")
        print(f"   {len(self.paragraphs)} paragraphs | dim={self.faiss_index.d}")
        
        print(f"\n🔤 Loading embedding model...")
        self.embed_model = SentenceTransformer(self.metadata["embedding_model"], device="cpu")
        print("   ✅ Ready")
        
        if ENABLE_LLM:
            print(f"\n🤖 Loading LLM: {LLM_MODEL_ID}")
            self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_ID,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True
            )
            self.llm_model.eval()
            print("   ✅ Ready")
        else:
            self.tokenizer = None
            self.llm_model = None
            print("\n🤖 LLM loading skipped (fast extractive mode enabled)")

        print(f"\n⚙️  Setting up search models...")
        self.bm25_data = setup_search_models(self.paragraphs, self.embed_model)
        
        print("\n✅ System fully loaded!\n")
        return True


# ════════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point."""
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  🌍 ARABIC BOOK RETRIEVAL & RAG SYSTEM".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    
    # Check if data exists
    if not os.path.exists(f"{OUTPUT_DIR}/faiss_index.bin"):
        print("\n⚠️  Index not found. Running Step 1: Book Preparation...")
        if not prepare_book_and_embeddings():
            print("\n❌ Failed to prepare book!")
            return
    
    # Load all models
    loader = SystemLoader(OUTPUT_DIR)
    loader.load_all()
    
    # Start web UI
    print("\n" + "─" * 80)
    print("🚀 Starting Web UI...")
    print("─" * 80)
    
    from web_ui import create_app
    app = create_app(loader)
    
    print(f"\n✨ Web UI available at: http://localhost:{WEB_UI_PORT}")
    print(f"   Press Ctrl+C to stop\n")
    
    app.run(debug=False, port=WEB_UI_PORT, use_reloader=False)


if __name__ == '__main__':
    main()
