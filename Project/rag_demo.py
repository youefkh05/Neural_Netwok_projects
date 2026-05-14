import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

# ============================================================
# Page Config
# ============================================================

st.set_page_config(
    page_title="RAG Interactive Demo",
    page_icon="🤖",
    layout="wide"
)

# ============================================================
# Custom CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.stTextInput input {
    background-color: #1E1E1E;
    color: white;
}

.block {
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 15px;
    background-color: #161B22;
    border: 1px solid #2F81F7;
}

.title-glow {
    color: #58A6FF;
    text-shadow: 0 0 10px #58A6FF;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# Title
# ============================================================

st.markdown(
    "<h1 class='title-glow'>Retrieval-Augmented Generation (RAG) Demo</h1>",
    unsafe_allow_html=True
)

st.write("""
This interactive demo shows how RAG systems:
1. Store documents
2. Convert text into embeddings
3. Retrieve relevant chunks
4. Generate context-aware answers
""")

# ============================================================
# Sample Knowledge Base
# ============================================================

documents = [
    "Diabetes symptoms include increased thirst, frequent urination, fatigue, and blurred vision.",

    "Artificial Neural Networks are computational models inspired by biological neurons.",

    "RAG stands for Retrieval-Augmented Generation and combines retrieval with text generation.",

    "Vector databases store embeddings and perform semantic similarity search.",

    "Transformers use self-attention mechanisms to process sequential data.",

    "Fine-tuning modifies model weights while RAG retrieves external knowledge dynamically.",

    "Machine learning enables systems to learn patterns from data.",

    "Embeddings convert text into numerical vector representations."
]

# ============================================================
# Display Documents
# ============================================================

st.subheader("📚 Knowledge Base")

for i, doc in enumerate(documents):
    st.markdown(f"""
    <div class='block'>
    <b>Document {i+1}</b><br>
    {doc}
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# User Query
# ============================================================

st.subheader("🔍 Ask a Question")

query = st.text_input(
    "Enter your query:",
    placeholder="Example: What are diabetes symptoms?"
)

# ============================================================
# Run RAG Pipeline
# ============================================================

if query:

    st.subheader("⚙️ Step 1 — Generate Embeddings")

    vectorizer = TfidfVectorizer()

    doc_vectors = vectorizer.fit_transform(documents)

    query_vector = vectorizer.transform([query])

    st.success("Text converted into vector embeddings.")

    # ========================================================
    # Similarity Search
    # ========================================================

    st.subheader("🧠 Step 2 — Semantic Retrieval")

    similarities = cosine_similarity(query_vector, doc_vectors).flatten()

    top_indices = similarities.argsort()[-3:][::-1]

    results = []

    for idx in top_indices:
        results.append({
            "Document": f"Document {idx+1}",
            "Similarity Score": round(similarities[idx], 3)
        })

    df = pd.DataFrame(results)

    st.dataframe(df, use_container_width=True)

    # ========================================================
    # Retrieved Chunks
    # ========================================================

    st.subheader("📄 Step 3 — Retrieved Documents")

    retrieved_text = ""

    for idx in top_indices:

        retrieved_text += documents[idx] + " "

        st.markdown(f"""
        <div class='block'>
        <b>Retrieved Chunk</b><br>
        {documents[idx]}
        </div>
        """, unsafe_allow_html=True)

    # ========================================================
    # Simulated LLM Generation
    # ========================================================

    st.subheader("🤖 Step 4 — Augmented Generation")

    generated_answer = f"""
Using the retrieved information, the RAG system generated the following answer:

{retrieved_text}
"""

    st.markdown(f"""
    <div class='block'>
    <b>Generated Response</b><br><br>
    {generated_answer}
    </div>
    """, unsafe_allow_html=True)

    # ========================================================
    # Explain Pipeline
    # ========================================================

    st.subheader("📌 RAG Pipeline Explanation")

    st.markdown("""
    - User query converted into embeddings
    - Similarity search performed
    - Most relevant chunks retrieved
    - Retrieved context passed to the generator
    - Final answer generated using external knowledge
    """)

# ============================================================
# Footer
# ============================================================

st.markdown("---")

st.caption("ANN Course Project — Interactive RAG Demonstration")

