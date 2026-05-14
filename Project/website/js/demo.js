// Lightweight TF-IDF + cosine similarity demo implemented in the browser
const documents = [
  "Diabetes symptoms include increased thirst, frequent urination, fatigue, and blurred vision.",
  "Artificial Neural Networks are computational models inspired by biological neurons.",
  "RAG stands for Retrieval-Augmented Generation and combines retrieval with text generation.",
  "Vector databases store embeddings and perform semantic similarity search.",
  "Transformers use self-attention mechanisms to process sequential data.",
  "Fine-tuning modifies model weights while RAG retrieves external knowledge dynamically.",
  "Machine learning enables systems to learn patterns from data.",
  "Embeddings convert text into numerical vector representations."
];

function tokenize(text){
  return text.toLowerCase().replace(/[^a-z0-9\s]/g,'').split(/\s+/).filter(Boolean);
}

function buildTfIdf(docs){
  const toks = docs.map(d=>tokenize(d));
  const vocab = new Map();
  toks.forEach(t=>t.forEach(w=>{ if(!vocab.has(w)) vocab.set(w,vocab.size); }));
  const N = docs.length;
  const df = new Array(vocab.size).fill(0);
  const tf = toks.map(t => {
    const row = new Array(vocab.size).fill(0);
    t.forEach(w=>{ row[vocab.get(w)] += 1; });
    const unique = new Set(t);
    unique.forEach(w=>{ df[vocab.get(w)] += 1; });
    return row;
  });
  const idf = df.map(c => Math.log((1 + N) / (1 + c)) + 1);
  const tfidf = tf.map(row => row.map((v,i) => v * idf[i]));
  return {vocab, tfidf, idf};
}

function dot(a,b){ let s=0; for(let i=0;i<a.length;i++) s+=a[i]*b[i]; return s }
function norm(a){ return Math.sqrt(dot(a,a)) }
function cosine(a,b){ const na=norm(a), nb=norm(b); if(na===0||nb===0) return 0; return dot(a,b)/(na*nb) }

const model = buildTfIdf(documents);

function encode(text){
  const toks = tokenize(text);
  const vec = new Array(model.vocab.size).fill(0);
  toks.forEach(w=>{ if(model.vocab.has(w)) vec[model.vocab.get(w)] += 1 });
  // apply idf weighting
  return vec.map((v,i)=>v * model.idf[i]);
}

function runQuery(q){
  const qv = encode(q);
  const scores = model.tfidf.map(dv=>cosine(qv,dv));
  const indices = scores.map((s,i)=>[s,i]).sort((a,b)=>b[0]-a[0]).slice(0,3);
  return indices.map(([score,i])=>({index:i,score:Math.round(score*1000)/1000,text:documents[i]}));
}

document.addEventListener('DOMContentLoaded', ()=>{
  const btn = document.getElementById('run');
  const input = document.getElementById('query');
  const resultsEl = document.getElementById('results');
  const answerEl = document.getElementById('answer');
  const explain = document.getElementById('explain');

  btn.addEventListener('click', ()=>{
    const q = input.value.trim();
    if(!q) return; explain.textContent = 'Running pipeline: embedding → retrieval → generation';
    const res = runQuery(q);
    resultsEl.innerHTML = '';
    let agg = '';
    res.forEach(r=>{
      const div = document.createElement('div'); div.className='block';
      div.innerHTML = `<strong>Doc ${r.index+1}</strong> <span class='muted'>(${r.score})</span><div>${r.text}</div>`;
      resultsEl.appendChild(div);
      agg += r.text + ' ';
    });
    answerEl.textContent = 'Using the retrieved chunks, a generator could produce:\n\n' + agg;
  });
});
