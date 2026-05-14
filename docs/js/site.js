const PROJECT = {
  liveDemo: 'https://neuralnetwokprojects-fspmceeeqzbjce4bxfnvj2.streamlit.app/',
  reportPdf: 'assets/final-project-report.pdf',
  mcqPdf: 'assets/final-project-mcq.pdf',
  pptxFile: 'assets/rag-presentation.pptx',
  briefPdf: 'assets/project-brief.pdf'
};

const RAG_EXPLAINER_STEPS = {
  definition: {
    title: 'Retrieval-Augmented Generation',
    summary: 'RAG combines document retrieval with language generation so answers are grounded in external information instead of relying on model memory alone.',
    metrics: [
      { value: '2 layers', label: 'Retrieval + generation' },
      { value: 'Less drift', label: 'Stronger factual grounding' },
      { value: 'Fresh data', label: 'Knowledge updates without retraining' }
    ],
    notes: [
      'Use RAG when the knowledge base changes often or needs citations.',
      'The retrieval stage controls which evidence the model receives.',
      'The generator then turns that evidence into a fluent answer.'
    ],
    tags: ['Retrieval', 'Embeddings', 'LLM'],
    meter: 28,
    activeNodes: ['query']
  },
  workflow: {
    title: 'The RAG workflow',
    summary: 'A user query is embedded, matched against a vector store, and then combined with the best passages before the model responds.',
    metrics: [
      { value: '3 stages', label: 'Query, search, answer' },
      { value: 'Vector search', label: 'Semantic matching' },
      { value: 'Prompt build', label: 'Context is assembled before generation' }
    ],
    notes: [
      'The query is converted into a vector representation first.',
      'Similarity search finds the closest document chunks.',
      'Those chunks are appended to the prompt for the LLM.'
    ],
    tags: ['Query', 'Vector DB', 'Prompting'],
    meter: 52,
    activeNodes: ['query', 'retrieve']
  },
  retrieval: {
    title: 'Retrieval is the grounding step',
    summary: 'This stage determines which chunks are relevant enough to trust, so chunking quality and similarity scoring matter a lot.',
    metrics: [
      { value: 'Chunking', label: 'Smaller sections improve precision' },
      { value: 'Similarity', label: 'Cosine scores rank evidence' },
      { value: 'Top-k', label: 'Only the best matches are kept' }
    ],
    notes: [
      'Good retrieval depends on clean preprocessing and sensible chunk sizes.',
      'Embeddings capture meaning, not just exact keywords.',
      'If the wrong passages are retrieved, the final answer suffers.'
    ],
    tags: ['Chunking', 'Cosine similarity', 'Top-k search'],
    meter: 72,
    activeNodes: ['retrieve']
  },
  generation: {
    title: 'Generation turns evidence into an answer',
    summary: 'Once context is attached, the language model can explain, summarize or synthesize a response that is more accurate and more useful.',
    metrics: [
      { value: 'Context-aware', label: 'Answer uses retrieved evidence' },
      { value: 'Readable', label: 'Produces fluent prose' },
      { value: 'Safer', label: 'Lower hallucination risk' }
    ],
    notes: [
      'The model does not invent the knowledge base itself; it reasons over what was retrieved.',
      'Prompt design affects how well the evidence is used.',
      'This is where clarity, tone and final answer quality are decided.'
    ],
    tags: ['LLM', 'Augmentation', 'Answering'],
    meter: 86,
    activeNodes: ['query', 'retrieve', 'llm']
  },
  value: {
    title: 'Why RAG matters in practice',
    summary: 'RAG is useful because it keeps answers current, makes the system more explainable, and avoids the cost of retraining for every knowledge update.',
    metrics: [
      { value: 'Lower cost', label: 'No full fine-tuning cycle' },
      { value: 'More control', label: 'Documents can be curated' },
      { value: 'Better UX', label: 'Users see a grounded workflow' }
    ],
    notes: [
      'It is especially strong for project demos, assistants and knowledge tools.',
      'The tradeoff is that retrieval quality must be engineered carefully.',
      'That is why this homepage links the report, quiz, slides and live demo together.'
    ],
    tags: ['Fresh knowledge', 'Explainability', 'Practical AI'],
    meter: 94,
    activeNodes: ['retrieve', 'llm']
  }
};

const PROJECT_STORY_STEPS = {
  architecture: {
    title: 'RAG architecture',
    summary: 'RAG combines a retriever and a generator. The retriever finds evidence, and the generator uses that evidence to produce a grounded answer.',
    quote: 'Search first, then generate.',
    metrics: [
      { value: '2 parts', label: 'Retriever + generator' },
      { value: 'Grounded', label: 'Answers rely on evidence' },
      { value: 'Dynamic', label: 'Works with updated documents' }
    ],
    notes: [
      'The key RAG idea is to attach relevant context before asking the LLM to answer.',
      'This makes the system more accurate than pure generation on many knowledge tasks.',
      'The architecture is simple, but the retrieval quality matters a lot.'
    ],
    tags: ['Retriever', 'Generator', 'Context'],
    active: ['query', 'retrieve', 'llm']
  },
  retrieval: {
    title: 'Retrieval layer',
    summary: 'The retriever converts the query into a searchable representation, compares it with stored chunks, and returns the most relevant passages.',
    quote: 'Retrieval quality decides how much useful evidence reaches the model.',
    metrics: [
      { value: 'Chunking', label: 'Documents are split into smaller pieces' },
      { value: 'Embeddings', label: 'Text is represented as vectors' },
      { value: 'Cosine similarity', label: 'Scores how closely query and chunk match' }
    ],
    notes: [
      'Good retrieval usually depends on sensible chunk size and clean text preparation.',
      'Semantic search lets the system find similar meaning rather than exact keywords.',
      'If retrieval is weak, the generator receives the wrong context and the answer degrades.'
    ],
    tags: ['Chunking', 'Embeddings', 'Semantic search'],
    active: ['query', 'retrieve']
  },
  generation: {
    title: 'Augmentation and generation',
    summary: 'Retrieved passages are appended to the prompt, giving the model the context it needs to produce a better answer.',
    quote: 'The generator should synthesize evidence, not invent it.',
    metrics: [
      { value: 'Prompt build', label: 'Retrieved context is added to the input' },
      { value: 'Better grounding', label: 'Reduces unsupported claims' },
      { value: 'Readable output', label: 'Produces fluent final answers' }
    ],
    notes: [
      'Prompt design affects how well the retrieved context is used.',
      'The generator can summarize, explain or compose a response from the evidence.',
      'This stage is where clarity and usefulness are decided.'
    ],
    tags: ['Prompt augmentation', 'LLM', 'Grounding'],
    active: ['retrieve', 'llm']
  },
  tradeoffs: {
    title: 'Strengths and limits',
    summary: 'RAG improves freshness and reduces hallucination risk, but it still depends on strong retrieval, good chunking and careful prompt design.',
    quote: 'RAG is a powerful pattern, but it is only as reliable as the evidence it retrieves.',
    metrics: [
      { value: 'Fresh knowledge', label: 'Documents can be updated without retraining' },
      { value: 'Lower hallucination', label: 'Responses are more grounded' },
      { value: 'Retrieval dependency', label: 'Bad search leads to bad answers' }
    ],
    notes: [
      'RAG is especially useful when information changes often.',
      'It does not remove all errors, because poor retrieval can still mislead the model.',
      'That tradeoff is what makes RAG interesting and practical at the same time.'
    ],
    tags: ['Fresh data', 'Hallucination risk', 'Tradeoff'],
    active: ['retrieve', 'llm']
  }
};

const RAG_GLOSSARY = {
  embeddings: {
    title: 'Embeddings',
    summary: 'Numerical vector representations of text that capture semantic meaning and relationships.',
    metrics: [
      { value: 'Text to vector', label: 'Turns language into math-friendly form' },
      { value: 'Semantic meaning', label: 'Similar ideas get nearby vectors' },
      { value: 'Retrieval input', label: 'Used to search the knowledge base' }
    ],
    notes: [
      'Embeddings are the bridge between text and vector search.',
      'They make it possible to compare a query with stored document chunks.',
      'The project demo uses this idea through TF-IDF-style vectorization.'
    ]
  },
  vector_database: {
    title: 'Vector database',
    summary: 'A database designed to store embeddings and perform semantic similarity search efficiently.',
    metrics: [
      { value: 'Fast search', label: 'Finds similar vectors quickly' },
      { value: 'Top-k results', label: 'Returns the best matches' },
      { value: 'RAG core', label: 'Holds the evidence for generation' }
    ],
    notes: [
      'The MCQ sheet explicitly connects vector databases with RAG retrieval.',
      'They are what allow the system to look beyond exact keyword matching.',
      'Without them, the retrieval stage becomes much less useful.'
    ]
  },
  cosine_similarity: {
    title: 'Cosine similarity',
    summary: 'A metric that measures angular similarity between embedding vectors and is commonly used for retrieval ranking.',
    metrics: [
      { value: 'Angular match', label: 'Compares direction, not just distance' },
      { value: 'Ranking score', label: 'Orders candidate chunks' },
      { value: 'Retrieval signal', label: 'Helps choose relevant evidence' }
    ],
    notes: [
      'The demo uses cosine similarity to score documents against the query.',
      'This is one of the easiest ways to show semantic retrieval in a classroom project.',
      'The better the score, the more relevant the retrieved chunk should be.'
    ]
  },
  chunking: {
    title: 'Chunking',
    summary: 'Splitting documents into smaller sections so retrieval becomes more precise and context is easier to manage.',
    metrics: [
      { value: 'Smaller pieces', label: 'Improves retrieval precision' },
      { value: 'Context control', label: 'Keeps passages focused' },
      { value: 'Tradeoff', label: 'Too small can lose meaning' }
    ],
    notes: [
      'The MCQ and report both emphasize that chunk size matters.',
      'Good chunking helps prevent overly broad or noisy retrieval results.',
      'It is one of the main engineering decisions in a RAG pipeline.'
    ]
  },
  semantic_search: {
    title: 'Semantic search',
    summary: 'Searching based on meaning and context rather than exact keyword matching.',
    metrics: [
      { value: 'Meaning-based', label: 'Finds related ideas even with different wording' },
      { value: 'Query driven', label: 'Uses the user question as the anchor' },
      { value: 'RAG friendly', label: 'Essential for evidence retrieval' }
    ],
    notes: [
      'This is the retrieval behavior that makes RAG feel intelligent.',
      'It helps the system find documents that answer the intent, not just the literal words.',
      'That is why vector embeddings are paired with similarity search.'
    ]
  },
  hallucination: {
    title: 'Hallucination',
    summary: 'Incorrect or invented output from a language model when it lacks grounded evidence.',
    metrics: [
      { value: 'Main risk', label: 'RAG is designed to reduce it' },
      { value: 'Grounding', label: 'Retrieval supplies factual context' },
      { value: 'Quality issue', label: 'Poor retrieval increases errors' }
    ],
    notes: [
      'The MCQ file frames this as one of the key problems RAG helps reduce.',
      'RAG does not magically remove hallucination, but it lowers the risk by grounding the answer.',
      'This is one reason the project emphasizes retrieval quality so heavily.'
    ]
  },
  fine_tuning: {
    title: 'Fine-tuning vs RAG',
    summary: 'Fine-tuning changes model weights, while RAG retrieves external knowledge dynamically for faster updates and lower cost.',
    metrics: [
      { value: 'Weight update', label: 'Fine-tuning retrains the model' },
      { value: 'External data', label: 'RAG pulls from documents' },
      { value: 'Update speed', label: 'RAG is easier to refresh' }
    ],
    notes: [
      'The report notes that some AI explanations drifted too deeply into fine-tuning details.',
      'This is why the homepage keeps the comparison short and practical.',
      'For a project demo, RAG gives a clearer and more flexible workflow.'
    ]
  }
};

const ADVANCED_RAG_TOPICS = {
  rewrite: {
    title: 'Improve the question before search',
    summary: 'Query rewriting turns a short or vague user question into a clearer retrieval query so the search stage has more useful terms and less ambiguity.',
    quote: 'Better search starts before search.',
    flow: ['Question', 'Rewrite', 'Search', 'Evidence'],
    metrics: [
      { value: 'Clarified intent', label: 'Rewrites missing context into search-friendly wording' },
      { value: 'Fewer misses', label: 'Reduces retrieval failures caused by vague phrasing' },
      { value: 'Better recall', label: 'Exposes more useful terms for matching' }
    ],
    notes: [
      'This is useful when users ask short or incomplete questions.',
      'A rewritten query can include synonyms, entities or domain hints.',
      'It often improves the first retrieval pass without changing the answer model.'
    ],
    tags: ['Query rewriting', 'Intent expansion', 'Recall']
  },
  hybrid: {
    title: 'Combine keyword and vector search',
    summary: 'Hybrid retrieval mixes lexical matching with semantic search so exact terms and meaning both contribute to ranking.',
    quote: 'Exact words and meaning can work together.',
    flow: ['Keyword', 'Vector', 'Merge', 'Rank'],
    metrics: [
      { value: 'Two signals', label: 'Keyword match plus semantic similarity' },
      { value: 'More robust', label: 'Helps when one method misses the answer' },
      { value: 'Flexible', label: 'Useful for technical and natural-language queries' }
    ],
    notes: [
      'Some questions need exact terms, especially for names or codes.',
      'Other questions are better solved by meaning-based search.',
      'Hybrid retrieval balances both and often improves real-world performance.'
    ],
    tags: ['Hybrid retrieval', 'Lexical search', 'Semantic search']
  },
  rerank: {
    title: 'Reorder candidates before answering',
    summary: 'A reranker inspects the first retrieved chunks again and moves the strongest evidence to the top before generation.',
    quote: 'The first search is not always the final ranking.',
    flow: ['Retrieve', 'Re-score', 'Sort', 'Select'],
    metrics: [
      { value: 'Second pass', label: 'Re-scores the retrieved candidates' },
      { value: 'Better precision', label: 'Pushes the most useful chunks upward' },
      { value: 'Cleaner prompt', label: 'Reduces noisy evidence before generation' }
    ],
    notes: [
      'A reranker is useful when retrieval returns many near-matches.',
      'It helps the model receive a more focused context window.',
      'This can noticeably improve answer quality in large document collections.'
    ],
    tags: ['Reranking', 'Precision', 'Prompt quality']
  },
  multihop: {
    title: 'Follow several evidence paths',
    summary: 'Multi-hop retrieval chains evidence across several passages when one chunk alone is not enough to answer the question.',
    quote: 'Some answers live across multiple documents.',
    flow: ['Chunk A', 'Chunk B', 'Chain', 'Answer'],
    metrics: [
      { value: 'Chain evidence', label: 'Uses several passages together' },
      { value: 'Complex questions', label: 'Helpful when answers require reasoning over multiple facts' },
      { value: 'Structured flow', label: 'Retrieval can be repeated in stages' }
    ],
    notes: [
      'This matters for questions that combine definitions, examples and explanations.',
      'The system may retrieve one chunk, then use it to search for a second chunk.',
      'It is a stronger pattern than single-pass lookup for deeper reasoning tasks.'
    ],
    tags: ['Multi-hop', 'Reasoning', 'Evidence chaining']
  },
  compress: {
    title: 'Trim retrieved text before prompting',
    summary: 'Context compression removes noise from retrieved chunks so the prompt window stays focused on the most relevant facts and details.',
    quote: 'Less noise leaves more room for the facts that matter.',
    flow: ['Chunks', 'Compress', 'Focus', 'Prompt'],
    metrics: [
      { value: 'Shorter prompt', label: 'Keeps the context window under control' },
      { value: 'Cleaner evidence', label: 'Removes repeated or weakly relevant text' },
      { value: 'Lower cost', label: 'Can reduce model input size' }
    ],
    notes: [
      'Compression is helpful when retrieved chunks are too long or repetitive.',
      'It preserves the key facts while removing unnecessary padding.',
      'That makes the generation step more focused and efficient.'
    ],
    tags: ['Compression', 'Context window', 'Noise reduction']
  },
  evaluate: {
    title: 'Check faithfulness and relevance',
    summary: 'Evaluation measures whether the answer is grounded, relevant to the query and consistent with the retrieved evidence.',
    quote: 'A good RAG system is measured, not just shown.',
    flow: ['Answer', 'Score', 'Review', 'Improve'],
    metrics: [
      { value: 'Faithfulness', label: 'Does the answer stay true to the evidence?' },
      { value: 'Answer relevance', label: 'Does it actually answer the question?' },
      { value: 'Context relevance', label: 'Were the retrieved chunks useful?' }
    ],
    notes: [
      'Evaluation matters because good-looking answers can still be wrong.',
      'These metrics help compare retrieval strategies and prompt designs.',
      'They are essential if the system will be used beyond a demo.'
    ],
    tags: ['Faithfulness', 'Evaluation', 'Quality control']
  }
};

const RAG_SIMULATION_DOCS = [
  'Diabetes symptoms include increased thirst, frequent urination, fatigue, and blurred vision.',
  'Artificial Neural Networks are computational models inspired by biological neurons.',
  'RAG stands for Retrieval-Augmented Generation and combines retrieval with text generation.',
  'Vector databases store embeddings and perform semantic similarity search.',
  'Transformers use self-attention mechanisms to process sequential data.',
  'Fine-tuning modifies model weights while RAG retrieves external knowledge dynamically.',
  'Machine learning enables systems to learn patterns from data.',
  'Embeddings convert text into numerical vector representations.'
];

const RAG_SIMULATION_STEPS = {
  query: {
    title: 'Query understanding',
    summary: 'The system reads the query and prepares it for semantic comparison with the knowledge base.',
    tags: ['Input query', 'Text cleaning', 'Intent capture'],
    detail: 'The input text is normalized and transformed into a small token set that captures the main semantic intent.',
    why: 'If this step is noisy, retrieval quality drops immediately because the system starts searching with weak intent signals.'
  },
  embed: {
    title: 'Embedding generation',
    summary: 'Both the query and the documents are turned into vector space so similarity can be measured mathematically.',
    tags: ['Vectors', 'Semantic space', 'Similarity ready'],
    detail: 'Each text is converted into a numeric vector where dimensions represent learned semantic features or token weights.',
    why: 'Embeddings make meaning computable. Without them, the system can only do shallow keyword matching.'
  },
  retrieve: {
    title: 'Semantic retrieval',
    summary: 'The system compares the query against the sample documents and returns the strongest matches.',
    tags: ['Top-k search', 'Cosine scores', 'Relevant chunks'],
    detail: 'Cosine similarity ranks the candidate chunks; the top-k highest scoring chunks are selected as supporting context.',
    why: 'This is the grounding core of RAG. Better retrieval gives the generator better evidence and fewer hallucinations.'
  },
  generate: {
    title: 'Augmented generation',
    summary: 'The retrieved chunks are passed to the generator, which composes a grounded answer from the evidence.',
    tags: ['Prompt augmentation', 'Grounded answer', 'LLM response'],
    detail: 'The model receives the retrieved context together with the query, then synthesizes a final answer that references this context.',
    why: 'Generation with context is more reliable than generation from memory only, especially for updated domain knowledge.'
  }
};

const SIMULATION_QUESTION_TOKENS = {
  diabetes: ['diabetes', 'symptoms', 'thirst', 'urination', 'fatigue', 'blurred', 'vision'],
  rag: ['rag', 'retrieval', 'generation', 'vector', 'database', 'embeddings', 'similarity'],
  vector: ['vector', 'database', 'search', 'similarity', 'embeddings']
};

function scoreDocument(queryTokens, document) {
  const words = document.toLowerCase().match(/[a-z]+/g) || [];
  const counts = new Map();
  words.forEach(word => counts.set(word, (counts.get(word) || 0) + 1));
  const docVector = new Map();
  counts.forEach((count, word) => {
    docVector.set(word, count);
  });

  let dot = 0;
  let queryNorm = 0;
  let docNorm = 0;

  const queryCounts = new Map();
  queryTokens.forEach(token => queryCounts.set(token, (queryCounts.get(token) || 0) + 1));

  queryCounts.forEach((count, token) => {
    queryNorm += count * count;
    if (docVector.has(token)) {
      dot += count * docVector.get(token);
    }
  });

  docVector.forEach(count => {
    docNorm += count * count;
  });

  if (!queryNorm || !docNorm) return 0;
  return dot / (Math.sqrt(queryNorm) * Math.sqrt(docNorm));
}

function initRagSimulation() {
  const root = document.querySelector('[data-rag-simulation]');
  if (!root) return;

  const queryInput = root.querySelector('[data-rag-query]');
  const runButton = root.querySelector('[data-rag-run]');
  const presetButtons = [...root.querySelectorAll('[data-query]')];
  const stageButtons = [...root.querySelectorAll('[data-stage]')];
  const titleEl = root.querySelector('[data-sim-title]');
  const summaryEl = root.querySelector('[data-sim-summary]');
  const tagsEl = root.querySelector('[data-sim-tags]');
  const docsEl = root.querySelector('[data-sim-docs]');
  const scoresEl = root.querySelector('[data-sim-scores]');
  const resultsEl = root.querySelector('[data-sim-results]');
  const answerEl = root.querySelector('[data-sim-answer]');
  const querySummaryEl = root.querySelector('[data-sim-query-display]');
  const topScoreEl = root.querySelector('[data-sim-top-score]');
  const retrievedCountEl = root.querySelector('[data-sim-retrieved-count]');
  const detailEl = root.querySelector('[data-sim-detail]');
  const whyEl = root.querySelector('[data-sim-why]');
  const detailWrapEl = root.querySelector('[data-sim-detail-wrap]');
  const whyWrapEl = root.querySelector('[data-sim-why-wrap]');
  const miniNodes = [...root.querySelectorAll('[data-mini-stage]')];

  function tokenize(query) {
    return (query.toLowerCase().match(/[a-z]+/g) || []).filter(Boolean);
  }

  function renderStage(stageKey) {
    const step = RAG_SIMULATION_STEPS[stageKey] || RAG_SIMULATION_STEPS.query;
    const currentIndex = ['query', 'embed', 'retrieve', 'generate'].indexOf(stageKey);
    stageButtons.forEach(button => {
      const index = ['query', 'embed', 'retrieve', 'generate'].indexOf(button.dataset.stage);
      button.classList.toggle('is-active', button.dataset.stage === stageKey);
      button.classList.toggle('is-complete', index > -1 && index < currentIndex);
    });

    miniNodes.forEach(node => {
      const index = ['query', 'embed', 'retrieve', 'generate'].indexOf(node.dataset.miniStage);
      node.classList.toggle('is-active', node.dataset.miniStage === stageKey);
      node.classList.toggle('is-complete', index > -1 && index < currentIndex);
    });

    titleEl.textContent = step.title;
    summaryEl.textContent = step.summary;
    tagsEl.innerHTML = step.tags.map(tag => `<span class="pill">${tag}</span>`).join('');
    detailEl.textContent = step.detail;
    whyEl.textContent = step.why;

    detailWrapEl.classList.remove('flash-in');
    whyWrapEl.classList.remove('flash-in');
    void detailWrapEl.offsetWidth;
    detailWrapEl.classList.add('flash-in');
    whyWrapEl.classList.add('flash-in');
  }

  function runSimulation(query) {
    const queryText = query.trim() || 'What are diabetes symptoms?';
    const tokens = tokenize(queryText);
    const scores = RAG_SIMULATION_DOCS.map((document, index) => ({
      index,
      document,
      score: scoreDocument(tokens, document)
    })).sort((left, right) => right.score - left.score);

    const topResults = scores.slice(0, 3);
    const maxScore = Math.max(...scores.map(item => item.score), 0.0001);

    querySummaryEl.textContent = queryText;
    topScoreEl.textContent = topResults[0] ? topResults[0].score.toFixed(3) : '0.000';
    retrievedCountEl.textContent = String(topResults.length);

    docsEl.innerHTML = RAG_SIMULATION_DOCS.map((document, index) => `
      <div class="simulation-doc ${topResults.some(result => result.index === index) ? 'is-retrieved' : ''}">
        <strong>Document ${index + 1}${topResults.some(result => result.index === index) ? ' • retrieved' : ''}</strong>
        <span>${document}</span>
      </div>
    `).join('');

    scoresEl.innerHTML = scores.map(result => `
      <div class="simulation-score-row ${topResults.some(match => match.index === result.index) ? 'is-top' : ''}">
        <div>
          <strong>Document ${result.index + 1}</strong>
          <span>${result.document}</span>
        </div>
        <div class="simulation-score-meta">
          <b>${result.score.toFixed(3)}</b>
          <div class="simulation-score-bar"><span style="width:${Math.max(4, (result.score / maxScore) * 100)}%"></span></div>
        </div>
      </div>
    `).join('');

    resultsEl.innerHTML = topResults.map(result => `
      <div class="simulation-result">
        <strong>Document ${result.index + 1}</strong>
        <p>${result.document}</p>
      </div>
    `).join('');

    const joinedContext = topResults.map(result => result.document).join(' ');
    answerEl.textContent = `Using the retrieved context, the model answers: ${joinedContext}`;

    renderStage('query');
  }

  presetButtons.forEach(button => {
    button.addEventListener('click', () => {
      queryInput.value = button.dataset.query;
      presetButtons.forEach(item => item.classList.toggle('is-active', item === button));
      runSimulation(queryInput.value);
    });
  });

  stageButtons.forEach(button => {
    button.addEventListener('click', () => renderStage(button.dataset.stage));
  });

  runButton.addEventListener('click', () => runSimulation(queryInput.value));
  queryInput.addEventListener('keydown', event => {
    if (event.key === 'Enter') {
      runSimulation(queryInput.value);
    }
  });

  runSimulation(queryInput.value);
}

const REPORT_SECTIONS = [
  {
    title: 'Project Objective',
    lead: 'Explain RAG as a modern ANN and LLM concept through a visual, interactive experience.',
    bullets: [
      'Turn a complex topic into a clearer educational flow.',
      'Use presentation design, workflow diagrams, simulations and AI-generated visuals.',
      'Demonstrate the use of AI tools while keeping the technical material understood by the team.'
    ]
  },
  {
    title: 'Presentation and Slides Generation',
    lead: 'The slides were produced through iterative prompt design and AI-assisted layout refinement.',
    bullets: [
      'ChatGPT Pro was used to structure the narrative and explain the technical ideas.',
      'Gamma helped generate a cinematic deck with a dark futuristic style and minimal text.',
      'The final visuals emphasise embeddings, retrieval, vector databases and generation.'
    ]
  },
  {
    title: 'Challenges and Limitations',
    lead: 'The team balanced clarity, accuracy and visual impact throughout the process.',
    bullets: [
      'Some generated explanations over-explored side topics and needed correction.',
      'A few diagrams required manual refinement for technical accuracy.',
      'Strong visuals occasionally reduced readability, so multiple iterations were needed.'
    ]
  },
  {
    title: 'Interactive RAG Demonstration',
    lead: 'A Streamlit demo was built locally, then published for public access.',
    bullets: [
      'The demo uses Streamlit, scikit-learn, pandas and NumPy.',
      'Users can enter queries, retrieve chunks and inspect the augmented response.',
      'It is hosted separately and linked from this website for convenience.'
    ]
  }
];

const QUIZ = [
  {
    question: 'What does RAG stand for?',
    options: ['Retrieval-Augmented Generation', 'Random Attention Generator', 'Retrieval Artificial Graph', 'Recursive Augmented Graph'],
    answer: 0,
    explanation: 'RAG combines information retrieval systems with language generation models to improve answer quality and reduce hallucinations.'
  },
  {
    question: 'What is the main purpose of RAG systems?',
    options: ['Compress neural networks', 'Retrieve external knowledge before generation', 'Replace transformers completely', 'Reduce dataset size'],
    answer: 1,
    explanation: 'RAG systems first retrieve relevant documents and then use them to generate context-aware responses.'
  },
  {
    question: 'Which problem does RAG primarily help reduce?',
    options: ['Overfitting', 'Vanishing gradients', 'Hallucination', 'Image noise'],
    answer: 2,
    explanation: 'RAG reduces hallucinations by grounding responses using retrieved external information.'
  },
  {
    question: 'What are embeddings in RAG systems?',
    options: ['Image compression algorithms', 'Numerical vector representations of text', 'Audio processing layers', 'Database encryption methods'],
    answer: 1,
    explanation: 'Embeddings convert text into vectors that capture semantic meaning and relationships.'
  },
  {
    question: 'Which database type is commonly used in RAG systems?',
    options: ['Relational database', 'Vector database', 'Blockchain database', 'Graphical database'],
    answer: 1,
    explanation: 'Vector databases store embeddings and perform semantic similarity search efficiently.'
  },
  {
    question: 'Which similarity metric is commonly used for retrieval?',
    options: ['Euclidean sorting', 'Binary matching', 'Cosine similarity', 'Histogram equalization'],
    answer: 2,
    explanation: 'Cosine similarity measures the angular similarity between embedding vectors.'
  },
  {
    question: 'What happens during the retrieval phase in RAG?',
    options: ['Model weights are retrained', 'Relevant document chunks are searched', 'Images are generated', 'Audio signals are filtered'],
    answer: 1,
    explanation: 'The retrieval stage identifies semantically similar documents related to the user query.'
  },
  {
    question: 'What is chunking in RAG systems?',
    options: ['Compressing neural weights', 'Splitting documents into smaller sections', 'Removing embeddings', 'Encrypting text files'],
    answer: 1,
    explanation: 'Large documents are divided into smaller chunks to improve retrieval accuracy.'
  },
  {
    question: 'Which of the following is an advantage of RAG compared to fine-tuning?',
    options: ['Requires retraining for every update', 'Easier knowledge updates', 'Uses no embeddings', 'Removes transformers completely'],
    answer: 1,
    explanation: 'RAG allows updating external documents without retraining the entire model.'
  },
  {
    question: 'Which ANN-related concept is strongly connected to RAG?',
    options: ['Backpropagation only', 'Embeddings and transformers', 'Edge detection only', 'Fourier transforms'],
    answer: 1,
    explanation: 'RAG relies heavily on transformer-based models and semantic vector embeddings.'
  },
  {
    question: 'What is semantic search?',
    options: ['Searching by exact keyword only', 'Searching based on meaning and context', 'Searching image colors', 'Searching file size'],
    answer: 1,
    explanation: 'Semantic search retrieves information using contextual meaning rather than exact keyword matching.'
  },
  {
    question: 'Why are vector databases important in RAG systems?',
    options: ['They store operating systems', 'They perform fast similarity search', 'They replace embeddings', 'They generate neural networks'],
    answer: 1,
    explanation: 'Vector databases efficiently retrieve semantically similar embeddings during the retrieval phase.'
  },
  {
    question: 'What is the role of the embedding model in RAG systems?',
    options: ['Generate images from text', 'Convert text into vector representations', 'Compress the database', 'Remove irrelevant documents'],
    answer: 1,
    explanation: 'Embedding models transform text into numerical vectors that capture semantic meaning for similarity search.'
  },
  {
    question: 'Which component generates the final response in a RAG pipeline?',
    options: ['Vector database', 'Embedding layer', 'Large Language Model (LLM)', 'Data loader'],
    answer: 2,
    explanation: 'After retrieval, the LLM uses the retrieved context to generate the final answer.'
  },
  {
    question: 'What is one limitation of RAG systems?',
    options: ['They cannot use transformers', 'Poor retrieval can produce incorrect answers', 'They completely eliminate latency', 'They require no storage'],
    answer: 1,
    explanation: 'If irrelevant or incorrect chunks are retrieved, the generated response quality decreases significantly.'
  },
  {
    question: 'Which of the following tools is commonly used for vector similarity search?',
    options: ['FAISS', 'Photoshop', 'AutoCAD', 'MySQL Workbench'],
    answer: 0,
    explanation: 'FAISS is a popular library developed for efficient vector similarity search and retrieval.'
  },
  {
    question: 'Why is chunk size important in RAG systems?',
    options: ['It controls internet speed', 'It affects retrieval quality and context understanding', 'It changes GPU temperature', 'It removes embeddings'],
    answer: 1,
    explanation: 'Very small chunks may lose context, while very large chunks may reduce retrieval precision.'
  },
  {
    question: 'Which type of search is performed in vector databases?',
    options: ['Lexical search only', 'Semantic similarity search', 'Binary search only', 'Pixel-based search'],
    answer: 1,
    explanation: 'Vector databases retrieve information based on semantic meaning rather than exact word matching.'
  },
  {
    question: 'What is prompt augmentation in RAG systems?',
    options: ['Reducing prompt size', 'Adding retrieved context to the prompt', 'Encrypting the prompt', 'Removing user queries'],
    answer: 1,
    explanation: 'Retrieved documents are appended to the user query to improve answer accuracy.'
  },
  {
    question: 'Why are RAG systems considered useful for modern AI applications?',
    options: ['They allow AI systems to use updated external knowledge', 'They completely replace neural networks', 'They remove the need for databases', 'They eliminate all AI errors'],
    answer: 0,
    explanation: 'RAG systems improve reliability by integrating external and dynamically updated information sources.'
  }
];

const SLIDES = [
  { title: 'Retrieval Augmented Generation', subtitle: 'Opening slide', summary: 'A clean title slide that frames the project around RAG and sets the visual tone.' },
  { title: 'Team Members', subtitle: 'People and supervision', summary: 'The team and supervisors are introduced before the technical narrative begins.' },
  { title: 'What We Will Cover', subtitle: 'Agenda slide', summary: 'The deck moves from definitions to architecture, then to applications and limitations.' },
  { title: 'Traditional LLM Vs RAG', subtitle: 'Why RAG matters', summary: 'A comparison slide that motivates retrieval-based augmentation.' },
  { title: 'The Problem with Traditional LLMs', subtitle: 'Hallucination and freshness', summary: 'The deck explains why pure generation can miss updated or external knowledge.' },
  { title: 'What is RAG?', subtitle: 'Core definition', summary: 'RAG combines retrieval with generation so the model can answer with grounded context.' },
  { title: 'How RAG Works?', subtitle: 'Pipeline overview', summary: 'User query, embeddings, retrieval and generation are shown as one flow.' },
  { title: 'RAG System Overview', subtitle: 'Big picture', summary: 'A system diagram presents the main stages and data flow in one glance.' },
  { title: 'Detailed RAG Architecture', subtitle: 'Component view', summary: 'The architecture expands the pipeline into preparation, retrieval and generation phases.' },
  { title: 'Phase 1: Data Preparation', subtitle: 'Indexing and chunking', summary: 'The source data is cleaned, split and prepared before retrieval begins.' },
  { title: 'Phase 2 & 3: Augmentation & Generation', subtitle: 'Context assembly', summary: 'Retrieved chunks are merged into the prompt before the final answer is produced.' },
  { title: 'Key Building Blocks of RAG', subtitle: 'Foundation concepts', summary: 'Chunking, embeddings and vector search are highlighted as the core pieces.' },
  { title: 'Chunking', subtitle: 'Why split documents', summary: 'Smaller chunks improve retrieval precision and make context assembly easier.' },
  { title: 'Embedding & Vector Databases', subtitle: 'Representation and storage', summary: 'Text becomes vectors and vectors are stored for semantic search.' },
  { title: 'Cosine Similarity Calculations', subtitle: 'Matching logic', summary: 'Similarity is calculated to find the closest document chunks to a query.' },
  { title: 'Building the Super-Prompt', subtitle: 'Prompt engineering', summary: 'The prompt is expanded with retrieved evidence for stronger generation.' },
  { title: 'Example of RAG', subtitle: 'Practical illustration', summary: 'A live example demonstrates how retrieval and generation work together.' },
  { title: 'RDBMS Vs VDBMS', subtitle: 'Storage comparison', summary: 'Traditional databases are contrasted with vector-aware retrieval systems.' },
  { title: 'Traditional RAG Vs Agentic RAG', subtitle: 'Evolution of workflows', summary: 'The deck compares classic RAG with newer agentic patterns.' },
  { title: 'Challenges of RAG', subtitle: 'Limitations and tradeoffs', summary: 'Retrieval quality, prompt size and technical correctness are discussed.' },
  { title: 'Applications of RAG', subtitle: 'Where it is used', summary: 'The final stretch shows how RAG applies to modern AI products and educational tools.' },
  { title: 'Thank You', subtitle: 'Closing slide', summary: 'A simple closing screen ends the presentation cleanly.' }
];

function initReveal() {
  const nodes = document.querySelectorAll('.reveal');
  if (!nodes.length) return;
  if (!('IntersectionObserver' in window)) {
    nodes.forEach(node => node.classList.add('visible'));
    return;
  }
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.15 });
  nodes.forEach(node => observer.observe(node));
}

function initRagExplainer() {
  const root = document.querySelector('[data-rag-explainer]');
  if (!root) return;

  const titleEl = root.querySelector('[data-rag-title]');
  const summaryEl = root.querySelector('[data-rag-summary]');
  const metricsEl = root.querySelector('[data-rag-metrics]');
  const notesEl = root.querySelector('[data-rag-notes]');
  const tagsEl = root.querySelector('[data-rag-tags]');
  const meterEl = root.querySelector('[data-rag-meter]');
  const tabButtons = [...root.querySelectorAll('[data-step]')];
  const flowNodes = [...root.querySelectorAll('[data-flow-node]')];

  function render(stepKey) {
    const step = RAG_EXPLAINER_STEPS[stepKey] || RAG_EXPLAINER_STEPS.definition;

    tabButtons.forEach(button => {
      const isActive = button.dataset.step === stepKey;
      button.classList.toggle('is-active', isActive);
      button.setAttribute('aria-selected', String(isActive));
    });

    titleEl.textContent = step.title;
    summaryEl.textContent = step.summary;
    metricsEl.innerHTML = step.metrics.map(metric => `
      <div class="explainer-metric">
        <strong>${metric.value}</strong>
        <span>${metric.label}</span>
      </div>
    `).join('');
    notesEl.innerHTML = step.notes.map(note => `
      <div class="explainer-note">${note}</div>
    `).join('');
    tagsEl.innerHTML = step.tags.map(tag => `<span class="pill">${tag}</span>`).join('');
    meterEl.style.width = `${step.meter}%`;

    flowNodes.forEach(node => {
      node.classList.toggle('is-active', step.activeNodes.includes(node.dataset.flowNode));
    });
  }

  tabButtons.forEach(button => {
    button.addEventListener('click', () => render(button.dataset.step));
  });

  render('definition');
}

function initProjectStory() {
  const root = document.querySelector('[data-project-story]');
  if (!root) return;

  const titleEl = root.querySelector('[data-project-title]');
  const summaryEl = root.querySelector('[data-project-summary]');
  const quoteEl = root.querySelector('[data-project-quote]');
  const metricsEl = root.querySelector('[data-project-metrics]');
  const notesEl = root.querySelector('[data-project-notes]');
  const tagsEl = root.querySelector('[data-project-tags]');
  const buttons = [...root.querySelectorAll('[data-project-step]')];

  function render(stepKey) {
    const step = PROJECT_STORY_STEPS[stepKey] || PROJECT_STORY_STEPS.architecture;
    buttons.forEach(button => {
      const isActive = button.dataset.projectStep === stepKey;
      button.classList.toggle('is-active', isActive);
      button.setAttribute('aria-selected', String(isActive));
    });
    titleEl.textContent = step.title;
    summaryEl.textContent = step.summary;
    quoteEl.textContent = step.quote;
    metricsEl.innerHTML = step.metrics.map(metric => `
      <div class="explainer-metric">
        <strong>${metric.value}</strong>
        <span>${metric.label}</span>
      </div>
    `).join('');
    notesEl.innerHTML = step.notes.map(note => `<div class="explainer-note">${note}</div>`).join('');
    tagsEl.innerHTML = step.tags.map(tag => `<span class="pill">${tag}</span>`).join('');
  }

  buttons.forEach(button => button.addEventListener('click', () => render(button.dataset.projectStep)));
  render('architecture');
}

function initGlossary() {
  const root = document.querySelector('.glossary-shell');
  if (!root) return;

  const navEl = root.querySelector('[data-glossary-nav]');
  const titleEl = root.querySelector('[data-glossary-title]');
  const summaryEl = root.querySelector('[data-glossary-summary]');
  const metricsEl = root.querySelector('[data-glossary-metrics]');
  const notesEl = root.querySelector('[data-glossary-notes]');

  const entries = Object.entries(RAG_GLOSSARY);

  function render(key) {
    const entry = RAG_GLOSSARY[key] || RAG_GLOSSARY.embeddings;
    [...navEl.children].forEach(child => child.classList.toggle('is-active', child.dataset.glossaryKey === key));
    titleEl.textContent = entry.title;
    summaryEl.textContent = entry.summary;
    metricsEl.innerHTML = entry.metrics.map(metric => `
      <div class="explainer-metric">
        <strong>${metric.value}</strong>
        <span>${metric.label}</span>
      </div>
    `).join('');
    notesEl.innerHTML = entry.notes.map(note => `<div class="explainer-note">${note}</div>`).join('');
  }

  entries.forEach(([key, entry]) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'glossary-card';
    button.dataset.glossaryKey = key;
    button.innerHTML = `<strong>${entry.title}</strong><span>${entry.summary}</span>`;
    button.addEventListener('click', () => render(key));
    navEl.appendChild(button);
  });

  render('embeddings');
}

function initAdvancedRagTopics() {
  const root = document.querySelector('[data-advanced-rag]');
  if (!root) return;

  const titleEl = root.querySelector('[data-advanced-title]');
  const summaryEl = root.querySelector('[data-advanced-summary]');
  const quoteEl = root.querySelector('[data-advanced-quote]');
  const flowEl = root.querySelector('[data-advanced-flow]');
  const metricsEl = root.querySelector('[data-advanced-metrics]');
  const notesEl = root.querySelector('[data-advanced-notes]');
  const tagsEl = root.querySelector('[data-advanced-tags]');
  const buttons = [...root.querySelectorAll('[data-advanced-step]')];

  function render(stepKey) {
    const topic = ADVANCED_RAG_TOPICS[stepKey] || ADVANCED_RAG_TOPICS.rewrite;
    buttons.forEach(button => {
      const isActive = button.dataset.advancedStep === stepKey;
      button.classList.toggle('is-active', isActive);
      button.setAttribute('aria-selected', String(isActive));
    });

    titleEl.textContent = topic.title;
    summaryEl.textContent = topic.summary;
    quoteEl.textContent = topic.quote;
    flowEl.innerHTML = topic.flow.map((step, index) => `
      <span class="advanced-flow-step ${index === 0 ? 'is-active' : ''}">${step}</span>${index < topic.flow.length - 1 ? '<span class="advanced-flow-link">→</span>' : ''}
    `).join('');
    metricsEl.innerHTML = topic.metrics.map(metric => `
      <div class="explainer-metric">
        <strong>${metric.value}</strong>
        <span>${metric.label}</span>
      </div>
    `).join('');
    notesEl.innerHTML = topic.notes.map(note => `<div class="explainer-note">${note}</div>`).join('');
    tagsEl.innerHTML = topic.tags.map(tag => `<span class="pill">${tag}</span>`).join('');
  }

  buttons.forEach(button => button.addEventListener('click', () => render(button.dataset.advancedStep)));
  render('rewrite');
}

function initQuiz() {
  const root = document.querySelector('[data-quiz-root]');
  if (!root) return;

  const questionEl = root.querySelector('[data-question]');
  const optionsEl = root.querySelector('[data-options]');
  const feedbackEl = root.querySelector('[data-feedback]');
  const progressFill = root.querySelector('[data-progress]');
  const counterEl = root.querySelector('[data-counter]');
  const prevBtn = root.querySelector('[data-prev]');
  const actionBtn = root.querySelector('[data-action]');
  const restartBtn = root.querySelector('[data-restart]');
  const summaryEl = root.querySelector('[data-summary]');
  const listEl = root.querySelector('[data-missed]');

  const state = {
    index: 0,
    selected: Array(QUIZ.length).fill(null),
    revealed: Array(QUIZ.length).fill(false)
  };

  function score() {
    return QUIZ.reduce((total, item, i) => total + (state.selected[i] === item.answer ? 1 : 0), 0);
  }

  function renderSummary() {
    const total = QUIZ.length;
    const correct = score();
    const percent = Math.round((correct / total) * 100);
    summaryEl.innerHTML = `
      <div class="score-card">
        <div class="eyebrow">Quiz complete</div>
        <h3 style="margin:10px 0 6px;font-size:2rem">${percent}% score</h3>
        <p class="muted" style="margin:0">You got ${correct} out of ${total} questions correct.</p>
        <div class="score-grid">
          <div class="stat"><strong>${correct}</strong><span class="muted">Correct</span></div>
          <div class="stat"><strong>${total - correct}</strong><span class="muted">Missed</span></div>
          <div class="stat"><strong>${percent}%</strong><span class="muted">Accuracy</span></div>
        </div>
      </div>
    `;
    listEl.innerHTML = '';
    QUIZ.forEach((item, i) => {
      if (state.selected[i] !== item.answer) {
        const row = document.createElement('div');
        row.className = 'feedback';
        row.innerHTML = `
          <strong>Q${i + 1}. ${item.question}</strong>
          <div class="muted">Correct answer: ${item.options[item.answer]}</div>
          <div class="muted">${item.explanation}</div>
        `;
        listEl.appendChild(row);
      }
    });
  }

  function renderQuestion() {
    const item = QUIZ[state.index];
    counterEl.textContent = `${state.index + 1} / ${QUIZ.length}`;
    progressFill.style.width = `${((state.index + 1) / QUIZ.length) * 100}%`;
    prevBtn.disabled = state.index === 0;
    restartBtn.hidden = false;

    questionEl.innerHTML = `
      <div class="slide-badge">Question ${state.index + 1}</div>
      <h3 class="quiz-question">${item.question}</h3>
      <div class="muted-small">Pick one option, check it, then move forward. The quiz reveals an explanation after every answer.</div>
    `;

    optionsEl.innerHTML = '';
    item.options.forEach((option, optionIndex) => {
      const button = document.createElement('button');
      button.className = 'option-btn';
      if (state.selected[state.index] === optionIndex) button.classList.add('selected');
      if (state.revealed[state.index]) {
        if (optionIndex === item.answer) button.classList.add('correct');
        if (state.selected[state.index] === optionIndex && optionIndex !== item.answer) button.classList.add('wrong');
      }
      button.textContent = `${String.fromCharCode(65 + optionIndex)}. ${option}`;
      button.addEventListener('click', () => {
        state.selected[state.index] = optionIndex;
        renderQuestion();
      });
      optionsEl.appendChild(button);
    });

    if (state.revealed[state.index]) {
      feedbackEl.innerHTML = `
        <strong>${state.selected[state.index] === item.answer ? 'Correct' : 'Not quite'}</strong>
        <div class="muted">${item.explanation}</div>
      `;
      actionBtn.textContent = state.index === QUIZ.length - 1 ? 'Finish quiz' : 'Next question';
    } else {
      feedbackEl.innerHTML = '<strong>Ready when you are</strong><div class="muted">Choose an answer to unlock the explanation.</div>';
      actionBtn.textContent = 'Check answer';
    }
  }

  function finalize() {
    renderSummary();
    questionEl.innerHTML = '<div class="score-card"><h3 style="margin-top:0">Review your result</h3><p class="muted">Use the missed question list below to review the concepts that need another pass.</p></div>';
    optionsEl.innerHTML = '';
    feedbackEl.innerHTML = '';
    prevBtn.disabled = true;
    actionBtn.disabled = true;
    progressFill.style.width = '100%';
    counterEl.textContent = `${QUIZ.length} / ${QUIZ.length}`;
  }

  prevBtn.addEventListener('click', () => {
    if (state.index > 0) {
      state.index -= 1;
      renderQuestion();
    }
  });

  actionBtn.addEventListener('click', () => {
    if (!state.revealed[state.index]) {
      if (state.selected[state.index] === null) return;
      state.revealed[state.index] = true;
      renderQuestion();
      return;
    }
    if (state.index === QUIZ.length - 1) {
      finalize();
      return;
    }
    state.index += 1;
    renderQuestion();
  });

  restartBtn.addEventListener('click', () => {
    state.index = 0;
    state.selected = Array(QUIZ.length).fill(null);
    state.revealed = Array(QUIZ.length).fill(false);
    summaryEl.innerHTML = '';
    listEl.innerHTML = '';
    actionBtn.disabled = false;
    renderQuestion();
  });

  renderQuestion();
}

function initDeck() {
  const root = document.querySelector('[data-deck-root]');
  if (!root) return;
  const titleEl = root.querySelector('[data-slide-title]');
  const subtitleEl = root.querySelector('[data-slide-subtitle]');
  const summaryEl = root.querySelector('[data-slide-summary]');
  const indexEl = root.querySelector('[data-slide-index]');
  const prevBtn = root.querySelector('[data-slide-prev]');
  const nextBtn = root.querySelector('[data-slide-next]');
  const listEl = root.querySelector('[data-slide-list]');
  const barEl = root.querySelector('[data-slide-progress]');

  let current = 0;

  function render() {
    const slide = SLIDES[current];
    titleEl.textContent = slide.title;
    subtitleEl.textContent = slide.subtitle;
    summaryEl.textContent = slide.summary;
    indexEl.textContent = `${String(current + 1).padStart(2, '0')} / ${String(SLIDES.length).padStart(2, '0')}`;
    prevBtn.disabled = current === 0;
    nextBtn.textContent = current === SLIDES.length - 1 ? 'Restart deck' : 'Next slide';
    barEl.style.width = `${((current + 1) / SLIDES.length) * 100}%`;
    [...listEl.children].forEach((child, i) => child.classList.toggle('active', i === current));
  }

  SLIDES.forEach((slide, i) => {
    const card = document.createElement('button');
    card.type = 'button';
    card.className = 'slide-thumb';
    card.innerHTML = `<strong>${String(i + 1).padStart(2, '0')}. ${slide.title}</strong><p>${slide.summary}</p>`;
    card.addEventListener('click', () => { current = i; render(); });
    listEl.appendChild(card);
  });

  prevBtn.addEventListener('click', () => {
    current = Math.max(0, current - 1);
    render();
  });

  nextBtn.addEventListener('click', () => {
    if (current === SLIDES.length - 1) {
      current = 0;
    } else {
      current += 1;
    }
    render();
  });

  render();
}

document.addEventListener('DOMContentLoaded', () => {
  initReveal();
  initRagExplainer();
  initProjectStory();
  initGlossary();
  initAdvancedRagTopics();
  initRagSimulation();
  initQuiz();
  initDeck();
});
