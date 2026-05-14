const PROJECT = {
  liveDemo: 'https://neuralnetwokprojects-fspmceeeqzbjce4bxfnvj2.streamlit.app/',
  reportPdf: 'assets/final-project-report.pdf',
  mcqPdf: 'assets/final-project-mcq.pdf',
  pptxFile: 'assets/rag-presentation.pptx',
  briefPdf: 'assets/project-brief.pdf'
};

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
  initQuiz();
  initDeck();
});
