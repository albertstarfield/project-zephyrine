import './style.css';
import * as THREE from 'three';

import { marked } from 'marked';
import DOMPurify from 'dompurify';
import mermaid from 'mermaid';
import katex from 'katex';
import 'katex/dist/katex.min.css';
import cytoscape from 'cytoscape';

// --------------------------------------------------------
// Chat Logic
// --------------------------------------------------------

async function renderMarkdownToElement(el: HTMLElement, mdText: string) {
  // Custom regex to handle KaTeX before marked parsing
  let preProcessed = mdText.replace(/\$\$(.*?)\$\$/gs, (match, math) => {
    try { return katex.renderToString(math, { displayMode: true }); } 
    catch(e) { return match; }
  });
  preProcessed = preProcessed.replace(/\$(.*?)\$/g, (match, math) => {
    try { return katex.renderToString(math, { displayMode: false }); } 
    catch(e) { return match; }
  });

  const rawHtml = marked.parse(preProcessed) as string;
  const cleanHtml = DOMPurify.sanitize(rawHtml, { ADD_TAGS: ['math', 'mrow', 'mi', 'mn', 'mo', 'ms', 'mspace', 'mtext', 'menclose', 'merror', 'mphantom', 'mpadded', 'mroot', 'mfrac', 'msqrt', 'mstyle', 'msub', 'msup', 'msubsup', 'munder', 'mover', 'munderover', 'mtable', 'mtr', 'mtd', 'annotation', 'semantics']});
  el.innerHTML = cleanHtml;

  // Render Mermaid diagrams
  const codeBlocks = el.querySelectorAll('pre code.language-mermaid');
  codeBlocks.forEach(async (block, index) => {
    const id = `mermaid-${Date.now()}-${index}`;
    const pre = block.parentElement;
    if (pre) {
      pre.outerHTML = `<div class="mermaid" id="${id}">${block.textContent}</div>`;
      try {
        const { svg } = await mermaid.render(`${id}-svg`, block.textContent || '');
        document.getElementById(id)!.innerHTML = svg;
      } catch (err) {
        console.error('Mermaid render error', err);
      }
    }
  });
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp?: string;
}

const messagesContainer = document.getElementById('messages') as HTMLDivElement;
const chatForm = document.getElementById('chat-form') as HTMLFormElement;
const chatInput = document.getElementById('chat-input') as HTMLInputElement;


const emptyState = document.getElementById('empty-state') as HTMLDivElement;
const chatContainerWrapper = document.getElementById('chat-container') as HTMLDivElement;

function addMessageToUI(msg: Message) {
  emptyState.classList.add('hidden');
  chatContainerWrapper.classList.remove('hidden');

  const msgEl = document.createElement('div');
  msgEl.className = `message ${msg.role}`;
  
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  
  // Render Markdown, LaTeX, and Mermaid directly into the bubble
  renderMarkdownToElement(bubble, msg.content);
  
  msgEl.appendChild(bubble);
  messagesContainer.appendChild(msgEl);
  
  // Auto-scroll to bottom
  chatContainerWrapper.scrollTop = chatContainerWrapper.scrollHeight;
}

async function loadHistory() {
  try {
    const res = await fetch('/api/messages');
    if (res.ok) {
      const msgs: Message[] = await res.json();
      if (msgs.length > 0) {
          msgs.forEach(addMessageToUI);
          
          // Populate the sidebar history topic
          const historySection = document.querySelector('.history-section');
          if (historySection) {
            const firstUserMsg = msgs.find(m => m.role === 'user');
            const topicName = firstUserMsg ? (firstUserMsg.content.substring(0, 20) + '...') : 'Current Session';
            
            const topicEl = document.createElement('div');
            topicEl.className = 'history-topic active';
            topicEl.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg> ${topicName}`;
            
            // Remove any existing topics
            const existingTopics = historySection.querySelectorAll('.history-topic');
            existingTopics.forEach(t => t.remove());
            
            historySection.appendChild(topicEl);
          }
      }
    }
  } catch (err) {
    console.error("Failed to load history:", err);
  }
}

const messageQueue: string[] = [];
let isProcessingQueue = false;

chatForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = chatInput.value.trim();
  if (!text) return;

  // Optimistically add user message
  addMessageToUI({ role: 'user', content: text });
  chatInput.value = '';

  messageQueue.push(text);
  if (!isProcessingQueue) {
    processQueue();
  }
});

async function processQueue() {
  isProcessingQueue = true;
  
  while (messageQueue.length > 0) {
    const text = messageQueue.shift();
    if (!text) continue;
    
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text })
      });
      
      if (res.ok) {
        const data = await res.json();
        addMessageToUI({ role: 'assistant', content: data.reply });
      } else {
        addMessageToUI({ role: 'assistant', content: "Error: " + res.statusText });
      }
    } catch (err) {
      addMessageToUI({ role: 'assistant', content: "Network error trying to reach the sidecar." });
    }
  }
  
  isProcessingQueue = false;
  chatInput.focus();
}

// Load history on startup
loadHistory();

const sidebarToggle = document.getElementById('sidebar-toggle') as HTMLButtonElement;
const sidebar = document.getElementById('sidebar') as HTMLElement;
if (sidebarToggle && sidebar) {
  sidebarToggle.addEventListener('click', () => {
    sidebar.classList.toggle('collapsed');
  });
}

// --------------------------------------------------------
// WebGL Glassmorphism Shader (adapted from LiquidGlass example)
// --------------------------------------------------------

const state = {
    x: window.innerWidth / 2,
    y: window.innerHeight / 2,
    gw: window.innerWidth * 0.9,
    gh: window.innerHeight * 0.8,
    gr: 40,
    thick: 50,
    bezel: 60,
    ior: 2.5,
    blur: 2.0,
    spec: 0.3,
    tint: 0.1,
    shadow: 0.3,
};

const canvas = document.getElementById('gl') as HTMLCanvasElement;
const renderer = new THREE.WebGLRenderer({ canvas, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);

const scene = new THREE.Scene();
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

const vertexShader = `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position, 1.0);
}`;

const fragmentShader = `
precision highp float;
varying vec2 vUv;

uniform vec2 uResolution;
uniform vec2 uGlassCenter;
uniform vec2 uGlassSize;
uniform float uRadius;
uniform float uBezel;
uniform float uThickness;
uniform float uIOR;
uniform float uBlur;
uniform float uSpecular;
uniform float uTint;
uniform float uShadow;

float sdRoundedRect(vec2 p, vec2 halfSize, float r) {
  vec2 q = abs(p) - halfSize + r;
  return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r;
}

float surfaceHeight(float t) {
  float s = 1.0 - t;
  return pow(1.0 - s*s*s*s, 0.25);
}

// Pseudo-background (blue gradient)
vec3 sampleBg(vec2 uv) {
    return mix(vec3(0.05, 0.1, 0.2), vec3(0.1, 0.3, 0.5), uv.y);
}

void main() {
  vec2 screenPx = vec2(vUv.x, 1.0 - vUv.y) * uResolution;
  vec2 p = screenPx - uGlassCenter;
  vec2 halfSize = uGlassSize * 0.5;

  float sd = sdRoundedRect(p, halfSize, uRadius);

  if (sd > 0.0) {
    // Outside the glass -> render background only
    gl_FragColor = vec4(0.0); 
    return;
  }

  float distFromEdge = -sd;
  float bezel = min(uBezel, min(uRadius, min(halfSize.x, halfSize.y)) - 1.0);
  float t = clamp(distFromEdge / bezel, 0.0, 1.0);

  float h = surfaceHeight(t);
  float dt = 0.001;
  float h2 = surfaceHeight(min(t + dt, 1.0));
  float dh = (h2 - h) / dt;

  float slopeAngle = atan(dh * (uThickness / bezel));
  float sinR = sin(slopeAngle) / uIOR;
  sinR = clamp(sinR, -1.0, 1.0);
  float thetaR = asin(sinR);
  float displacement = h * uThickness * (tan(slopeAngle) - tan(thetaR));

  vec2 grad;
  float eps = 0.5;
  grad.x = sdRoundedRect(p + vec2(eps, 0.0), halfSize, uRadius) - sd;
  grad.y = sdRoundedRect(p + vec2(0.0, eps), halfSize, uRadius) - sd;
  grad = normalize(grad);

  vec2 offset = -grad * displacement / uResolution;

  vec2 screenUV = screenPx / uResolution;
  vec2 refractedUV = screenUV + offset;

  // Sample blurred background
  vec3 color = sampleBg(refractedUV);

  vec2 lightDir = normalize(vec2(0.5, -0.7));
  float rimDot = abs(dot(grad, lightDir));
  float rimFalloff = 1.0 - smoothstep(0.0, bezel * 0.4, distFromEdge);
  float specHighlight = pow(rimDot * rimFalloff, 1.5);
  color += vec3(specHighlight * uSpecular);

  float innerShadow = 1.0 - smoothstep(0.0, bezel * 0.6, distFromEdge);
  color *= mix(1.0, 0.7, innerShadow * 0.3);

  float innerRim = smoothstep(0.0, 2.0, distFromEdge) * (1.0 - smoothstep(2.0, 5.0, distFromEdge));
  color += vec3(innerRim * 0.15 * uSpecular);

  color = mix(color, vec3(1.0), uTint);

  float alpha = smoothstep(0.0, 1.5, distFromEdge);
  gl_FragColor = vec4(color, alpha * 0.6); // slight transparency overall
}`;

const material = new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader,
    uniforms: {
        uResolution: { value: new THREE.Vector2(window.innerWidth, window.innerHeight) },
        uGlassCenter: { value: new THREE.Vector2(state.x, state.y) },
        uGlassSize: { value: new THREE.Vector2(state.gw, state.gh) },
        uRadius: { value: state.gr },
        uBezel: { value: state.bezel },
        uThickness: { value: state.thick },
        uIOR: { value: state.ior },
        uBlur: { value: state.blur },
        uSpecular: { value: state.spec },
        uTint: { value: state.tint },
        uShadow: { value: state.shadow },
    },
    transparent: true,
    depthTest: false,
});

scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material));

function render() {
    // Keep glass centered
    state.x = window.innerWidth / 2;
    state.y = window.innerHeight / 2 + 30; // offset slightly for header
    state.gw = window.innerWidth * 0.95;
    state.gh = window.innerHeight * 0.85;

    const u = material.uniforms;
    u.uResolution.value.set(window.innerWidth, window.innerHeight);
    u.uGlassCenter.value.set(state.x, state.y);
    u.uGlassSize.value.set(state.gw, state.gh);
    u.uRadius.value = state.gr;
    u.uBezel.value = state.bezel;
    u.uThickness.value = state.thick;
    u.uIOR.value = state.ior;
    u.uBlur.value = state.blur;
    u.uSpecular.value = state.spec;
    u.uTint.value = state.tint;
    u.uShadow.value = state.shadow;
    
    renderer.render(scene, camera);
    requestAnimationFrame(render);
}
render();

window.addEventListener('resize', () => {
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// --------------------------------------------------------
// Navigation Handlers
// --------------------------------------------------------
const navVoice = document.getElementById('nav-voice');
const navImage = document.getElementById('nav-image');
const navSettings = document.getElementById('nav-settings');
const navExit = document.getElementById('nav-exit');
const userEmailText = document.getElementById('user-email-text');
const greetingTitleText = document.getElementById('greeting-title-text');

// Fetch user info
fetch('/api/user_info')
  .then(res => res.json())
  .then(data => {
    if (data.username && userEmailText && greetingTitleText) {
      userEmailText.textContent = data.username;
      greetingTitleText.textContent = `Greetings, ${data.username}`;
    }
  })
  .catch(err => console.error("Error fetching user info:", err));

if (navVoice) {
  navVoice.addEventListener('click', (e) => {
    e.preventDefault();
    alert("INOP");
  });
}

if (navImage) {
  navImage.addEventListener('click', (e) => {
    e.preventDefault();
    alert("INOP");
  });
}

if (navSettings) {
  navSettings.addEventListener('click', (e) => {
    e.preventDefault();
    alert("Not implemented yet but things are being settings automatically for now");
  });
}

if (navExit) {
  navExit.addEventListener('click', async (e) => {
    e.preventDefault();
    try {
      await fetch('/api/exit', { method: 'POST' });
    } catch (err) {}
    window.close(); // Fallback in case fetch fails
  });
}

// --------------------------------------------------------
// About Section Logic
// --------------------------------------------------------

const navMain = document.getElementById('nav-main');
const navAbout = document.getElementById('nav-about');
const aboutContainer = document.getElementById('about-container');
const inputContainer = document.querySelector('.input-container') as HTMLElement;
const tabReadme = document.getElementById('tab-readme');
const tabLicense = document.getElementById('tab-license');
const aboutMarkdown = document.getElementById('about-markdown');
const aboutToc = document.getElementById('about-toc');
const searchInput = document.getElementById('about-search-input') as HTMLInputElement;

mermaid.initialize({ startOnLoad: false, theme: 'dark' });

let currentAboutContent = '';

// View transition helper
async function switchView(hideEls: HTMLElement[], showEls: HTMLElement[]) {
  // Fade out
  hideEls.forEach(el => el.classList.add('fade-out'));
  if (hideEls.length > 0) {
    await new Promise(r => setTimeout(r, 250));
    hideEls.forEach(el => el.classList.add('hidden'));
  }
  
  // Fade in
  showEls.forEach(el => {
    el.classList.remove('hidden');
    // Force reflow to ensure transition runs
    void el.offsetWidth;
    el.classList.remove('fade-out');
  });
}



const navTuning = document.getElementById('nav-tuning');
const knowledgeContainer = document.getElementById('knowledge-container');
const uploadInput = document.getElementById('knowledge-file-input') as HTMLInputElement;
const uploadBtn = document.getElementById('knowledge-upload-btn');
const domainSelect = document.getElementById('knowledge-domain-select') as HTMLSelectElement;
const fileListContainer = document.getElementById('knowledge-file-list');
const searchInputK = document.getElementById('knowledge-search-input') as HTMLInputElement;
const searchBtnK = document.getElementById('knowledge-search-btn');
const searchResultsContainer = document.getElementById('knowledge-search-results');
const tabGraph = document.getElementById('k-tab-graph');
const tabSearch = document.getElementById('k-tab-search');
const contentGraph = document.getElementById('k-content-graph');
const contentSearch = document.getElementById('k-content-search');
const cyContainer = document.getElementById('cy-container');

// Memory DOM Elements
const segmentLiterature = document.getElementById('segment-literature');
const segmentMemory = document.getElementById('segment-memory');
const wrapperLiterature = document.getElementById('knowledge-literature-wrapper');
const wrapperMemory = document.getElementById('knowledge-memory-wrapper');

const memSessionInput = document.getElementById('memory-session-input') as HTMLInputElement;
const memTopicInput = document.getElementById('memory-topic-input') as HTMLInputElement;
const memContentInput = document.getElementById('memory-content-input') as HTMLTextAreaElement;
const memUploadBtn = document.getElementById('memory-upload-btn');
const memProgressContainer = document.getElementById('memory-progress-container');
const memProgressBar = document.getElementById('memory-progress-bar');
const memProgressText = document.getElementById('memory-progress-text');
const cyMemoryContainer = document.getElementById('cy-memory-container');

let cyInstance: any = null;
let cyMemoryInstance: any = null;

if (navMain) {
  navMain.addEventListener('click', async (e) => {
    e.preventDefault();
    
    // Clear chat history to go to home default page
    messagesContainer.innerHTML = '';
    
    const hideEls = [];
    if (!aboutContainer?.classList.contains('hidden')) hideEls.push(aboutContainer!);
    if (!knowledgeContainer?.classList.contains('hidden')) hideEls.push(knowledgeContainer!);
    if (!chatContainerWrapper?.classList.contains('hidden')) hideEls.push(chatContainerWrapper!);
    
    const showEls = [inputContainer, emptyState];
    await switchView(hideEls, showEls);
  });
}

const newChatBtn = document.querySelector('.new-chat-btn');
if (newChatBtn) {
  newChatBtn.addEventListener('click', async (e) => {
    e.preventDefault();
    
    // Clear chat history to go to home default page
    messagesContainer.innerHTML = '';
    
    const hideEls = [];
    if (!aboutContainer?.classList.contains('hidden')) hideEls.push(aboutContainer!);
    if (!knowledgeContainer?.classList.contains('hidden')) hideEls.push(knowledgeContainer!);
    if (!chatContainerWrapper?.classList.contains('hidden')) hideEls.push(chatContainerWrapper!);
    
    const showEls = [inputContainer, emptyState];
    await switchView(hideEls, showEls);
  });
}

if (navAbout) {
  navAbout.addEventListener('click', async (e) => {
    e.preventDefault();
    if (aboutContainer?.classList.contains('hidden')) {
      const hideEls = [inputContainer];
      if (!emptyState.classList.contains('hidden')) hideEls.push(emptyState);
      if (!chatContainerWrapper.classList.contains('hidden')) hideEls.push(chatContainerWrapper);
      if (!knowledgeContainer?.classList.contains('hidden')) hideEls.push(knowledgeContainer!);
      
      await switchView(hideEls, [aboutContainer!]);
      await loadAboutContent('/api/docs/readme');
      tabReadme?.classList.add('active');
      tabLicense?.classList.remove('active');
    }
  });
}

if (navTuning) {
  navTuning.addEventListener('click', async (e) => {
    e.preventDefault();
    if (knowledgeContainer?.classList.contains('hidden')) {
      const hideEls = [inputContainer];
      if (!emptyState.classList.contains('hidden')) hideEls.push(emptyState);
      if (!chatContainerWrapper.classList.contains('hidden')) hideEls.push(chatContainerWrapper);
      if (!aboutContainer?.classList.contains('hidden')) hideEls.push(aboutContainer!);
      
      await switchView(hideEls, [knowledgeContainer!]);
      if (wrapperLiterature && !wrapperLiterature.classList.contains('hidden')) {
        loadKnowledgeFiles();
        loadGraph();
      } else {
        loadMemoryGraph();
      }
    }
  });
}

// Segment Toggles
if (segmentLiterature && segmentMemory) {
  segmentLiterature.addEventListener('click', () => {
    segmentLiterature.style.background = 'var(--primary)';
    segmentLiterature.style.color = '#fff';
    segmentMemory.style.background = 'transparent';
    segmentMemory.style.color = '#a0a0a0';
    wrapperLiterature?.classList.remove('hidden');
    wrapperMemory?.classList.add('hidden');
    loadKnowledgeFiles();
    loadGraph();
  });
  segmentMemory.addEventListener('click', () => {
    segmentMemory.style.background = 'var(--primary)';
    segmentMemory.style.color = '#fff';
    segmentLiterature.style.background = 'transparent';
    segmentLiterature.style.color = '#a0a0a0';
    wrapperMemory?.classList.remove('hidden');
    wrapperLiterature?.classList.add('hidden');
    loadMemoryGraph();
  });
}

// Knowledge Stack Tabs
if (tabGraph && tabSearch) {
  tabGraph.addEventListener('click', () => {
    tabGraph.classList.add('active');
    tabSearch.classList.remove('active');
    contentGraph?.classList.add('active');
    contentSearch?.classList.remove('active');
  });
  tabSearch.addEventListener('click', () => {
    tabSearch.classList.add('active');
    tabGraph.classList.remove('active');
    contentSearch?.classList.add('active');
    contentGraph?.classList.remove('active');
  });
}

// Knowledge Upload
const progressContainer = document.getElementById('upload-progress-container');
const progressBar = document.getElementById('upload-progress-bar');
const progressText = document.getElementById('upload-progress-text');

if (uploadBtn && uploadInput) {
  uploadBtn.addEventListener('click', async () => {
    if (!uploadInput.files || uploadInput.files.length === 0) return;
    const formData = new FormData();
    for (let i = 0; i < uploadInput.files.length; i++) {
      formData.append('files', uploadInput.files[i]);
    }
    formData.append('domain', domainSelect.value);

    uploadBtn.textContent = 'Uploading...';
    if (uploadBtn instanceof HTMLButtonElement) uploadBtn.disabled = true;
    if (progressContainer) progressContainer.classList.remove('hidden');
    if (progressBar) progressBar.style.width = '0%';
    if (progressText) progressText.textContent = '0%';

    try {
      const res = await fetch('/api/knowledgestackfrontend/upload', {
        method: 'POST',
        body: formData
      });
      if (res.ok && res.body) {
        const reader = res.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let done = false;

        while (!done) {
          const { value, done: readerDone } = await reader.read();
          done = readerDone;
          if (value) {
            const chunk = decoder.decode(value, { stream: !done });
            const lines = chunk.split('\\n');
            for (const line of lines) {
              if (line.trim()) {
                try {
                  const data = JSON.parse(line);
                  if (data.progress !== undefined) {
                    if (progressBar) progressBar.style.width = `${data.progress}%`;
                    if (progressText) progressText.textContent = `Embedding... ${data.progress}%`;
                  }
                  if (data.status === 'success') {
                    uploadInput.value = '';
                    loadKnowledgeFiles();
                    loadGraph();
                  }
                } catch (e) {
                  // Partial JSON chunk, ignore and continue
                }
              }
            }
          }
        }
      } else {
        alert('Upload failed.');
      }
    } catch (e) {
      if (progressText) progressText.innerText = 'Error parsing stream';
    } finally {
      uploadBtn.textContent = 'Upload to Database';
      if (uploadBtn instanceof HTMLButtonElement) uploadBtn.disabled = false;
    }
  });
}

// Memory Upload
if (memUploadBtn) {
  memUploadBtn.addEventListener('click', async () => {
    const session = memSessionInput.value.trim();
    const topic = memTopicInput.value.trim();
    const content = memContentInput.value.trim();
    if (!session || !topic || !content) {
      alert('Please fill out all memory fields');
      return;
    }
    
    if (memProgressContainer) memProgressContainer.classList.remove('hidden');
    if (memProgressBar) memProgressBar.style.width = '50%';
    if (memProgressText) memProgressText.innerText = 'Embedding memory...';
    
    const formData = new FormData();
    formData.append('session', session);
    formData.append('topic', topic);
    formData.append('content', content);
    
    try {
      await fetch('/api/knowledgestackfrontend/memory/upload', {
        method: 'POST',
        body: formData
      });
      if (memProgressBar) memProgressBar.style.width = '100%';
      if (memProgressText) memProgressText.innerText = 'Memory embedded successfully!';
      
      setTimeout(() => {
        if (memProgressContainer) memProgressContainer.classList.add('hidden');
        memSessionInput.value = '';
        memTopicInput.value = '';
        memContentInput.value = '';
        loadMemoryGraph();
      }, 2000);
    } catch (e) {
      if (memProgressText) memProgressText.innerText = 'Error embedding memory';
    }
  });
}

// Load Files
async function loadKnowledgeFiles() {
  if (!fileListContainer) return;
  try {
    const res = await fetch('/api/knowledgestackfrontend/documents');
    const data = await res.json();
    fileListContainer.innerHTML = '';
    
    for (const [domain, files] of Object.entries(data)) {
      const group = document.createElement('div');
      group.className = 'domain-group';
      group.innerHTML = `<h4>${domain}</h4>`;
      (files as any[]).forEach(f => {
        const item = document.createElement('div');
        item.className = 'file-item';
        item.textContent = f;
        group.appendChild(item);
      });
      fileListContainer.appendChild(group);
    }
  } catch (e) {}
}

// Search
if (searchBtnK && searchInputK) {
  searchBtnK.addEventListener('click', async () => {
    const q = searchInputK.value.trim();
    if (!q) return;
    
    // Switch to search tab
    tabSearch?.click();
    if (searchResultsContainer) searchResultsContainer.innerHTML = '<p>Searching embeddings...</p>';
    
    try {
      const res = await fetch(`/api/knowledgestackfrontend/search?q=${encodeURIComponent(q)}`);
      const data = await res.json();
      if (!searchResultsContainer) return;
      searchResultsContainer.innerHTML = '';
      if (data.results && data.results.length > 0) {
        data.results.forEach((r: any) => {
          const div = document.createElement('div');
          div.className = 'search-result';
          div.innerHTML = `<h4>${r.filename} (${r.domain})</h4><p>Score: ${r.score.toFixed(3)}</p><p>${r.snippet}</p>`;
          searchResultsContainer.appendChild(div);
        });
      } else {
        searchResultsContainer.innerHTML = '<p>No matching knowledge found.</p>';
      }
    } catch (e) {}
  });
}

// Load Literature Graph
async function loadGraph() {
  if (!cyContainer || (wrapperLiterature && wrapperLiterature.classList.contains('hidden'))) return;
  try {
    const res = await fetch('/api/knowledgestackfrontend/graph');
    if (!res.ok) return;
    const elements = await res.json();
    
    if (cyInstance) cyInstance.destroy();
    
    cyInstance = cytoscape({
      container: cyContainer,
      elements: elements,
      minZoom: 0.1,
      maxZoom: 1.5,
      style: [
        {
          selector: 'node',
          style: {
            'background-color': '#00d4ff',
            'label': 'data(label)',
            'color': '#fff',
            'text-valign': 'center',
            'text-outline-width': 2,
            'text-outline-color': '#000',
            'font-size': '10px'
          }
        },
        {
          selector: 'node[type="domain"]',
          style: {
            'background-color': '#ff4d4d',
            'width': 60,
            'height': 60,
            'font-size': '16px',
            'font-weight': 'bold'
          }
        },
        {
          selector: 'node[type="document"]',
          style: {
            'background-color': '#ffb347',
            'width': 40,
            'height': 40,
            'font-size': '12px'
          }
        },
        {
          selector: 'node[type="chunk"]',
          style: {
            'background-color': '#00d4ff',
            'width': 20,
            'height': 20,
            'font-size': '8px'
          }
        },
        {
          selector: 'edge',
          style: {
            'width': 2,
            'line-color': '#444',
            'target-arrow-color': '#444',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier'
          }
        }
      ],
      layout: {
        name: 'cose',
        padding: 10
      }
    });
  } catch (e) {}
}

// Load Memory Graph
async function loadMemoryGraph() {
  if (!cyMemoryContainer || (wrapperMemory && wrapperMemory.classList.contains('hidden'))) return;
  try {
    const res = await fetch('/api/knowledgestackfrontend/memory/graph');
    if (!res.ok) return;
    const elements = await res.json();
    
    if (cyMemoryInstance) cyMemoryInstance.destroy();
    
    cyMemoryInstance = cytoscape({
      container: cyMemoryContainer,
      elements: elements,
      minZoom: 0.1,
      maxZoom: 1.5,
      style: [
        {
          selector: 'node',
          style: {
            'background-color': '#e056fd',
            'label': 'data(label)',
            'color': '#fff',
            'text-valign': 'center',
            'text-outline-width': 2,
            'text-outline-color': '#000',
            'font-size': '10px'
          }
        },
        {
          selector: 'node[type="session"]',
          style: {
            'background-color': '#686de0',
            'width': 60,
            'height': 60,
            'font-size': '16px',
            'font-weight': 'bold'
          }
        },
        {
          selector: 'node[type="topic"]',
          style: {
            'background-color': '#f0932b',
            'width': 40,
            'height': 40,
            'font-size': '12px'
          }
        },
        {
          selector: 'node[type="memory"]',
          style: {
            'background-color': '#e056fd',
            'width': 20,
            'height': 20,
            'font-size': '8px'
          }
        },
        {
          selector: 'edge',
          style: {
            'width': 2,
            'line-color': '#666',
            'target-arrow-color': '#666',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier'
          }
        }
      ],
      layout: {
        name: 'cose',
        padding: 10
      }
    });
  } catch (e) {}
}

if (tabReadme) {
  tabReadme.addEventListener('click', async () => {
    tabReadme.classList.add('active');
    tabLicense?.classList.remove('active');
    await loadAboutContent('/api/docs/readme');
  });
}

if (tabLicense) {
  tabLicense.addEventListener('click', async () => {
    tabLicense.classList.add('active');
    tabReadme?.classList.remove('active');
    await loadAboutContent('/api/docs/license');
  });
}

async function loadAboutContent(url: string) {
  if (!aboutMarkdown) return;
  try {
    const res = await fetch(url);
    const data = await res.json();
    currentAboutContent = data.content || data.error || 'Empty';
    renderMarkdown(currentAboutContent);
  } catch (err) {
    aboutMarkdown.innerHTML = 'Error loading content.';
  }
}

async function renderMarkdown(mdText: string) {
  if (!aboutMarkdown || !aboutToc) return;

  await renderMarkdownToElement(aboutMarkdown, mdText);

  // Generate TOC
  aboutToc.innerHTML = '';
  const headers = aboutMarkdown.querySelectorAll('h1, h2, h3');
  headers.forEach((h, i) => {
    const id = `heading-${i}`;
    h.id = id;
    const a = document.createElement('a');
    a.href = `#${id}`;
    a.textContent = h.textContent;
    a.className = `toc-${h.tagName.toLowerCase()}`;
    a.addEventListener('click', (e) => {
      e.preventDefault();
      h.scrollIntoView({ behavior: 'smooth' });
    });
    aboutToc.appendChild(a);
  });
}

// Simple text search / highlight
if (searchInput) {
  searchInput.addEventListener('input', () => {
    const term = searchInput.value.trim().toLowerCase();
    // Reset view first
    renderMarkdown(currentAboutContent);
    if (!term || !aboutMarkdown) return;

    // Simple traverse and highlight text nodes (primitive approach)
    const walker = document.createTreeWalker(aboutMarkdown, NodeFilter.SHOW_TEXT, null);
    const nodes = [];
    let node;
    while ((node = walker.nextNode())) nodes.push(node);

    nodes.forEach(textNode => {
      if (textNode.nodeValue && textNode.parentElement && !['SCRIPT','STYLE'].includes(textNode.parentElement.tagName)) {
        const text = textNode.nodeValue;
        const lower = text.toLowerCase();
        const index = lower.indexOf(term);
        if (index >= 0) {
          const before = text.substring(0, index);
          const match = text.substring(index, index + term.length);
          const after = text.substring(index + term.length);
          
          const span = document.createElement('span');
          span.style.backgroundColor = 'rgba(255, 165, 0, 0.5)';
          span.textContent = match;
          
          const parent = textNode.parentElement;
          if (before) parent.insertBefore(document.createTextNode(before), textNode);
          parent.insertBefore(span, textNode);
          if (after) parent.insertBefore(document.createTextNode(after), textNode);
          parent.removeChild(textNode);
          
          span.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
      }
    });
  });
}

// --------------------------------------------------------
// MangoHUD Engine Stats Overlay Logic
// --------------------------------------------------------
const mangoText = document.getElementById('mango-text') as HTMLDivElement;
const mangoJSText = document.getElementById('mango-js-text') as HTMLDivElement;
const mangoCanvas = document.getElementById('mango-canvas') as HTMLCanvasElement;
const mangoJSCanvas = document.getElementById('mango-js-canvas') as HTMLCanvasElement;

let mangoCtx: CanvasRenderingContext2D | null = null;
if (mangoCanvas) {
  mangoCtx = mangoCanvas.getContext('2d');
}
let mangoJSCtx: CanvasRenderingContext2D | null = null;
if (mangoJSCanvas) {
  mangoJSCtx = mangoJSCanvas.getContext('2d');
}

let lastHUDUpdate = Date.now();
const jsLoopHistory: {ts: number, val: number}[] = [];

async function updateMangoHUD() {
  if (!mangoText || !mangoJSText || !mangoCtx || !mangoJSCtx) return;
  
  const now = Date.now();
  const jsLoopTime = now - lastHUDUpdate;
  lastHUDUpdate = now;
  jsLoopHistory.push({ts: now / 1000, val: jsLoopTime});
  if (jsLoopHistory.length > 60) jsLoopHistory.shift();

  try {
    const res = await fetch(`/api/adelaideenginestats?queue_len=${messageQueue.length}`);
    if (res.ok) {
      const stats = await res.json();
      
      const uptimeH = Math.floor(stats.Current_Uptime / 3600);
      const uptimeM = Math.floor((stats.Current_Uptime % 3600) / 60);
      const uptimeS = Math.floor(stats.Current_Uptime % 60);
      
      mangoText.textContent = `
ADELAIDE_ADA_ENGINE
--------------------
WCET_ELP0 : ${stats.WCET_ELP0.toFixed(3)}s (\u0394 ${(stats.WCET_ELP0_delta).toFixed(3)}s)
WCET_ELP1 : ${stats.WCET_ELP1.toFixed(3)}s (\u0394 ${(stats.WCET_ELP1_delta).toFixed(3)}s)
WCET_ELP2 : ${stats.WCET_ELP2.toFixed(3)}s (\u0394 ${(stats.WCET_ELP2_delta).toFixed(3)}s)
WCET_WtDog: ${stats.WCET_WatchdogLoop_uS.toFixed(1)} us (\u0394 ${(stats.WCET_WatchdogLoop_uS_delta).toFixed(1)}us)
WCET_mLoop: ${stats.WCET_mainLoop_uS.toFixed(1)} us (\u0394 ${(stats.WCET_mainLoop_uS_delta).toFixed(1)}us)
Memory    : ${stats.MemoryConsumption_MB.toFixed(1)} MB
CPU       : ${stats.CPU_Consumption.toFixed(1)} %
Tokens/s  : ${stats.WCETR.toFixed(2)}
Total Tok : ${stats.Total_Tokens_Processed}
Uptime    : ${uptimeH}h ${uptimeM}m ${uptimeS}s
Queue     : ${stats.Current_Queue}
`;

      mangoJSText.textContent = `
ADELAIDE_JAVASHIT_ENGINE
--------------------
WCEL      : ${(stats.WCEL / 1000).toFixed(2)} ms
WCEL \u0394 1m: ${(stats.WCEL_delta_1m / 1000).toFixed(2)} ms
JS Loop   : ${jsLoopTime} ms
`;
      
      // Draw Ada Graph (Tokens/s History)
      const width = mangoCanvas.width;
      const height = mangoCanvas.height;
      mangoCtx.clearRect(0, 0, width, height);
      
      const history: {ts: number, val: number}[] = stats.History_1m;
      if (history.length > 0) {
        const maxVal = Math.max(...history.map(h => h.val), 10);
        const minTime = Date.now() / 1000 - 60;
        
        mangoCtx.beginPath();
        mangoCtx.strokeStyle = '#0f0';
        mangoCtx.lineWidth = 1;
        
        for (let i = 0; i < history.length; i++) {
          const pt = history[i];
          const x = ((pt.ts - minTime) / 60) * width;
          const y = height - ((pt.val / maxVal) * height);
          if (i === 0) mangoCtx.moveTo(x, y);
          else mangoCtx.lineTo(x, y);
        }
        mangoCtx.stroke();
      }

      // Draw JS Graph (WCEL and JS Loop)
      const jsWidth = mangoJSCanvas.width;
      const jsHeight = mangoJSCanvas.height;
      mangoJSCtx.clearRect(0, 0, jsWidth, jsHeight);

      const wcelHistory: {ts: number, val: number}[] = stats.WCEL_History_1m || [];
      const minTimeJS = Date.now() / 1000 - 60;

      // Plot JS Loop Time in Cyan
      if (jsLoopHistory.length > 0) {
        const maxLoop = Math.max(...jsLoopHistory.map(h => h.val), 2000); // Max 2s
        mangoJSCtx.beginPath();
        mangoJSCtx.strokeStyle = '#0ff';
        mangoJSCtx.lineWidth = 1;
        for (let i = 0; i < jsLoopHistory.length; i++) {
          const pt = jsLoopHistory[i];
          const x = ((pt.ts - minTimeJS) / 60) * jsWidth;
          const y = jsHeight - ((pt.val / maxLoop) * jsHeight);
          if (i === 0) mangoJSCtx.moveTo(x, y);
          else mangoJSCtx.lineTo(x, y);
        }
        mangoJSCtx.stroke();
      }

      // Plot WCEL in Red/Orange for contrast
      if (wcelHistory.length > 0) {
        const maxWCEL = Math.max(...wcelHistory.map(h => h.val), 1000000); // 1s in us
        mangoJSCtx.beginPath();
        mangoJSCtx.strokeStyle = '#f50';
        mangoJSCtx.lineWidth = 1;
        for (let i = 0; i < wcelHistory.length; i++) {
          const pt = wcelHistory[i];
          const x = ((pt.ts - minTimeJS) / 60) * jsWidth;
          const y = jsHeight - ((pt.val / maxWCEL) * jsHeight);
          if (i === 0) mangoJSCtx.moveTo(x, y);
          else mangoJSCtx.lineTo(x, y);
        }
        mangoJSCtx.stroke();
      }
    }
  } catch (err) {
    mangoText.textContent = "Engine Offline";
    mangoJSText.textContent = "Engine Offline";
  }
}

setInterval(updateMangoHUD, 1000);
