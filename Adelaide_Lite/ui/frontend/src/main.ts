import './style.css';
import * as THREE from 'three';

// --------------------------------------------------------
// Chat Logic
// --------------------------------------------------------

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp?: string;
}

const messagesContainer = document.getElementById('messages') as HTMLDivElement;
const chatForm = document.getElementById('chat-form') as HTMLFormElement;
const chatInput = document.getElementById('chat-input') as HTMLInputElement;
const sendBtn = document.getElementById('send-btn') as HTMLButtonElement;

const emptyState = document.getElementById('empty-state') as HTMLDivElement;
const chatContainerWrapper = document.getElementById('chat-container') as HTMLDivElement;

function addMessageToUI(msg: Message) {
  emptyState.classList.add('hidden');
  chatContainerWrapper.classList.remove('hidden');

  const msgEl = document.createElement('div');
  msgEl.className = `message ${msg.role}`;
  
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = msg.content;
  
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
      }
    }
  } catch (err) {
    console.error("Failed to load history:", err);
  }
}

chatForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = chatInput.value.trim();
  if (!text) return;

  // Optimistically add user message
  addMessageToUI({ role: 'user', content: text });
  chatInput.value = '';
  sendBtn.disabled = true;

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
  } finally {
    sendBtn.disabled = false;
    chatInput.focus();
  }
});

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
const navTuning = document.getElementById('nav-tuning');
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

if (navTuning) {
  navTuning.addEventListener('click', (e) => {
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
import { marked } from 'marked';
import DOMPurify from 'dompurify';
import mermaid from 'mermaid';
import katex from 'katex';
import 'katex/dist/katex.min.css';

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

if (navMain) {
  navMain.addEventListener('click', (e) => {
    e.preventDefault();
    aboutContainer?.classList.add('hidden');
    inputContainer.classList.remove('hidden');
    // Decide whether to show empty state or chat container based on messages
    if (messagesContainer.children.length === 0) {
      emptyState.classList.remove('hidden');
    } else {
      chatContainerWrapper.classList.remove('hidden');
    }
  });
}

if (navAbout) {
  navAbout.addEventListener('click', async (e) => {
    e.preventDefault();
    emptyState.classList.add('hidden');
    chatContainerWrapper.classList.add('hidden');
    inputContainer.classList.add('hidden');
    aboutContainer?.classList.remove('hidden');
    await loadAboutContent('/api/docs/readme');
    tabReadme?.classList.add('active');
    tabLicense?.classList.remove('active');
  });
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

function renderMarkdown(mdText: string) {
  if (!aboutMarkdown || !aboutToc) return;

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
  aboutMarkdown.innerHTML = cleanHtml;

  // Render Mermaid diagrams
  const codeBlocks = aboutMarkdown.querySelectorAll('pre code.language-mermaid');
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
