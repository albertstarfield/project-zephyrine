import * as THREE from 'three';

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
uniform int uNumRects;
uniform vec4 uRects[50]; // [cx, cy, hw, hh] - Center X/Y and Half Width/Height in pixel coords
uniform float uRadii[50]; // Corner radius

// Glass configuration
uniform float uBezel;
uniform float uThickness;
uniform float uIOR;
uniform float uBlur;
uniform float uSpecular;
uniform float uTint;
uniform float uShadow;
uniform float uTime;

float sdRoundedRect(vec2 p, vec2 halfSize, float r) {
  vec2 q = abs(p) - halfSize + r;
  return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r;
}

float surfaceHeight(float t) {
  float s = 1.0 - t;
  return pow(1.0 - s*s*s*s, 0.25);
}

// Hash function for stars
float hash(vec2 p) {
  return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

vec3 sampleBg(vec2 uv) {
  vec2 p = uv * 2.0 - 1.0;
  p.x *= uResolution.x / uResolution.y;
  
  // Base very deep dark color
  vec3 color = vec3(0.04, 0.045, 0.06);
  
  // Radial glow orb in center
  float dist = length(p);
  float glow = exp(-dist * 1.8) * 0.4;
  color += vec3(0.12, 0.24, 0.6) * glow;
  
  // Procedural stars
  float n = hash(floor(uv * 800.0));
  if(n > 0.995) {
      float brightness = (n - 0.995) * 200.0;
      float twinkle = 0.5 + 0.5 * sin(uTime * 3.0 + n * 100.0);
      color += vec3(brightness * twinkle);
  }
  
  return color;
}

vec3 sampleBgBlurred(vec2 uv, float radius) {
  if (radius < 0.5) return sampleBg(uv);
  vec3 sum = vec3(0.0);
  vec2 px = 1.0 / uResolution;
  vec2 offsets[16];
  offsets[0]  = vec2(-0.94201, -0.39906);
  offsets[1]  = vec2( 0.94558, -0.76890);
  offsets[2]  = vec2(-0.09418, -0.92938);
  offsets[3]  = vec2( 0.34495,  0.29387);
  offsets[4]  = vec2(-0.91588, -0.45771);
  offsets[5]  = vec2(-0.81544,  0.48568);
  offsets[6]  = vec2(-0.38277, -0.56071);
  offsets[7]  = vec2(-0.12675,  0.84686);
  offsets[8]  = vec2( 0.89642,  0.41254);
  offsets[9]  = vec2( 0.18150, -0.30020);
  offsets[10] = vec2(-0.01445, -0.16001);
  offsets[11] = vec2( 0.59614,  0.71118);
  offsets[12] = vec2( 0.49742, -0.47280);
  offsets[13] = vec2( 0.80685,  0.04588);
  offsets[14] = vec2(-0.32490, -0.03965);
  offsets[15] = vec2(-0.60975,  0.06566);
  for (int i = 0; i < 16; i++) {
    sum += sampleBg(uv + offsets[i] * radius * px);
  }
  return sum / 16.0;
}

void main() {
  vec2 screenPx = vec2(vUv.x, 1.0 - vUv.y) * uResolution;
  
  float sd = 99999.0;
  float minBezel = uBezel;
  
  for (int i = 0; i < 50; i++) {
    if (i >= uNumRects) break;
    vec2 p = screenPx - uRects[i].xy;
    vec2 halfSize = uRects[i].zw;
    float r = uRadii[i];
    float d = sdRoundedRect(p, halfSize, r);
    if (d < sd) {
      sd = d;
      minBezel = min(uBezel, min(r, min(halfSize.x, halfSize.y)) - 1.0);
    }
  }

  if (sd > 0.0) {
    // Outside the glass -> render background maybe with shadow
    float shadowFalloff = exp(-sd * sd / 800.0);
    float shadowAlpha = uShadow * shadowFalloff * 0.6;
    
    vec3 baseColor = sampleBg(vUv);
    vec3 outColor = mix(baseColor, vec3(0.0), shadowAlpha);
    gl_FragColor = vec4(outColor, 1.0);
    return;
  }

  float distFromEdge = -sd;
  float t = clamp(distFromEdge / minBezel, 0.0, 1.0);

  float h = surfaceHeight(t);
  float dt = 0.001;
  float h2 = surfaceHeight(min(t + dt, 1.0));
  float dh = (h2 - h) / dt;

  float slopeAngle = atan(dh * (uThickness / minBezel));
  float sinR = sin(slopeAngle) / uIOR;
  sinR = clamp(sinR, -1.0, 1.0);
  float thetaR = asin(sinR);
  float displacement = h * uThickness * (tan(slopeAngle) - tan(thetaR));

  vec2 grad = vec2(0.0);
  float eps = 0.5;
  float sdRight = 99999.0;
  float sdUp = 99999.0;
  
  for (int i = 0; i < 50; i++) {
    if (i >= uNumRects) break;
    vec2 pRight = screenPx + vec2(eps, 0.0) - uRects[i].xy;
    vec2 pUp = screenPx + vec2(0.0, eps) - uRects[i].xy;
    vec2 halfSize = uRects[i].zw;
    float r = uRadii[i];
    
    sdRight = min(sdRight, sdRoundedRect(pRight, halfSize, r));
    sdUp = min(sdUp, sdRoundedRect(pUp, halfSize, r));
  }
  
  grad.x = sdRight - sd;
  grad.y = sdUp - sd;
  grad = normalize(grad);

  vec2 offset = -grad * displacement / uResolution;

  vec2 screenUV = screenPx / uResolution;
  vec2 refractedUV = screenUV + offset;

  vec3 color = sampleBgBlurred(refractedUV, uBlur);

  vec2 lightDir = normalize(vec2(0.5, -0.7));
  float rimDot = abs(dot(grad, lightDir));
  float rimFalloff = 1.0 - smoothstep(0.0, minBezel * 0.4, distFromEdge);
  float specHighlight = pow(rimDot * rimFalloff, 1.5);
  color += vec3(specHighlight * uSpecular);

  float innerShadow = 1.0 - smoothstep(0.0, minBezel * 0.6, distFromEdge);
  color *= mix(1.0, 0.7, innerShadow * 0.3);

  float innerRim = smoothstep(0.0, 2.0, distFromEdge) * (1.0 - smoothstep(2.0, 5.0, distFromEdge));
  color += vec3(innerRim * 0.15 * uSpecular);

  // Apply dark tint specific to Adelaide Lite design
  vec3 tintColor = vec3(0.0, 0.05, 0.02); // slight dark greenish tint
  color = mix(color, tintColor, uTint);

  gl_FragColor = vec4(color, 1.0);
}`;

export class LiquidGlassSystem {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.OrthographicCamera;
  material: THREE.ShaderMaterial;
  bgTexture: THREE.Texture | null = null;
  animationFrameId: number | null = null;
  needsRender = true;

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, alpha: false, antialias: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    
    this.scene = new THREE.Scene();
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

    // Initial state based on the requested aesthetic
    this.material = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: {
        uResolution: { value: new THREE.Vector2(window.innerWidth, window.innerHeight) },
        uNumRects: { value: 0 },
        uRects: { value: Array(50).fill(new THREE.Vector4()) },
        uRadii: { value: Array(50).fill(0) },
        uBezel: { value: 40.0 },
        uThickness: { value: 30.0 },
        uIOR: { value: 2.0 },
        uBlur: { value: 2.5 },
        uSpecular: { value: 0.6 },
        uTint: { value: 0.2 }, // slight tint over procedural
        uShadow: { value: 0.8 },
        uTime: { value: 0.0 },
      },
      depthTest: false,
    });

    this.scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.material));
    
    this.resize();
    window.addEventListener('resize', this.resize.bind(this));
    
    this.renderLoop = this.renderLoop.bind(this);
    this.renderLoop();
  }
  
  setBackgroundImage(url: string) {
    new THREE.TextureLoader().load(url, (tex) => {
      tex.minFilter = THREE.LinearFilter;
      tex.magFilter = THREE.LinearFilter;
      this.bgTexture = tex;
      this.material.uniforms.uBgTex.value = tex;
      this.material.uniforms.uBgAspect.value = tex.image.width / tex.image.height;
      this.needsRender = true;
    });
  }
  
  updateRects(rects: { x: number, y: number, w: number, h: number, r: number }[]) {
    const num = Math.min(rects.length, 50);
    this.material.uniforms.uNumRects.value = num;
    
    const uRects = this.material.uniforms.uRects.value;
    const uRadii = this.material.uniforms.uRadii.value;
    
    for (let i = 0; i < num; i++) {
      const r = rects[i];
      // center X, center Y, half width, half height
      uRects[i] = new THREE.Vector4(r.x + r.w / 2, r.y + r.h / 2, r.w / 2, r.h / 2);
      uRadii[i] = r.r;
    }
    this.needsRender = true;
  }

  resize() {
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.material.uniforms.uResolution.value.set(window.innerWidth, window.innerHeight);
    this.needsRender = true;
  }

  renderLoop() {
    if (this.needsRender || true) { // Always render if uTime is animating
      this.material.uniforms.uTime.value = performance.now() / 1000.0;
      this.renderer.render(this.scene, this.camera);
      this.needsRender = false;
    }
    this.animationFrameId = requestAnimationFrame(this.renderLoop);
  }
  
  dispose() {
    if (this.animationFrameId) cancelAnimationFrame(this.animationFrameId);
    window.removeEventListener('resize', this.resize.bind(this));
    this.renderer.dispose();
  }
}
