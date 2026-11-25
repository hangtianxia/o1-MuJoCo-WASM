// WebGL2 fallback for the WebGPU liquid-glass compositor. Same SDF glass,
// dispersion, bevel and ambient math, for browsers/drivers where WebGPU
// rendering is unavailable.

const MAX_PANELS = 4;

const VERT = `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const FRAG = `#version 300 es
precision highp float;
uniform vec2 uCss;
uniform vec2 uCanvas;
uniform float uCount;
uniform float uTime;
uniform vec4 uPanels[4];
uniform vec4 uRadii;
uniform vec4 uFaceTop;
uniform vec4 uFaceBottom;
uniform vec4 uRim;
uniform vec4 uBevel;
uniform vec4 uShade;
uniform sampler2D uScene;
uniform sampler2D uAmbient;
out vec4 outColor;

float sdRoundBox(vec2 p, vec2 halfSize, float radius) {
  vec2 q = abs(p) - halfSize + vec2(radius);
  return length(max(q, vec2(0.0))) + min(max(q.x, q.y), 0.0) - radius;
}
float panelSdf(vec2 p, vec4 rect, float radius) {
  return sdRoundBox(p - (rect.xy + rect.zw * 0.5), rect.zw * 0.5, radius);
}
float radiusAt(int index) {
  if (index == 0) return uRadii.x;
  if (index == 1) return uRadii.y;
  if (index == 2) return uRadii.z;
  return uRadii.w;
}
vec3 sceneAt(vec2 uv, vec2 offset) {
  vec2 texel = 1.0 / vec2(textureSize(uScene, 0));
  vec2 base = clamp(uv + offset, vec2(0.0), vec2(1.0));
  vec3 c = texture(uScene, base).rgb * 0.44;
  c += texture(uScene, clamp(base + vec2(texel.x * 1.4, 0.0), vec2(0.0), vec2(1.0))).rgb * 0.14;
  c += texture(uScene, clamp(base - vec2(texel.x * 1.4, 0.0), vec2(0.0), vec2(1.0))).rgb * 0.14;
  c += texture(uScene, clamp(base + vec2(0.0, texel.y * 1.4), vec2(0.0), vec2(1.0))).rgb * 0.14;
  c += texture(uScene, clamp(base - vec2(0.0, texel.y * 1.4), vec2(0.0), vec2(1.0))).rgb * 0.14;
  return c;
}
void main() {
  vec2 uv = vec2(gl_FragCoord.x / uCanvas.x, 1.0 - gl_FragCoord.y / uCanvas.y);
  vec2 p = uv * uCss;
  int chosen = -1;
  float chosenD = 1.0e9;
  vec4 chosenRect = vec4(0.0);
  float chosenRadius = 22.0;
  for (int i = 0; i < 4; i++) {
    if (float(i) >= uCount) break;
    vec4 rect = uPanels[i];
    if (rect.z <= 0.0 || rect.w <= 0.0) continue;
    float radius = radiusAt(i);
    float d = panelSdf(p, rect, radius);
    if (d < 0.0 && d < chosenD) {
      chosenD = d;
      chosen = i;
      chosenRect = rect;
      chosenRadius = radius;
    }
  }
  if (chosen < 0) discard;
  float d = chosenD;
  vec4 rect = chosenRect;
  float w = max(rect.z, 1.0);
  float h = max(rect.w, 1.0);
  float band = min(min(w, h) * 0.30, 104.0);
  float t = clamp(-d / max(band, 1.0), 0.0, 1.0);
  float lens = pow(1.0 - t, 2.0);
  float amount = min(min(w, h) * 0.32, 112.0);
  float e = 1.0;
  float gx = panelSdf(p + vec2(e, 0.0), rect, chosenRadius) - panelSdf(p - vec2(e, 0.0), rect, chosenRadius);
  float gy = panelSdf(p + vec2(0.0, e), rect, chosenRadius) - panelSdf(p - vec2(0.0, e), rect, chosenRadius);
  vec2 normal = normalize(vec2(gx, gy) + vec2(0.0001));
  float pulse = 1.0 + 0.18 * sin(uTime * 1.05 + p.y * 0.013);
  vec2 offset = normal * (amount * lens * pulse) / uCss;
  offset += vec2(-normal.y, normal.x) * (sin(uTime * 0.82 + t * 8.0) * 5.0) / uCss;
  vec2 centerUv = (rect.xy + rect.zw * 0.5) / uCss;
  vec2 sampleUv = centerUv + (uv - centerUv) * 0.90;
  vec3 cr = sceneAt(sampleUv, offset * 1.00);
  vec3 cg = sceneAt(sampleUv, offset * 1.12);
  vec3 cb = sceneAt(sampleUv, offset * 1.24);
  vec4 amb = vec4(0.0);

  vec3 color = vec3(cr.r, cg.g, cb.b);
  color = color * (1.0 - amb.a) + amb.rgb;
  vec2 local = p - rect.xy;
  float faceT = clamp(local.x / w * 0.28 + local.y / h * 0.72, 0.0, 1.0);
  vec4 face = mix(uFaceTop, uFaceBottom, faceT);
  face.a *= 0.85 * smoothstep(0.0, 0.55, t);
  color = color * (1.0 - face.a) + face.rgb * face.a;
  float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
  color = mix(vec3(luma), color, 1.16);
  float edge = 1.0 - smoothstep(0.0, 2.0, -d);
  float glow = smoothstep(0.0, 5.0, -d) * (1.0 - smoothstep(12.0, 26.0, -d));
  color += vec3(1.0) * (1.0 - smoothstep(0.0, 1.2, -d)) * 0.12;
  color *= 1.0 - uShade.a * smoothstep(0.0, 18.0, -d) * 0.10;
  outColor = vec4(clamp(sceneAt(sampleUv, vec2(0.0)), vec3(0.0), vec3(1.0)), 1.0);
}`;

function parseColor(value) {
  if (!value) return [1, 1, 1, 0];
  const text = value.trim();
  if (text.startsWith('#')) {
    const hex = text.slice(1);
    const n = hex.length === 3 ? hex.split('').map((c) => c + c).join('') : hex.slice(0, 6);
    return [parseInt(n.slice(0, 2), 16) / 255, parseInt(n.slice(2, 4), 16) / 255, parseInt(n.slice(4, 6), 16) / 255, 1];
  }
  const nums = text.match(/[\d.]+/g);
  if (!nums || nums.length < 3) return [1, 1, 1, 0];
  return [Number(nums[0]) / 255, Number(nums[1]) / 255, Number(nums[2]) / 255, nums.length > 3 ? Number(nums[3]) : 1];
}

function cssColor(style, name) {
  return parseColor(style.getPropertyValue(name));
}

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error('[glass-gl] shader', gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

export async function createGlassGL({ canvas, sceneCanvas, ambientCanvas = null, panelSelectors = ['#controls', '#telemetry'] } = {}) {
  if (!canvas || !sceneCanvas) return null;
  let gl = null;
  try {
    gl = canvas.getContext('webgl2', { alpha: true, premultipliedAlpha: true, antialias: false });
  } catch (_) {}
  if (!gl) return null;
  const vs = compile(gl, gl.VERTEX_SHADER, VERT);
  const fs = compile(gl, gl.FRAGMENT_SHADER, FRAG);
  if (!vs || !fs) return null;
  const program = gl.createProgram();
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error('[glass-gl] link', gl.getProgramInfoLog(program));
    return null;
  }
  gl.useProgram(program);

  const U = {};
  for (const name of ['uCss', 'uCanvas', 'uCount', 'uTime', 'uPanels', 'uRadii', 'uFaceTop', 'uFaceBottom', 'uRim', 'uBevel', 'uShade', 'uScene', 'uAmbient']) {
    U[name] = gl.getUniformLocation(program, name);
  }
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  const ambientTexture = gl.createTexture();
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, ambientTexture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.activeTexture(gl.TEXTURE0);
  gl.uniform1i(U.uScene, 0);
  gl.uniform1i(U.uAmbient, 1);
  gl.disable(gl.DEPTH_TEST);
  gl.disable(gl.BLEND);

  let panels = [];
  let colors = null;
  let ambientSource = ambientCanvas;
  let ambientDirty = true;
  let lastDraw = 0;
  let destroyed = false;

  const readColors = () => {
    const style = getComputedStyle(document.documentElement);
    const faceTop = cssColor(style, '--glass-top');
    const faceBottom = cssColor(style, '--glass-bottom');
    const rim = cssColor(style, '--glass-topglow');
    const bevel = cssColor(style, '--glass-bevel');
    const shade = cssColor(style, '--glass-shade');
    return {
      faceTop, faceBottom,
      rim: [rim[0], rim[1], rim[2], Math.min(1, rim[3] * 1.6)],
      bevel: [bevel[0], bevel[1], bevel[2], Math.min(1, bevel[3] * 1.2)],
      shade,
    };
  };

  const updatePanels = () => {
    panels = panelSelectors.map((selector) => document.querySelector(selector)).filter(Boolean).map((element) => {
      const rect = element.getBoundingClientRect();
      const radius = Number.parseFloat(getComputedStyle(element).borderTopLeftRadius) || 22;
      return { x: rect.left, y: rect.top, w: rect.width, h: rect.height, radius };
    }).filter((p) => p.w > 0 && p.h > 0);
  };

  const resize = () => {
    const cssW = innerWidth || 1;
    const cssH = innerHeight || 1;
    const w = Math.max(1, Math.round(cssW * 0.6));
    const h = Math.max(1, Math.round(cssH * 0.6));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    gl.viewport(0, 0, w, h);
  };

  const render = (now) => {
    if (destroyed) return;
    if (now - lastDraw < 1000 / 30) return;
    lastDraw = now;
    updatePanels();
    if (!colors) colors = readColors();
    resize();
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    try {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, sceneCanvas);
    } catch (error) {
      console.warn('[glass-gl] scene upload failed', error);
      return;
    }
    if (ambientDirty && ambientSource) {
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, ambientTexture);
      try {
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, ambientSource);
        ambientDirty = false;
      } catch (error) {
        console.warn('[glass-gl] ambient upload failed', error);
      }
      gl.activeTexture(gl.TEXTURE0);
    }
    gl.useProgram(program);
    gl.uniform2f(U.uCss, innerWidth || 1, innerHeight || 1);
    gl.uniform2f(U.uCanvas, canvas.width, canvas.height);
    gl.uniform1f(U.uCount, panels.length);
    gl.uniform1f(U.uTime, now / 1000);
    const rects = new Float32Array(MAX_PANELS * 4);
    const radii = new Float32Array(MAX_PANELS);
    panels.forEach((panel, index) => {
      rects.set([panel.x, panel.y, panel.w, panel.h], index * 4);
      radii[index] = panel.radius;
    });
    gl.uniform4fv(U.uPanels, rects);
    gl.uniform4fv(U.uRadii, radii);
    gl.uniform4fv(U.uFaceTop, colors.faceTop);
    gl.uniform4fv(U.uFaceBottom, colors.faceBottom);
    gl.uniform4fv(U.uRim, colors.rim);
    gl.uniform4fv(U.uBevel, colors.bevel);
    gl.uniform4fv(U.uShade, colors.shade);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.flush();
  };

  const onResize = () => { resize(); updatePanels(); };
  const onTheme = () => { colors = readColors(); };
  const observer = new ResizeObserver(() => updatePanels());
  document.querySelectorAll(panelSelectors.join(',')).forEach((element) => observer.observe(element));
  addEventListener('resize', onResize);
  document.addEventListener('themechange', onTheme);
  resize();
  updatePanels();
  colors = readColors();
  document.body.classList.add('glass-gpu');
  window.__glassGL = { get active() { return !destroyed; }, get panels() { return panels.length; }, destroy };

  function destroy() {
    if (destroyed) return;
    destroyed = true;
    removeEventListener('resize', onResize);
    document.removeEventListener('themechange', onTheme);
    observer.disconnect();
    document.body.classList.remove('glass-gpu');
    gl.deleteTexture(texture);
    gl.deleteProgram(program);
  }

  const setAmbient = (canvasSource) => { ambientSource = canvasSource; ambientDirty = true; };

  return { render, destroy, setAmbient, get active() { return !destroyed; }, get backend() { return 'webgl2'; } };
}
