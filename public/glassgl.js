// Zero-copy same-context WebGL2 liquid-glass post-pass inside Three.js's own
// WebGL2 context (eliminates cross-canvas readback/stalls on Safari/WebKit).
// Implements the same iOS 26.5 optical pipeline as glassgpu.js:
// G2 continuous superellipse SDF, mipmapped variable blur, dual-layer spherical
// refraction with subtle edge dispersion, YCC face & bleed color matrices, soft
// optical shadows, and 45-degree key/fill vibrant specular rims.

import { MAX_PANELS, DEFAULT_PANEL_SELECTORS, getThemeGlassParams, collectPanels } from './glass-math.js';

const VERT = `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const FRAG = `#version 300 es
precision highp float;

uniform vec4 uResMeta;           // cssW, cssH, count, lightFlag
uniform vec2 uCanvas;            // backing framebuffer px
uniform vec4 uPanels[12];        // x, y, w, h (CSS px)
uniform float uRadii[12];
uniform vec4 uFaceCM0;
uniform vec4 uFaceCM1;
uniform vec4 uFaceCM2;
uniform vec4 uBleedCM0;
uniform vec4 uBleedCM1;
uniform vec4 uBleedCM2;
uniform vec4 uRimCM0;
uniform vec4 uRimCM1;
uniform vec4 uRimCM2;
uniform vec4 uShadowParams;      // offsetX, offsetY, opacity, radius
uniform vec4 uRimParams;         // widthPx, curvature, keyOpacity, fillOpacity
uniform vec4 uOptics;            // bleedOpacity, bleedDarken, dispersion, blurScale
uniform sampler2D uScene;

out vec4 outColor;

float sdSquircle(vec2 p, vec2 halfSize, float radius) {
  float r = min(radius, min(halfSize.x, halfSize.y));
  const float K = 1.5286649465560913;
  float E = r * K;
  vec2 ap = abs(p);
  vec2 corner = halfSize - min(vec2(E), halfSize);
  if (ap.x > corner.x && ap.y > corner.y && r > 0.5) {
    vec2 wts = clamp(vec2(2.0 * 1.4457785) - (2.0 * halfSize * 1.4457785) / max(E, 1e-4), vec2(0.0), vec2(1.0));
    vec2 qq = ap - halfSize + vec2(E + (r - E) * max(wts.x, wts.y));
    vec2 v = max(vec2(0.0), (ap - halfSize + vec2(E)) / max(E, 1e-4));
    float L = length(v);
    float hi = max(v.x, v.y);
    float t = clamp(hi > 0.0 ? min(v.x, v.y) / hi : 0.0, 0.0, 1.0);
    float poly = (((-0.9260540008544922 * t + 3.1560099124908447) * t - 3.6412200927734375) * t + 1.268030047416687) * t + 0.2685309946537018;
    float cont = L + 1.0 - 1.0 / (1.0 - t * t * clamp(L, 0.0, 1.0) * poly);
    float circ = length(max(vec2(0.0), v * K - vec2(0.5286650061607361))) * 0.6541655659675598 + 0.3458344340324402;
    float s = v.y > v.x ? 1.0 : -1.0;
    float b = clamp(0.5 - s + s * t, 0.0, 1.0);
    float c = cont + (circ - cont) * wts.x;
    c = c + ((cont + (circ - cont) * wts.y) - c) * b;
    return min(max(qq.x, qq.y), 0.0) + E * (c - 1.0);
  }
  vec2 q = ap - halfSize;
  return max(q.x, q.y);
}

float panelSdf(vec2 p, vec4 rect, float radius) {
  return sdSquircle(p - (rect.xy + rect.zw * 0.5), rect.zw * 0.5, radius);
}

float sphereProfile(float amount, float invHeight, float d) {
  float t = clamp(-d * invHeight, 0.0, 1.0);
  return amount * (1.0 - clamp(sqrt((2.0 - t) * t), 0.0, 1.0));
}

float shadowCDF(float shiftedDistance, float invRadius) {
  float x = clamp(0.25 * (invRadius * shiftedDistance) + 0.5, 0.0, 1.0);
  x = 4.0 * x - 2.0;
  float x2 = x * x;
  float poly = 0.002954483 * x2 - 0.034454346;
  poly = poly * x2 + 0.168212891;
  poly = poly * x2 - 0.560546875;
  return clamp(poly * x + 0.5, 0.0, 1.0);
}

float radiusLOD(float radius) {
  float logIn = radius >= 2.0 ? radius : 1.0 + 0.5 * radius;
  return clamp(log2(max(1.0, logIn)), 0.0, 4.5);
}

vec3 applyCM(vec3 rgb, vec4 r0, vec4 r1, vec4 r2) {
  return vec3(
    dot(rgb, r0.xyz) + r0.w,
    dot(rgb, r1.xyz) + r1.w,
    dot(rgb, r2.xyz) + r2.w
  );
}

float keyFillBand(float depth, float width, float aa, float curvature) {
  float t = clamp(depth / max(width, 1e-4), 0.0, 1.0);
  float hard = t < 1.0 ? 1.0 : 0.0;
  return mix(hard, 1.0 - t, curvature)
    * clamp(depth / aa + 0.5, 0.0, 1.0)
    * clamp((width - depth) / aa + 0.5, 0.0, 1.0);
}

vec3 sampleScene(vec2 uv, float lod) {
  return textureLod(uScene, clamp(uv, vec2(0.001), vec2(0.999)), lod).rgb;
}

void main() {
  vec2 res = uResMeta.xy;
  int count = int(uResMeta.z + 0.5);
  // uScene is copied directly from the default WebGL framebuffer (origin bottom-left).
  vec2 texUv = gl_FragCoord.xy / uCanvas;
  vec2 cssUv = vec2(texUv.x, 1.0 - texUv.y);
  vec2 p = cssUv * res;

  int chosen = -1;
  float chosenD = 1.0e9;
  vec4 chosenRect = vec4(0.0);
  float chosenRadius = 18.0;
  float shadowAccum = 0.0;

  for (int i = 0; i < 12; i++) {
    if (i >= count) break;
    vec4 rect = uPanels[i];
    if (rect.z <= 0.0 || rect.w <= 0.0) continue;
    vec2 halfSize = rect.zw * 0.5;
    vec2 qBox = abs(p - (rect.xy + halfSize)) - halfSize;
    float boxD = max(qBox.x, qBox.y);
    if (boxD > 60.0) continue;
    float radius = uRadii[i];
    float d = panelSdf(p, rect, radius);
    if (d < chosenD) {
      chosenD = d;
      chosen = i;
      chosenRect = rect;
      chosenRadius = radius;
    }
    float D = min(rect.z, rect.w);
    float sScale = clamp(D / 140.0, 0.45, 1.0);
    float sRad = uShadowParams.w * sScale;
    vec2 sOff = uShadowParams.xy * sScale;
    if (d > -2.0 && d < sRad * 2.2) {
      float sd = panelSdf(p - sOff, rect, radius);
      float sFactor = shadowCDF(sd, 1.0 / max(sRad, 1.0)) * uShadowParams.z * mix(0.65, 1.0, sScale);
      shadowAccum = max(shadowAccum, sFactor);
    }
  }

  if (chosen < 0) discard;

  float d = chosenD;
  float aa = max(fwidth(d), 0.75);
  float coverage = clamp(0.5 - d / aa, 0.0, 1.0);

  if (coverage <= 0.001 && shadowAccum <= 0.002) discard;

  vec3 shadowTint = uResMeta.w > 0.5 ? vec3(0.05, 0.09, 0.15) : vec3(0.01, 0.02, 0.04);
  vec4 shadowMat = vec4(shadowTint * shadowAccum, shadowAccum);

  if (coverage <= 0.001) {
    outColor = shadowMat;
    return;
  }

  vec4 rect = chosenRect;
  float D = min(rect.z, rect.w);
  float largeT = clamp((D - 44.0) / 116.0, 0.0, 1.0);

  float eps = 0.75;
  float gx = panelSdf(p + vec2(eps, 0.0), rect, chosenRadius) - panelSdf(p - vec2(eps, 0.0), rect, chosenRadius);
  float gy = panelSdf(p + vec2(0.0, eps), rect, chosenRadius) - panelSdf(p - vec2(0.0, eps), rect, chosenRadius);
  vec2 normal = normalize(vec2(gx, gy) + vec2(1e-5, 0.0));
  // Convert CSS-space normal (+y down) to WebGL texture space (+v up)
  vec2 texNormal = vec2(normal.x, -normal.y);

  float innerHeight = clamp(D * 0.24, 10.0, 22.0);
  float innerAmount = -min(D * 0.60, 48.0);
  float outerHeight = clamp(D * 0.12, 4.0, 13.0);
  float outerAmount = min(D * 0.15, 13.0);

  float innerDisp = sphereProfile(innerAmount, 1.0 / innerHeight, d);
  float outerDisp = sphereProfile(outerAmount, 1.0 / outerHeight, d);

  float baseBlur = mix(0.45, 1.35, largeT) * uOptics.w;
  float rimClear = clamp(-d / max(innerHeight * 0.95, 1.0), 0.0, 1.0);
  float innerLod = radiusLOD(baseBlur * smoothstep(0.0, 1.0, rimClear));
  float outerLod = radiusLOD(baseBlur * 0.18);

  float disp = uOptics.z;
  vec2 innerVec = (texNormal * innerDisp) / res;
  vec2 outerVec = (texNormal * outerDisp) / res;

  float cr = sampleScene(texUv + innerVec * (1.0 - disp), innerLod).r;
  float cg = sampleScene(texUv + innerVec, innerLod).g;
  float cb = sampleScene(texUv + innerVec * (1.0 + disp), innerLod).b;
  vec3 refracted = vec3(cr, cg, cb);

  vec3 outerSrc = sampleScene(texUv + outerVec, outerLod);
  float outerMix = clamp((d + 1.5) / 1.5, 0.0, 1.0) * 0.30;
  refracted = mix(refracted, outerSrc, outerMix);

  vec3 faceRGB = applyCM(refracted, uFaceCM0, uFaceCM1, uFaceCM2);

  float bleedOpacity = uOptics.x * mix(0.25, 1.0, largeT);
  if (bleedOpacity > 0.001) {
    float bleedHeight = clamp(D * 0.35, 12.0, 52.0);
    float bleedAmount = clamp(D * 0.35, 12.0, 48.0);
    float bleedDisp = sphereProfile(bleedAmount, 1.0 / bleedHeight, d);
    float bleedLod = radiusLOD(clamp(D * 0.22, 8.0, 28.0));
    vec3 bleedSrc = sampleScene(texUv + (texNormal * bleedDisp) / res, bleedLod);
    vec3 bleedRGB = applyCM(bleedSrc, uBleedCM0, uBleedCM1, uBleedCM2);
    float bleedRamp = 1.0 - clamp(-d / max(bleedHeight, 1.0), 0.0, 1.0);
    float luma = clamp(dot(faceRGB, vec3(0.2125, 0.7154, 0.0721)), 0.0, 1.0);
    float darken = uOptics.y > 0.5 ? luma : (1.0 - luma);
    float w = darken * darken * bleedRamp;
    faceRGB = mix(faceRGB, bleedRGB, clamp(w * w * bleedOpacity, 0.0, 0.65));
  }

  float holdW = 1.0 - clamp((-d - 1.0) / 2.0, 0.0, 1.0);
  faceRGB = mix(faceRGB, faceRGB * (uResMeta.w > 0.5 ? 0.935 : 0.975), holdW * (uResMeta.w > 0.5 ? 0.50 : 0.35));

  float depth = -d;
  float rimW = uRimParams.x;
  float curv = uRimParams.y;
  vec2 keyDir = vec2(-0.70710678, -0.70710678);
  vec2 fillDir = vec2(0.70710678, 0.70710678);

  float band = keyFillBand(depth, rimW, aa, curv);
  float keyAlign = clamp(dot(keyDir, normal), 0.0, 1.0);
  float fillAlign = clamp(dot(fillDir, normal), 0.0, 1.0);

  float grazing = 1.0 - max(keyAlign, fillAlign);
  float contour = keyFillBand(depth, 1.05, aa, 0.5) * (uResMeta.w > 0.5 ? (0.10 + 0.08 * grazing) : 0.05);
  faceRGB *= 1.0 - contour;

  float keyMask = band * pow(keyAlign, 1.25) * uRimParams.z;
  float fillMask = band * pow(fillAlign, 1.35) * uRimParams.w;
  float hairline = keyFillBand(depth, 1.0, aa, 0.85) * (uResMeta.w > 0.5 ? 0.22 : 0.18);
  float rimAlpha = clamp(keyMask + fillMask + hairline, 0.0, 1.0);

  if (rimAlpha > 0.001) {
    vec3 vibrantRim = clamp(applyCM(faceRGB, uRimCM0, uRimCM1, uRimCM2), vec3(0.0), vec3(1.2));
    faceRGB = mix(faceRGB, clamp(vibrantRim, vec3(0.0), vec3(1.0)), rimAlpha);
  }

  float innerSheen = smoothstep(0.5, 2.5, depth) * (1.0 - smoothstep(3.0, 11.0, depth))
    * pow(keyAlign, 1.6) * (uResMeta.w > 0.5 ? 0.05 : 0.07);
  faceRGB = clamp(faceRGB + vec3(innerSheen), vec3(0.0), vec3(1.0));

  vec4 faceMat = vec4(faceRGB, 1.0);
  outColor = mix(shadowMat, faceMat, coverage);
}`;

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

export async function createGlassGL({
  renderer = null,
  canvas = null,
  sceneCanvas = null,
  panelSelectors = DEFAULT_PANEL_SELECTORS,
} = {}) {
  const inline = Boolean(renderer && typeof renderer.getContext === 'function');
  const gl = inline ? renderer.getContext() : canvas?.getContext('webgl2', { alpha: true, premultipliedAlpha: true, antialias: false });
  if (!gl || !(gl instanceof WebGL2RenderingContext)) return null;

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

  const U = {};
  for (const name of [
    'uResMeta', 'uCanvas', 'uPanels', 'uRadii',
    'uFaceCM0', 'uFaceCM1', 'uFaceCM2',
    'uBleedCM0', 'uBleedCM1', 'uBleedCM2',
    'uRimCM0', 'uRimCM1', 'uRimCM2',
    'uShadowParams', 'uRimParams', 'uOptics', 'uScene',
  ]) {
    U[name] = gl.getUniformLocation(program, name);
  }

  const vao = gl.createVertexArray();
  const texture = gl.createTexture();
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAX_LEVEL, 4);
  if (inline) renderer.resetState();

  let panels = [];
  let panelsDirty = true;
  let themeParams = getThemeGlassParams(document.documentElement.dataset.theme || 'dark');
  let destroyed = false;
  let texW = 0;
  let texH = 0;
  const rects = new Float32Array(MAX_PANELS * 4);
  const radii = new Float32Array(MAX_PANELS);

  const updatePanels = () => {
    panels = collectPanels(panelSelectors, MAX_PANELS);
    panelsDirty = false;
  };

  const render = () => {
    if (destroyed) return;
    if (panelsDirty) updatePanels();
    if (!panels.length) return;

    const w = gl.drawingBufferWidth;
    const h = gl.drawingBufferHeight;
    if (!w || !h) return;

    // Zero-copy on-GPU copy from Three.js's default framebuffer into our mipmapped texture.
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    if (inline) {
      if (texW !== w || texH !== h) {
        texW = w;
        texH = h;
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      }
      gl.copyTexSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 0, 0, w, h);
    } else if (sceneCanvas) {
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, sceneCanvas);
    } else {
      return;
    }
    gl.generateMipmap(gl.TEXTURE_2D);

    gl.viewport(0, 0, w, h);
    gl.disable(gl.SCISSOR_TEST);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);
    gl.depthMask(false);
    gl.colorMask(true, true, true, true);
    if (inline) {
      gl.enable(gl.BLEND);
      gl.blendEquation(gl.FUNC_ADD);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    } else {
      gl.disable(gl.BLEND);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }

    gl.useProgram(program);
    gl.bindVertexArray(vao);
    gl.uniform1i(U.uScene, 0);
    gl.uniform4f(U.uResMeta, innerWidth || 1, innerHeight || 1, panels.length, themeParams.light);
    gl.uniform2f(U.uCanvas, w, h);

    rects.fill(0);
    radii.fill(0);
    for (let i = 0; i < panels.length; i++) {
      const p = panels[i];
      rects[i * 4] = p.x;
      rects[i * 4 + 1] = p.y;
      rects[i * 4 + 2] = p.w;
      rects[i * 4 + 3] = p.h;
      radii[i] = p.radius;
    }
    gl.uniform4fv(U.uPanels, rects);
    gl.uniform1fv(U.uRadii, radii);
    gl.uniform4fv(U.uFaceCM0, themeParams.faceCM0);
    gl.uniform4fv(U.uFaceCM1, themeParams.faceCM1);
    gl.uniform4fv(U.uFaceCM2, themeParams.faceCM2);
    gl.uniform4fv(U.uBleedCM0, themeParams.bleedCM0);
    gl.uniform4fv(U.uBleedCM1, themeParams.bleedCM1);
    gl.uniform4fv(U.uBleedCM2, themeParams.bleedCM2);
    gl.uniform4fv(U.uRimCM0, themeParams.rimCM0);
    gl.uniform4fv(U.uRimCM1, themeParams.rimCM1);
    gl.uniform4fv(U.uRimCM2, themeParams.rimCM2);
    gl.uniform4fv(U.uShadowParams, themeParams.shadow);
    gl.uniform4fv(U.uRimParams, themeParams.rimParams);
    gl.uniform4fv(U.uOptics, themeParams.optics);

    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
    if (inline) renderer.resetState();
  };

  const onResize = () => { panelsDirty = true; };
  const onTheme = () => {
    themeParams = getThemeGlassParams(document.documentElement.dataset.theme || 'dark');
    panelsDirty = true;
  };

  const observer = new ResizeObserver(() => { panelsDirty = true; });
  for (const selector of panelSelectors) {
    document.querySelectorAll(selector).forEach((el) => observer.observe(el));
  }
  addEventListener('resize', onResize);
  document.addEventListener('themechange', onTheme);
  updatePanels();
  document.body.classList.add('glass-gpu');
  if (inline) document.body.classList.add('glass-inline');
  window.__glassGL = { get active() { return !destroyed; }, get panels() { return panels.length; }, inline, render, destroy };

  function destroy() {
    if (destroyed) return;
    destroyed = true;
    removeEventListener('resize', onResize);
    document.removeEventListener('themechange', onTheme);
    observer.disconnect();
    document.body.classList.remove('glass-gpu', 'glass-inline');
    gl.deleteTexture(texture);
    gl.deleteVertexArray(vao);
    gl.deleteProgram(program);
    if (inline) renderer.resetState();
  }

  const setAmbient = () => {};
  return { render, destroy, setAmbient, get active() { return !destroyed; }, get backend() { return 'webgl2'; } };
}
