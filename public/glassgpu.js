// WebGPU liquid-glass compositor modeled after iOS 26.5 Liquid Glass.
// Pipeline per frame:
// 1. Copy the three.js canvas into mip 0 of a 5-level mipmapped texture.
// 2. Build mips 1..4 via 13-tap Gaussian downsample passes (smooth variable blur).
// 3. Render floating UI panels with G2 continuous superellipse SDF, dual-layer
//    spherical refraction, subtle edge dispersion, distance-driven blur LOD,
//    YCC face & edge-bleed color matrices, soft optical shadows, and 45-deg
//    key/fill vibrant specular rims.

import { MAX_PANELS, DEFAULT_PANEL_SELECTORS, getThemeGlassParams, collectPanels } from './glass-math.js';

const MIP_LEVELS = 5;
const UNIFORM_FLOATS = 128; // 512 bytes (16-byte aligned)

const DOWNSAMPLE_WGSL = /* wgsl */ `
struct DownsampleUniform {
  dx: f32,
  dy: f32,
  pad0: f32,
  pad1: f32,
};

struct VsOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
};

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var srcSamp: sampler;
@group(0) @binding(2) var<uniform> params: DownsampleUniform;

@vertex
fn vs(@builtin(vertex_index) idx: u32) -> VsOut {
  let uv = vec2f(f32((idx << 1u) & 2u), f32(idx & 2u));
  var o: VsOut;
  o.pos = vec4f(uv * vec2f(2.0, -2.0) + vec2f(-1.0, 1.0), 0.0, 1.0);
  o.uv = uv;
  return o;
}

fn s(uv: vec2f, offset: vec2f) -> vec4f {
  return textureSampleLevel(srcTex, srcSamp, clamp(uv + offset * vec2f(params.dx, params.dy), vec2f(0.0), vec2f(1.0)), 0.0);
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
  let innerOffset = 1.9600850343704224;
  let outerOffset = 3.9206759929656982;

  let inner = s(in.uv, vec2f(0.0, -innerOffset))
    + s(in.uv, vec2f(-innerOffset, 0.0))
    + s(in.uv, vec2f(0.0, innerOffset))
    + s(in.uv, vec2f(innerOffset, 0.0));

  let diag = s(in.uv, vec2f(-innerOffset, innerOffset))
    + s(in.uv, vec2f(-innerOffset, -innerOffset))
    + s(in.uv, vec2f(innerOffset, -innerOffset))
    + s(in.uv, vec2f(innerOffset, innerOffset));

  let outer = s(in.uv, vec2f(-outerOffset, 0.0))
    + s(in.uv, vec2f(0.0, -outerOffset))
    + s(in.uv, vec2f(0.0, outerOffset))
    + s(in.uv, vec2f(outerOffset, 0.0));

  return textureSampleLevel(srcTex, srcSamp, in.uv, 0.0) * 0.1054970026
    + inner * 0.0901859999
    + diag * 0.0770979971
    + outer * 0.0563409999;
}
`;

const GLASS_WGSL = /* wgsl */ `
struct Uniforms {
  resMeta: vec4f,          // cssW, cssH, count, lightFlag
  panels: array<vec4f, 12>,// x, y, w, h (CSS px)
  radii: array<vec4f, 3>,  // 12 radii packed in 3 vec4f
  faceCM0: vec4f,
  faceCM1: vec4f,
  faceCM2: vec4f,
  bleedCM0: vec4f,
  bleedCM1: vec4f,
  bleedCM2: vec4f,
  rimCM0: vec4f,
  rimCM1: vec4f,
  rimCM2: vec4f,
  shadowParams: vec4f,     // offsetX, offsetY, opacity, radius
  rimParams: vec4f,        // widthPx, curvature, keyOpacity, fillOpacity
  optics: vec4f,           // bleedOpacity, bleedDarken, dispersion, blurScale
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var sceneTex: texture_2d<f32>;
@group(0) @binding(2) var sceneSamp: sampler;

struct VsOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vs(@builtin(vertex_index) idx: u32) -> VsOut {
  let uv = vec2f(f32((idx << 1u) & 2u), f32(idx & 2u));
  var o: VsOut;
  o.pos = vec4f(uv * vec2f(2.0, -2.0) + vec2f(-1.0, 1.0), 0.0, 1.0);
  o.uv = uv;
  return o;
}

fn radiusAt(idx: u32) -> f32 {
  let v = u.radii[idx >> 2u];
  let lane = idx & 3u;
  if (lane == 0u) { return v.x; }
  if (lane == 1u) { return v.y; }
  if (lane == 2u) { return v.z; }
  return v.w;
}

// G2 continuous superellipse (Apple squircle) signed distance in CSS pixels.
fn sdSquircle(p: vec2f, halfSize: vec2f, radius: f32) -> f32 {
  let r = min(radius, min(halfSize.x, halfSize.y));
  let K = 1.5286649465560913;
  let E = r * K;
  let ap = abs(p);
  let corner = halfSize - min(vec2f(E), halfSize);
  if (all(ap > corner) && r > 0.5) {
    let wts = clamp(vec2f(2.0 * 1.4457785) - (2.0 * halfSize * 1.4457785) / max(E, 1e-4), vec2f(0.0), vec2f(1.0));
    let qq = ap - halfSize + vec2f(E + (r - E) * max(wts.x, wts.y));
    let v = max(vec2f(0.0), (ap - halfSize + vec2f(E)) / max(E, 1e-4));
    let L = length(v);
    let hi = max(v.x, v.y);
    let t = clamp(select(0.0, min(v.x, v.y) / hi, hi > 0.0), 0.0, 1.0);
    let poly = (((-0.9260540008544922 * t + 3.1560099124908447) * t - 3.6412200927734375) * t + 1.268030047416687) * t + 0.2685309946537018;
    let cont = L + 1.0 - 1.0 / (1.0 - t * t * clamp(L, 0.0, 1.0) * poly);
    let circ = length(max(vec2f(0.0), v * K - vec2f(0.5286650061607361))) * 0.6541655659675598 + 0.3458344340324402;
    let s = select(-1.0, 1.0, v.y > v.x);
    let b = clamp(0.5 - s + s * t, 0.0, 1.0);
    var c = cont + (circ - cont) * wts.x;
    c = c + ((cont + (circ - cont) * wts.y) - c) * b;
    return min(max(qq.x, qq.y), 0.0) + E * (c - 1.0);
  }
  let q = ap - halfSize;
  return max(q.x, q.y);
}

fn panelSdf(p: vec2f, rect: vec4f, radius: f32) -> f32 {
  return sdSquircle(p - (rect.xy + rect.zw * 0.5), rect.zw * 0.5, radius);
}

// Spherical lens profile from iOS 26.5 Liquid Glass: steep at the rim, zero inside.
fn sphereProfile(amount: f32, invHeight: f32, d: f32) -> f32 {
  let t = clamp(-d * invHeight, 0.0, 1.0);
  return amount * (1.0 - clamp(sqrt((2.0 - t) * t), 0.0, 1.0));
}

// Soft Gaussian-like shadow CDF polynomial from iOS 26.5.
fn shadowCDF(shiftedDistance: f32, invRadius: f32) -> f32 {
  var x = clamp(0.25 * (invRadius * shiftedDistance) + 0.5, 0.0, 1.0);
  x = 4.0 * x - 2.0;
  let x2 = x * x;
  var poly = 0.002954483 * x2 - 0.034454346;
  poly = poly * x2 + 0.168212891;
  poly = poly * x2 - 0.560546875;
  return clamp(poly * x + 0.5, 0.0, 1.0);
}

fn radiusLOD(radius: f32) -> f32 {
  let logIn = select(1.0 + 0.5 * radius, radius, radius >= 2.0);
  return clamp(log2(max(1.0, logIn)), 0.0, 4.0);
}

fn applyCM(rgb: vec3f, r0: vec4f, r1: vec4f, r2: vec4f) -> vec3f {
  return vec3f(
    dot(rgb, r0.xyz) + r0.w,
    dot(rgb, r1.xyz) + r1.w,
    dot(rgb, r2.xyz) + r2.w
  );
}

fn keyFillBand(depth: f32, width: f32, aa: f32, curvature: f32) -> f32 {
  let t = clamp(depth / max(width, 1e-4), 0.0, 1.0);
  let hard = select(0.0, 1.0, t < 1.0);
  return mix(hard, 1.0 - t, curvature)
    * clamp(depth / aa + 0.5, 0.0, 1.0)
    * clamp((width - depth) / aa + 0.5, 0.0, 1.0);
}

fn sampleScene(uv: vec2f, lod: f32) -> vec3f {
  return textureSampleLevel(sceneTex, sceneSamp, clamp(uv, vec2f(0.001), vec2f(0.999)), lod).rgb;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
  let res = u.resMeta.xy;
  let count = u32(u.resMeta.z + 0.5);
  let p = in.uv * res;

  // Find closest panel (for both interior glass and exterior shadow)
  var chosen: i32 = -1;
  var chosenD = 1.0e9;
  var chosenRect = vec4f(0.0);
  var chosenRadius = 18.0;

  // Also accumulate soft exterior shadow from nearby panels
  var shadowAccum = 0.0;

  for (var i: u32 = 0u; i < 12u; i = i + 1u) {
    if (i >= count) { break; }
    let rect = u.panels[i];
    if (rect.z <= 0.0 || rect.w <= 0.0) { continue; }
    // Fast AABB early-out so pixels in the open 3D viewport skip squircle SDF math
    let half = rect.zw * 0.5;
    let qBox = abs(p - (rect.xy + half)) - half;
    let boxD = max(qBox.x, qBox.y);
    if (boxD > 60.0) { continue; }
    let radius = radiusAt(i);
    let d = panelSdf(p, rect, radius);
    if (d < chosenD) {
      chosenD = d;
      chosen = i32(i);
      chosenRect = rect;
      chosenRadius = radius;
    }
    let D = min(rect.z, rect.w);
    let sScale = clamp(D / 140.0, 0.45, 1.0);
    let sRad = u.shadowParams.w * sScale;
    let sOff = u.shadowParams.xy * sScale;
    if (d > -2.0 && d < sRad * 2.2) {
      let sd = panelSdf(p - sOff, rect, radius);
      let sFactor = shadowCDF(sd, 1.0 / max(sRad, 1.0)) * u.shadowParams.z * mix(0.65, 1.0, sScale);
      shadowAccum = max(shadowAccum, sFactor);
    }
  }

  if (chosen < 0) { discard; }

  let d = chosenD;
  let aa = max(fwidth(d), 0.75);
  let coverage = clamp(0.5 - d / aa, 0.0, 1.0);

  if (coverage <= 0.001 && shadowAccum <= 0.002) {
    discard;
  }

  // Exterior shadow premultiplied color (subtle cool-neutral tint)
  let shadowTint = select(vec3f(0.01, 0.02, 0.04), vec3f(0.05, 0.09, 0.15), u.resMeta.w > 0.5);
  let shadowMat = vec4f(shadowTint * shadowAccum, shadowAccum);

  if (coverage <= 0.001) {
    return shadowMat;
  }

  let rect = chosenRect;
  let D = min(rect.z, rect.w);
  let largeT = clamp((D - 44.0) / 116.0, 0.0, 1.0);

  // Analytic SDF gradient (outward normal)
  let eps = 0.75;
  let gx = panelSdf(p + vec2f(eps, 0.0), rect, chosenRadius) - panelSdf(p - vec2f(eps, 0.0), rect, chosenRadius);
  let gy = panelSdf(p + vec2f(0.0, eps), rect, chosenRadius) - panelSdf(p - vec2f(0.0, eps), rect, chosenRadius);
  let normal = normalize(vec2f(gx, gy) + vec2f(1e-5, 0.0));

  // Dual-layer spherical refraction (unified across themes)
  let innerHeight = clamp(D * 0.24, 10.0, 22.0);
  let innerAmount = -min(D * 0.60, 48.0);
  let outerHeight = clamp(D * 0.12, 4.0, 13.0);
  let outerAmount = min(D * 0.15, 13.0);

  let innerDisp = sphereProfile(innerAmount, 1.0 / innerHeight, d);
  let outerDisp = sphereProfile(outerAmount, 1.0 / outerHeight, d);

  // Distance-driven variable blur LOD: razor-sharp at the refracted rim, whisper-soft inside
  let baseBlur = mix(0.45, 1.35, largeT) * u.optics.w;
  let rimClear = clamp(-d / max(innerHeight * 0.95, 1.0), 0.0, 1.0);
  let innerLod = radiusLOD(baseBlur * smoothstep(0.0, 1.0, rimClear));
  let outerLod = radiusLOD(baseBlur * 0.18);

  // Subtle chromatic dispersion concentrated on the refracted rim
  let disp = u.optics.z;
  let innerVec = (normal * innerDisp) / res;
  let outerVec = (normal * outerDisp) / res;

  let cr = sampleScene(in.uv + innerVec * (1.0 - disp), innerLod).r;
  let cg = sampleScene(in.uv + innerVec, innerLod).g;
  let cb = sampleScene(in.uv + innerVec * (1.0 + disp), innerLod).b;
  var refracted = vec3f(cr, cg, cb);

  // Outer wall refraction mix right at the perimeter
  let outerSrc = sampleScene(in.uv + outerVec, outerLod);
  let outerMix = clamp((d + 1.5) / 1.5, 0.0, 1.0) * 0.30;
  refracted = mix(refracted, outerSrc, outerMix);

  // YCC Face color matrix (preserves luminosity & glass depth)
  var faceRGB = applyCM(refracted, u.faceCM0, u.faceCM1, u.faceCM2);

  // Edge Bleed: wide-blurred background color softly flowing into the glass edge
  let bleedOpacity = u.optics.x * mix(0.25, 1.0, largeT);
  if (bleedOpacity > 0.001) {
    let bleedHeight = clamp(D * 0.35, 12.0, 52.0);
    let bleedAmount = clamp(D * 0.35, 12.0, 48.0);
    let bleedDisp = sphereProfile(bleedAmount, 1.0 / bleedHeight, d);
    let bleedLod = radiusLOD(clamp(D * 0.22, 8.0, 28.0));
    let bleedSrc = sampleScene(in.uv + (normal * bleedDisp) / res, bleedLod);
    let bleedRGB = applyCM(bleedSrc, u.bleedCM0, u.bleedCM1, u.bleedCM2);
    let bleedRamp = 1.0 - clamp(-d / max(bleedHeight, 1.0), 0.0, 1.0);
    let luma = clamp(dot(faceRGB, vec3f(0.2125, 0.7154, 0.0721)), 0.0, 1.0);
    let darken = select(1.0 - luma, luma, u.optics.y > 0.5);
    let w = darken * darken * bleedRamp;
    faceRGB = mix(faceRGB, bleedRGB, clamp(w * w * bleedOpacity, 0.0, 0.65));
  }

  // Subtle SDR holding tone + Fresnel edge groove (keeps bright-mode glass crisp against pale backgrounds)
  let holdW = 1.0 - clamp((-d - 1.0) / 2.0, 0.0, 1.0);
  faceRGB = mix(faceRGB, faceRGB * select(0.975, 0.935, u.resMeta.w > 0.5), holdW * select(0.35, 0.50, u.resMeta.w > 0.5));

  // Directional 45-degree Key/Fill Specular Rim + Vibrancy Color Matrix (iOS 26.5)
  let depth = -d;
  let rimW = u.rimParams.x;
  let curv = u.rimParams.y;
  let keyDir = vec2f(-0.70710678, -0.70710678); // -45 deg (top-left)
  let fillDir = vec2f(0.70710678, 0.70710678);  // +135 deg (bottom-right)

  let band = keyFillBand(depth, rimW, aa, curv);
  let keyAlign = clamp(dot(keyDir, normal), 0.0, 1.0);
  let fillAlign = clamp(dot(fillDir, normal), 0.0, 1.0);

  // Subtle cool-slate contour line on grazing edges in light mode so white highlights pop
  let grazing = 1.0 - max(keyAlign, fillAlign);
  let contour = keyFillBand(depth, 1.05, aa, 0.5) * select(0.05, 0.10 + 0.08 * grazing, u.resMeta.w > 0.5);
  faceRGB *= 1.0 - contour;

  // Nonlinear specular rolloff (amount = 0.5 => shape factor 0.0; crisp falloff)
  let keyMask = band * pow(keyAlign, 1.25) * u.rimParams.z;
  let fillMask = band * pow(fillAlign, 1.35) * u.rimParams.w;
  // Subtle continuous ambient hairline around the full perimeter so edges never vanish
  let hairline = keyFillBand(depth, 1.0, aa, 0.85) * select(0.18, 0.22, u.resMeta.w > 0.5);
  let rimAlpha = clamp(keyMask + fillMask + hairline, 0.0, 1.0);

  if (rimAlpha > 0.001) {
    let vibrantRim = clamp(applyCM(faceRGB, u.rimCM0, u.rimCM1, u.rimCM2), vec3f(0.0), vec3f(1.2));
    faceRGB = mix(faceRGB, clamp(vibrantRim, vec3f(0.0), vec3f(1.0)), rimAlpha);
  }

  // Secondary soft inner bevel sheen along top-left curvature (restrained in light mode to avoid milky edge)
  let innerSheen = smoothstep(0.5, 2.5, depth) * (1.0 - smoothstep(3.0, 11.0, depth))
    * pow(keyAlign, 1.6) * select(0.07, 0.05, u.resMeta.w > 0.5);
  faceRGB = clamp(faceRGB + vec3f(innerSheen), vec3f(0.0), vec3f(1.0));

  let faceMat = vec4f(faceRGB, 1.0);
  return mix(shadowMat, faceMat, coverage);
}
`;

export async function createGlassGPU({
  canvas,
  sceneCanvas,
  ambientCanvas = null,
  panelSelectors = DEFAULT_PANEL_SELECTORS,
  forceCss = false,
  onLost = null,
} = {}) {
  if (forceCss || !canvas || !sceneCanvas || !navigator.gpu) {
    console.warn('[glass-gpu] early', { forceCss, canvas: !!canvas, scene: !!sceneCanvas, gpu: !!navigator.gpu });
    return null;
  }

  let adapter;
  let device;
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (adapter) break;
    } catch (error) {
      console.warn('[glass-gpu] adapter attempt failed', error);
    }
    if (attempt === 0) await new Promise((resolve) => setTimeout(resolve, 50));
  }
  if (!adapter) { console.warn('[glass-gpu] no adapter'); return null; }

  try {
    device = await adapter.requestDevice();
  } catch (error) {
    console.warn('[glass-gpu] unavailable', error);
    return null;
  }
  if (!device) { console.warn('[glass-gpu] no device'); return null; }

  const context = canvas.getContext('webgpu');
  if (!context) { console.warn('[glass-gpu] no context'); return null; }
  const format = navigator.gpu.getPreferredCanvasFormat();

  const downsampleModule = device.createShaderModule({ code: DOWNSAMPLE_WGSL });
  const glassModule = device.createShaderModule({ code: GLASS_WGSL });
  const [dsInfo, glInfo] = await Promise.all([
    downsampleModule.getCompilationInfo(),
    glassModule.getCompilationInfo(),
  ]);
  const compileErrors = [...dsInfo.messages, ...glInfo.messages].filter((m) => m.type === 'error');
  if (compileErrors.length) {
    console.error('[glass-gpu] shader compile failed', compileErrors.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('\n'));
    return null;
  }

  device.addEventListener('uncapturederror', (event) => {
    console.error('[glass-gpu] uncaptured', event.error?.message || event.error);
  });

  let downsamplePipeline;
  let glassPipeline;
  try {
    [downsamplePipeline, glassPipeline] = await Promise.all([
      device.createRenderPipelineAsync({
        layout: 'auto',
        vertex: { module: downsampleModule, entryPoint: 'vs' },
        fragment: { module: downsampleModule, entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }] },
        primitive: { topology: 'triangle-list' },
      }),
      device.createRenderPipelineAsync({
        layout: 'auto',
        vertex: { module: glassModule, entryPoint: 'vs' },
        fragment: { module: glassModule, entryPoint: 'fs', targets: [{ format }] },
        primitive: { topology: 'triangle-list' },
      }),
    ]);
  } catch (error) {
    console.error('[glass-gpu] pipeline failed', error);
    return null;
  }

  const uniformBuffer = device.createBuffer({
    size: UNIFORM_FLOATS * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const uniformData = new Float32Array(UNIFORM_FLOATS);

  const mipUniformBuffers = Array.from({ length: MIP_LEVELS - 1 }, () =>
    device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
  );

  const linearSampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  });
  const trilinearSampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  });

  let sceneTexture = null;
  let sceneFullView = null;
  let mipViews = [];
  let downsampleBindGroups = [];
  let glassBindGroup = null;
  let sceneWidth = 0;
  let sceneHeight = 0;
  let destroyed = false;
  let panels = [];
  let panelsDirty = true;
  let themeParams = getThemeGlassParams(document.documentElement.dataset.theme || 'dark');

  const resize = () => {
    // Match sceneCanvas 1x CSS resolution to avoid wasteful 4x oversampling on Retina displays.
    const width = Math.max(1, Math.round(innerWidth || 1));
    const height = Math.max(1, Math.round(innerHeight || 1));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    context.configure({ device, format, alphaMode: 'premultiplied' });
  };

  const ensureSceneTexture = () => {
    const w = sceneCanvas.width | 0;
    const h = sceneCanvas.height | 0;
    if (!w || !h) return false;
    if (!sceneTexture || w !== sceneWidth || h !== sceneHeight) {
      sceneTexture?.destroy();
      sceneWidth = w;
      sceneHeight = h;
      const maxDim = Math.max(w, h);
      const levels = Math.min(MIP_LEVELS, Math.floor(Math.log2(maxDim)) + 1);
      sceneTexture = device.createTexture({
        size: [w, h],
        mipLevelCount: levels,
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      sceneFullView = sceneTexture.createView({ baseMipLevel: 0, mipLevelCount: levels });
      mipViews = Array.from({ length: levels }, (_, level) =>
        sceneTexture.createView({ baseMipLevel: level, mipLevelCount: 1 }),
      );
      downsampleBindGroups = [];
      let mw = w;
      let mh = h;
      for (let level = 0; level < levels - 1; level++) {
        device.queue.writeBuffer(mipUniformBuffers[level], 0, new Float32Array([1 / Math.max(1, mw), 1 / Math.max(1, mh), 0, 0]));
        downsampleBindGroups.push(device.createBindGroup({
          layout: downsamplePipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: mipViews[level] },
            { binding: 1, resource: linearSampler },
            { binding: 2, resource: { buffer: mipUniformBuffers[level] } },
          ],
        }));
        mw = Math.max(1, mw >> 1);
        mh = Math.max(1, mh >> 1);
      }
      glassBindGroup = device.createBindGroup({
        layout: glassPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: sceneFullView },
          { binding: 2, resource: trilinearSampler },
        ],
      });
    }
    return true;
  };

  const updatePanels = () => {
    panels = collectPanels(panelSelectors, MAX_PANELS);
    panelsDirty = false;
    return panels;
  };

  const writeUniforms = () => {
    const cssW = innerWidth || 1;
    const cssH = innerHeight || 1;
    uniformData.fill(0);
    // resMeta (0..3)
    uniformData[0] = cssW;
    uniformData[1] = cssH;
    uniformData[2] = panels.length;
    uniformData[3] = themeParams.light;
    // panels (4..51)
    for (let i = 0; i < panels.length; i++) {
      const p = panels[i];
      const base = 4 + i * 4;
      uniformData[base] = p.x;
      uniformData[base + 1] = p.y;
      uniformData[base + 2] = p.w;
      uniformData[base + 3] = p.h;
      // radii (52..63)
      uniformData[52 + i] = p.radius;
    }
    // Color matrices & params (64..111)
    uniformData.set(themeParams.faceCM0, 64);
    uniformData.set(themeParams.faceCM1, 68);
    uniformData.set(themeParams.faceCM2, 72);
    uniformData.set(themeParams.bleedCM0, 76);
    uniformData.set(themeParams.bleedCM1, 80);
    uniformData.set(themeParams.bleedCM2, 84);
    uniformData.set(themeParams.rimCM0, 88);
    uniformData.set(themeParams.rimCM1, 92);
    uniformData.set(themeParams.rimCM2, 96);
    uniformData.set(themeParams.shadow, 100);
    uniformData.set(themeParams.rimParams, 104);
    uniformData.set(themeParams.optics, 108);

    device.queue.writeBuffer(uniformBuffer, 0, uniformData);
  };

  const render = () => {
    if (destroyed) return;
    if (!ensureSceneTexture()) return;
    try {
      device.queue.copyExternalImageToTexture(
        { source: sceneCanvas },
        { texture: sceneTexture, mipLevel: 0 },
        [sceneWidth, sceneHeight],
      );
    } catch (error) {
      console.warn('[glass-gpu] scene copy failed', error);
      return;
    }

    if (panelsDirty) {
      updatePanels();
      writeUniforms();
    }

    const encoder = device.createCommandEncoder();

    // Generate Gaussian mip pyramid (levels 1..N-1)
    for (let level = 0; level < downsampleBindGroups.length; level++) {
      const dsPass = encoder.beginRenderPass({
        colorAttachments: [{
          view: mipViews[level + 1],
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        }],
      });
      dsPass.setPipeline(downsamplePipeline);
      dsPass.setBindGroup(0, downsampleBindGroups[level]);
      dsPass.draw(3);
      dsPass.end();
    }

    // Main liquid-glass composite pass
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
    pass.setPipeline(glassPipeline);
    pass.setBindGroup(0, glassBindGroup);
    pass.draw(3);
    pass.end();

    device.queue.submit([encoder.finish()]);
  };

  const onResize = () => {
    resize();
    panelsDirty = true;
  };
  const onTheme = () => {
    themeParams = getThemeGlassParams(document.documentElement.dataset.theme || 'dark');
    panelsDirty = true;
  };

  resize();
  updatePanels();
  writeUniforms();
  addEventListener('resize', onResize);
  document.addEventListener('themechange', onTheme);
  const observer = new ResizeObserver(() => { panelsDirty = true; });
  for (const selector of panelSelectors) {
    document.querySelectorAll(selector).forEach((el) => observer.observe(el));
  }
  document.body.classList.add('glass-gpu');

  device.lost.then((info) => {
    if (destroyed) return;
    console.warn('[glass-gpu] device lost', info.reason, info.message || '');
    destroy();
    onLost?.();
  });

  function destroy() {
    if (destroyed) return;
    destroyed = true;
    removeEventListener('resize', onResize);
    document.removeEventListener('themechange', onTheme);
    observer.disconnect();
    document.body.classList.remove('glass-gpu');
    sceneTexture?.destroy();
    uniformBuffer.destroy();
    mipUniformBuffers.forEach((b) => b.destroy());
    device.destroy();
  }

  const setAmbient = () => {};

  window.__glassGPU = { get active() { return !destroyed; }, get panels() { return panels.length; }, render, destroy };
  return { render, destroy, setAmbient, get active() { return !destroyed; }, get backend() { return 'webgpu'; } };
}
