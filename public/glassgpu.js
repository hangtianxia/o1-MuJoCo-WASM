// WebGPU liquid-glass compositor for the preview's floating panels.
// The three.js canvas is copied into a texture each frame; a fragment shader
// renders the panel glass with refraction, RGB dispersion, bevel glow and the
// page's ambient mesh. DOM text and controls stay above it, so the UI remains
// crisp and accessible.

const MAX_PANELS = 4;
const UNIFORM_FLOATS = 64;

const WGSL = /* wgsl */ `
struct Uniforms {
  resTime: vec4f,
  panels: array<vec4f, 4>,
  radii: vec4f,
  faceTop: vec4f,
  faceBottom: vec4f,
  rim: vec4f,
  bevel: vec4f,
  shade: vec4f,
  amb0: vec4f,
  amb1: vec4f,
  amb2: vec4f,
  tint: vec4f,
  params: vec4f,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var sceneTex: texture_2d<f32>;
@group(0) @binding(2) var sceneSampler: sampler;
@group(0) @binding(3) var ambientTex: texture_2d<f32>;

struct VsOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@vertex
fn vs(@builtin(vertex_index) index: u32) -> VsOut {
  var positions = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f(3.0, -1.0),
    vec2f(-1.0, 3.0)
  );
  let p = positions[index];
  var out: VsOut;
  out.position = vec4f(p, 0.0, 1.0);
  out.uv = vec2f(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
  return out;
}

fn sdRoundBox(p: vec2f, halfSize: vec2f, radius: f32) -> f32 {
  let q = abs(p) - halfSize + vec2f(radius);
  return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0) - radius;
}

fn panelSdf(p: vec2f, rect: vec4f, radius: f32) -> f32 {
  return sdRoundBox(p - (rect.xy + rect.zw * 0.5), rect.zw * 0.5, radius);
}

fn radiusAt(index: u32) -> f32 {
  if (index == 0u) { return u.radii.x; }
  if (index == 1u) { return u.radii.y; }
  if (index == 2u) { return u.radii.z; }
  return u.radii.w;
}

fn sceneAt(uv: vec2f, offset: vec2f) -> vec3f {
  let texel = 1.0 / vec2f(textureDimensions(sceneTex));
  let base = clamp(uv + offset, vec2f(0.0), vec2f(1.0));
  var c = textureSample(sceneTex, sceneSampler, base).rgb * 0.44;
  c += textureSample(sceneTex, sceneSampler, clamp(base + vec2f(texel.x * 1.4, 0.0), vec2f(0.0), vec2f(1.0))).rgb * 0.14;
  c += textureSample(sceneTex, sceneSampler, clamp(base - vec2f(texel.x * 1.4, 0.0), vec2f(0.0), vec2f(1.0))).rgb * 0.14;
  c += textureSample(sceneTex, sceneSampler, clamp(base + vec2f(0.0, texel.y * 1.4), vec2f(0.0), vec2f(1.0))).rgb * 0.14;
  c += textureSample(sceneTex, sceneSampler, clamp(base - vec2f(0.0, texel.y * 1.4), vec2f(0.0), vec2f(1.0))).rgb * 0.14;
  return c;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
  let res = u.resTime.xy;
  let p = in.uv * res;

  var chosen: i32 = -1;
  var chosenD = 1.0e9;
  var chosenRect = vec4f(0.0);
  var chosenRadius = 22.0;
  for (var i: u32 = 0u; i < 4u; i = i + 1u) {
    if (f32(i) >= u.resTime.w) { break; }
    let rect = u.panels[i];
    if (rect.z <= 0.0 || rect.w <= 0.0) { continue; }
    let radius = radiusAt(i);
    let d = panelSdf(p, rect, radius);
    if (d < 0.0 && d < chosenD) {
      chosenD = d;
      chosen = i32(i);
      chosenRect = rect;
      chosenRadius = radius;
    }
  }
  if (chosen < 0) { discard; }

  let d = chosenD;
  let rect = chosenRect;
  let w = max(rect.z, 1.0);
  let h = max(rect.w, 1.0);
  let band = min(min(w, h) * 0.30, 104.0);
  let t = clamp(-d / max(band, 1.0), 0.0, 1.0);
  let lens = pow(1.0 - t, 2.0);
  let amount = min(min(w, h) * 0.32, 112.0);

  let e = 1.0;
  let gx = panelSdf(p + vec2f(e, 0.0), rect, chosenRadius) - panelSdf(p - vec2f(e, 0.0), rect, chosenRadius);
  let gy = panelSdf(p + vec2f(0.0, e), rect, chosenRadius) - panelSdf(p - vec2f(0.0, e), rect, chosenRadius);
  let normal = normalize(vec2f(gx, gy) + vec2f(0.0001));

  let pulse = 1.0 + 0.18 * sin(u.resTime.z * 0.00105 + p.y * 0.013);
  var offset = normal * (amount * lens * pulse) / res;
  offset += vec2f(-normal.y, normal.x) * (sin(u.resTime.z * 0.00082 + t * 8.0) * 5.0) / res;
  let centerUv = (rect.xy + rect.zw * 0.5) / res;
  let sampleUv = centerUv + (in.uv - centerUv) * 0.90;
  let cr = sceneAt(sampleUv, offset * 1.00);
  let cg = sceneAt(sampleUv, offset * 1.12);
  let cb = sceneAt(sampleUv, offset * 1.24);

  var amb = vec4f(0.0);

  var color = vec3f(cr.r, cg.g, cb.b);
  color = color * (1.0 - amb.a) + amb.rgb;

  let local = p - rect.xy;
  let faceT = clamp(local.x / w * 0.28 + local.y / h * 0.72, 0.0, 1.0);
  let face = mix(u.faceTop, u.faceBottom, faceT);
  var fa = face.a * 0.85 * smoothstep(0.0, 0.55, t);
  color = color * (1.0 - fa) + face.rgb * fa;

  let luma = dot(color, vec3f(0.2126, 0.7152, 0.0722));
  color = mix(vec3f(luma), color, 1.16);

  let edge = 1.0 - smoothstep(0.0, 2.0, -d);
  let glow = smoothstep(0.0, 5.0, -d) * (1.0 - smoothstep(12.0, 26.0, -d));
  color += vec3f(1.0) * (1.0 - smoothstep(0.0, 1.2, -d)) * 0.12;
  color *= 1.0 - u.shade.a * smoothstep(0.0, 18.0, -d) * 0.10;

  return vec4f(clamp(color, vec3f(0.0), vec3f(1.0)), 1.0);
}
`;

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

export async function createGlassGPU({ canvas, sceneCanvas, ambientCanvas = null, panelSelectors = ['#controls', '#telemetry'], forceCss = false, onLost = null } = {}) {
  if (forceCss || !canvas || !sceneCanvas || !navigator.gpu) { console.warn('[glass-gpu] early', {forceCss, canvas:!!canvas, scene:!!sceneCanvas, gpu:!!navigator.gpu}); return null; }
  let adapter;
  let device;
  for (let attempt = 0; attempt < 8; attempt++) {
    try {
      adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (adapter) break;
    } catch (error) {
      console.warn('[glass-gpu] adapter attempt failed', error);
    }
    await new Promise((resolve) => setTimeout(resolve, 300));
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
  const module = device.createShaderModule({ code: WGSL });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  if (errors.length) {
    console.error('[glass-gpu] shader compile failed', errors.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('\n'));
    return null;
  }
  device.addEventListener('uncapturederror', (event) => {
    console.error('[glass-gpu] uncaptured', event.error?.message || event.error);
  });
  device.pushErrorScope('validation');
  let pipeline;
  try {
    pipeline = await device.createRenderPipelineAsync({
      layout: 'auto',
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
    });
  } catch (error) {
    console.error('[glass-gpu] pipeline failed', error);
    return null;
  }
  const pipelineError = await device.popErrorScope().catch(() => null);
  if (pipelineError) console.error('[glass-gpu] pipeline validation', pipelineError.message);

  const uniformBuffer = device.createBuffer({ size: UNIFORM_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
  let sceneTexture = null;
  let sceneView = null;
  let ambientTexture = null;
  let ambientView = null;
  let bindGroup = null;
  let sceneWidth = 0;
  let sceneHeight = 0;
  let ambientWidth = 0;
  let ambientHeight = 0;
  let ambientSource = ambientCanvas;
  let ambientDirty = true;
  let destroyed = false;
  let lastDraw = 0;
  let panels = [];

  const resize = () => {
    const width = Math.max(1, Math.round((innerWidth || 1) * 0.6));
    const height = Math.max(1, Math.round((innerHeight || 1) * 0.6));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    context.configure({ device, format, alphaMode: 'premultiplied' });
  };

  const makeBindGroup = () => {
    if (!sceneView) return;
    bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: sceneView },
        { binding: 2, resource: sampler },
      ],
    });
  };

  const ensureAmbientTexture = () => {
    if (!ambientSource) {
      if (!ambientTexture) {
        ambientTexture = device.createTexture({
          size: [1, 1],
          format: 'rgba8unorm',
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
        });
        device.queue.writeTexture({ texture: ambientTexture }, new Uint8Array([0, 0, 0, 0]), { bytesPerRow: 4 }, { width: 1, height: 1 });
        ambientView = ambientTexture.createView();
        ambientWidth = 1;
        ambientHeight = 1;
        bindGroup = null;
      }
      return;
    }
    const w = ambientSource.width | 0;
    const h = ambientSource.height | 0;
    if (!w || !h) return;
    if (!ambientTexture || w !== ambientWidth || h !== ambientHeight) {
      ambientTexture?.destroy();
      ambientWidth = w;
      ambientHeight = h;
      ambientTexture = device.createTexture({
        size: [w, h],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      ambientView = ambientTexture.createView();
      ambientDirty = true;
      bindGroup = null;
    }
    if (ambientDirty) {
      try {
        device.queue.copyExternalImageToTexture({ source: ambientSource }, { texture: ambientTexture }, [w, h]);
        ambientDirty = false;
      } catch (error) {
        console.warn('[glass-gpu] ambient copy failed', error);
      }
    }
  };

  const ensureSceneTexture = () => {
    const w = sceneCanvas.width | 0;
    const h = sceneCanvas.height | 0;
    if (!w || !h) return false;
    if (!sceneTexture || w !== sceneWidth || h !== sceneHeight) {
      sceneTexture?.destroy();
      sceneWidth = w;
      sceneHeight = h;
      sceneTexture = device.createTexture({
        size: [w, h],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      sceneView = sceneTexture.createView();
      bindGroup = null;
    }
    if (!bindGroup) makeBindGroup();
    return true;
  };

  const updatePanels = () => {
    panels = panelSelectors.map((selector) => document.querySelector(selector)).filter(Boolean).map((element) => {
      const rect = element.getBoundingClientRect();
      const style = getComputedStyle(element);
      const radius = Number.parseFloat(style.borderTopLeftRadius) || 22;
      return { element, x: rect.left, y: rect.top, w: rect.width, h: rect.height, radius };
    }).filter((p) => p.w > 0 && p.h > 0 && p.element.offsetParent !== null);
    return panels;
  };

  const writeUniforms = (now) => {
    const data = new Float32Array(UNIFORM_FLOATS);
    const style = getComputedStyle(document.documentElement);
    const cssW = innerWidth || 1;
    const cssH = innerHeight || 1;
    data[0] = cssW;
    data[1] = cssH;
    data[2] = now;
    data[3] = panels.length;
    panels.forEach((panel, index) => {
      const base = 4 + index * 4;
      data[base] = panel.x;
      data[base + 1] = panel.y;
      data[base + 2] = panel.w;
      data[base + 3] = panel.h;
      data[20 + index] = panel.radius;
    });
    for (let i = panels.length; i < MAX_PANELS; i++) {
      const base = 4 + i * 4;
      data[base] = 0;
      data[base + 1] = 0;
      data[base + 2] = 0;
      data[base + 3] = 0;
      data[20 + i] = 0;
    }
    const faceTop = cssColor(style, '--glass-top');
    const faceBottom = cssColor(style, '--glass-bottom');
    const rim = cssColor(style, '--glass-topglow');
    const bevel = cssColor(style, '--glass-bevel');
    const shade = cssColor(style, '--glass-shade');
    const amb0 = cssColor(style, '--ambient-peach');
    const amb1 = cssColor(style, '--ambient-sky');
    const amb2 = cssColor(style, '--ambient-aqua');
    data.set(faceTop, 24);
    data.set(faceBottom, 28);
    data.set([rim[0], rim[1], rim[2], Math.min(1, rim[3] * 1.6)], 32);
    data.set([bevel[0], bevel[1], bevel[2], Math.min(1, bevel[3] * 1.2)], 36);
    data.set(shade, 40);
    data.set(amb0, 44);
    data.set(amb1, 48);
    data.set(amb2, 52);
    data.set([1, 1, 1, 0], 56);
    data.set([0, 0, 0, 0], 60);
    device.queue.writeBuffer(uniformBuffer, 0, data);
  };

  const draw = (now) => {
    if (destroyed) return;
    ensureAmbientTexture();
    if (!ensureSceneTexture()) return;
    try {
      device.queue.copyExternalImageToTexture({ source: sceneCanvas }, { texture: sceneTexture }, [sceneWidth, sceneHeight]);
    } catch (error) {
      console.warn('[glass-gpu] scene copy failed', error);
      return;
    }
    device.pushErrorScope('validation');
    updatePanels();
    writeUniforms(now);
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
    device.queue.submit([encoder.finish()]);
    device.popErrorScope().then((error) => {
      if (error) console.error('[glass-gpu] draw validation', error.message);
    }).catch(() => {});
  };

  const render = (now) => {
    if (destroyed) return;
    if (now - lastDraw < 1000 / 30) return;
    lastDraw = now;
    draw(now);
  };

  const onResize = () => {
    resize();
    updatePanels();
  };
  const onTheme = () => {
    updatePanels();
  };

  resize();
  updatePanels();
  addEventListener('resize', onResize);
  document.addEventListener('themechange', onTheme);
  const observer = new ResizeObserver(() => updatePanels());
  document.querySelectorAll(panelSelectors.join(',')).forEach((element) => observer.observe(element));
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
    ambientTexture?.destroy();
    uniformBuffer.destroy();
    device.destroy();
  }

  const setAmbient = (canvasSource) => { ambientSource = canvasSource; ambientDirty = true; bindGroup = null; };

  window.__glassGPU = { get active() { return !destroyed; }, get panels() { return panels.length; }, render, destroy };
  return { render, destroy, setAmbient, get active() { return !destroyed; }, get backend() { return 'webgpu'; } };
}
