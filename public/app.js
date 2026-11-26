import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import loadMujoco from '/node_modules/@mujoco/mujoco/mujoco.js';
import * as ort from '/node_modules/onnxruntime-web/dist/ort.wasm.bundle.min.mjs';
import { DIRECTIONS, buildTerrain, buildUniformTerrain, buildGapTerrain, buildPlatformTerrain, buildGratingTerrain, buildSlopeTerrain, buildHillsideTerrain, terrainGeomXml, terrainTopAt } from './terrain.js';
import { createGlassGPU } from './glassgpu.js';
import { createGlassGL } from './glassgl.js';
import { createAmbientLayer } from './ambient.js';

const IS_SAFARI = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);
const urlThreads = Number(new URLSearchParams(location.search).get('threads'));
// Safari/WebKit pthread Atomics.wait overhead makes 4-thread WASM ~8-10x slower
// than 1-thread SIMD on this compact GRU policy and slows initial load.
const CORES = Number.isFinite(urlThreads) && urlThreads >= 1
  ? Math.min(8, Math.floor(urlThreads))
  : IS_SAFARI ? 1 : Math.max(1, Math.min(4, navigator.hardwareConcurrency || 1));
const urlProxy = new URLSearchParams(location.search).get('proxy');
ort.env.wasm.numThreads = CORES;
// Offload ONNX inference to a dedicated worker on Safari so main-thread rAF & camera drag never stall.
ort.env.wasm.proxy = urlProxy !== null ? urlProxy === '1' : IS_SAFARI;
ort.env.wasm.wasmPaths = '/node_modules/onnxruntime-web/dist/';
ort.env.logLevel = 'error';

const TERMS = ['base_ang_vel', 'projected_gravity', 'velocity_commands', 'joint_pos_rel', 'joint_vel_rel', 'last_action'];
const TERM_WIDTH = { base_ang_vel: 3, projected_gravity: 3, velocity_commands: 3, joint_pos_rel: 29, joint_vel_rel: 29, last_action: 29 };
const HISTORY = 8;
const DEPTH_W = 64;
const DEPTH_H = 36;
const DEPTH_CROP_LEFT = 14;
const DEPTH_CROP_RIGHT = 14;
const DEPTH_OUT = 36;
const DEPTH_MAX = 2.5;
const CAM_FOV_X_DEG = 89.51;
const CAM_FOV_Y_DEG = 58.29;
const DEPTH_FAR = 20.0;
const SUPPORTED_OBS = 2064;

const state = {
  sceneClear: 0x0b0e13,
  hemiLight: null,
  keyLight: null,
  ready: false,
  config: null,
  template: null,
  assetBase: '/',
  exportId: null,
  ckptChoice: null,
  exportJob: null,
  mujoco: null,
  model: null,
  data: null,
  session: null,
  bodyNodes: [],
  robotRoot: null,
  terrainGroup: null,
  terrainMaterials: null,
  terrain: null,
  direction: 'up',
  level: 7,
  terrainMode: 'uniform',
  lastAppliedTerrainMode: null,
  uniform: { riser: 0.12, tread: 0.35, steps: 12, width: 4.0 },
  sim2real: {
    friction: 0.0, damping: 0.0, actuatorDelayMs: 0, encoderZero: 0.0,
    imuPitchBiasDeg: 0.0, gyroBias: 0.0, jointVelFilterMs: 0.0, backlash: 0.0,
    shoe: false, zeroOffsets: new Float64Array(29), velFiltered: new Float64Array(29),
    targetQueue: [], lastApplied: null,
  },
  gap: { width: 0.30, depth: 0.45, center: 2.0 },
  platform: { height: 0.20, start: 1.5 },
  grating: { riser: 0.20, tread: 0.25, steps: 12, width: 1.2, bar: 0.03, gap: 0.015 },
  slope: { angle: 12, run: 3.1, width: 4.0 },
  lockWaist: false,
  depthDisplay: 'real',
  running: true,
  speed: 1.0,
  command: [0.0, 0.0, 0.0],
  commandScale: { vx: 0.45, vy: 0.0, wz: 0.8 },
  keys: new Set(),
  keyboardEnabled: true,
  substeps: 4,
  controlDt: 0.02,
  steps: 0,
  simTime: 0,
  hidden: null,
  histories: {},
  lastAction: null,
  depth: new Float32Array(DEPTH_OUT * DEPTH_OUT),
  depthVisRange: null,
  lastControlMs: 0,
  lastInferenceMs: 0,
  follow: true,
  startPos: null,
  minPelvisZ: Infinity,
  maxPelvisZ: -Infinity,
  minClearance: Infinity,
  recentSpeeds: [],
  checking: false,
  depthPending: false,
  depthGen: 0,
  renderLimited: false,
  renderIntervalMs: 0,
  renderNoteBase: '',
  lowRateTicks: 0,
  highRateTicks: 0,
  perf: { renderMs: 0, depthMs: 0, depthLagMs: 0, inferMs: 0, physicsMs: 0, stepMs: 0, controlHz: 0, simRate: 0, fps: 0, cores: 0, gl: '' },
};

let renderer, scene, camera, controls, depthCamera, depthTarget, depthMaterial, depthPixels;
let cameraTarget, cameraPixels, lastCameraViewAt = 0;

function asset(p) { return `${state.assetBase}${p}`; }

function groundHeight(x, y = 0) {
  if (state.terrain?.topAt) return state.terrain.topAt(x, y);
  return terrainTopAt(state.terrain.boxes, x, y);
}
let skeletonGroup, gridHelper;
let lastFrame = performance.now();
let accumulator = 0;
let ticking = false;
let lastHud = 0;
let lastRenderAt = 0;

// ---------------------------------------------------------------------------
// math helpers (MuJoCo quaternions are w, x, y, z)
// ---------------------------------------------------------------------------
function quatMul(a, b) {
  const [w1, x1, y1, z1] = a;
  const [w2, x2, y2, z2] = b;
  return [
    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
  ];
}

function quatConj(q) { return [q[0], -q[1], -q[2], -q[3]]; }

function quatRotateInverse(q, v) {
  const r = quatMul(quatMul(quatConj(q), [0, v[0], v[1], v[2]]), q);
  return [r[1], r[2], r[3]];
}

function quatToMatrix(q) {
  const [w, x, y, z] = q;
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
  ];
}

function yawFromQuat(q) {
  return Math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] * q[2] + q[3] * q[3]));
}

class HistoryBuffer {
  constructor(width) {
    this.width = width;
    this.frames = Array.from({ length: HISTORY }, () => new Float64Array(width));
  }
  reset(values) {
    for (const f of this.frames) f.set(values);
  }
  push(values) {
    const first = this.frames.shift();
    first.set(values);
    this.frames.push(first);
  }
  write(out, offset) {
    for (const f of this.frames) {
      for (let i = 0; i < this.width; i++) out[offset++] = f[i];
    }
    return offset;
  }
}

// ---------------------------------------------------------------------------
// MuJoCo model / scene
// ---------------------------------------------------------------------------
function parseNumbers(text, fallback) {
  if (!text) return fallback;
  return text.trim().split(/\s+/).map(Number);
}

function collectVisualGeoms(templateXml) {
  const doc = new DOMParser().parseFromString(templateXml, 'text/xml');
  const out = [];
  for (const g of doc.querySelectorAll('geom[group="2"]')) {
    const mesh = g.getAttribute('mesh');
    const body = g.parentElement?.getAttribute('name');
    if (!mesh || !body) continue;
    out.push({
      body,
      mesh,
      name: g.getAttribute('name') || '',
      pos: parseNumbers(g.getAttribute('pos'), [0, 0, 0]),
      quat: parseNumbers(g.getAttribute('quat'), [1, 0, 0, 0]),
      scale: parseNumbers(g.getAttribute('scale'), [1, 1, 1]),
    });
  }
  return out;
}

function buildModelXml(templateXml, terrain) {
  const doc = new DOMParser().parseFromString(templateXml, 'text/xml');
  // Visual mesh geoms are rendered by three.js from the same STL files. MuJoCo
  // only needs the collision primitives, so drop mesh assets to keep loading
  // self-contained in the browser.
  for (const node of doc.querySelectorAll('asset')) node.remove();
  for (const node of doc.querySelectorAll('geom[group="2"]')) node.remove();
  doc.querySelector('compiler')?.removeAttribute('meshdir');

  // Optional shoe: flat rubber sole, 9 mm lower than the bare-foot capsules,
  // with torsional friction (condim=4) like a real flat sole.
  if (state.sim2real?.shoe) {
    for (const body of doc.querySelectorAll('body')) {
      const name = body.getAttribute('name') || '';
      if (!name.endsWith('ankle_roll_link')) continue;
      const sole = doc.createElement('geom');
      sole.setAttribute('name', `${name}_shoe_sole`);
      sole.setAttribute('type', 'box');
      sole.setAttribute('size', '0.095 0.045 0.0045');
      sole.setAttribute('pos', '0.065 0 -0.0385');
      sole.setAttribute('condim', '4');
      sole.setAttribute('friction', '0.8 0.02 0.0001');
      sole.setAttribute('solref', '0.01 1');
      sole.setAttribute('contype', '2');
      sole.setAttribute('conaffinity', '1');
      sole.setAttribute('group', '3');
      sole.setAttribute('rgba', '.1 .1 .1 .35');
      body.appendChild(sole);
    }
  }

  const world = doc.querySelector('worldbody');
  if (!world) throw new Error('robot template has no worldbody');
  if (terrain.hfield) {
    const { nrow, ncol, data, size, center } = terrain.hfield;
    if (!state.mujoco.FS?.writeFile) throw new Error('Hillside requires MuJoCo FS.writeFile support');
    if (nrow < 2 || ncol < 2 || data.length !== nrow * ncol || !size.every((value) => Number.isFinite(value) && value > 0)) {
      throw new Error('Invalid hillside heightfield dimensions or size');
    }
    const buffer = new ArrayBuffer(8 + data.length * 4);
    const view = new DataView(buffer);
    view.setInt32(0, nrow, true);
    view.setInt32(4, ncol, true);
    for (let index = 0; index < data.length; index++) view.setFloat32(8 + index * 4, data[index], true);
    state.mujoco.FS.writeFile('/paper_hillside.hfield', new Uint8Array(buffer));
    const assets = doc.createElement('asset');
    const hfield = doc.createElement('hfield');
    hfield.setAttribute('name', 'paper_hillside');
    hfield.setAttribute('file', '/paper_hillside.hfield');
    hfield.setAttribute('size', size.join(' '));
    assets.appendChild(hfield);
    world.parentNode.insertBefore(assets, world);
    const geom = doc.createElement('geom');
    for (const [name, value] of Object.entries({ name: 'terrain_hfield', type: 'hfield', hfield: 'paper_hillside', pos: center.join(' '), rgba: '.24 .27 .31 1', contype: '1', conaffinity: '3' })) {
      geom.setAttribute(name, value);
    }
    world.appendChild(geom);
  }
  const fragment = new DOMParser().parseFromString(
    `<mujoco>${terrainGeomXml(terrain.boxes)}</mujoco>`, 'text/xml',
  );
  for (const child of Array.from(fragment.documentElement.children)) {
    world.appendChild(doc.importNode(child, true));
  }
  // Physical waist tilt lock: equality joint constraints at the configured default.
  // yaw stays free so the robot can still turn; roll/pitch are the two tilts.
  // They are always present but toggled through data.eq_active.
  const equality = doc.createElement('equality');
  for (const name of ['waist_roll_joint', 'waist_pitch_joint']) {
    const joint = state.config?.joints?.find((j) => j.name === name);
    if (!joint) continue;
    const eq = doc.createElement('joint');
    eq.setAttribute('joint1', name);
    eq.setAttribute('polycoef', `${Number(joint.default).toFixed(9)} 0 0 0 0`);
    eq.setAttribute('solref', '0.004 1');
    eq.setAttribute('solimp', '0.99 0.999 0.001');
    equality.appendChild(eq);
  }
  if (equality.childElementCount) doc.documentElement.appendChild(equality);
  return new XMLSerializer().serializeToString(doc);
}


function applyRobotVisibility() {
  const lowPoly = Boolean(state.lowPoly);
  if (state.visualMeshesHigh) for (const m of state.visualMeshesHigh) m.visible = !lowPoly;
  if (state.visualMeshesLow) for (const m of state.visualMeshesLow) m.visible = lowPoly;
  if (state.proxyMeshes) for (const m of state.proxyMeshes) m.visible = false;
}

function buildCollisionProxies() {
  const { model, mujoco, bodyNodes } = state;
  if (!model) return;
  if (state.proxyMeshes) {
    for (const mesh of state.proxyMeshes) {
      mesh.parent?.remove(mesh);
      mesh.geometry?.dispose();
    }
    state.proxyMeshes = [];
  }
  if (state.proxyGroup) {
    state.proxyGroup.traverse((o) => { if (o.geometry) o.geometry.dispose(); });
    scene.remove(state.proxyGroup);
  }
  state.proxyGroup = new THREE.Group();
  state.proxyGroup.visible = false;
  scene.add(state.proxyGroup);
  state.proxyMeshes = [];
  const material = new THREE.MeshLambertMaterial({ color: 0xaebccd });
  const T = mujoco.mjtGeom;
  for (let g = 0; g < model.ngeom; g++) {
    const geom = model.geom(g);
    if (Number(geom.group) !== 3) continue;
    const type = Number(geom.type);
    const size = geom.size;
    const r = Number(size[0]);
    let geometry = null;
    if (type === T.mjGEOM_CAPSULE.value) {
      geometry = new THREE.CapsuleGeometry(r, Number(size[1]) * 2, 2, 6);
      geometry.rotateX(Math.PI / 2);
    } else if (type === T.mjGEOM_CYLINDER.value) {
      geometry = new THREE.CylinderGeometry(r, r, Number(size[1]) * 2, 6);
      geometry.rotateX(Math.PI / 2);
    } else if (type === T.mjGEOM_SPHERE.value) {
      geometry = new THREE.SphereGeometry(r, 8, 6);
    } else if (type === T.mjGEOM_BOX.value) {
      geometry = new THREE.BoxGeometry(Number(size[0]) * 2, Number(size[1]) * 2, Number(size[2]) * 2);
    } else if (type === T.mjGEOM_ELLIPSOID.value) {
      geometry = new THREE.SphereGeometry(1, 8, 6);
      geometry.scale(Number(size[0]), Number(size[1]), Number(size[2]));
    }
    if (!geometry) continue;
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(Number(geom.pos[0]), Number(geom.pos[1]), Number(geom.pos[2]));
    mesh.quaternion.set(Number(geom.quat[1]), Number(geom.quat[2]), Number(geom.quat[3]), Number(geom.quat[0]));
    const bodyId = Number(geom.bodyid);
    const parent = bodyNodes[bodyId];
    if (parent) parent.add(mesh);
    state.proxyMeshes.push(mesh);
  }
  applyRobotVisibility();
  console.log('collision proxies:', state.proxyMeshes.length);
}

function lowestCollisionZ() {
  const { model, data, mujoco } = state;
  const GEOM_SPHERE = mujoco.mjtGeom.mjGEOM_SPHERE.value;
  const GEOM_CAPSULE = mujoco.mjtGeom.mjGEOM_CAPSULE.value;
  const GEOM_CYLINDER = mujoco.mjtGeom.mjGEOM_CYLINDER.value;
  const GEOM_BOX = mujoco.mjtGeom.mjGEOM_BOX.value;
  const GEOM_ELLIPSOID = mujoco.mjtGeom.mjGEOM_ELLIPSOID.value;
  let best = Infinity;
  for (let g = 0; g < model.ngeom; g++) {
    const geom = model.geom(g);
    if (Number(geom.group) !== 3) continue;
    const type = Number(geom.type);
    const size = geom.size;
    const pz = data.geom_xpos[g * 3 + 2];
    const R = data.geom_xmat;
    let low;
    if (type === GEOM_SPHERE) {
      low = pz - Number(size[0]);
    } else if (type === GEOM_CAPSULE || type === GEOM_CYLINDER) {
      low = pz - (Math.abs(Number(R[g * 9 + 8])) * Number(size[1]) + Number(size[0]));
    } else if (type === GEOM_BOX) {
      low = pz - (
        Math.abs(Number(R[g * 9 + 6])) * Number(size[0]) +
        Math.abs(Number(R[g * 9 + 7])) * Number(size[1]) +
        Math.abs(Number(R[g * 9 + 8])) * Number(size[2])
      );
    } else if (type === GEOM_ELLIPSOID) {
      low = pz - Math.hypot(
        Number(R[g * 9 + 6]) * Number(size[0]),
        Number(R[g * 9 + 7]) * Number(size[1]),
        Number(R[g * 9 + 8]) * Number(size[2]),
      );
    } else {
      low = pz - Number(geom.rbound);
    }
    if (low < best) best = low;
  }
  return best;
}

function applySim2RealModel() {
  const { model, config } = state;
  if (!model || !config) return;
  const s = state.sim2real;
  for (const j of config.joints) {
    model.dof_frictionloss[j.qvel] = Number(s.friction) || 0;
    model.dof_damping[j.qvel] = Number(s.damping) || 0;
  }
}

function buildModel(terrain) {
  const { mujoco } = state;
  const xml = buildModelXml(state.template, terrain);
  state.modelXml = xml;
  if (state.data) { try { state.data.delete(); } catch (_) {} }
  if (state.model) { try { state.model.delete(); } catch (_) {} }
  state.model = mujoco.MjModel.from_xml_string(xml);
  state.data = new mujoco.MjData(state.model);
  applySim2RealModel();
  applyEqualityActivity();
  buildCollisionProxies();
  const physicsDt = Number(state.model?.opt?.timestep ?? state.config.engine.physics_dt ?? 0.005);
  state.substeps = Math.max(1, Math.round(state.controlDt / physicsDt));
}

function terrainMeshes(terrain) {
  const group = new THREE.Group();
  state.terrainEdges = [];
  state.terrainVisuals = [];
  const palette = SCENE_THEMES[document.documentElement.dataset.theme] || SCENE_THEMES.dark;
  const fill = new THREE.MeshLambertMaterial({ color: palette.fill });
  const steel = new THREE.MeshStandardMaterial({ color: palette.steel, metalness: palette.steelMetalness, roughness: palette.steelRoughness, side: THREE.DoubleSide });
  const edge = new THREE.LineBasicMaterial({ color: palette.edge, transparent: true, opacity: palette.edgeOpacity });
  state.terrainMaterials = { fill, steel, edge };
  if (terrain.hfield) {
    const { nrow, ncol, heights, x0, y0, dx, dy } = terrain.hfield;
    const positions = new Float32Array(nrow * ncol * 3);
    const indices = [];
    const grid = [];
    for (let row = 0; row < nrow; row++) {
      for (let col = 0; col < ncol; col++) {
        const index = row * ncol + col;
        positions.set([x0 + col * dx, y0 + row * dy, heights[index]], index * 3);
        if (row < nrow - 1 && col < ncol - 1) {
          indices.push(index, index + 1, index + ncol + 1, index, index + ncol + 1, index + ncol);
        }
        const segment = (next) => {
          grid.push(...positions.subarray(index * 3, index * 3 + 2), heights[index] + 0.003,
            x0 + (next % ncol) * dx, y0 + Math.floor(next / ncol) * dy, heights[next] + 0.003);
        };
        if ((row % 10 === 0 || row === nrow - 1) && col < ncol - 1) segment(index + 1);
        if ((col % 10 === 0 || col === ncol - 1) && row < nrow - 1) segment(index + ncol);
      }
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();
    group.add(new THREE.Mesh(geometry, fill));
    const gridGeometry = new THREE.BufferGeometry();
    gridGeometry.setAttribute('position', new THREE.Float32BufferAttribute(grid, 3));
    edge.opacity *= 0.55;
    const lines = new THREE.LineSegments(gridGeometry, edge);
    group.add(lines);
    state.terrainEdges.push(lines);
  }
  for (const b of terrain.boxes) {
    const geo = new THREE.BoxGeometry(b.hx * 2, b.hy * 2, b.hz * 2);
    const mesh = new THREE.Mesh(geo, b.steel ? steel : fill);
    mesh.position.set(b.cx, b.cy, b.cz);
    if (b.quat) mesh.quaternion.set(b.quat[1], b.quat[2], b.quat[3], b.quat[0]);
    group.add(mesh);
    const lines = new THREE.LineSegments(new THREE.EdgesGeometry(geo), edge);
    lines.position.copy(mesh.position);
    if (b.quat) lines.quaternion.copy(mesh.quaternion);
    group.add(lines);
    state.terrainEdges.push(lines);
  }
  for (const v of terrain.visuals || []) {
    const geo = new THREE.BoxGeometry(v.hx * 2, v.hy * 2, v.hz * 2);
    const mesh = new THREE.Mesh(geo, steel);
    mesh.position.set(v.cx, v.cy, v.cz);
    if (v.rotY) mesh.rotation.y = v.rotY;
    group.add(mesh);
    state.terrainVisuals.push(mesh);
  }
  return group;
}

// ---------------------------------------------------------------------------
// observations / control
// ---------------------------------------------------------------------------
const KEY_CODES = new Set(['KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE']);
const ARROW_ALIASES = { ArrowUp: 'KeyW', ArrowDown: 'KeyS', ArrowLeft: 'KeyA', ArrowRight: 'KeyD' };

function installKeyboard() {
  const targetAllowed = (event) => {
    const el = event.target;
    const tag = el?.tagName;
    if (tag === 'SELECT' || tag === 'TEXTAREA') return false;
    if (tag === 'INPUT' && el.type !== 'range') return false;
    return true;
  };
  window.addEventListener('keydown', (event) => {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    const code = ARROW_ALIASES[event.code] || event.code;
    if (!KEY_CODES.has(code)) return;
    if (!targetAllowed(event)) return;
    event.preventDefault();
    if (!event.repeat) state.keys.add(code);
  });
  window.addEventListener('keyup', (event) => {
    const code = ARROW_ALIASES[event.code] || event.code;
    if (!KEY_CODES.has(code)) return;
    state.keys.delete(code);
  });
  window.addEventListener('blur', () => state.keys.clear());
}

function updateCommandFromKeys() {
  if (!state.keyboardEnabled) return;
  const k = state.keys;
  const s = state.commandScale;
  const target = [
    (k.has('KeyW') ? s.vx : 0) + (k.has('KeyS') ? -s.vx * 0.6 : 0),
    (k.has('KeyE') ? s.vy : 0) + (k.has('KeyQ') ? -s.vy : 0),
    (k.has('KeyA') ? s.wz : 0) + (k.has('KeyD') ? -s.wz : 0),
  ];
  // ~0.15 s ramp at 50 Hz, matching the smooth command routing used in training.
  const alpha = 0.18;
  for (let i = 0; i < 3; i++) {
    state.command[i] += (target[i] - state.command[i]) * alpha;
    if (Math.abs(state.command[i]) < 1e-4) state.command[i] = 0;
  }
}

function observationTerms() {
  const { data, config, lastAction } = state;
  const s = state.sim2real;
  const rootQuat = Array.from(data.qpos.slice(3, 7));
  const jointPos = new Float64Array(29);
  const jointVel = new Float64Array(29);
  const filterMs = Number(s.jointVelFilterMs) || 0;
  const filterAlpha = filterMs > 0 ? Math.exp(-state.controlDt / (filterMs / 1000)) : 0;
  for (let i = 0; i < 29; i++) {
    const j = config.joints[i];
    jointPos[i] = data.qpos[j.qpos] - j.default + s.zeroOffsets[i];
    let v = data.qvel[j.qvel];
    if (filterAlpha > 0) {
      s.velFiltered[i] = filterAlpha * s.velFiltered[i] + (1 - filterAlpha) * v;
      v = s.velFiltered[i];
    }
    jointVel[i] = v;
  }
  let gravity = quatRotateInverse(rootQuat, [0, 0, -1]);
  const pitch = (Number(s.imuPitchBiasDeg) || 0) * Math.PI / 180;
  if (pitch) {
    const c = Math.cos(pitch), sn = Math.sin(pitch);
    gravity = [gravity[0] * c - gravity[2] * sn, gravity[1], gravity[0] * sn + gravity[2] * c];
  }
  return {
    base_ang_vel: [data.qvel[3] * 0.25, data.qvel[4] * 0.25, (data.qvel[5] + (Number(s.gyroBias) || 0)) * 0.25],
    projected_gravity: gravity,
    velocity_commands: state.command.slice(),
    joint_pos_rel: jointPos,
    joint_vel_rel: Array.from(jointVel, (v) => v * 0.05),
    last_action: Array.from(lastAction),
  };
}

function initHistories() {
  const terms = observationTerms();
  for (const name of TERMS) state.histories[name].reset(terms[name]);
}

function flattenObservation() {
  const obs = new Float32Array(state.config.observation_dim);
  let offset = 0;
  for (const name of TERMS) offset = state.histories[name].write(obs, offset);
  obs.set(state.depth, offset);
  offset += state.depth.length;
  if (offset !== obs.length) throw new Error(`observation layout mismatch: ${offset} vs ${obs.length}`);
  return obs;
}

function pdStep(target) {
  const { data, config } = state;
  const s = state.sim2real;
  const backlash = Number(s.backlash) || 0;
  for (let i = 0; i < 29; i++) {
    const j = config.joints[i];
    const q = data.qpos[j.qpos] + s.zeroOffsets[i];
    const qd = data.qvel[j.qvel];
    let error = target[i] - q;
    if (backlash > 0) {
      const half = backlash * 0.5;
      error = Math.abs(error) <= half ? 0 : error - Math.sign(error) * half;
    }
    let tau = j.kp * error - j.kd * qd;
    if (tau > j.effort) tau = j.effort;
    else if (tau < -j.effort) tau = -j.effort;
    data.ctrl[j.actuator] = tau;
  }
}

async function policyStep() {
  const { session, config } = state;
  const obs = flattenObservation();
  const t0 = performance.now();
  const feeds = {
    observations: new ort.Tensor('float32', obs, [1, config.observation_dim]),
    hidden: new ort.Tensor('float32', state.hidden, [1, 1, 256]),
  };
  const results = await session.run(feeds);
  const actions = results.actions.data;
  state.lastInferenceMs = performance.now() - t0;
  state.perf.inferMs = state.perf.inferMs ? state.perf.inferMs * 0.85 + state.lastInferenceMs * 0.15 : state.lastInferenceMs;
  state.hidden = new Float32Array(results.hidden_next.data);

  const target = new Float64Array(29);
  for (let i = 0; i < 29; i++) {
    const j = config.joints[i];
    target[i] = j.default + j.scale * actions[i];
    state.lastAction[i] = actions[i];
  }
  return target;
}

function validState() {
  const q = state.data.qpos;
  for (let i = 0; i < q.length; i++) if (!Number.isFinite(q[i])) return false;
  const v = state.data.qvel;
  for (let i = 0; i < v.length; i++) if (!Number.isFinite(v[i])) return false;
  return true;
}

async function controlStep() {
  const tStep = performance.now();
  updateCommandFromKeys();
  const target = await policyStep();
  const s = state.sim2real;
  const delayFrames = Math.max(0, Math.round((Number(s.actuatorDelayMs) || 0) / (state.controlDt * 1000)));
  let pdTarget = target;
  if (delayFrames > 0) {
    if (!s.lastApplied) s.lastApplied = Float64Array.from(target);
    s.targetQueue.push(Float64Array.from(target));
    if (s.targetQueue.length > delayFrames) pdTarget = s.targetQueue.shift();
    else pdTarget = s.lastApplied;
    s.lastApplied = Float64Array.from(pdTarget);
  } else {
    s.targetQueue.length = 0;
    s.lastApplied = null;
  }
  const t0 = performance.now();
  for (let s = 0; s < state.substeps; s++) {
    pdStep(pdTarget);
    state.mujoco.mj_step(state.model, state.data);
  }
  const physicsMs = performance.now() - t0;
  state.perf.physicsMs = state.perf.physicsMs ? state.perf.physicsMs * 0.85 + physicsMs * 0.15 : physicsMs;
  state.steps += 1;
  state.simTime += state.controlDt;
  state.lastControlMs = performance.now() - t0;

  const terms = observationTerms();
  for (const name of TERMS) state.histories[name].push(terms[name]);
  const depthGen = state.depthGen;
  if (state.checking) {
    state.depth = renderDepth();
  } else if (!state.depthPending) {
    state.depthPending = true;
    renderDepthAsync()
      .then((depth) => { if (depthGen === state.depthGen) state.depth = depth; })
      .catch((err) => { console.warn('async depth failed, falling back to sync', err); if (depthGen === state.depthGen) state.depth = renderDepth(); })
      .finally(() => { state.depthPending = false; });
  }

  const pelvisZ = state.data.qpos[2];
  const clearance = pelvisZ - groundHeight(state.data.qpos[0], state.data.qpos[1]);
  state.minPelvisZ = Math.min(state.minPelvisZ, pelvisZ);
  state.maxPelvisZ = Math.max(state.maxPelvisZ, pelvisZ);
  state.minClearance = Math.min(state.minClearance, clearance);
  const stepMs = performance.now() - tStep;
  state.perf.stepMs = state.perf.stepMs ? state.perf.stepMs * 0.85 + stepMs * 0.15 : stepMs;
  return validState();
}

// ---------------------------------------------------------------------------
// depth camera
// ---------------------------------------------------------------------------
function cameraPose() {
  const { data, config } = state;
  const torso = Number(config.torso_body);
  const q = Array.from(data.xquat.slice(torso * 4, torso * 4 + 4));
  const R = quatToMatrix(q);
  const p = [
    data.xpos[torso * 3] + R[0][0] * config.camera.offset.pos[0] + R[0][1] * config.camera.offset.pos[1] + R[0][2] * config.camera.offset.pos[2],
    data.xpos[torso * 3 + 1] + R[1][0] * config.camera.offset.pos[0] + R[1][1] * config.camera.offset.pos[1] + R[1][2] * config.camera.offset.pos[2],
    data.xpos[torso * 3 + 2] + R[2][0] * config.camera.offset.pos[0] + R[2][1] * config.camera.offset.pos[1] + R[2][2] * config.camera.offset.pos[2],
  ];
  const qc = quatMul(q, config.camera.offset.rot);
  const Rc = quatToMatrix(qc);
  // Isaac camera frame is x-forward, y-left, z-up. three.js camera basis is
  // x-right, y-up, z-backward, so X=-y_isaac, Y=z_isaac, Z=-x_isaac.
  const X = new THREE.Vector3(-Rc[0][1], -Rc[1][1], -Rc[2][1]);
  const Y = new THREE.Vector3(Rc[0][2], Rc[1][2], Rc[2][2]);
  const Z = new THREE.Vector3(-Rc[0][0], -Rc[1][0], -Rc[2][0]);
  const m = new THREE.Matrix4().makeBasis(X, Y, Z);
  const quaternion = new THREE.Quaternion().setFromRotationMatrix(m);
  return { position: new THREE.Vector3(p[0], p[1], p[2]), quaternion };
}

function unpackDepth(pixels, i) {
  const r = pixels[i] / 255;
  const g = pixels[i + 1] / 255;
  const b = pixels[i + 2] / 255;
  const a = pixels[i + 3] / 255;
  return (r + g / 255 + b / 65025 + a / 16581375) * DEPTH_FAR;
}

function renderDepth() {
  const tDepth = performance.now();
  if (!depthTarget) return new Float32Array(DEPTH_OUT * DEPTH_OUT);
  const pose = cameraPose();
  depthCamera.position.copy(pose.position);
  depthCamera.quaternion.copy(pose.quaternion);
  depthCamera.updateMatrixWorld(true);

  const oldOverride = scene.overrideMaterial;
  const oldBg = scene.background;
  scene.overrideMaterial = depthMaterial;
  scene.background = null;
  const gridWasVisible = gridHelper ? gridHelper.visible : false;
  if (gridHelper) gridHelper.visible = false;
  // Fidelity: the training depth camera raycasts the original visual meshes,
  // never the decimated low-poly preview or the collision proxies. The policy
  // input must stay identical when the user selects performance quality.
  if (state.visualMeshesHigh) for (const m of state.visualMeshesHigh) m.visible = true;
  if (state.visualMeshesLow) for (const m of state.visualMeshesLow) m.visible = false;
  if (state.proxyMeshes) for (const m of state.proxyMeshes) m.visible = false;
  if (state.terrainEdges) for (const l of state.terrainEdges) l.visible = false;
  if (state.terrainVisuals) for (const m of state.terrainVisuals) m.visible = false;
  renderer.setRenderTarget(depthTarget);
  renderer.setClearColor(0x000000, 0);
  renderer.clear();
  renderer.render(scene, depthCamera);
  renderer.readRenderTargetPixels(depthTarget, 0, 0, DEPTH_W, DEPTH_H, depthPixels);
  renderer.setRenderTarget(null);
  renderer.setClearColor(state.sceneClear, 1);
  scene.overrideMaterial = oldOverride;
  scene.background = oldBg;
  if (gridHelper) gridHelper.visible = gridWasVisible;
  applyRobotVisibility();
  if (state.terrainEdges) for (const l of state.terrainEdges) l.visible = true;
  if (state.terrainVisuals) for (const m of state.terrainVisuals) m.visible = true;

  const out = unpackDepthBuffer(depthPixels);
  const ms = performance.now() - tDepth;
  state.perf.depthMs = state.perf.depthMs ? state.perf.depthMs * 0.85 + ms * 0.15 : ms;
  return out;
}

function unpackDepthBuffer(pixels) {
  if (!state.depthScratch) state.depthScratch = new Float32Array(DEPTH_OUT * DEPTH_OUT);
  const out = state.depthScratch;
  for (let row = 0; row < DEPTH_OUT; row++) {
    // readRenderTargetPixels is bottom-up; training images are top-down.
    const srcRow = DEPTH_H - 1 - row;
    for (let col = 0; col < DEPTH_OUT; col++) {
      const srcCol = DEPTH_CROP_LEFT + col;
      const idx = (srcRow * DEPTH_W + srcCol) * 4;
      let d = unpackDepth(pixels, idx);
      if (pixels[idx + 3] === 0) d = DEPTH_MAX; // background / miss
      if (!Number.isFinite(d)) d = DEPTH_MAX;
      d = Math.min(DEPTH_MAX, Math.max(0, d));
      out[row * DEPTH_OUT + col] = d / DEPTH_MAX;
    }
  }
  return out;
}

let depthPbo = null;
let depthPboSize = 0;
let depthReadJob = null;

function pollDepthReadback(flush = false) {
  const job = depthReadJob;
  if (!job || !renderer) return false;
  const gl = renderer.getContext();
  const status = gl.clientWaitSync(job.sync, flush ? gl.SYNC_FLUSH_COMMANDS_BIT : 0, 0);
  if (status === gl.TIMEOUT_EXPIRED) return false;
  depthReadJob = null;
  if (status !== gl.WAIT_FAILED) {
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, depthPbo);
    gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, job.buffer);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
  }
  gl.deleteSync(job.sync);
  if (job.timer) clearTimeout(job.timer);
  job.resolve(status !== gl.WAIT_FAILED);
  return true;
}

function readDepthTargetAsync(buffer) {
  const gl = renderer.getContext();
  if (!(gl instanceof WebGL2RenderingContext)) {
    return renderer.readRenderTargetPixelsAsync(depthTarget, 0, 0, DEPTH_W, DEPTH_H, buffer).then(() => true);
  }
  if (!depthPbo || depthPboSize !== buffer.byteLength) {
    if (depthPbo) gl.deleteBuffer(depthPbo);
    depthPbo = gl.createBuffer();
    depthPboSize = buffer.byteLength;
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, depthPbo);
    gl.bufferData(gl.PIXEL_PACK_BUFFER, depthPboSize, gl.STREAM_READ);
  } else {
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, depthPbo);
  }
  gl.readPixels(0, 0, DEPTH_W, DEPTH_H, gl.RGBA, gl.UNSIGNED_BYTE, 0);
  gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
  const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
  if (document.hidden) gl.flush();
  return new Promise((resolve) => {
    const job = { sync, buffer, resolve, timer: 0 };
    depthReadJob = job;
    // Fallback poll only when rAF is paused (e.g. hidden tab or throttled render)
    const fallback = () => {
      if (depthReadJob !== job) return;
      if (!pollDepthReadback(true)) job.timer = setTimeout(fallback, 16);
    };
    job.timer = setTimeout(fallback, 24);
  });
}

async function renderDepthAsync() {
  if (!depthTarget) return new Float32Array(DEPTH_OUT * DEPTH_OUT);
  const t0 = performance.now();
  const pose = cameraPose();
  depthCamera.position.copy(pose.position);
  depthCamera.quaternion.copy(pose.quaternion);
  depthCamera.updateMatrixWorld(true);

  const oldOverride = scene.overrideMaterial;
  const oldBg = scene.background;
  scene.overrideMaterial = depthMaterial;
  scene.background = null;
  const gridWasVisible = gridHelper ? gridHelper.visible : false;
  if (gridHelper) gridHelper.visible = false;
  // Fidelity: the training depth camera raycasts the original visual meshes,
  // never the decimated low-poly preview or the collision proxies. The policy
  // input must stay identical when the user selects performance quality.
  if (state.visualMeshesHigh) for (const m of state.visualMeshesHigh) m.visible = true;
  if (state.visualMeshesLow) for (const m of state.visualMeshesLow) m.visible = false;
  if (state.proxyMeshes) for (const m of state.proxyMeshes) m.visible = false;
  if (state.terrainEdges) for (const l of state.terrainEdges) l.visible = false;
  if (state.terrainVisuals) for (const m of state.terrainVisuals) m.visible = false;
  renderer.setRenderTarget(depthTarget);
  renderer.setClearColor(0x000000, 0);
  renderer.clear();
  renderer.render(scene, depthCamera);
  if (!state.depthAsyncBuffer) state.depthAsyncBuffer = new Uint8Array(DEPTH_W * DEPTH_H * 4);
  const buffer = state.depthAsyncBuffer;
  const promise = readDepthTargetAsync(buffer);
  renderer.setRenderTarget(null);
  renderer.setClearColor(state.sceneClear, 1);
  scene.overrideMaterial = oldOverride;
  scene.background = oldBg;
  if (gridHelper) gridHelper.visible = gridWasVisible;
  applyRobotVisibility();
  if (state.terrainEdges) for (const l of state.terrainEdges) l.visible = true;
  if (state.terrainVisuals) for (const m of state.terrainVisuals) m.visible = true;

  await promise;
  const lag = performance.now() - t0;
  state.perf.depthLagMs = state.perf.depthLagMs ? state.perf.depthLagMs * 0.85 + lag * 0.15 : lag;
  return Float32Array.from(unpackDepthBuffer(buffer));
}

// ---------------------------------------------------------------------------
// reset / terrain change
// ---------------------------------------------------------------------------
function applyEqualityActivity() {
  const eq = state.data?.eq_active;
  if (!eq) return;
  const active = state.lockWaist ? 1 : 0;
  for (let i = 0; i < eq.length; i++) eq[i] = active;
}

function resetSim() {
  const { data, config, terrain } = state;
  state.mujoco.mj_resetData(state.model, data);
  applyEqualityActivity();
  for (const j of config.joints) data.qpos[j.qpos] = j.default;
  data.qpos[0] = terrain.spawn.x;
  data.qpos[1] = terrain.spawn.y;
  data.qpos[2] = 0;
  data.qpos[3] = 1; data.qpos[4] = 0; data.qpos[5] = 0; data.qpos[6] = 0;
  data.qvel.fill(0);
  data.ctrl.fill(0);
  state.mujoco.mj_forward(state.model, data);
  const low = lowestCollisionZ();
  data.qpos[2] = terrain.spawn.z - low + 0.004;
  data.qvel.fill(0);
  data.ctrl.fill(0);
  state.mujoco.mj_forward(state.model, data);

  state.depthGen += 1;
  state.command = [0, 0, 0];
  state.steps = 0;
  state.simTime = 0;
  state.hidden = new Float32Array(256);
  state.lastAction = new Float32Array(29);
  const s = state.sim2real;
  for (let i = 0; i < 29; i++) {
    s.zeroOffsets[i] = Math.sin(i * 1.7) * (Number(s.encoderZero) || 0);
    s.velFiltered[i] = 0;
  }
  s.targetQueue = [];
  s.lastApplied = null;
  state.lastInferenceMs = 0;
  state.lastControlMs = 0;
  for (const name of TERMS) state.histories[name] = new HistoryBuffer(TERM_WIDTH[name]);
  initHistories();
  updateBodyTransforms();
  state.depth = renderDepth();
  const pelvis = [data.qpos[0], data.qpos[1], data.qpos[2]];
  state.startPos = pelvis;
  state.minPelvisZ = pelvis[2];
  state.maxPelvisZ = pelvis[2];
  state.minClearance = pelvis[2] - groundHeight(pelvis[0], pelvis[1]);
  state.recentSpeeds = [];
  updateBodyTransforms();
}

function persistTerrainPrefs() {
  try {
    localStorage.setItem('preview.terrain', JSON.stringify({
      mode: state.terrainMode,
      direction: state.direction,
      level: state.level,
      uniform: state.uniform,
      gap: state.gap,
      platform: state.platform,
      grating: state.grating,
      slope: state.slope,
      lockWaist: state.lockWaist,
    }));
  } catch (_) { /* storage unavailable */ }
}

function applyUrlParams() {
  const params = new URLSearchParams(location.search);
  const number = (key, fallback) => (params.has(key) ? Number(params.get(key)) : fallback);
  if (params.has('mode')) state.terrainMode = String(params.get('mode'));
  if (params.has('direction')) state.direction = String(params.get('direction'));
  if (params.has('level')) state.level = Math.max(0, Math.min(19, Math.round(number('level', state.level))));
  const u = state.uniform;
  u.riser = number('riser', u.riser * 100) / 100;
  u.tread = number('tread', u.tread * 100) / 100;
  u.steps = Math.round(number('steps', u.steps));
  u.width = number('width', u.width * 100) / 100;
  const s = state.sim2real;
  s.friction = number('friction', s.friction);
  s.damping = number('damping', s.damping);
  s.actuatorDelayMs = number('delay', s.actuatorDelayMs);
  s.encoderZero = number('zero', 0) * Math.PI / 180;
  s.imuPitchBiasDeg = number('pitch', 0);
  s.gyroBias = number('gyro', 0) * Math.PI / 180;
  s.jointVelFilterMs = number('velfilter', 0);
  s.backlash = number('backlash', 0) * Math.PI / 180;
  if (params.has('shoe')) {
    s.shoe = !['0', 'false', 'off', 'no'].includes(String(params.get('shoe')).toLowerCase());
  }
  if (params.has('vx')) {
    state.commandScale.vx = number('vx', state.commandScale.vx);
    const vxInput = qs('cmd-vx');
    if (vxInput) vxInput.value = String(state.commandScale.vx);
    const vxLabel = qs('cmd-vx-value');
    if (vxLabel) vxLabel.textContent = state.commandScale.vx.toFixed(2);
  }
  // Reflect URL values in the DOM before setupUi() reads the controls.
  const write = (id, value) => { const el = qs(id); if (el) el.value = String(value); };
  write('uniform-riser', u.riser * 100);
  write('uniform-tread', u.tread * 100);
  write('uniform-steps', u.steps);
  write('uniform-width', u.width);
  write('s2r-friction', s.friction);
  write('s2r-damping', s.damping);
  write('s2r-delay', s.actuatorDelayMs);
  write('s2r-zero', s.encoderZero * 180 / Math.PI);
  write('s2r-pitch', s.imuPitchBiasDeg);
  write('s2r-gyro', s.gyroBias * 180 / Math.PI);
  write('s2r-velfilter', s.jointVelFilterMs);
  write('s2r-backlash', s.backlash * 180 / Math.PI);
  const shoe = qs('s2r-shoe');
  if (shoe) shoe.checked = Boolean(s.shoe);
}

function restoreTerrainPrefs() {
  try {
    const saved = JSON.parse(localStorage.getItem('preview.terrain') || 'null');
    if (!saved) return;
    if (['uniform', 'curriculum', 'gap', 'platform', 'grating', 'slope', 'hillside', 'flat'].includes(saved.mode)) state.terrainMode = saved.mode;
    if (DIRECTIONS.includes(saved.direction)) state.direction = saved.direction;
    if (Number.isFinite(Number(saved.level))) state.level = Math.max(0, Math.min(19, Number(saved.level)));
    if (saved.uniform) {
      const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, Number(v)));
      state.uniform = {
        riser: clamp(saved.uniform.riser ?? 0.12, 0.03, 0.30),
        tread: clamp(saved.uniform.tread ?? 0.35, 0.15, 0.60),
        steps: Math.round(clamp(saved.uniform.steps ?? 12, 3, 40)),
        width: clamp(saved.uniform.width ?? 4.0, 0.8, 4.0),
      };
      if (qs('uniform-riser')) {
        qs('uniform-riser').value = String(state.uniform.riser * 100);
        qs('uniform-tread').value = String(state.uniform.tread * 100);
        qs('uniform-steps').value = String(state.uniform.steps);
        if (qs('uniform-width')) qs('uniform-width').value = String(state.uniform.width);
      }
    }
    const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, Number(v)));
    if (saved.gap) {
      state.gap = {
        width: clamp(saved.gap.width ?? 0.30, 0.10, 0.80),
        depth: clamp(saved.gap.depth ?? 0.45, 0.30, 0.60),
        center: clamp(saved.gap.center ?? 2.0, 0.5, 6.0),
      };
      if (qs('gap-width')) {
        qs('gap-width').value = String(state.gap.width * 100);
        qs('gap-depth').value = String(state.gap.depth * 100);
        qs('gap-center').value = String(state.gap.center);
      }
    }
    if (saved.platform) {
      state.platform = {
        height: clamp(saved.platform.height ?? 0.20, 0.05, 0.40),
        start: clamp(saved.platform.start ?? 1.5, 0.5, 6.0),
      };
      if (qs('platform-height')) {
        qs('platform-height').value = String(state.platform.height * 100);
        qs('platform-start').value = String(state.platform.start);
      }
    }
    if (saved.grating) {
      state.grating = {
        riser: clamp(saved.grating.riser ?? 0.20, 0.10, 0.30),
        tread: clamp(saved.grating.tread ?? 0.25, 0.20, 0.35),
        steps: Math.round(clamp(saved.grating.steps ?? 12, 3, 30)),
        width: clamp(saved.grating.width ?? 1.2, 0.8, 2.0),
        bar: clamp(saved.grating.bar ?? 0.03, 0.02, 0.05),
        gap: clamp(saved.grating.gap ?? 0.015, 0.01, 0.03),
      };
      if (qs('grating-riser')) {
        qs('grating-riser').value = String(state.grating.riser * 100);
        qs('grating-tread').value = String(state.grating.tread * 100);
        qs('grating-steps').value = String(state.grating.steps);
        qs('grating-width').value = String(state.grating.width);
        qs('grating-bar').value = String(state.grating.bar * 100);
        qs('grating-gap').value = String(state.grating.gap * 100);
      }
    }
    if (typeof saved.lockWaist === 'boolean') state.lockWaist = saved.lockWaist;
    if (qs('lock-waist')) qs('lock-waist').checked = Boolean(state.lockWaist);
    if (saved.slope) {
      state.slope = {
        angle: clamp(saved.slope.angle ?? 12, 0, 20),
        run: clamp(saved.slope.run ?? 3.1, 1.5, 3.5),
        width: clamp(saved.slope.width ?? 4.0, 2.0, 4.0),
      };
      if (qs('slope-angle')) {
        qs('slope-angle').value = String(state.slope.angle);
        qs('slope-run').value = String(state.slope.run);
        qs('slope-width').value = String(state.slope.width);
      }
    }
  } catch (_) { /* ignore corrupt storage */ }
}

function applyTerrain(direction = state.direction, level = state.level) {
  state.direction = direction;
  state.level = level;
  if (state.terrainMode === 'hillside') {
    state.terrain = buildHillsideTerrain({ difficulty: Math.max(0, Math.min(1, level / 19)) });
  } else if (state.terrainMode === 'gap') {
    state.terrain = buildGapTerrain({ gapWidth: state.gap.width, gapDepth: state.gap.depth, centerX: state.gap.center });
  } else if (state.terrainMode === 'platform') {
    state.terrain = buildPlatformTerrain({ platformHeight: state.platform.height, startX: state.platform.start });
  } else if (state.terrainMode === 'grating') {
    state.terrain = buildGratingTerrain({
      riser: state.grating.riser,
      tread: state.grating.tread,
      steps: state.grating.steps,
      width: state.grating.width,
      barWidth: state.grating.bar,
      gapWidth: state.grating.gap,
    });
  } else if (state.terrainMode === 'slope') {
    state.terrain = buildSlopeTerrain({
      angle: state.slope.angle,
      run: state.slope.run,
      width: state.slope.width,
    });
  } else if (state.terrainMode === 'flat' || direction === 'flat') {
    state.terrain = buildTerrain(state.config, 'flat', 0);
  } else if (state.terrainMode === 'uniform') {
    state.terrain = buildUniformTerrain({
      direction,
      riser: state.uniform.riser,
      tread: state.uniform.tread,
      steps: state.uniform.steps,
      width: state.uniform.width,
    });
  } else {
    state.terrain = buildTerrain(state.config, direction, level);
  }
  if (state.terrainGroup) {
    scene.remove(state.terrainGroup);
    const materials = new Set(Object.values(state.terrainMaterials));
    state.terrainGroup.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) {
        for (const material of Array.isArray(o.material) ? o.material : [o.material]) materials.add(material);
      }
    });
    for (const material of materials) material.dispose();
    state.terrainMaterials = null;
  }
  state.terrainGroup = terrainMeshes(state.terrain);
  scene.add(state.terrainGroup);
  if (gridHelper) gridHelper.visible = !(state.terrain?.spec?.gap || state.terrain?.spec?.grating || state.terrain?.hfield);
  try {
    buildModel(state.terrain);
  } catch (err) {
    setStatus(`terrain failed: ${err.message}`, true);
    throw err;
  }
  resetSim();
  syncTerrainUi();
  if (controls) {
    if (state.terrainMode === 'hillside') {
      if (state.lastAppliedTerrainMode !== 'hillside') {
        const { x, y, z } = state.terrain.spawn;
        const compact = renderer.domElement.clientWidth < 600;
        const width = renderer.domElement.clientWidth;
        const height = renderer.domElement.clientHeight;
        camera.position.set(x - (compact ? 8 : 4.6), y - (compact ? 3 : 6.4), z + (compact ? 3.8 : 3.6));
        camera.zoom = compact ? 0.5 : 1;
        camera.setViewOffset(width, height, compact ? -125 : 140, 0, width, height);
        controls.target.set(x, y, state.data.qpos[2]);
      }
    } else {
      if (state.lastAppliedTerrainMode === 'hillside') {
        camera.zoom = 1;
        camera.clearViewOffset();
      }
      controls.target.set(state.terrain.spawn.x + 0.8, 0, state.terrain.spawn.z + 0.7);
    }
    controls.update();
  }
  state.lastAppliedTerrainMode = state.terrainMode;
  persistTerrainPrefs();
}

function updateBodyTransforms() {
  const { data, bodyNodes } = state;
  for (let i = 0; i < bodyNodes.length; i++) {
    const node = bodyNodes[i];
    if (!node) continue;
    node.position.set(data.xpos[i * 3], data.xpos[i * 3 + 1], data.xpos[i * 3 + 2]);
    node.quaternion.set(data.xquat[i * 4 + 1], data.xquat[i * 4 + 2], data.xquat[i * 4 + 3], data.xquat[i * 4]);
  }
}

// ---------------------------------------------------------------------------
// scene setup
// ---------------------------------------------------------------------------
const SCENE_THEMES = {
  dark: {
    clear: 0x0b0e13, fog: 0x0b0e13, gridMajor: 0x2a3642, gridMinor: 0x1b232c,
    sky: 0xa9c4e6, ground: 0x10141a, hemiIntensity: 1.35, keyIntensity: 2.4,
    fill: 0x2c3742, steel: 0x8c99a6, edge: 0x4a5e70, edgeOpacity: 0.65,
    steelMetalness: 0.55, steelRoughness: 0.42,
  },
  light: {
    clear: 0xe8edf4, fog: 0xe8edf4, gridMajor: 0x9aaec4, gridMinor: 0xc5d2e0,
    sky: 0xe3edfc, ground: 0x78889c, hemiIntensity: 1.15, keyIntensity: 2.0,
    fill: 0xb6c5d6, steel: 0xa1b1c2, edge: 0x5d728a, edgeOpacity: 0.65,
    steelMetalness: 0.2, steelRoughness: 0.55,
  },
};

function applySceneTheme(theme) {
  const palette = SCENE_THEMES[theme] || SCENE_THEMES.dark;
  state.sceneClear = palette.clear;
  if (state.terrainMaterials) {
    const { fill, steel, edge } = state.terrainMaterials;
    fill.color.setHex(palette.fill);
    steel.color.setHex(palette.steel);
    steel.metalness = palette.steelMetalness;
    steel.roughness = palette.steelRoughness;
    edge.color.setHex(palette.edge);
    edge.opacity = palette.edgeOpacity * (state.terrain?.hfield ? 0.55 : 1);
  }
  if (!scene || !state.hemiLight) return;
  renderer.setClearColor(state.sceneClear, 1);
  scene.fog.color.setHex(palette.fog);
  state.hemiLight.color.setHex(palette.sky);
  state.hemiLight.groundColor.setHex(palette.ground);
  state.hemiLight.intensity = palette.hemiIntensity;
  state.keyLight.intensity = palette.keyIntensity;
  const parent = gridHelper.parent;
  const visible = gridHelper.visible;
  gridHelper.removeFromParent();
  gridHelper.geometry.dispose();
  gridHelper.material.dispose();
  gridHelper = new THREE.GridHelper(40, 40, palette.gridMajor, palette.gridMinor);
  gridHelper.rotation.x = Math.PI / 2;
  gridHelper.position.z = -0.002;
  gridHelper.visible = visible;
  if (parent) parent.add(gridHelper);
}

document.addEventListener('themechange', (event) => applySceneTheme(event.detail.theme));

async function initThree() {
  const host = document.getElementById('viewport');
  renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1));
  renderer.setSize(host.clientWidth, host.clientHeight);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.setClearColor(state.sceneClear, 1);
  host.appendChild(renderer.domElement);
  try {
    const gl = renderer.getContext();
    const dbg = gl.getExtension('WEBGL_debug_renderer_info');
    state.perf.gl = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : (gl.getParameter(gl.RENDERER) || '');
    console.log('WebGL renderer:', state.perf.gl);
  } catch (_) {}

  scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x0b0e13, 18, 42);

  camera = new THREE.PerspectiveCamera(46, host.clientWidth / host.clientHeight, 0.05, 120);
  camera.up.set(0, 0, 1);
  camera.position.set(3.8, -4.2, 2.35);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(1.2, 0, 0.72);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 1.2;
  controls.maxDistance = 26;
  controls.maxPolarAngle = Math.PI * 0.495;
  controls.update();

  const hemi = new THREE.HemisphereLight(0xa9c4e6, 0x10141a, 1.35);
  state.hemiLight = hemi;
  scene.add(hemi);
  const key = new THREE.DirectionalLight(0xffffff, 2.4);
  state.keyLight = key;
  key.position.set(5, -7, 11);
  scene.add(key);

  gridHelper = new THREE.GridHelper(40, 40, 0x2a3642, 0x1b232c);
  gridHelper.rotation.x = Math.PI / 2;
  gridHelper.position.z = -0.002;
  scene.add(gridHelper);
  applySceneTheme(document.documentElement.dataset.theme);

  skeletonGroup = new THREE.Group();
  scene.add(skeletonGroup);

  depthCamera = new THREE.PerspectiveCamera(
    CAM_FOV_Y_DEG, DEPTH_W / DEPTH_H, 0.05, DEPTH_FAR,
  );
  depthCamera.up.set(0, 0, 1);
  depthTarget = new THREE.WebGLRenderTarget(DEPTH_W, DEPTH_H, {
    minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter,
    format: THREE.RGBAFormat,
    type: THREE.UnsignedByteType,
    depthBuffer: true,
    stencilBuffer: false,
  });
  depthPixels = new Uint8Array(DEPTH_W * DEPTH_H * 4);
  cameraTarget = new THREE.WebGLRenderTarget(DEPTH_W, DEPTH_H, {
    minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter,
    format: THREE.RGBAFormat,
    type: THREE.UnsignedByteType,
    depthBuffer: true,
    stencilBuffer: false,
  });
  cameraTarget.texture.colorSpace = THREE.SRGBColorSpace;
  cameraPixels = new Uint8Array(DEPTH_W * DEPTH_H * 4);
  depthMaterial = new THREE.ShaderMaterial({
    uniforms: { far: { value: DEPTH_FAR } },
    vertexShader: `
      varying float vDepth;
      void main() {
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        vDepth = -mv.z;
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: `
      varying float vDepth;
      uniform float far;
      vec4 packDepth(float v) {
        const vec4 bitShift = vec4(1.0, 255.0, 65025.0, 16581375.0);
        const vec4 bitMask = vec4(1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0, 0.0);
        vec4 res = fract(v * bitShift);
        res -= res.xxyz * bitMask;
        res.a = 1.0;
        return res;
      }
      void main() {
        gl_FragColor = packDepth(clamp(vDepth / far, 0.0, 1.0));
      }`,
    side: THREE.DoubleSide,
    toneMapped: false,
  });

  window.addEventListener('resize', () => {
    const w = host.clientWidth;
    const h = host.clientHeight;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  });
}

async function loadRobotVisuals(base = state.assetBase) {
  const { config, template } = state;
  const geoms = collectVisualGeoms(template);
  if (state.bodyNodes?.length) {
    const disposedGeo = new Set();
    const disposedMat = new Set();
    for (const node of state.bodyNodes) {
      node.traverse((o) => {
        if (o.geometry && !disposedGeo.has(o.geometry)) { disposedGeo.add(o.geometry); o.geometry.dispose(); }
        if (o.material && !disposedMat.has(o.material)) { disposedMat.add(o.material); o.material.dispose(); }
      });
      scene.remove(node);
    }
  }
  state.bodyNodes = [];
  state.proxyMeshes = [];
  const unique = Array.from(new Set(geoms.map((g) => g.mesh)));
  const loadSet = async (dir) => {
    const loader = new STLLoader();
    const map = new Map();
    await Promise.all(unique.map(async (file) => {
      try {
        const geometry = await loader.loadAsync(`${dir}${file}`);
        geometry.computeVertexNormals();
        map.set(file, geometry);
      } catch (err) {
        if (!dir.endsWith('/meshes/')) console.warn(`low-poly mesh missing: ${file}`);
        else console.warn(`mesh load failed: ${file}`, err);
      }
    }));
    return map;
  };
  const [highSet, lowSet] = await Promise.all([
    loadSet(`${base}meshes/`),
    loadSet(`${base}meshes_low/`),
  ]);

  state.visualMeshesHigh = [];
  state.visualMeshesLow = [];
  const bodyMaterial = new THREE.MeshLambertMaterial({ color: 0xaebccd, side: THREE.DoubleSide });
  const torsoMaterial = new THREE.MeshLambertMaterial({ color: 0xd4deea, side: THREE.DoubleSide });

  state.bodyNodes = config.body_names.map((name, id) => {
    const node = new THREE.Group();
    node.name = name;
    node.userData.bodyId = id;
    scene.add(node);
    return node;
  });

  for (const g of geoms) {
    const node = state.bodyNodes[config.body_names.indexOf(g.body)];
    if (!node) continue;
    const highGeo = highSet.get(g.mesh);
    const lowGeo = lowSet.get(g.mesh) || highGeo;
    const isTorso = g.body === 'torso_link';
    const addMesh = (geometry, target) => {
      if (!geometry) return;
      const mesh = new THREE.Mesh(geometry, isTorso ? torsoMaterial : bodyMaterial);
      mesh.position.set(g.pos[0], g.pos[1], g.pos[2]);
      mesh.quaternion.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]);
      mesh.scale.set(g.scale[0], g.scale[1], g.scale[2]);
      node.add(mesh);
      target.push(mesh);
    };
    addMesh(highGeo, state.visualMeshesHigh);
    addMesh(lowGeo, state.visualMeshesLow);
  }
}

// ---------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------
function qs(id) { return document.getElementById(id); }

function syncSim2RealUi() {
  const s = state.sim2real;
  const set = (id, value, text) => {
    const el = qs(id);
    if (el) el.value = String(value);
    const label = qs(`${id}-value`);
    if (label) label.textContent = text;
  };
  set('s2r-friction', s.friction, s.friction.toFixed(2));
  set('s2r-damping', s.damping, s.damping.toFixed(2));
  set('s2r-delay', s.actuatorDelayMs, `${Math.round(s.actuatorDelayMs)} ms`);
  set('s2r-zero', s.encoderZero * 180 / Math.PI, `${(s.encoderZero * 180 / Math.PI).toFixed(1)}°`);
  set('s2r-pitch', s.imuPitchBiasDeg, `${s.imuPitchBiasDeg.toFixed(1)}°`);
  set('s2r-gyro', s.gyroBias * 180 / Math.PI, `${(s.gyroBias * 180 / Math.PI).toFixed(1)}°/s`);
  set('s2r-velfilter', s.jointVelFilterMs, `${Math.round(s.jointVelFilterMs)} ms`);
  set('s2r-backlash', s.backlash * 180 / Math.PI, `${(s.backlash * 180 / Math.PI).toFixed(1)}°`);
  const shoe = qs('s2r-shoe');
  if (shoe) shoe.checked = Boolean(s.shoe);
}

function setupUi() {
  const cfg = state.config;
  updateCheckpointUi();

  const dir = qs('terrain-dir');
  for (const d of DIRECTIONS) {
    const opt = document.createElement('option');
    opt.value = d;
    opt.textContent = d === 'flat' ? 'flat ground' : d;
    dir.appendChild(opt);
  }
  dir.value = state.direction;
  qs('terrain-level').value = String(state.level);
  qs('level-value').textContent = String(state.level);

  const setCommand = () => {
    state.commandScale = {
      vx: Number(qs('cmd-vx').value),
      vy: Number(qs('cmd-vy').value),
      wz: Number(qs('cmd-wz').value),
    };
    qs('cmd-vx-value').textContent = Number(qs('cmd-vx').value).toFixed(2);
    qs('cmd-vy-value').textContent = Number(qs('cmd-vy').value).toFixed(2);
    qs('cmd-wz-value').textContent = Number(qs('cmd-wz').value).toFixed(2);
  };
  for (const id of ['cmd-vx', 'cmd-vy', 'cmd-wz']) {
    qs(id).addEventListener('input', setCommand);
  }
  setCommand();

  dir.addEventListener('change', () => {
    const flat = dir.value === 'flat';
    qs('level-row').style.opacity = flat ? '0.35' : '1';
    qs('terrain-level').disabled = flat;
    applyTerrain(dir.value, Number(qs('terrain-level').value));
    refreshTerrainLabel();
  });
  qs('terrain-level').addEventListener('input', () => {
    qs('level-value').textContent = qs('terrain-level').value;
  });
  qs('terrain-level').addEventListener('change', () => {
    applyTerrain(dir.value, Number(qs('terrain-level').value));
    refreshTerrainLabel();
  });

  const mode = qs('terrain-mode');
  mode.addEventListener('change', () => {
    state.terrainMode = mode.value;
    applyTerrain(state.direction, state.level);
  });

  const setUniform = () => {
    state.uniform = {
      riser: Number(qs('uniform-riser').value) / 100,
      tread: Number(qs('uniform-tread').value) / 100,
      steps: Number(qs('uniform-steps').value),
      width: Number(qs('uniform-width')?.value ?? state.uniform.width),
    };
    qs('uniform-riser-value').textContent = `${Number(qs('uniform-riser').value).toFixed(1)} cm`;
    qs('uniform-tread-value').textContent = `${Number(qs('uniform-tread').value).toFixed(0)} cm`;
    qs('uniform-steps-value').textContent = String(qs('uniform-steps').value);
    if (qs('uniform-width-value')) qs('uniform-width-value').textContent = `${state.uniform.width.toFixed(1)} m`;
  };
  for (const id of ['uniform-riser', 'uniform-tread', 'uniform-steps', 'uniform-width']) {
    if (!qs(id)) continue;
    qs(id).addEventListener('input', setUniform);
    qs(id).addEventListener('change', () => {
      setUniform();
      if (state.terrainMode === 'uniform') applyTerrain(state.direction, state.level);
    });
  }
  setUniform();

  const setSim2Real = () => {
    const s = state.sim2real;
    s.friction = Number(qs('s2r-friction').value);
    s.damping = Number(qs('s2r-damping').value);
    s.actuatorDelayMs = Number(qs('s2r-delay').value);
    s.encoderZero = Number(qs('s2r-zero').value) * Math.PI / 180;
    s.imuPitchBiasDeg = Number(qs('s2r-pitch').value);
    s.gyroBias = Number(qs('s2r-gyro').value) * Math.PI / 180;
    s.jointVelFilterMs = Number(qs('s2r-velfilter').value);
    s.backlash = Number(qs('s2r-backlash').value) * Math.PI / 180;
    for (let i = 0; i < 29; i++) s.zeroOffsets[i] = Math.sin(i * 1.7) * s.encoderZero;
    applySim2RealModel();
    syncSim2RealUi();
  };
  for (const id of ['s2r-friction', 's2r-damping', 's2r-delay', 's2r-zero', 's2r-pitch', 's2r-gyro', 's2r-velfilter', 's2r-backlash']) {
    if (qs(id)) qs(id).addEventListener('input', setSim2Real);
  }
  if (qs('s2r-shoe')) {
    qs('s2r-shoe').addEventListener('change', () => {
      state.sim2real.shoe = Boolean(qs('s2r-shoe').checked);
      applyTerrain(state.direction, state.level);
    });
  }
  syncSim2RealUi();

  const setGap = () => {
    state.gap = {
      width: Number(qs('gap-width').value) / 100,
      depth: Number(qs('gap-depth').value) / 100,
      center: Number(qs('gap-center').value),
    };
    qs('gap-width-value').textContent = `${Number(qs('gap-width').value).toFixed(0)} cm`;
    qs('gap-depth-value').textContent = `${Number(qs('gap-depth').value).toFixed(0)} cm`;
    qs('gap-center-value').textContent = `${Number(qs('gap-center').value).toFixed(1)} m`;
  };
  for (const id of ['gap-width', 'gap-depth', 'gap-center']) {
    qs(id).addEventListener('input', setGap);
    qs(id).addEventListener('change', () => {
      setGap();
      if (state.terrainMode === 'gap') applyTerrain(state.direction, state.level);
    });
  }
  setGap();

  const setPlatform = () => {
    state.platform = {
      height: Number(qs('platform-height').value) / 100,
      start: Number(qs('platform-start').value),
    };
    qs('platform-height-value').textContent = `${Number(qs('platform-height').value).toFixed(0)} cm`;
    qs('platform-start-value').textContent = `${Number(qs('platform-start').value).toFixed(1)} m`;
  };
  for (const id of ['platform-height', 'platform-start']) {
    qs(id).addEventListener('input', setPlatform);
    qs(id).addEventListener('change', () => {
      setPlatform();
      if (state.terrainMode === 'platform') applyTerrain(state.direction, state.level);
    });
  }
  setPlatform();

  const setGrating = () => {
    state.grating = {
      riser: Number(qs('grating-riser').value) / 100,
      tread: Number(qs('grating-tread').value) / 100,
      steps: Number(qs('grating-steps').value),
      width: Number(qs('grating-width').value),
      bar: Number(qs('grating-bar').value) / 100,
      gap: Number(qs('grating-gap').value) / 100,
    };
    qs('grating-riser-value').textContent = `${Number(qs('grating-riser').value).toFixed(0)} cm`;
    qs('grating-tread-value').textContent = `${Number(qs('grating-tread').value).toFixed(0)} cm`;
    qs('grating-steps-value').textContent = String(qs('grating-steps').value);
    qs('grating-width-value').textContent = `${Number(qs('grating-width').value).toFixed(1)} m`;
    qs('grating-bar-value').textContent = `${Number(qs('grating-bar').value).toFixed(1)} cm`;
    qs('grating-gap-value').textContent = `${(Number(qs('grating-gap').value)).toFixed(1)} cm`;
  };
  for (const id of ['grating-riser', 'grating-tread', 'grating-steps', 'grating-width', 'grating-bar', 'grating-gap']) {
    qs(id).addEventListener('input', setGrating);
    qs(id).addEventListener('change', () => {
      setGrating();
      if (state.terrainMode === 'grating') applyTerrain(state.direction, state.level);
    });
  }
  setGrating();

  const setSlope = () => {
    state.slope = {
      angle: Number(qs('slope-angle').value),
      run: Number(qs('slope-run').value),
      width: Number(qs('slope-width').value),
    };
    qs('slope-angle-value').textContent = `${Number(qs('slope-angle').value).toFixed(1)}°`;
    qs('slope-run-value').textContent = `${Number(qs('slope-run').value).toFixed(2)} m`;
    qs('slope-width-value').textContent = `${Number(qs('slope-width').value).toFixed(1)} m`;
  };
  for (const id of ['slope-angle', 'slope-run', 'slope-width']) {
    qs(id).addEventListener('input', setSlope);
    qs(id).addEventListener('change', () => {
      setSlope();
      if (state.terrainMode === 'slope') applyTerrain(state.direction, state.level);
    });
  }
  setSlope();

  qs('btn-play').addEventListener('click', () => {
    state.running = !state.running;
    qs('btn-play').textContent = state.running ? 'Pause' : 'Resume';
    qs('btn-play').dataset.running = String(state.running);
  });
  qs('btn-reset').addEventListener('click', () => resetSim());
  qs('btn-check').addEventListener('click', async () => {
    qs('btn-check').disabled = true;
    qs('check-result').textContent = 'running 250 steps...';
    try {
      resetSim();
      const result = await runCheck({ steps: 250 });
      qs('check-result').textContent =
        `${result.fell ? 'FELL' : 'OK'} · dz ${result.deltaZ.toFixed(3)} m · dx ${result.deltaX.toFixed(2)} m · clearance ${result.end.minClearance.toFixed(2)} m`;
    } catch (err) {
      qs('check-result').textContent = `check failed: ${err.message}`;
    } finally {
      qs('btn-check').disabled = false;
    }
  });
  const applyQuality = () => {
    const quality = qs('quality').value;
    const dpr = window.devicePixelRatio || 1;
    let scale = 1;
    if (quality === 'performance') scale = 0.7;
    else if (quality === 'balanced') scale = Math.min(dpr, 1);
    else if (quality === 'quality') scale = Math.max(1, Math.min(dpr, 2));
    state.lowPoly = quality === 'performance';
    renderer.setPixelRatio(scale);
    const host = qs('viewport');
    renderer.setSize(host.clientWidth, host.clientHeight);
    state.perf.pixelRatio = scale;
    applyRobotVisibility();
  };
  qs('quality').addEventListener('change', applyQuality);
  applyQuality();
  if (/swiftshader|llvmpipe|software/i.test(state.perf.gl)) {
    qs('quality').value = 'performance';
    applyQuality();
    state.renderNoteBase = 'software WebGL → performance mode';
    updateRenderNote();
  } else if (state.perf.gl) {
    state.renderNoteBase = `GPU: ${state.perf.gl.slice(0, 46)}`;
    updateRenderNote();
  }

  qs('speed').addEventListener('input', () => {
    state.speed = Number(qs('speed').value);
    qs('speed-value').textContent = `${state.speed.toFixed(2)}×`;
  });
  qs('follow').addEventListener('change', () => { state.follow = qs('follow').checked; });
  const depthDisplay = qs('depth-display');
  if (depthDisplay) {
    depthDisplay.value = state.depthDisplay;
    const applyDepthDisplay = () => {
      state.depthDisplay = depthDisplay.value;
      state.depthVisRange = null;
      const profile = qs('depth-profile');
      if (profile) profile.hidden = state.depthDisplay !== 'analysis';
      const bar = qs('depth-bar');
      if (bar) {
        bar.style.background = state.depthDisplay === 'real'
          ? 'linear-gradient(90deg,#ffffff,#000000)'
          : 'linear-gradient(90deg,#ff6450,#f0b45c,#2db4a5,#1e5ca0,#142860)';
      }
      drawDepthPreview();
    };
    depthDisplay.addEventListener('change', applyDepthDisplay);
    const profile = qs('depth-profile');
    if (profile) profile.hidden = state.depthDisplay !== 'analysis';
    const bar = qs('depth-bar');
    if (bar) {
      bar.style.background = state.depthDisplay === 'real'
        ? 'linear-gradient(90deg,#ffffff,#000000)'
        : 'linear-gradient(90deg,#ff6450,#f0b45c,#2db4a5,#1e5ca0,#142860)';
    }
  }
  const showCamera = qs('show-camera');
  if (showCamera) {
    showCamera.addEventListener('change', () => {
      const wrap = qs('camera-wrap');
      if (wrap) wrap.hidden = !showCamera.checked;
    });
  }
  const lockWaist = qs('lock-waist');
  if (lockWaist) {
    lockWaist.checked = Boolean(state.lockWaist);
    lockWaist.addEventListener('change', () => {
      state.lockWaist = lockWaist.checked;
      applyEqualityActivity();
      persistTerrainPrefs();
    });
  }
  refreshTerrainLabel();

  for (const name of TERMS) {
    state.histories[name] = new HistoryBuffer(TERM_WIDTH[name]);
  }
  setupCheckpointUi();
}

function updateCheckpointUi() {
  const cfg = state.config;
  qs('ckpt-iter').textContent = `iter ${cfg.checkpoint.iteration}`;
  qs('ckpt-sha').textContent = cfg.checkpoint.sha256.slice(0, 12);
  qs('ckpt-name').textContent = cfg.checkpoint.label || cfg.checkpoint.path.split('/').pop();
  if (qs('ckpt-current-label')) {
    qs('ckpt-current-label').textContent = `${cfg.checkpoint.label || 'checkpoint'} · ${cfg.checkpoint.iteration}`;
  }
  const terms = cfg.observation_terms.map((t) => `${t.name}:${t.width}×${t.history}`).join('  ');
  qs('obs-layout').textContent = `${terms}  depth:36×36 → ${cfg.observation_dim}`;
}

// ---------------------------------------------------------------------------
// checkpoint browser / export jobs
// ---------------------------------------------------------------------------
function setupCheckpointUi() {
  const modal = qs('ckpt-modal');
  if (!modal) return;
  const closeModal = () => {
    modal.hidden = true;
    qs('ckpt-browse').focus();
  };
  modal.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      event.preventDefault();
      closeModal();
    } else if (event.key === 'Tab') {
      const focusable = Array.from(modal.querySelector('.modal-card').querySelectorAll(
        'button, input, select, textarea, a[href], [tabindex]',
      )).filter((element) => !element.disabled && element.tabIndex >= 0 && element.getClientRects().length);
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
  });
  qs('ckpt-close').addEventListener('click', closeModal);
  qs('ckpt-cancel').addEventListener('click', closeModal);
  modal.addEventListener('click', (e) => { if (e.target === modal) closeModal(); });
  qs('ckpt-browse').addEventListener('click', async () => {
    modal.hidden = false;
    qs('ckpt-close').focus();
    if (!qs('ckpt-dir').value.trim()) {
      try {
        const info = await fetch('/api/info', { cache: 'no-store' }).then((r) => r.json());
        qs('ckpt-dir').value = info.defaultDir || '';
      } catch (_) { /* keep empty */ }
    }
    scanCheckpointDir();
    refreshUploads();
  });
  qs('ckpt-load').addEventListener('click', async () => {
    const id = qs('ckpt-select').value;
    qs('ckpt-status').textContent = 'loading...';
    try {
      const cp = await hotLoadExport(id);
      qs('ckpt-status').textContent = `loaded ${cp.iteration} · ${cp.sha256.slice(0, 8)}`;
    } catch (err) {
      qs('ckpt-status').textContent = `load failed: ${err.message}`;
    }
  });
  for (const tab of ['server', 'upload']) {
    qs(`tab-${tab}`).addEventListener('click', () => {
      for (const name of ['server', 'upload']) {
        qs(`tab-${name}`).classList.toggle('active', name === tab);
        qs(`pane-${name}`).hidden = name !== tab;
      }
    });
  }
  qs('ckpt-scan').addEventListener('click', scanCheckpointDir);
  qs('ckpt-upload').addEventListener('click', uploadCheckpoint);
  qs('ckpt-export').addEventListener('click', startCheckpointExport);
  qs('ckpt-export').disabled = true;
  populateExportSelect();
}

function shortRunName(name) {
  if (!name) return '';
  const m = /stage_(\d+)/.exec(name);
  if (m) return `stage ${m[1]}`;
  return name.length > 26 ? `${name.slice(0, 24)}…` : name;
}

function checkpointOptionLabel(item) {
  const iter = item.iteration != null ? `iter ${item.iteration}` : 'latest';
  const name = item.displayName || item.name;
  const run = item.source === 'upload' ? '' : shortRunName(item.runName || '');
  return `${name} · ${run ? `${run} · ` : ''}${iter} · ${(item.size / 2 ** 20).toFixed(0)} MB`;
}

function renderCheckpointList(containerId, items, kind) {
  const root = qs(containerId);
  if (!root) return;
  root.replaceChildren();
  if (!items.length) {
    const empty = document.createElement('div');
    empty.className = 'check-line';
    empty.style.padding = '10px';
    empty.textContent = kind === 'upload' ? 'no uploaded checkpoints' : 'no .pt checkpoints found in this directory';
    root.appendChild(empty);
    return;
  }
  for (const item of items) {
    const row = document.createElement('button');
    row.type = 'button';
    row.className = 'ckpt-item';
    const name = document.createElement('div');
    name.className = 'name';
    name.textContent = checkpointOptionLabel(item);
    const meta = document.createElement('div');
    meta.className = 'meta';
    const bits = [];
    if (item.runName && item.source !== 'upload') bits.push(item.runName);
    if (!item.hasParams && kind === 'server') bits.push('config falls back to active run');
    if (item.source === 'upload') bits.push('uploaded from browser');
    meta.textContent = bits.join(' · ') || item.path;
    row.append(name, meta);
    row.title = item.path;
    row.addEventListener('click', () => selectCheckpointItem(item, row, root, kind));
    root.appendChild(row);
  }
}

function selectCheckpointItem(item, row, root, kind) {
  for (const child of root.children) child.classList?.remove('selected');
  row.classList.add('selected');
  state.ckptChoice = { ...item, kind };
  qs('ckpt-export').disabled = false;
  qs('ckpt-export-status').textContent =
    `${item.label || item.name} · ${(item.size / 2 ** 20).toFixed(0)} MB · ready to export`;
}

async function scanCheckpointDir() {
  const dir = qs('ckpt-dir').value.trim();
  const note = qs('ckpt-dir-note');
  if (!dir) { note.textContent = 'enter an absolute server directory'; return; }
  note.textContent = 'scanning...';
  try {
    const res = await fetch(`/api/checkpoints?dir=${encodeURIComponent(dir)}`, { cache: 'no-store' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `scan failed (${res.status})`);
    renderCheckpointList('ckpt-server-list', data.checkpoints || [], 'server');
    note.textContent = `${(data.checkpoints || []).length} checkpoint(s) found`;
    note.title = data.dir;
    if (data.uploads) renderCheckpointList('ckpt-upload-list', data.uploads, 'upload');
  } catch (err) {
    note.textContent = err.message;
    renderCheckpointList('ckpt-server-list', [], 'server');
  }
}

async function refreshUploads() {
  try {
    const data = await fetch('/api/uploads', { cache: 'no-store' }).then((r) => r.json());
    renderCheckpointList('ckpt-upload-list', data.uploads || [], 'upload');
    return data.uploads || [];
  } catch (_) {
    return [];
  }
}

async function uploadCheckpoint() {
  const file = qs('ckpt-file').files?.[0];
  const note = qs('ckpt-upload-note');
  if (!file) { note.textContent = 'choose a .pt file first'; return; }
  note.textContent = `uploading ${file.name} (${(file.size / 2 ** 20).toFixed(0)} MB)...`;
  try {
    const res = await fetch('/api/upload', {
      method: 'POST',
      headers: { 'X-Filename': encodeURIComponent(file.name) },
      body: file,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `upload failed (${res.status})`);
    note.textContent = `uploaded ${data.originalName || data.name}`;
    const uploads = await refreshUploads();
    const item = uploads.find((u) => u.name === data.name) || data;
    const root = qs('ckpt-upload-list');
    const rows = Array.from(root.children);
    const row = rows.find((r) => r.title === item.path);
    if (row) selectCheckpointItem(item, row, root, 'upload');
  } catch (err) {
    note.textContent = err.message;
  }
}

async function populateExportSelect() {
  const select = qs('ckpt-select');
  if (!select) return;
  let defaultIter = state.config.checkpoint.iteration;
  try {
    const c = await fetch('/config.json', { cache: 'no-store' }).then((r) => r.json());
    defaultIter = c.checkpoint.iteration;
  } catch (_) { /* keep */ }
  const options = [{ id: 'default', label: `public · ${defaultIter}` }];
  try {
    const data = await fetch('/api/exports', { cache: 'no-store' }).then((r) => r.json());
    for (const e of data.exports || []) {
      const label = e.checkpoint?.label || e.id;
      const shortLabel = label.length > 18 ? `${label.slice(0, 17)}…` : label;
      options.push({
        id: e.id,
        label: `iter ${e.checkpoint?.iteration ?? '?'} · ${shortLabel}`,
      });
    }
  } catch (_) { /* API unavailable */ }
  const current = state.exportId || 'default';
  select.replaceChildren(...options.map((o) => {
    const opt = document.createElement('option');
    opt.value = o.id;
    opt.textContent = o.label;
    return opt;
  }));
  select.value = options.some((o) => o.id === current) ? current : 'default';
}

async function hotLoadExport(id) {
  const base = id === 'default' ? '/' : `/exports/${id}/`;
  if (state.session && base === state.assetBase) return state.config.checkpoint;
  const wasRunning = state.running;
  const previous = { config: state.config, template: state.template, base: state.assetBase, exportId: state.exportId };
  state.running = false;
  state.checking = true;
  setStatus(`loading ${id === 'default' ? 'default export' : id}...`);
  try {
    const [config, template] = await Promise.all([
      fetch(`${base}config.json`, { cache: 'no-store' }).then((r) => {
        if (!r.ok) throw new Error(`config.json ${r.status}`);
        return r.json();
      }),
      fetch(`${base}robot_template.xml`, { cache: 'no-store' }).then((r) => {
        if (!r.ok) throw new Error(`robot_template.xml ${r.status}`);
        return r.text();
      }),
    ]);
    if (!config.checkpoint || Number(config.observation_dim) !== SUPPORTED_OBS) {
      throw new Error(`unsupported export layout: observation_dim=${config.observation_dim}`);
    }
    const session = await ort.InferenceSession.create(
      `${base}policy.onnx?v=${config.checkpoint.sha256.slice(0, 12)}`,
      { executionProviders: ['wasm'], graphOptimizationLevel: 'all' },
    );
    const sourceChanged = config.checkpoint.source_sha !== state.config.checkpoint.source_sha;
    const oldSession = state.session;
    state.config = config;
    state.template = template;
    state.assetBase = base;
    state.exportId = id === 'default' ? null : id;
    state.controlDt = Number(config.engine.control_dt ?? config.engine.step_dt ?? 0.02);
    if (sourceChanged) await loadRobotVisuals(base);
    state.session = session;
    if (oldSession && oldSession !== session) { try { oldSession.release?.(); } catch (_) {} }
    applyTerrain(state.direction, state.level);
    updateCheckpointUi();
    await populateExportSelect();
    if (id === 'default') localStorage.removeItem('preview.export');
    else localStorage.setItem('preview.export', id);
    setStatus(`ready · checkpoint ${config.checkpoint.iteration} · ${CORES} wasm threads`);
    return config.checkpoint;
  } catch (err) {
    state.config = previous.config;
    state.template = previous.template;
    state.assetBase = previous.base;
    state.exportId = previous.exportId;
    setStatus(`load failed: ${err.message}`, true);
    throw err;
  } finally {
    state.checking = false;
    state.running = wasRunning;
    const play = qs('btn-play');
    if (play) {
      play.textContent = state.running ? 'Pause' : 'Resume';
      play.dataset.running = String(state.running);
    }
  }
}

let exportPollTimer = null;

async function startCheckpointExport() {
  const choice = state.ckptChoice;
  const status = qs('ckpt-export-status');
  const log = qs('ckpt-log');
  if (!choice) { status.textContent = 'select a checkpoint first'; return; }
  if (exportPollTimer) { clearInterval(exportPollTimer); exportPollTimer = null; }
  qs('ckpt-export').disabled = true;
  log.textContent = 'starting exporter...';
  status.textContent = 'starting exporter...';
  const body = choice.source === 'upload'
    ? { uploadId: choice.name, label: choice.displayName || choice.name }
    : {
        checkpoint: choice.path,
        runDir: choice.hasParams ? choice.runDir : null,
        label: choice.name,
      };
  try {
    const res = await fetch('/api/export', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `export failed (${res.status})`);
    if (data.state === 'done' || data.cached) {
      status.textContent = 'using cached export';
      await finishCheckpointExport(data.id);
      return;
    }
    status.textContent = `exporting ${data.id}...`;
    exportPollTimer = setInterval(() => pollExportJob(data.id), 900);
  } catch (err) {
    status.textContent = err.message;
    log.textContent = err.message;
    qs('ckpt-export').disabled = false;
  }
}

async function pollExportJob(id) {
  try {
    const res = await fetch(`/api/export/${encodeURIComponent(id)}`, { cache: 'no-store' });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `status ${res.status}`);
    if (data.log) {
      const log = qs('ckpt-log');
      log.textContent = data.log;
      log.scrollTop = log.scrollHeight;
    }
    if (data.state === 'done') {
      clearInterval(exportPollTimer); exportPollTimer = null;
      qs('ckpt-export-status').textContent = 'export done, loading...';
      await finishCheckpointExport(id);
    } else if (data.state === 'error') {
      clearInterval(exportPollTimer); exportPollTimer = null;
      qs('ckpt-export-status').textContent = `export failed: ${(data.error || 'see log').split('\n').slice(-1)[0]}`;
      qs('ckpt-export').disabled = false;
    } else {
      qs('ckpt-export-status').textContent = `exporting... ${(data.elapsed / 1000).toFixed(0)} s`;
    }
  } catch (err) {
    clearInterval(exportPollTimer); exportPollTimer = null;
    qs('ckpt-export-status').textContent = `status check failed: ${err.message}`;
    qs('ckpt-export').disabled = false;
  }
}

async function finishCheckpointExport(id) {
  try {
    const cp = await hotLoadExport(id);
    qs('ckpt-export-status').textContent = `loaded iter ${cp.iteration} · ${cp.sha256.slice(0, 8)}`;
    qs('ckpt-status').textContent = `loaded ${cp.iteration} · ${cp.sha256.slice(0, 8)}`;
    qs('ckpt-log').textContent = `loaded ${cp.path}`;
    setTimeout(() => { qs('ckpt-close').click(); }, 900);
  } catch (err) {
    qs('ckpt-export-status').textContent = `exported but load failed: ${err.message}`;
  } finally {
    qs('ckpt-export').disabled = false;
  }
}

function syncTerrainUi() {
  const dir = qs('terrain-dir');
  if (dir) dir.value = state.direction;
  const mode = qs('terrain-mode');
  if (mode) mode.value = state.terrainMode;
  const type = state.terrainMode;
  const flat = type === 'flat' || ((type === 'uniform' || type === 'curriculum') && state.direction === 'flat');
  const stairs = type === 'uniform' || type === 'curriculum';
  const curriculum = type === 'curriculum' && !flat;
  const uniform = type === 'uniform' && !flat;
  const gap = type === 'gap';
  const platform = type === 'platform';
  const grating = type === 'grating';
  const slope = type === 'slope';
  const level = qs('terrain-level');
  if (level) {
    level.value = String(state.level);
    level.disabled = flat;
    qs('level-value').textContent = String(state.level);
    qs('level-row').style.display = curriculum || type === 'hillside' ? '' : 'none';
    qs('level-row').style.opacity = flat ? '0.35' : '1';
  }
  qs('hillside-hint').hidden = type !== 'hillside';
  const dirRow = qs('direction-row');
  if (dirRow) dirRow.style.display = stairs ? '' : 'none';
  const uniformRow = qs('uniform-row');
  if (uniformRow) uniformRow.style.display = uniform ? '' : 'none';
  const stepsRow = qs('uniform-steps-row');
  if (stepsRow) stepsRow.style.display = uniform ? '' : 'none';
  const widthRow = qs('uniform-width-row');
  if (widthRow) widthRow.style.display = uniform ? '' : 'none';
  const gapRow = qs('gap-row');
  if (gapRow) gapRow.style.display = gap ? '' : 'none';
  const gapCenter = qs('gap-center-row');
  if (gapCenter) gapCenter.style.display = gap ? '' : 'none';
  const platformRow = qs('platform-row');
  if (platformRow) platformRow.style.display = platform ? '' : 'none';
  for (const id of ['grating-row1', 'grating-row2', 'grating-row3']) {
    const row = qs(id);
    if (row) row.style.display = grating ? '' : 'none';
  }
  for (const id of ['slope-row1', 'slope-row2']) {
    const row = qs(id);
    if (row) row.style.display = slope ? '' : 'none';
  }
  refreshTerrainLabel();
}

function refreshTerrainLabel() {
  const t = state.terrain;
  const spec = t?.spec;
  if (!spec) {
    qs('terrain-meta').textContent = 'flat';
    qs('terrain-meta').title = 'flat ground';
    return;
  }
  if (spec.hillside) {
    const text = `difficulty ${spec.difficulty.toFixed(2)} · peaks ${spec.peak_m.toFixed(2)} m · 9.2 × 4.0 m`;
    qs('terrain-meta').textContent = text;
    qs('terrain-meta').title = text;
    qs('hillside-hint').textContent = text;
    return;
  }
  if (spec.slope) {
    qs('terrain-meta').textContent =
      `slope ${spec.slope_angle_deg.toFixed(1)}° · run ${spec.run_m.toFixed(1)} m`;
    qs('terrain-meta').title =
      `slope: ${spec.slope_angle_deg.toFixed(1)}°, up/down ramps ${spec.run_m.toFixed(2)} m each, ` +
      `rise ${spec.rise_m.toFixed(3)} m, width ${spec.width_m.toFixed(2)} m, ` +
      `middle platform ${spec.middle_platform_m.toFixed(2)} m`;
    return;
  }
  if (spec.grating) {
    qs('terrain-meta').textContent =
      `grating ${(spec.riser_height_m * 100).toFixed(0)}/${(spec.step_length_m * 100).toFixed(0)} cm · ${spec.steps} st`;
    qs('terrain-meta').title =
      `steel grating: riser ${(spec.riser_height_m * 100).toFixed(1)} cm, tread ${(spec.step_length_m * 100).toFixed(1)} cm, ` +
      `${spec.steps} steps, width ${spec.width_m.toFixed(2)} m, bars ${(spec.bar_width_m * 100).toFixed(0)} mm / ` +
      `gaps ${(spec.gap_width_m * 1000).toFixed(0)} mm, open risers`;
    return;
  }
  if (spec.gap) {
    qs('terrain-meta').textContent =
      `gap ${(spec.gap_width_m * 100).toFixed(0)}×${(spec.gap_depth_m * 100).toFixed(0)} cm @${spec.center_x_m.toFixed(1)} m`;
    qs('terrain-meta').title =
      `gap: width ${(spec.gap_width_m * 100).toFixed(1)} cm, depth ${(spec.gap_depth_m * 100).toFixed(1)} cm, ` +
      `centre x ${spec.center_x_m.toFixed(2)} m, full width ${spec.width_m.toFixed(1)} m`;
    return;
  }
  if (spec.platform) {
    qs('terrain-meta').textContent =
      `platform ${(spec.platform_height_m * 100).toFixed(0)} cm @${spec.start_x_m.toFixed(1)} m`;
    qs('terrain-meta').title =
      `platform: height ${(spec.platform_height_m * 100).toFixed(1)} cm, starts at x ${spec.start_x_m.toFixed(2)} m, ` +
      `width ${spec.width_m.toFixed(1)} m`;
    return;
  }
  if (spec.uniform) {
    qs('terrain-meta').textContent =
      `${(spec.riser_height_m * 100).toFixed(1)}/${(spec.step_length_m * 100).toFixed(0)} cm · ${spec.steps_per_run} st`;
    qs('terrain-meta').title =
      `uniform ${t.direction}: riser ${(spec.riser_height_m * 100).toFixed(2)} cm, ` +
      `tread ${(spec.step_length_m * 100).toFixed(1)} cm, ${spec.steps_per_run} steps per run ` +
      `(${spec.num_steps} total)`;
    return;
  }
  qs('terrain-meta').textContent = `L${t.level} · ${spec.num_steps} st · ${(spec.riser_height_m * 100).toFixed(1)} cm`;
  qs('terrain-meta').title =
    `${t.direction} level ${t.level}: ${spec.num_steps} steps, mean riser ${(spec.riser_height_m * 100).toFixed(2)} cm`;
}

function updateHud() {
  const { data, config } = state;
  const q = Array.from(data.qpos.slice(3, 7));
  const z = data.qpos[2];
  const yaw = yawFromQuat(q);
  const torsoZ = data.xpos[Number(config.torso_body) * 3 + 2];
  const forwardSpeed = Math.hypot(data.qvel[0], data.qvel[1]);
  if (state.steps > 0) {
    state.recentSpeeds.push(forwardSpeed);
    if (state.recentSpeeds.length > 50) state.recentSpeeds.shift();
  }
  const avgSpeed = state.recentSpeeds.length
    ? state.recentSpeeds.reduce((a, b) => a + b, 0) / state.recentSpeeds.length
    : 0;
  qs('stat-time').textContent = `${state.simTime.toFixed(2)} s`;
  qs('stat-steps').textContent = String(state.steps);
  qs('stat-z').textContent = `${z.toFixed(3)} m`;
  qs('stat-torso').textContent = `${torsoZ.toFixed(3)} m`;
  const ground = groundHeight(data.qpos[0], data.qpos[1]);
  qs('stat-clearance').textContent = `${(z - ground).toFixed(3)} m`;
  qs('stat-speed').textContent = `${avgSpeed.toFixed(2)} m/s`;
  qs('stat-xy').textContent = `${data.qpos[0].toFixed(2)}, ${data.qpos[1].toFixed(2)}`;
  qs('stat-yaw').textContent = `${(yaw * 180 / Math.PI).toFixed(1)}°`;
  qs('stat-contacts').textContent = String(data.ncon);
  qs('stat-infer').textContent = `${state.lastInferenceMs.toFixed(1)} ms`;
  qs('stat-fps').textContent = String(Math.round(fps));
  qs('stat-chz').textContent = `${state.perf.controlHz.toFixed(0)} Hz`;
  qs('stat-simrate').textContent = `${state.perf.simRate.toFixed(2)}×`;
  qs('stat-step').textContent = `${state.perf.stepMs.toFixed(1)} ms`;
  qs('stat-render').textContent = `${state.perf.renderMs.toFixed(1)} ms`;
  qs('stat-depth').textContent = `${state.perf.depthMs.toFixed(1)} ms`;
  qs('stat-dlag').textContent = `${state.perf.depthLagMs.toFixed(0)} ms`;
  if (qs('stat-threads')) qs('stat-threads').textContent = String(state.perf.cores);
  const maxAction = state.lastAction.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
  qs('stat-action').textContent = maxAction.toFixed(2);
  if (qs('cmd-active')) {
    qs('cmd-active').textContent =
      `active  vx ${state.command[0].toFixed(2)} · vy ${state.command[1].toFixed(2)} · wz ${state.command[2].toFixed(2)}`;
  }
}

// Near = warm (coral) -> far = deep blue. The policy still receives the raw
// normalized depth; this mapping only affects the preview.
const DEPTH_STOPS = [
  [0.00, 20, 40, 96],
  [0.30, 30, 92, 160],
  [0.55, 45, 180, 165],
  [0.78, 240, 180, 92],
  [1.00, 255, 100, 80],
];

function depthColor(t) {
  t = Math.max(0, Math.min(1, t));
  for (let i = 1; i < DEPTH_STOPS.length; i++) {
    const a = DEPTH_STOPS[i - 1];
    const b = DEPTH_STOPS[i];
    if (t <= b[0]) {
      const k = (t - a[0]) / Math.max(1e-6, b[0] - a[0]);
      return [
        Math.round(a[1] + (b[1] - a[1]) * k),
        Math.round(a[2] + (b[2] - a[2]) * k),
        Math.round(a[3] + (b[3] - a[3]) * k),
      ];
    }
  }
  return [255, 100, 80];
}

function renderCameraView() {
  const canvas = qs('camera-canvas');
  if (!canvas || !cameraTarget) return;
  const debugMain = new URLSearchParams(location.search).has('debugMain');
  const viewCamera = debugMain ? camera : depthCamera;
  if (!debugMain) {
    const pose = cameraPose();
    depthCamera.position.copy(pose.position);
    depthCamera.quaternion.copy(pose.quaternion);
    depthCamera.updateMatrixWorld(true);
  }
  const gridWas = gridHelper ? gridHelper.visible : false;
  if (gridHelper) gridHelper.visible = false;
  if (state.terrainEdges) for (const l of state.terrainEdges) l.visible = false;
  // The inset deliberately uses collision proxies: the visual torso shell can
  // enclose the camera and turn the inset black. The policy depth pass above
  // uses the high-detail visual meshes instead.
  if (state.visualMeshesHigh) for (const m of state.visualMeshesHigh) m.visible = false;
  if (state.visualMeshesLow) for (const m of state.visualMeshesLow) m.visible = false;
  if (state.proxyMeshes) for (const m of state.proxyMeshes) m.visible = true;
  if (state.terrainVisuals) for (const m of state.terrainVisuals) m.visible = true;
  const oldOverride = scene.overrideMaterial;
  const oldBg = scene.background;
  const headlight = new THREE.DirectionalLight(0xffffff, 2.4);
  headlight.position.copy(depthCamera.position);
  headlight.target.position.copy(depthCamera.position).add(
    new THREE.Vector3(0, 0, -1).applyQuaternion(depthCamera.quaternion),
  );
  scene.add(headlight);
  scene.add(headlight.target);
  scene.overrideMaterial = null;
  scene.background = null;
  renderer.setRenderTarget(cameraTarget);
  renderer.setClearColor(state.sceneClear, 1);
  renderer.clear();
  renderer.render(scene, viewCamera);
  renderer.readRenderTargetPixels(cameraTarget, 0, 0, DEPTH_W, DEPTH_H, cameraPixels);
  renderer.setRenderTarget(null);
  renderer.setClearColor(state.sceneClear, 1);
  scene.remove(headlight);
  scene.remove(headlight.target);
  headlight.dispose();
  scene.overrideMaterial = oldOverride;
  scene.background = oldBg;
  if (gridHelper) gridHelper.visible = gridWas;
  if (state.terrainEdges) for (const l of state.terrainEdges) l.visible = true;
  applyRobotVisibility();
  if (state.terrainVisuals) for (const m of state.terrainVisuals) m.visible = true;
  if (state.proxyMeshes) for (const m of state.proxyMeshes) m.visible = false;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(DEPTH_W, DEPTH_H);
  for (let row = 0; row < DEPTH_H; row++) {
    const srcRow = DEPTH_H - 1 - row;
    for (let col = 0; col < DEPTH_W; col++) {
      const si = (srcRow * DEPTH_W + col) * 4;
      const di = (row * DEPTH_W + col) * 4;
      img.data[di] = cameraPixels[si];
      img.data[di + 1] = cameraPixels[si + 1];
      img.data[di + 2] = cameraPixels[si + 2];
      img.data[di + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

function drawDepthPreview() {
  const canvas = qs('depth-canvas');
  if (!canvas || !state.depth) return;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(DEPTH_OUT, DEPTH_OUT);
  if (state.depthDisplay === 'real') {
    // Faithful policy depth rendering: fixed [0, 2.5] m, near=white, far=black.
    for (let i = 0; i < state.depth.length; i++) {
      const v = state.depth[i];
      const g = v >= 0.999 ? 0 : Math.round(255 * (1 - Math.min(1, Math.max(0, v))));
      img.data[i * 4] = g;
      img.data[i * 4 + 1] = g;
      img.data[i * 4 + 2] = g;
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    const rangeEl = qs('depth-range');
    if (rangeEl) rangeEl.textContent = '0–2.5 m';
    const profile = qs('depth-profile');
    if (profile) profile.hidden = true;
    return;
  }
  const profile = qs('depth-profile');
  if (profile) profile.hidden = false;
  let rawMin = 1;
  let rawMax = 0;
  let hits = 0;
  for (let i = 0; i < state.depth.length; i++) {
    const v = state.depth[i];
    if (v < 0.999) {
      if (v < rawMin) rawMin = v;
      if (v > rawMax) rawMax = v;
      hits += 1;
    }
  }
  let lo = 0;
  let hi = 1;
  if (hits) {
    lo = Math.max(0, rawMin - 0.02);
    hi = Math.min(1, rawMax + 0.02);
    if (hi - lo < 0.03) {
      const mid = (lo + hi) / 2;
      lo = Math.max(0, mid - 0.015);
      hi = Math.min(1, mid + 0.015);
    }
  }
  if (!state.depthVisRange) state.depthVisRange = { lo, hi };
  else {
    const smooth = 0.2;
    state.depthVisRange.lo += (lo - state.depthVisRange.lo) * smooth;
    state.depthVisRange.hi += (hi - state.depthVisRange.hi) * smooth;
  }
  const visLo = state.depthVisRange.lo;
  const span = Math.max(0.02, state.depthVisRange.hi - visLo);
  const edge = new Uint8Array(DEPTH_OUT * DEPTH_OUT);
  for (let r = 0; r < DEPTH_OUT; r++) {
    for (let c = 0; c < DEPTH_OUT; c++) {
      const i = r * DEPTH_OUT + c;
      const v = state.depth[i];
      if (v >= 0.999) continue;
      let grad = 0;
      if (c + 1 < DEPTH_OUT && state.depth[i + 1] < 0.999) grad = Math.max(grad, Math.abs(v - state.depth[i + 1]));
      if (r + 1 < DEPTH_OUT && state.depth[i + DEPTH_OUT] < 0.999) grad = Math.max(grad, Math.abs(v - state.depth[i + DEPTH_OUT]));
      if (grad > 0.035) edge[i] = 1;
    }
  }
  for (let i = 0; i < state.depth.length; i++) {
    const v = state.depth[i];
    let r = 10; let g = 14; let b = 20;
    if (v < 0.999) {
      // 1 = near / warm, 0 = far / blue
      [r, g, b] = depthColor(1 - (v - visLo) / span);
      if (edge[i]) { r = Math.min(255, r + 120); g = Math.min(255, g + 120); b = Math.min(255, b + 120); }
    }
    img.data[i * 4] = r;
    img.data[i * 4 + 1] = g;
    img.data[i * 4 + 2] = b;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  const rangeEl = qs('depth-range');
  if (rangeEl) {
    rangeEl.textContent = hits
      ? `${(rawMin * DEPTH_MAX).toFixed(2)}–${(rawMax * DEPTH_MAX).toFixed(2)} m`
      : 'no returns';
  }
  drawDepthProfile();
}

function drawDepthProfile() {
  const canvas = qs('depth-profile');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width; const H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = 'rgba(120,150,180,0.35)';
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    const y = Math.round(H * i / 4) + 0.5;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }
  const col = Math.floor(DEPTH_OUT / 2);
  let mn = 1; let mx = 0;
  for (let r = 0; r < DEPTH_OUT; r++) {
    const v = state.depth[r * DEPTH_OUT + col];
    if (v < 0.999) { mn = Math.min(mn, v); mx = Math.max(mx, v); }
  }
  const profileSpan = Math.max(0.02, mx - mn);
  ctx.strokeStyle = '#67d2c4';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let r = 0; r < DEPTH_OUT; r++) {
    const v = state.depth[r * DEPTH_OUT + col];
    const x = (r / (DEPTH_OUT - 1)) * (W - 1);
    const y = v >= 0.999 ? H - 1 : (1 - (v - mn) / profileSpan) * (H - 6) + 3;
    if (r === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.fillStyle = 'rgba(131,148,168,0.9)';
  ctx.font = '9px ui-monospace, monospace';
  ctx.fillText(`${(mn * DEPTH_MAX).toFixed(2)}m`, 4, 11);
  ctx.fillText(`${(mx * DEPTH_MAX).toFixed(2)}m`, 4, H - 4);
}

let fps = 0;
let fpsCount = 0;
let fpsTime = performance.now();

function updateVisuals() {
  updateBodyTransforms();
  if (state.follow && controls) {
    const p = new THREE.Vector3(state.data.qpos[0], state.data.qpos[1], state.data.qpos[2]);
    const delta = p.clone().sub(controls.target);
    delta.z *= 0.35;
    if (delta.length() > 0.001) {
      controls.target.add(delta.multiplyScalar(0.12));
      camera.position.add(state.terrainMode === 'hillside' ? delta : delta.multiplyScalar(0.12));
    }
  }
  controls.update();
}

let controlTimer = null;
let nextControlAt = 0;
let controlCount = 0;
let controlRateTime = performance.now();

let rateSimTime = 0;
function bumpControlRate() {
  controlCount += 1;
  const now = performance.now();
  if (now - controlRateTime > 1000) {
    const wall = (now - controlRateTime) / 1000;
    state.perf.controlHz = controlCount / wall;
    state.perf.simRate = (state.simTime - rateSimTime) / wall;
    controlCount = 0;
    controlRateTime = now;
    rateSimTime = state.simTime;
    updateControlPressure();
  }
}

function updateRenderNote() {
  const base = state.renderNoteBase;
  if (state.renderLimited) {
    const fpsCap = Math.round(1000 / state.renderIntervalMs);
    qs('render-note').textContent = `${base ? base + ' · ' : ''}control priority: render ${fpsCap} fps`;
    qs('render-note').title = `${state.perf.gl} (render throttled to keep 50 Hz control)`;
  } else {
    qs('render-note').textContent = base;
    qs('render-note').title = state.perf.gl;
  }
}

function updateControlPressure() {
  const slow = state.perf.controlHz > 0 && state.perf.controlHz < 42 && state.perf.simRate < 0.88;
  const fast = state.perf.controlHz >= 46 && state.perf.simRate >= 0.94;
  if (!state.renderLimited) {
    state.lowRateTicks = slow ? state.lowRateTicks + 1 : 0;
    if (state.lowRateTicks >= 3) {
      state.renderLimited = true;
      state.renderIntervalMs = 1000 / 30;
      state.highRateTicks = 0;
      state.lowRateTicks = 0;
      updateRenderNote();
    }
  } else {
    if (slow && state.renderIntervalMs > 1000 / 24) {
      state.lowRateTicks += 1;
      if (state.lowRateTicks >= 3) {
        state.renderIntervalMs = 1000 / 20;
        updateRenderNote();
      }
    }
    state.highRateTicks = fast ? state.highRateTicks + 1 : 0;
    if (state.highRateTicks >= 2) {
      state.renderLimited = false;
      state.renderIntervalMs = 0;
      state.lowRateTicks = 0;
      updateRenderNote();
    }
  }
}

let controlInFlight = false;
let lastRafControlAt = 0;

async function pumpControl() {
  if (controlInFlight || !state.ready) return;
  const now = performance.now();
  if (!state.running || state.checking) {
    nextControlAt = now;
    return;
  }
  if (now < nextControlAt - 2) return;
  controlInFlight = true;
  try {
    let catchup = 0;
    while (performance.now() >= nextControlAt - 2 && catchup < 2) {
      await controlStep();
      bumpControlRate();
      nextControlAt += (state.controlDt * 1000) / state.speed;
      catchup += 1;
      if (nextControlAt < performance.now() - 200) {
        nextControlAt = performance.now();
        break;
      }
      if (performance.now() - now > 12) break;
    }
  } catch (err) {
    state.running = false;
    setStatus(`error: ${err.message}`, true);
    console.error(err);
  } finally {
    controlInFlight = false;
  }
}

async function controlTick() {
  controlTimer = null;
  if (document.hidden || performance.now() - lastRafControlAt > 80) {
    await pumpControl();
  }
  controlTimer = setTimeout(controlTick, document.hidden ? 25 : 40);
}

function startControlLoop() {
  if (controlTimer) return;
  nextControlAt = performance.now();
  controlTimer = setTimeout(controlTick, 40);
}

function loop(now) {
  requestAnimationFrame(loop);
  lastFrame = now;

  pollDepthReadback(true);

  const canRender = !state.renderIntervalMs || (now - lastRenderAt >= state.renderIntervalMs);
  if (canRender && renderer && scene && camera) {
    lastRenderAt = now;
    fpsCount += 1;
    if (now - fpsTime > 500) {
      fps = (fpsCount * 1000) / (now - fpsTime);
      state.perf.fps = fps;
      fpsCount = 0;
      fpsTime = now;
    }
    updateVisuals();
    const tRender = performance.now();
    renderer.render(scene, camera);
    if (state.glass) state.glass.render(now);
    const ms = performance.now() - tRender;
    state.perf.renderMs = state.perf.renderMs ? state.perf.renderMs * 0.85 + ms * 0.15 : ms;
  }

  lastRafControlAt = performance.now();
  pumpControl();

  if (now - lastHud > 100) {
    lastHud = now;
    updateHud();
    drawDepthPreview();
    if (now - lastCameraViewAt > 200) {
      lastCameraViewAt = now;
      if (!qs('camera-wrap')?.hidden) renderCameraView();
    }
  }
}

function setStatus(text, isError = false) {
  const el = qs('status');
  el.textContent = text;
  el.classList.toggle('error', isError);
}

// ---------------------------------------------------------------------------
// check harness
// ---------------------------------------------------------------------------
function bodyPos(name) {
  const body = state.data.body(name);
  return body ? Array.from(body.xpos) : null;
}

function sim2realSnapshot() {
  const s = state.sim2real;
  return {
    friction: Number(s.friction) || 0,
    damping: Number(s.damping) || 0,
    actuatorDelayMs: Number(s.actuatorDelayMs) || 0,
    encoderZeroDeg: (Number(s.encoderZero) || 0) * 180 / Math.PI,
    imuPitchBiasDeg: Number(s.imuPitchBiasDeg) || 0,
    gyroBiasDeg: (Number(s.gyroBias) || 0) * 180 / Math.PI,
    jointVelFilterMs: Number(s.jointVelFilterMs) || 0,
    backlashDeg: (Number(s.backlash) || 0) * 180 / Math.PI,
    shoe: Boolean(s.shoe),
  };
}

function snapshot() {
  const { data } = state;
  return {
    x: data.qpos[0], y: data.qpos[1], z: data.qpos[2],
    foot: {
      left: bodyPos('left_ankle_roll_link'),
      right: bodyPos('right_ankle_roll_link'),
    },
    yaw: yawFromQuat(Array.from(data.qpos.slice(3, 7))),
    steps: state.steps,
    simTime: state.simTime,
    minZ: state.minPelvisZ,
    maxZ: state.maxPelvisZ,
    clearance: data.qpos[2] - groundHeight(data.qpos[0], data.qpos[1]),
    minClearance: state.minClearance,
    depth: { min: Math.min(...state.depth), max: Math.max(...state.depth) },
    valid: validState(),
  };
}

async function runCheck({ steps = 250, command = null } = {}) {
  state.running = false;
  state.checking = true;
  const previousCommand = state.command.slice();
  const previousKeyboard = state.keyboardEnabled;
  state.keyboardEnabled = false;
  const direction = state.terrain?.direction || 'up';
  const autoSpeed = { flat: 0.45, up: 0.45, down: 0.30, up_down: 0.40, down_up: 0.35 }[direction] ?? 0.45;
  state.command = command ? command.slice(0, 3) : [autoSpeed, 0, 0];
  const start = snapshot();
  let fell = false;
  let invalid = false;
  let maxAction = 0;
  try {
  for (let i = 0; i < steps; i++) {
    const ok = await controlStep();
    if (!ok) { invalid = true; break; }
    const clearance = state.data.qpos[2] -
      groundHeight(state.data.qpos[0], state.data.qpos[1]);
    if (clearance < 0.35) { fell = true; break; }
    maxAction = Math.max(maxAction, ...Array.from(state.lastAction, Math.abs));
  }
  const end = snapshot();
  return {
    start, end,
    deltaX: end.x - start.x,
    deltaZ: end.z - start.z,
    fell, invalid, maxAction,
    depthRange: [end.depth.min, end.depth.max],
  };
  } finally {
    state.checking = false;
    state.command = previousCommand;
    state.keyboardEnabled = previousKeyboard;
  }
}

// ---------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------
  const initGlass = async () => {
    const badge = document.getElementById('glass-badge');
    const setBadge = (text, mode) => { if (badge) { badge.textContent = text; badge.dataset.mode = mode; } };
    setBadge('glass · css', 'css');
    try {
      const glassMode = new URLSearchParams(location.search).get('glass');
      if (glassMode === 'css') { setBadge('glass · css', 'css'); return; }
      const startGL = async () => {
        setBadge('glass · webgl2', 'webgl2');
        state.glass = await createGlassGL({
          renderer,
          sceneCanvas: renderer.domElement,
        });
        if (state.glass) console.log('webgl2 inline glass active');
      };
      if (glassMode !== 'css') {
        if (glassMode === 'webgpu' || (!IS_SAFARI && glassMode !== 'gl')) {
          setBadge('glass · webgpu', 'webgpu');
          state.glass = await createGlassGPU({
            canvas: document.getElementById('glass-gpu'),
            sceneCanvas: renderer.domElement,
            ambientCanvas: state.ambient?.canvas,
            onLost: () => { setBadge('glass · css', 'css'); startGL().catch((error) => console.warn('WebGL glass fallback failed', error)); },
          });
        }
        if (!state.glass) await startGL();
      }
      if (state.glass) console.log(`${state.glass.backend} glass active`);
    } catch (error) {
      setBadge('glass · css', 'css');
      console.warn('GPU glass unavailable, keeping CSS glass', error);
    }
  };

async function boot() {
  setStatus('loading config + Mujoco WASM...');
  let base = '/';
  const urlExport = new URLSearchParams(location.search).get('export');
  const savedExport = urlExport || localStorage.getItem('preview.export');
  if (urlExport) localStorage.setItem('preview.export', urlExport);
  if (savedExport) {
    try {
      const probe = await fetch(`/exports/${savedExport}/config.json`, { cache: 'no-store' });
      if (probe.ok) base = `/exports/${savedExport}/`;
      else localStorage.removeItem('preview.export');
    } catch (_) {
      localStorage.removeItem('preview.export');
    }
  }
  state.assetBase = base;
  state.exportId = base === '/' ? null : savedExport;
  const [config, template] = await Promise.all([
    fetch(`${base}config.json`, { cache: 'no-store' }).then((r) => {
      if (!r.ok) throw new Error(`config.json ${r.status}`);
      return r.json();
    }),
    fetch(`${base}robot_template.xml`, { cache: 'no-store' }).then((r) => {
      if (!r.ok) throw new Error(`robot_template.xml ${r.status}`);
      return r.text();
    }),
  ]);
  state.config = config;
  state.template = template;
  state.controlDt = Number(config.engine.control_dt ?? config.engine.step_dt ?? 0.02);
  state.mujoco = await loadMujoco();

  await initThree();
  const ambient = createAmbientLayer();
  state.ambient = ambient;
  const syncAmbient = () => {
    ambient.refresh();
    state.glass?.setAmbient?.(ambient.canvas);
  };
  document.addEventListener('themechange', syncAmbient);
  let ambientResize = 0;
  window.addEventListener('resize', () => {
    clearTimeout(ambientResize);
    ambientResize = setTimeout(syncAmbient, 120);
  });
  restoreTerrainPrefs();
  applyUrlParams();
  setupUi();
  installKeyboard();
  setStatus('loading meshes + ONNX policy...');
  await loadRobotVisuals(base);

  setStatus('creating ONNX session...');
  state.session = await ort.InferenceSession.create(
    `${base}policy.onnx?v=${config.checkpoint.sha256.slice(0, 12)}`, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });

  applyTerrain(state.direction, state.level);
  state.ready = true;
  state.perf.cores = CORES;
  rateSimTime = state.simTime;
  setStatus(`ready · checkpoint ${config.checkpoint.iteration} · ${CORES} wasm threads`);
  document.body.dataset.ready = 'true';
  setTimeout(() => { initGlass().catch((error) => console.warn('GPU glass init failed', error)); }, 700);

  window.__preview = {
    get ready() { return state.ready; },
    get config() { return state.config; },
    get checkpoint() { return state.config.checkpoint; },
    get observationLayout() {
      return {
        dimension: state.config.observation_dim,
        terms: state.config.observation_terms,
        depth: `${DEPTH_OUT}x${DEPTH_OUT}`,
      };
    },
    async reset(opts = {}) {
      if (opts.direction || opts.level !== undefined) {
        applyTerrain(opts.direction ?? state.direction, opts.level ?? state.level);
      } else {
        resetSim();
      }
      return snapshot();
    },
    async runCheck(opts = {}) { return runCheck(opts); },
    sim2realState: () => sim2realSnapshot(),
    footState: () => ({
      left: bodyPos('left_ankle_roll_link'),
      right: bodyPos('right_ankle_roll_link'),
    }),
    setSim2real(opts = {}) {
      const s = state.sim2real;
      const shoeChanged = opts.shoe !== undefined && Boolean(opts.shoe) !== s.shoe;
      if (opts.friction !== undefined) s.friction = Number(opts.friction);
      if (opts.damping !== undefined) s.damping = Number(opts.damping);
      if (opts.actuatorDelayMs !== undefined) s.actuatorDelayMs = Number(opts.actuatorDelayMs);
      if (opts.encoderZeroDeg !== undefined) s.encoderZero = Number(opts.encoderZeroDeg) * Math.PI / 180;
      if (opts.imuPitchBiasDeg !== undefined) s.imuPitchBiasDeg = Number(opts.imuPitchBiasDeg);
      if (opts.gyroBiasDeg !== undefined) s.gyroBias = Number(opts.gyroBiasDeg) * Math.PI / 180;
      if (opts.jointVelFilterMs !== undefined) s.jointVelFilterMs = Number(opts.jointVelFilterMs);
      if (opts.backlashDeg !== undefined) s.backlash = Number(opts.backlashDeg) * Math.PI / 180;
      if (opts.shoe !== undefined) s.shoe = Boolean(opts.shoe);
      for (let i = 0; i < 29; i++) s.zeroOffsets[i] = Math.sin(i * 1.7) * s.encoderZero;
      applySim2RealModel();
      syncSim2RealUi();
      if (shoeChanged) applyTerrain(state.direction, state.level);
      return sim2realSnapshot();
    },
    setTerrain(opts = {}) {
      if (opts.mode) state.terrainMode = String(opts.mode);
      if (opts.direction) state.direction = String(opts.direction);
      if (opts.level !== undefined) state.level = Math.max(0, Math.min(19, Math.round(Number(opts.level))));
      if (opts.uniform) Object.assign(state.uniform, opts.uniform);
      applyTerrain(state.direction, state.level);
      const write = (id, value) => { const el = qs(id); if (el) el.value = String(value); };
      write('uniform-riser', state.uniform.riser * 100);
      write('uniform-tread', state.uniform.tread * 100);
      write('uniform-steps', state.uniform.steps);
      write('uniform-width', state.uniform.width);
      const label = (id, text) => { const el = qs(`${id}-value`); if (el) el.textContent = text; };
      label('uniform-riser', `${(state.uniform.riser * 100).toFixed(1)} cm`);
      label('uniform-tread', `${(state.uniform.tread * 100).toFixed(0)} cm`);
      label('uniform-steps', String(state.uniform.steps));
      label('uniform-width', `${state.uniform.width.toFixed(1)} m`);
      return { mode: state.terrainMode, direction: state.direction, uniform: { ...state.uniform } };
    },
    stats: () => snapshot(),
    loadExport: (id) => hotLoadExport(id),
    listExports: () => fetch('/api/exports', { cache: 'no-store' }).then((r) => r.json()),
    setCommand(cmd) { state.command = cmd.slice(0, 3); },
    terrainInfo: () => ({
      mode: state.terrainMode,
      direction: state.direction,
      level: state.level,
      uniform: { ...state.uniform },
      spec: state.terrain?.spec || null,
      spawn: state.terrain?.spawn || null,
      boxes: state.terrain?.boxes || [],
    }),
    perf: () => ({
      ...state.perf,
      pixelRatio: renderer.getPixelRatio(),
      canvas: [renderer.domElement.width, renderer.domElement.height],
      controlHz: state.perf.controlHz,
    }),
    rawState: () => ({
      qpos: Array.from(state.data.qpos),
      qvel: Array.from(state.data.qvel),
      depth: Array.from(state.depth),
      steps: state.steps,
      direction: state.direction,
      level: state.level,
    }),
  };

  state.running = !new URLSearchParams(location.search).has('autostart');
  document.getElementById('btn-play').textContent = state.running ? 'Pause' : 'Resume';
  document.getElementById('btn-play').dataset.running = String(state.running);
  requestAnimationFrame(loop);
  lastFrame = performance.now();
  startControlLoop();
}

boot().catch((err) => {
  console.error(err);
  setStatus(`boot failed: ${err.message}`, true);
  const el = document.getElementById('status');
  if (el) el.textContent = `boot failed: ${err.message}`;
});
