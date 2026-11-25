// Terrain construction shared by the MuJoCo model and the three.js renderer.
// Heights in config.json are per-step riser magnitudes; absolute tread height is
// the cumulative sum in the direction of travel (matching the training height field).

export const DIRECTIONS = ['flat', 'up', 'down', 'up_down', 'down_up'];

const SUPPORT_THICKNESS = 1.0;

function slab(boxes, x0, x1, top, width) {
  if (!(x1 - x0 > 1e-6)) return;
  boxes.push({
    cx: (x0 + x1) / 2,
    cy: 0,
    cz: top - SUPPORT_THICKNESS / 2,
    hx: (x1 - x0) / 2,
    hy: width / 2,
    hz: SUPPORT_THICKNESS / 2,
    top,
  });
}

export function terrainTopAt(boxes, x, y = 0) {
  let top = -Infinity;
  for (const b of boxes) {
    if (Math.abs(x - b.cx) <= b.hx + 1e-9 && Math.abs(y - b.cy) <= b.hy + 1e-9) {
      top = Math.max(top, b.top);
    }
  }
  return Number.isFinite(top) ? top : 0;
}

export function buildTerrain(config, direction, level) {
  if (direction === 'flat') {
    const boxes = [{
      cx: 0, cy: 0, cz: -SUPPORT_THICKNESS / 2,
      hx: 40, hy: 40, hz: SUPPORT_THICKNESS / 2, top: 0,
    }];
    return {
      direction, level: 0, label: 'flat',
      boxes, spawn: { x: 0.2, y: 0, z: 0 },
      spec: null,
    };
  }

  const levels = config?.terrain?.[direction];
  if (!Array.isArray(levels) || levels.length === 0) {
    throw new Error(`No terrain data for direction ${direction}`);
  }
  const clamped = Math.max(0, Math.min(levels.length - 1, level | 0));
  const spec = levels[clamped];
  const width = Number(spec.width_m ?? 4.0);
  const extent = Number(spec.extent ?? 9.2);
  const runs = spec.runs ?? [];
  if (runs.length === 0) throw new Error(`Terrain ${direction}/${clamped} has no runs`);

  const boxes = [];
  const firstStart = Number(runs[0].starts[0]);
  slab(boxes, -2.0, firstStart, 0, width);

  let top = 0;
  let prevEnd = firstStart;
  for (let r = 0; r < runs.length; r++) {
    const run = runs[r];
    if (r > 0) slab(boxes, prevEnd, Number(run.starts[0]), top, width);
    const sign = Number(run.direction) < 0 ? -1 : 1;
    for (let i = 0; i < run.starts.length; i++) {
      const start = Number(run.starts[i]);
      const end = Number(run.ends[i]);
      top += sign * Number(run.heights[i]);
      slab(boxes, start, end, top, width);
      prevEnd = end;
    }
  }
  slab(boxes, prevEnd, extent, top, width);

  const spawnX = Number(spec.spawn_origin_x ?? 0.2);
  return {
    direction, level: clamped, label: `${direction} L${clamped}`,
    boxes,
    spawn: { x: spawnX, y: 0, z: terrainTopAt(boxes, spawnX, 0) },
    spec,
  };
}


// Uniform straight staircase with adjustable riser height / tread depth.
// steps is the number of steps per run (two runs for up_down / down_up).
export function buildUniformTerrain({
  direction = 'up', riser = 0.12, tread = 0.35, steps = 12,
  width = 4.0, startX = 1.0, middlePlatform = 1.0, endPlatform = 2.0,
} = {}) {
  if (!['up', 'down', 'up_down', 'down_up'].includes(direction)) {
    throw new Error(`uniform terrain does not support direction ${direction}`);
  }
  const r = Number(riser);
  const d = Number(tread);
  const n = Math.max(1, Math.round(Number(steps)));
  if (!(r > 0) || !(d > 0)) throw new Error('riser and tread must be positive');

  const boxes = [];
  slab(boxes, -2.0, startX, 0, width);
  let x = startX;
  let top = 0;
  const totalSteps = direction === 'up' || direction === 'down' ? n : n * 2;

  if (direction === 'up' || direction === 'down') {
    const sign = direction === 'up' ? 1 : -1;
    for (let i = 0; i < n; i++) {
      top += sign * r;
      slab(boxes, x, x + d, top, width);
      x += d;
    }
  } else {
    // up_down: ascend then descend; down_up: descend then ascend.
    const sign = direction === 'up_down' ? 1 : -1;
    for (let i = 0; i < n; i++) {
      top += sign * r;
      slab(boxes, x, x + d, top, width);
      x += d;
    }
    slab(boxes, x, x + middlePlatform, top, width);
    x += middlePlatform;
    for (let i = 0; i < n; i++) {
      top -= sign * r;
      slab(boxes, x, x + d, top, width);
      x += d;
    }
  }
  const extent = Math.max(9.2, x + endPlatform + 0.5);
  slab(boxes, x, extent, top, width);
  const spawnX = 0.2;
  return {
    direction,
    level: null,
    label: `uniform ${direction}`,
    boxes,
    spawn: { x: spawnX, y: 0, z: terrainTopAt(boxes, spawnX, 0) },
    spec: {
      uniform: true,
      direction,
      num_steps: totalSteps,
      steps_per_run: n,
      riser_height_m: r,
      step_length_m: d,
      width_m: width,
      middle_platform_m: middlePlatform,
      start_x: startX,
      extent,
    },
  };
}

// A single rectangular trench (gap) across the route, matching the training
// paper_gap parameter ranges (width 0.10-0.45 m, depth 0.30-0.60 m).
export function buildGapTerrain({
  gapWidth = 0.30, gapDepth = 0.45, centerX = 2.0, width = 4.0, extent = 9.2,
} = {}) {
  const w = Math.max(0.05, Number(gapWidth));
  const depth = Math.max(0.05, Number(gapDepth));
  const center = Number(centerX);
  const half = w / 2;
  const start = Math.max(0.5, center - half);
  const end = start + w;
  const ext = Math.max(Number(extent) || 9.2, end + 2.0);
  const boxes = [];
  slab(boxes, -2.0, start, 0, width);
  slab(boxes, end, ext, 0, width);
  // Pit bottom; the floor slabs' sides form the vertical walls.
  slab(boxes, start, end, -depth, width);
  const spawnX = 0.2;
  return {
    direction: 'gap', level: null, label: 'gap', boxes,
    spawn: { x: spawnX, y: 0, z: terrainTopAt(boxes, spawnX, 0) },
    spec: {
      gap: true, gap_width_m: w, gap_depth_m: depth,
      center_x_m: (start + end) / 2, width_m: width, extent: ext,
    },
  };
}

// A single raised platform from startX to the end of the tile, matching the
// training paper_platform ranges (height 0.05-0.40 m).
export function buildPlatformTerrain({
  platformHeight = 0.20, startX = 1.5, width = 4.0, extent = 9.2,
} = {}) {
  const height = Math.max(0.02, Number(platformHeight));
  const start = Math.max(0.5, Number(startX));
  const ext = Math.max(Number(extent) || 9.2, start + 3.0);
  const boxes = [];
  slab(boxes, -2.0, start, 0, width);
  slab(boxes, start, ext, height, width);
  const spawnX = 0.2;
  return {
    direction: 'platform', level: null, label: 'platform', boxes,
    spawn: { x: spawnX, y: 0, z: terrainTopAt(boxes, spawnX, 0) },
    spec: {
      platform: true, platform_height_m: height, start_x_m: start,
      width_m: width, extent: ext,
    },
  };
}

// Industrial open-riser steel stair: each tread is a grating of transverse
// bars (real collision per bar), with no solid riser. Side stringers and a
// handrail are visual-only so the depth camera sees the factory geometry.
export function buildGratingTerrain({
  riser = 0.20, tread = 0.25, steps = 12, width = 1.2,
  barWidth = 0.03, gapWidth = 0.015, startX = 1.0, endLanding = 1.0, extent = 9.2,
} = {}) {
  const r = Math.max(0.05, Number(riser));
  const d = Math.max(0.12, Number(tread));
  const n = Math.max(1, Math.round(Number(steps)));
  const w = Math.max(0.5, Number(width));
  const bw = Math.max(0.01, Number(barWidth));
  const gw = Math.max(0.005, Number(gapWidth));
  const boxes = [];
  const visuals = [];
  const approachWidth = Math.max(4.0, w + 1.0);
  slab(boxes, -2.0, startX, 0, approachWidth);

  const pitch = bw + gw;
  const barCount = Math.max(1, Math.floor(d / pitch));
  const stairEnd = startX + n * d;
  const top = n * r;

  for (let i = 0; i < n; i++) {
    const stepTop = (i + 1) * r;
    const x0 = startX + i * d;
    for (let j = 0; j < barCount; j++) {
      const cx = x0 + bw / 2 + j * pitch;
      if (cx + bw / 2 > x0 + d - 1e-9) break;
      boxes.push({
        cx, cy: 0, cz: stepTop - 0.015, hx: bw / 2, hy: w / 2, hz: 0.015,
        top: stepTop, steel: true,
      });
    }
  }

  // Steel landing plate at the top of the flight.
  const landStart = stairEnd;
  const landEnd = Math.max(landStart + Math.max(0.2, Number(endLanding)), Number(extent) || 9.2);
  boxes.push({
    cx: (landStart + landEnd) / 2, cy: 0, cz: top - 0.02,
    hx: (landEnd - landStart) / 2, hy: w / 2, hz: 0.02, top, steel: true,
  });

  // Visual stringers and handrail (not part of MuJoCo collision).
  const run = Math.max(0.01, stairEnd - startX);
  const angle = Math.atan2(top, run);
  const slopeLength = Math.hypot(run, top);
  for (const side of [-1, 1]) {
    const y = side * (w / 2 + 0.05);
    visuals.push({
      cx: startX + run / 2, cy: y, cz: top / 2, hx: slopeLength / 2,
      hy: 0.025, hz: 0.08, rotY: -angle, steel: true, part: 'stringer',
    });
    visuals.push({
      cx: startX + run / 2, cy: y, cz: top / 2 + 0.9, hx: slopeLength / 2,
      hy: 0.015, hz: 0.015, rotY: -angle, steel: true, part: 'handrail',
    });
    for (let k = 1; k <= 4; k++) {
      const t = k / 5;
      const px = startX + t * run;
      const pz = top * t;
      visuals.push({
        cx: px, cy: y, cz: pz + 0.45, hx: 0.012, hy: 0.012, hz: 0.45,
        steel: true, part: 'post',
      });
    }
  }

  const spawnX = 0.2;
  return {
    direction: 'grating', level: null, label: 'steel grating', boxes, visuals,
    spawn: { x: spawnX, y: 0, z: terrainTopAt(boxes, spawnX, 0) },
    spec: {
      grating: true, riser_height_m: r, step_length_m: d, steps: n, width_m: w,
      bar_width_m: bw, gap_width_m: gw, stair_end_x_m: stairEnd,
      top_height_m: top, extent: landEnd,
    },
  };
}

// Smooth up-down slope (up ramp, middle platform, down ramp) with a rotated
// box for the ramp surface. The analytic topAt() is used for spawn/clearance.
export function buildSlopeTerrain({
  angle = 12, run = 3.1, width = 4.0, startX = 1.0, middle = 1.0,
  endPlatform = 1.0, extent = 9.2,
} = {}) {
  const deg = Math.max(0, Math.min(30, Number(angle)));
  const a = deg * Math.PI / 180;
  const r = Math.max(0.5, Number(run));
  const w = Math.max(0.5, Number(width));
  const m = Math.tan(a);
  const rise = r * m;
  const x0 = startX;
  const x1 = x0 + r;
  const x2 = x1 + Math.max(0, Number(middle));
  const x3 = x2 + r;
  const ext = Math.max(Number(extent) || 9.2, x3 + Math.max(0, Number(endPlatform)));
  const boxes = [];
  slab(boxes, -2.0, x0, 0, w);
  slab(boxes, x1, x2, rise, w);
  slab(boxes, x3, ext, 0, w);

  const t = 0.12;
  const S = Math.hypot(r, rise);
  const sin = Math.sin(a);
  const cos = Math.cos(a);
  // Up ramp: local +x ascends; local +z is the surface normal (-sin, cos).
  const qUp = [Math.cos(a / 2), 0, -Math.sin(a / 2), 0];
  boxes.push({
    cx: (x0 + x1) / 2 + (t / 2) * sin, cy: 0, cz: rise / 2 - (t / 2) * cos,
    hx: S / 2, hy: w / 2, hz: t / 2, quat: qUp, top: rise / 2,
  });
  // Down ramp: local +x descends; normal (+sin, cos).
  const qDn = [Math.cos(a / 2), 0, Math.sin(a / 2), 0];
  boxes.push({
    cx: (x2 + x3) / 2 - (t / 2) * sin, cy: 0, cz: rise / 2 - (t / 2) * cos,
    hx: S / 2, hy: w / 2, hz: t / 2, quat: qDn, top: rise / 2,
  });

  const topAt = (x, y = 0) => {
    if (Math.abs(y) > w / 2 + 1e-9) return 0;
    if (x <= x0) return 0;
    if (x <= x1) return Math.min(rise, (x - x0) * m);
    if (x <= x2) return rise;
    if (x <= x3) return Math.max(0, rise - (x - x2) * m);
    return 0;
  };
  const spawnX = 0.2;
  return {
    direction: 'slope', level: null, label: 'slope', boxes, topAt,
    spawn: { x: spawnX, y: 0, z: topAt(spawnX, 0) },
    spec: {
      slope: true, slope_angle_deg: deg, run_m: r, width_m: w, rise_m: rise,
      start_x_m: x0, top_start_x_m: x1, top_end_x_m: x2, end_x_m: x3,
      middle_platform_m: Math.max(0, Number(middle)), extent: ext,
    },
  };
}

export function buildHillsideTerrain({ difficulty = 0.5, seed = 0x5eed } = {}) {
  if (!Number.isFinite(difficulty)) throw new Error('Hillside difficulty must be finite');
  const d = Math.max(0, Math.min(1, difficulty));
  const ncol = 214;
  const nrow = 81;
  const tileCols = 185;
  const dx = 10.7 / (ncol - 1);
  const dy = 0.05;
  const noiseAmps = [[0.45, 3.5], [0.10, 1.2], [0.018, 0.3]];
  let randomState = seed >>> 0;
  const random = () => {
    randomState = (randomState + 0x6d2b79f5) >>> 0;
    let value = randomState;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
  const reflect = (index, length) => {
    const folded = ((index % (2 * length)) + 2 * length) % (2 * length);
    return folded < length ? folded : 2 * length - folded - 1;
  };
  const blur = (input, output, radius, horizontal) => {
    const length = horizontal ? tileCols : nrow;
    const lines = horizontal ? nrow : tileCols;
    const stride = horizontal ? 1 : tileCols;
    const divisor = 2 * radius + 1;
    for (let line = 0; line < lines; line++) {
      const base = horizontal ? line * tileCols : line;
      let sum = 0;
      for (let offset = -radius; offset <= radius; offset++) sum += input[base + reflect(offset, length) * stride];
      for (let index = 0; index < length; index++) {
        output[base + index * stride] = sum / divisor;
        sum += input[base + reflect(index + radius + 1, length) * stride]
          - input[base + reflect(index - radius, length) * stride];
      }
    }
  };
  const tile = new Float64Array(tileCols * nrow);
  for (let row = 0; row < nrow; row++) {
    const y = row * dy;
    for (let col = 0; col < tileCols; col++) {
      const x = col * 0.05;
      const denominator = 2 * (0.3 * 9.2) ** 2;
      tile[row * tileCols + col] = d * (
        1.6 * Math.exp(-((x - 0.65 * 9.2) ** 2 + (y - 2) ** 2) / denominator)
        + 0.9 * Math.exp(-((x - 0.3 * 9.2) ** 2 + (y - 3) ** 2) / denominator)
        + 0.02 * (y - 2));
    }
  }
  for (const [amplitude, scale] of noiseAmps) {
    const noise = Float64Array.from(tile, () => Math.sqrt(-2 * Math.log(Math.max(random(), 1e-12))) * Math.cos(2 * Math.PI * random()));
    const scratch = new Float64Array(noise.length);
    const radius = Math.round((Math.sqrt(1 + 4 * (scale / 0.05) ** 2) - 1) / 2);
    for (let pass = 0; pass < 3; pass++) {
      blur(noise, scratch, radius, true);
      blur(scratch, noise, radius, false);
    }
    let mean = 0;
    for (const value of noise) mean += value / noise.length;
    let variance = 0;
    for (const value of noise) variance += (value - mean) ** 2 / noise.length;
    const std = Math.max(Math.sqrt(variance), 1e-9);
    for (let index = 0; index < tile.length; index++) tile[index] += d * amplitude * noise[index] / std;
  }
  const heights = new Float32Array(nrow * ncol);
  let min = Infinity;
  for (let row = 0; row < nrow; row++) {
    for (let col = 0; col < ncol; col++) {
      const tileX = Math.max(0, Math.min(tileCols - 1, (-1.5 + col * dx) / 0.05));
      const left = Math.min(tileCols - 2, Math.floor(tileX));
      const fraction = tileX - left;
      const index = row * tileCols + left;
      const height = tile[index] * (1 - fraction) + tile[index + 1] * fraction;
      heights[row * ncol + col] = height;
      min = Math.min(min, heights[row * ncol + col]);
    }
  }
  let peak = 0;
  for (let index = 0; index < heights.length; index++) {
    heights[index] -= min;
    peak = Math.max(peak, heights[index]);
  }
  const maxHeight = Math.max(0.01, peak);
  const data = Float32Array.from(heights, (height) => height / maxHeight);
  const topAt = (x, y = 0) => {
    if (x < -1.5 || x > 9.2 || y < -2 || y > 2) return 0;
    const gridX = Math.max(0, Math.min(ncol - 1, (x + 1.5) / dx));
    const gridY = Math.max(0, Math.min(nrow - 1, (y + 2) / dy));
    const col = Math.min(ncol - 2, Math.floor(gridX));
    const row = Math.min(nrow - 2, Math.floor(gridY));
    const fx = gridX - col;
    const fy = gridY - row;
    const index = row * ncol + col;
    return (heights[index] * (1 - fx) + heights[index + 1] * fx) * (1 - fy)
      + (heights[index + ncol] * (1 - fx) + heights[index + ncol + 1] * fx) * fy;
  };
  // Floating preview tile needs a surrounding apron so the heightfield does
  // not read as an isolated slab; it sits 1 cm below the terrain's lowest point.
  const boxes = [];
  slab(boxes, -40, 40, -0.01, 80);
  return {
    direction: 'hillside', level: d, label: 'paper hillside', boxes, topAt,
    spawn: { x: 0.2, y: 0, z: topAt(0.2, 0) },
    hfield: { nrow, ncol, data, heights, size: [5.35, 2, maxHeight, 0.1], center: [3.85, 0, 0], x0: -1.5, y0: -2, dx, dy, maxHeight },
    spec: {
      hillside: true, difficulty: d, peak_m: peak, size_x_m: 10.7, tile_x_m: 9.2, tile_y_m: 4,
      noise: { seed: seed >>> 0, layers: noiseAmps, approximation: 'Seeded Gaussian white noise, three separable reflected box passes per axis, each layer normalized to unit standard deviation on the training tile.' },
    },
  };
}

export function terrainGeomXml(boxes) {
  return boxes.map((b, i) => {
    const quat = b.quat
      ? `quat="${b.quat.map((v) => Number(v).toFixed(12)).join(' ')}" `
      : '';
    return `<geom name="terrain_${i}" type="box" ` +
      `pos="${b.cx.toFixed(6)} ${b.cy.toFixed(6)} ${b.cz.toFixed(6)} " ` +
      `${quat}size="${b.hx.toFixed(6)} ${b.hy.toFixed(6)} ${b.hz.toFixed(6)}" ` +
      `rgba=".24 .27 .31 1" contype="1" conaffinity="3"/>`;
  }).join('\n');
}
