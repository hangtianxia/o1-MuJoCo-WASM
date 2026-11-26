// Shared optical parameters and YCC color-matrix helpers for the
// WebGPU and WebGL2 liquid-glass compositors, modeled after iOS 26.5 Liquid Glass.

export const MAX_PANELS = 12;

export const DEFAULT_PANEL_SELECTORS = [
  '#controls',
  '#telemetry',
  '.chips .chip',
  '#theme-toggle',
  '#status',
  '#mobile-dock',
  '#touch-joy',
];

const RGB_TO_YCC = [
  0.2126, 0.7152, 0.0722, 0, 0,
  -0.114572, -0.385428, 0.5, 0, 0.5,
  0.5, -0.454153, -0.045847, 0, 0.5,
  0, 0, 0, 1, 0,
];

const YCC_TO_RGB = [
  1, 0, 1.5748, 0, -0.7874,
  1, -0.187324, -0.468124, 0, 0.327724,
  1, 1.8556, 0, 0, -0.9278,
  0, 0, 0, 1, 0,
];

function matMul4x5(a, b) {
  const out = new Array(20);
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 5; col++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) {
        sum += a[row * 5 + k] * b[k * 5 + col];
      }
      if (col === 4) sum += a[row * 5 + 4];
      out[row * 5 + col] = sum;
    }
  }
  return out;
}

export function yccMatrix(black, white, saturation) {
  const luma = [
    white - black, 0, 0, 0, black,
    0, 1, 0, 0, 0,
    0, 0, 1, 0, 0,
    0, 0, 0, 1, 0,
  ];
  const chroma = [
    1, 0, 0, 0, 0,
    0, saturation, 0, 0, 0.5 - saturation * 0.5,
    0, 0, saturation, 0, 0.5 - saturation * 0.5,
    0, 0, 0, 1, 0,
  ];
  return matMul4x5(YCC_TO_RGB, matMul4x5(chroma, matMul4x5(luma, RGB_TO_YCC)));
}

// Returns 3 vec4 rows [r, g, b, offset] for dot(rgb, row.xyz) + row.w
export function yccRows(white, black, saturation, fill = [0, 0, 0, 0]) {
  const m = yccMatrix(black, white, saturation);
  const a = fill[3];
  const rest = 1 - a;
  return [0, 1, 2].map((row) => [
    m[row * 5 + 0] * rest,
    m[row * 5 + 1] * rest,
    m[row * 5 + 2] * rest,
    m[row * 5 + 4] * rest + fill[row] * a,
  ]);
}

function vibrancyFill(m, rgb, alpha) {
  const cap = 0.996;
  const gains = rgb.map((x) => {
    const p = Math.min(alpha * x, cap);
    return 1 + p / (1 - p);
  });
  return m.map((v, i) => (i < 15 ? v * gains[Math.floor(i / 5)] : v));
}

// Returns 3 vec4 rows for the rim vibrancy color matrix
export function rimVibrancyRows(theme) {
  if (theme === 'light') {
    // iOS 26.5 light rim: controls [0.9, 1.0, 1.5], fill null
    const m = yccMatrix(0.9, 1.0, 1.5);
    return [0, 1, 2].map((row) => [
      m[row * 5 + 0],
      m[row * 5 + 1],
      m[row * 5 + 2],
      m[row * 5 + 4],
    ]);
  }
  // iOS 26.5 dark rim: controls [0.06, 0.6, 1.2], fill [0.6, 0.6, 0.6, 1]
  let m = yccMatrix(0.06, 0.6, 1.2);
  m = vibrancyFill(m, [0.6, 0.6, 0.6], 1.0);
  return [0, 1, 2].map((row) => [
    m[row * 5 + 0],
    m[row * 5 + 1],
    m[row * 5 + 2],
    m[row * 5 + 4],
  ]);
}

export function getThemeGlassParams(theme) {
  const light = theme === 'light';
  // Ultra-clear liquid glass: in light mode, avoid lifting whites toward 1.0
  // (which bleaches out light gray geometry); gently preserve midtone contrast
  // so refracted edges and grid lines stay crisp through the pane.
  const face = light
    ? yccRows(0.95, 0.0, 1.24, [0, 0, 0, 0])
    : yccRows(0.94, 0.02, 1.16, [0.03, 0.06, 0.10, 0.06]);
  const bleed = light
    ? yccRows(0.96, 0.35, 1.30, [0, 0, 0, 0])
    : yccRows(0.75, 0.02, 1.22, [0, 0, 0, 0]);
  const rim = rimVibrancyRows(theme);

  return {
    light: light ? 1.0 : 0.0,
    faceCM0: face[0],
    faceCM1: face[1],
    faceCM2: face[2],
    bleedCM0: bleed[0],
    bleedCM1: bleed[1],
    bleedCM2: bleed[2],
    rimCM0: rim[0],
    rimCM1: rim[1],
    rimCM2: rim[2],
    // Shadow parameters: [offsetX, offsetY, opacity, radius]
    shadow: light ? [0.0, 8.0, 0.24, 24.0] : [0.0, 10.0, 0.38, 24.0],
    // Rim light parameters: [widthPx, curvature, keyOpacity, fillOpacity]
    rimParams: light ? [1.5, 0.65, 0.98, 0.84] : [1.5, 0.65, 0.98, 0.72],
    // Misc optical parameters: [bleedOpacity, bleedDarken, dispersion, blurScale]
    // Keep dispersion and blurScale identical across themes so refractive index matches.
    optics: [0.16, light ? 1.0 : 0.0, 0.045, 0.45],
  };
}

export function collectPanels(selectors = DEFAULT_PANEL_SELECTORS, maxPanels = MAX_PANELS) {
  const seen = new Set();
  const out = [];
  for (const selector of selectors) {
    const nodes = document.querySelectorAll(selector);
    for (const element of nodes) {
      if (out.length >= maxPanels) return out;
      if (seen.has(element)) continue;
      seen.add(element);
      if (element.hidden || element.offsetParent === null) continue;
      const style = getComputedStyle(element);
      if (style.display === 'none' || style.visibility === 'hidden' || Number.parseFloat(style.opacity) === 0) continue;
      const rect = element.getBoundingClientRect();
      if (rect.width <= 1 || rect.height <= 1) continue;
      const rawRadius = Number.parseFloat(style.borderTopLeftRadius) || 18;
      const radius = Math.min(rawRadius, rect.width * 0.5, rect.height * 0.5);
      out.push({ element, x: rect.left, y: rect.top, w: rect.width, h: rect.height, radius });
    }
  }
  return out;
}
