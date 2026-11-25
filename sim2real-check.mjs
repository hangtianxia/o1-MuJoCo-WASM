#!/usr/bin/env node
// Headless sim2real sensitivity check for the MuJoCo WASM preview.
// Usage: node sim2real-check.mjs [smoke|matrix]
import { chromium } from 'playwright-core';

const URL_BASE = process.env.PREVIEW_URL || 'http://127.0.0.1:8099';
const EXPORT = process.env.PREVIEW_EXPORT || 'ckpt_fe73c1c39d69';
const CHROME = process.env.CHROME_PATH || '/usr/bin/google-chrome';
const mode = process.argv[2] || 'smoke';

const browser = await chromium.launch({
  executablePath: CHROME,
  headless: true,
  args: [
    '--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu-sandbox',
    '--ignore-gpu-blocklist', '--enable-webgl', '--enable-unsafe-swiftshader',
    '--use-gl=angle', '--use-angle=swiftshader', '--window-size=1440,900',
  ],
});
const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
await context.addInitScript((id) => { localStorage.setItem('preview.export', id); }, EXPORT);
const page = await context.newPage();
const errors = [];
page.on('pageerror', (err) => { errors.push(String(err)); console.error('PAGEERROR', err.stack || err); });
page.on('console', (msg) => { if (msg.type() === 'error') errors.push(msg.text()); });

await page.goto(`${URL_BASE}/?autostart=check`, { waitUntil: 'domcontentloaded', timeout: 60000 });
await page.waitForFunction(() => window.__preview?.ready === true, null, { timeout: 180000 });
const info = await page.evaluate(() => ({
  iteration: window.__preview.checkpoint.iteration,
  sha: window.__preview.checkpoint.sha256.slice(0, 12),
}));
console.log('checkpoint', JSON.stringify(info));

await page.evaluate(() => window.__preview.setTerrain({
  mode: 'uniform', direction: 'down',
  uniform: { riser: 0.18, tread: 0.28, steps: 9, width: 1.0 },
}));
console.log('terrain', JSON.stringify(await page.evaluate(() => window.__preview.terrainInfo())));

async function run(label, sim2real, steps, command = null) {
  await page.evaluate((s2r) => window.__preview.setSim2real(s2r), sim2real);
  await page.evaluate(() => window.__preview.reset());
  const started = Date.now();
  const result = await page.evaluate(({ n, command }) => window.__preview.runCheck({ steps: n, command }), { n: steps, command });
  const foot = result.end.foot;
  const row = {
    label,
    seconds: ((Date.now() - started) / 1000).toFixed(1),
    deltaX: Number(result.deltaX.toFixed(3)),
    deltaZ: Number(result.deltaZ.toFixed(3)),
    fell: result.fell,
    invalid: result.invalid,
    endX: Number(result.end.x.toFixed(2)),
    startZ: Number(result.start.z.toFixed(2)),
    endZ: Number(result.end.z.toFixed(2)),
    steps: result.end.steps,
    simTime: Number(result.end.simTime.toFixed(2)),
    clearance: Number(result.end.clearance.toFixed(2)),
    footLeft: foot?.left ? foot.left.map((v) => Number(v.toFixed(3))) : null,
    footRight: foot?.right ? foot.right.map((v) => Number(v.toFixed(3))) : null,
  };
  console.log('RUN', JSON.stringify(row));
  await page.screenshot({ path: `/tmp/sim2real-${label.replace(/[^a-z0-9]+/gi, '_')}.png` });
  return row;
}

if (mode === 'quick') {
  const steps = Number(process.env.CHECK_STEPS || 200);
  await page.evaluate(() => window.__preview.setTerrain({ mode: 'flat', direction: 'flat' }));
  await run('flat-baseline', {}, steps);
  await run('flat-shoe', { shoe: true }, steps);
  await run('flat-shoe-notorsion-noextrude', { shoe: false }, steps);
  await page.evaluate(() => window.__preview.setTerrain({
    mode: 'uniform', direction: 'down', uniform: { riser: 0.18, tread: 0.28, steps: 9, width: 1.0 },
  }));
  const long = Number(process.env.CHECK_LONG || 800);
  await run('stairs-baseline', {}, long);
  await run('stairs-shoe', { shoe: true }, long);
  await run('stairs-friction-delay', { friction: 0.3, damping: 0.2, actuatorDelayMs: 25 }, long);
  await run('stairs-state-bias', { encoderZeroDeg: 1.5, imuPitchBiasDeg: 1.5, gyroBiasDeg: 2, jointVelFilterMs: 25 }, long);
  await run('stairs-backlash', { backlashDeg: 1.0 }, long);
  await run('stairs-combined', {
    friction: 0.3, damping: 0.2, actuatorDelayMs: 25, encoderZeroDeg: 1.5,
    imuPitchBiasDeg: 1.5, gyroBiasDeg: 2, jointVelFilterMs: 25, backlashDeg: 1.0, shoe: true,
  }, long);
} else if (mode === 'real') {
  const steps = Number(process.env.CHECK_STEPS || 600);
  const ZERO = {
    friction: 0, damping: 0, actuatorDelayMs: 0, encoderZeroDeg: 0,
    imuPitchBiasDeg: 0, gyroBiasDeg: 0, jointVelFilterMs: 0, backlashDeg: 0, shoe: false,
  };
  const noEnc = { imuPitchBiasDeg: 1.5, gyroBiasDeg: 2, jointVelFilterMs: 25 };
  const noEnc2 = { imuPitchBiasDeg: 2.0, gyroBiasDeg: 2, jointVelFilterMs: 25 };
  const mech = { friction: 0.2, damping: 0.2, actuatorDelayMs: 25 };
  const mechHard = { friction: 0.3, damping: 0.2, actuatorDelayMs: 30 };
  const worst = { ...noEnc, ...mech, backlashDeg: 0.5, shoe: true };
  const geoms = [
    ['16-28', 0.16, 0.28, 4.0],
    ['14-40', 0.14, 0.40, 4.0],
    ['16-28-w12', 0.16, 0.28, 1.2],
    ['16-28-w10', 0.16, 0.28, 1.0],
  ];
  for (const [name, riser, tread, width] of geoms) {
    for (const direction of ['down', 'up']) {
      await page.evaluate(({ direction, riser, tread, width }) => window.__preview.setTerrain({
        mode: 'uniform', direction, uniform: { riser, tread, steps: 9, width },
      }), { direction, riser, tread, width });
      const v = [0.6, 0, 0];
      await run(`${name}-${direction}-nominal`, { ...ZERO }, steps, v);
      await run(`${name}-${direction}-state-p15`, { ...ZERO, ...noEnc }, steps, v);
      await run(`${name}-${direction}-state-p20`, { ...ZERO, ...noEnc2 }, steps, v);
      await run(`${name}-${direction}-mech`, { ...ZERO, ...mech }, steps, v);
      await run(`${name}-${direction}-mech-hard`, { ...ZERO, ...mechHard }, steps, v);
      await run(`${name}-${direction}-backlash`, { ...ZERO, backlashDeg: 1.0 }, steps, v);
      await run(`${name}-${direction}-shoe`, { ...ZERO, shoe: true }, steps, v);
      await run(`${name}-${direction}-worst`, { ...ZERO, ...worst }, steps, v);
    }
  }
} else if (mode === 'threshold') {
  const steps = Number(process.env.CHECK_STEPS || 600);
  const ZERO = {
    friction: 0, damping: 0, actuatorDelayMs: 0, encoderZeroDeg: 0,
    imuPitchBiasDeg: 0, gyroBiasDeg: 0, jointVelFilterMs: 0, backlashDeg: 0, shoe: false,
  };
  const mech = { friction: 0.3, damping: 0.2, actuatorDelayMs: 25 };
  const stateb = { encoderZeroDeg: 1.5, imuPitchBiasDeg: 1.5, gyroBiasDeg: 2, jointVelFilterMs: 25 };
  await page.evaluate(() => window.__preview.setTerrain({
    mode: 'uniform', direction: 'down', uniform: { riser: 0.18, tread: 0.28, steps: 9, width: 4.0 },
  }));
  for (const speed of [0.45, 0.50, 0.55, 0.60]) {
    await run(`down-r18-t28-v${Math.round(speed * 100)}-baseline`, { ...ZERO }, steps, [speed, 0, 0]);
  }
  await run('down-r18-t28-v60-state', { ...ZERO, ...stateb }, steps, [0.60, 0, 0]);
  await run('down-r18-t28-v60-mech', { ...ZERO, ...mech }, steps, [0.60, 0, 0]);
  await run('down-r18-t28-v60-backlash', { ...ZERO, backlashDeg: 1.0 }, steps, [0.60, 0, 0]);
  await run('down-r18-t28-v60-shoe', { ...ZERO, shoe: true }, steps, [0.60, 0, 0]);
  await run('down-r18-t28-v60-combined-noshoe', { ...ZERO, ...mech, ...stateb, backlashDeg: 1.0 }, steps, [0.60, 0, 0]);
  await run('down-r18-t28-v60-combined-shoe', { ...ZERO, ...mech, ...stateb, backlashDeg: 1.0, shoe: true }, steps, [0.60, 0, 0]);
  await run('down-r18-t28-v50-state', { ...ZERO, ...stateb }, steps, [0.50, 0, 0]);
  await run('down-r18-t28-v50-mech', { ...ZERO, ...mech }, steps, [0.50, 0, 0]);
  await run('down-r18-t28-v50-backlash', { ...ZERO, backlashDeg: 1.0 }, steps, [0.50, 0, 0]);
  await page.evaluate(() => window.__preview.setTerrain({ mode: 'uniform', direction: 'up', uniform: { riser: 0.18, tread: 0.28, steps: 9, width: 4.0 } }));
  await run('up-r18-t28-v45-baseline', { ...ZERO }, steps, [0.45, 0, 0]);
  await run('up-r18-t28-v45-state', { ...ZERO, ...stateb }, steps, [0.45, 0, 0]);
} else if (mode === 'geometry') {
  const steps = Number(process.env.CHECK_STEPS || 600);
  const cases = [
    ['down-r12-t35', 'down', 0.12, 0.35, 0.30],
    ['down-r12-t35-v45', 'down', 0.12, 0.35, 0.45],
    ['down-r12-t35-v60', 'down', 0.12, 0.35, 0.60],
    ['down-r18-t28-v30', 'down', 0.18, 0.28, 0.30],
    ['down-r18-t28-v45', 'down', 0.18, 0.28, 0.45],
    ['down-r18-t28-v60', 'down', 0.18, 0.28, 0.60],
    ['down-r18-t28-v75', 'down', 0.18, 0.28, 0.75],
    ['down-r18-t35-v45', 'down', 0.18, 0.35, 0.45],
    ['down-r15-t28-v45', 'down', 0.15, 0.28, 0.45],
    ['up-r18-t28-v45', 'up', 0.18, 0.28, 0.45],
  ];
  for (const [label, direction, riser, tread, speed] of cases) {
    await page.evaluate(({ direction, riser, tread }) => window.__preview.setTerrain({
      mode: 'uniform', direction, uniform: { riser, tread, steps: 9, width: 4.0 },
    }), { direction, riser, tread });
    await run(label, {}, steps, [speed, 0, 0]);
  }
} else if (mode === 'width') {
  const steps = Number(process.env.CHECK_STEPS || 800);
  for (const width of [4.0, 1.6, 1.0]) {
    await page.evaluate((w) => window.__preview.setTerrain({
      mode: 'uniform', direction: 'down', uniform: { riser: 0.18, tread: 0.28, steps: 9, width: w },
    }), width);
    await run(`stairs-w${width}-r18-t28`, {}, steps);
  }
  await page.evaluate(() => window.__preview.setTerrain({
    mode: 'uniform', direction: 'down', uniform: { riser: 0.12, tread: 0.35, steps: 12, width: 4.0 },
  }));
  await run('stairs-w4-r12-t35-default', {}, steps);
} else if (mode === 'smoke') {
  console.log('sim2real default', JSON.stringify(await page.evaluate(() => window.__preview.sim2realState())));
  await run('smoke-baseline', {}, 40);
  await run('smoke-shoe', { shoe: true }, 40);
} else {
  const steps = Number(process.env.CHECK_STEPS || 300);
  const rows = [];
  rows.push(await run('baseline', {}, steps));
  rows.push(await run('mech-friction-delay', { friction: 0.3, damping: 0.2, actuatorDelayMs: 25 }, steps));
  rows.push(await run('state-bias', { encoderZeroDeg: 1.5, imuPitchBiasDeg: 1.5, gyroBiasDeg: 2, jointVelFilterMs: 25 }, steps));
  rows.push(await run('shoe', { shoe: true }, steps));
  rows.push(await run('backlash', { backlashDeg: 1.0 }, steps));
  rows.push(await run('combined', {
    friction: 0.3, damping: 0.2, actuatorDelayMs: 25, encoderZeroDeg: 1.5,
    imuPitchBiasDeg: 1.5, gyroBiasDeg: 2, jointVelFilterMs: 25, backlashDeg: 1.0, shoe: true,
  }, steps));
  console.log('SUMMARY', JSON.stringify(rows, null, 2));
}

console.log('errors', JSON.stringify(errors.slice(0, 10)));
await browser.close();
