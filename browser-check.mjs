#!/usr/bin/env node
// Headless verification for the MuJoCo WASM preview.
// Usage: node browser-check.mjs [url]

import { chromium } from 'playwright-core';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const URL_BASE = process.argv[2] || process.env.PREVIEW_URL || 'http://127.0.0.1:8099';
const CHROME = process.env.CHROME_PATH || '/usr/bin/google-chrome';

function fail(message) {
  throw new Error(message);
}

const browser = await chromium.launch({
  executablePath: CHROME,
  headless: true,
  args: [
    '--no-sandbox',
    '--disable-dev-shm-usage',
    '--disable-gpu-sandbox',
    '--ignore-gpu-blocklist',
    '--enable-webgl',
    '--enable-unsafe-swiftshader',
    '--use-gl=angle',
    '--use-angle=swiftshader',
    '--window-size=1440,900',
  ],
});

const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
const page = await context.newPage();
const consoleErrors = [];
const pageErrors = [];
page.on('console', (msg) => {
  if (msg.type() === 'error') consoleErrors.push(msg.text());
});
page.on('pageerror', (err) => { pageErrors.push(String(err)); console.error('PAGEERROR', err.stack || err); });

let result;
try {
  console.log(`opening ${URL_BASE}/?autostart=check`);
  await page.goto(`${URL_BASE}/?autostart=check`, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForFunction(() => window.__preview?.ready === true, null, { timeout: 180000 });
  const info = await page.evaluate(() => ({
    checkpoint: window.__preview.checkpoint,
    layout: window.__preview.observationLayout,
  }));
  console.log('checkpoint:', info.checkpoint.iteration, info.checkpoint.sha256.slice(0, 12));
  console.log('layout:', JSON.stringify(info.layout));

  const run = async (direction, level, steps, label) => {
    const started = Date.now();
    const data = await page.evaluate(async ({ direction, level, steps }) => {
      await window.__preview.reset({ direction, level });
      return window.__preview.runCheck({ steps });
    }, { direction, level, steps });
    const elapsed = Date.now() - started;
    console.log(`[${label}] raw: ${JSON.stringify(data)}`);
    console.log(`[${label}] ${steps} steps in ${elapsed} ms (${(elapsed / steps).toFixed(1)} ms/step)`);
    console.log(`  x ${data.start.x.toFixed(3)} -> ${data.end.x.toFixed(3)} (dx ${data.deltaX.toFixed(3)})`);
    console.log(`  z ${data.start.z.toFixed(3)} -> ${data.end.z.toFixed(3)} (dz ${data.deltaZ.toFixed(3)})`);
    console.log(`  clearance ${data.end.clearance.toFixed(3)} min ${data.end.minClearance.toFixed(3)}  maxAction ${data.maxAction.toFixed(2)}  fell=${data.fell}  invalid=${data.invalid}`);
    console.log(`  depth range ${data.depthRange[0].toFixed(2)}..${data.depthRange[1].toFixed(2)}`);
    return data;
  };

  const flat = await run('flat', 0, 250, 'flat');
  await page.screenshot({ path: path.join(ROOT, 'check-flat.png') });

  const stairs = await run('up', 7, 250, 'up uniform');
  await page.waitForTimeout(250);
  await page.screenshot({ path: path.join(ROOT, 'check-stairs.png') });

  if (pageErrors.length) fail(`page errors:\n${pageErrors.join('\n')}`);
  if (consoleErrors.length) fail(`console errors:\n${consoleErrors.join('\n')}`);
  if (!flat.end.valid || flat.fell) fail('flat check failed: state invalid or robot fell');
  if (flat.deltaX < 0.6) fail(`flat check failed: forward progress ${flat.deltaX.toFixed(3)} m < 0.6 m`);
  if (flat.end.minZ < 0.35) fail(`flat check failed: pelvis dropped to ${flat.end.minZ.toFixed(3)} m`);
  if (flat.maxAction > 30) fail(`flat check failed: action magnitude ${flat.maxAction.toFixed(2)} out of range`);

  if (!stairs.end.valid || stairs.fell) fail('stairs check failed: state invalid or robot fell');
  if (stairs.deltaX < 0.5) fail(`stairs check failed: forward progress ${stairs.deltaX.toFixed(3)} m < 0.5 m`);
  if (stairs.deltaZ < 0.05) fail(`stairs check failed: climbed only ${stairs.deltaZ.toFixed(3)} m`);
  if (stairs.depthRange[0] < 0 || stairs.depthRange[1] > 1.0001) fail('depth normalized range out of [0,1]');

  result = {
    ok: true,
    checkpoint: info.checkpoint,
    layout: info.layout,
    flat,
    stairs,
    screenshots: ['check-flat.png', 'check-stairs.png'],
  };
  console.log('\nRESULT ' + JSON.stringify(result));
} catch (err) {
  result = { ok: false, error: err.message, consoleErrors, pageErrors };
  console.error('\nCHECK FAILED:', err.message);
  console.error('stack:', err.stack);
  console.error('consoleErrors:', consoleErrors);
  console.error('pageErrors:', pageErrors);
  try {
    await page.screenshot({ path: path.join(ROOT, 'check-failure.png') });
  } catch (_) {}
  process.exitCode = 1;
} finally {
  await browser.close();
}
