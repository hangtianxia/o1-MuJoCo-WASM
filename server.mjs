#!/usr/bin/env node
// Static preview server + checkpoint export API.
// Serves public/ at /, node_modules used by the browser, and the /api/*
// endpoints that scan checkpoint directories, accept uploads and run the
// CPU-only ONNX exporter.

import http from 'node:http';
import { createReadStream, createWriteStream, existsSync, mkdirSync, readFileSync } from 'node:fs';
import { mkdir, open, readFile, readdir, realpath, rename, rm, stat } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const PUBLIC = path.join(ROOT, 'public');
const NODE_MODULES = path.join(ROOT, 'node_modules');
const EXPORTS = path.join(PUBLIC, 'exports');
const UPLOADS = path.join(ROOT, 'uploads');
const EXPORT_LOGS = path.join(ROOT, 'export-logs');
const TRAIN = '/home/ubuntu/workspace/instinct-ssr-stair-audited-fromzero-20261012';
const ACTIVE_STATE = path.join(TRAIN, 'outputs/continuous60000_training_execution.json');
const PYTHON = process.env.PYTHON || '/home/ubuntu/.local/share/mamba/envs/instinctlab/bin/python';
const PORT = Number(process.env.PORT || 8099);
const HOST = process.env.HOST || '0.0.0.0';
const MAX_UPLOAD = 1.5 * 1024 * 1024 * 1024;

const CACHEABLE_EXT = new Set(['.wasm', '.onnx', '.stl', '.png', '.map']);
const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.wasm': 'application/wasm',
  '.onnx': 'application/octet-stream',
  '.stl': 'model/stl',
  '.xml': 'application/xml; charset=utf-8',
  '.png': 'image/png',
  '.map': 'application/json; charset=utf-8',
  '.txt': 'text/plain; charset=utf-8',
};

mkdirSync(EXPORTS, { recursive: true });
mkdirSync(UPLOADS, { recursive: true });
mkdirSync(EXPORT_LOGS, { recursive: true });

const jobs = new Map();
let activeJob = null;

function safeJoin(base, requestPath) {
  const decoded = decodeURIComponent(requestPath.split('?')[0]);
  const normalized = path.normalize(decoded).replace(/^(\.\.[/\\])+/, '');
  const full = path.join(base, normalized);
  if (!full.startsWith(base + path.sep) && full !== base) return null;
  return full;
}

function sendJson(res, status, value) {
  const body = JSON.stringify(value);
  res.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': Buffer.byteLength(body),
    'Cache-Control': 'no-store',
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Cross-Origin-Embedder-Policy': 'require-corp',
  });
  res.end(body);
}

async function readJson(req, limit = 1024 * 1024) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > limit) throw new Error('request body too large');
    chunks.push(chunk);
  }
  if (size === 0) return {};
  return JSON.parse(Buffer.concat(chunks).toString('utf8'));
}

function activeState() {
  return JSON.parse(readFileSync(ACTIVE_STATE, 'utf8'));
}

function assertDirectory(p) {
  if (typeof p !== 'string' || !path.isAbsolute(p)) throw new Error('path must be absolute');
  const resolved = path.resolve(p);
  const blocked = ['/proc', '/sys', '/dev'];
  if (blocked.some((b) => resolved === b || resolved.startsWith(b + path.sep))) {
    throw new Error('this directory is not available');
  }
  return resolved;
}

function parseIteration(name) {
  const m = /(?:^|_)model_(\d+)\.pt$/i.exec(name);
  return m ? Number(m[1]) : null;
}

async function listDirectory(dir) {
  const root = assertDirectory(dir);
  const info = await stat(root);
  if (!info.isDirectory()) throw new Error('not a directory');
  const found = [];
  const seen = new Set();
  const walk = async (current, depth) => {
    let entries;
    try { entries = await readdir(current, { withFileTypes: true }); }
    catch { return; }
    for (const entry of entries) {
      if (entry.name.startsWith('.')) continue;
      const full = path.join(current, entry.name);
      if (entry.isFile() && /\.pt$/i.test(entry.name)) {
        const real = await realpath(full).catch(() => full);
        if (seen.has(real)) continue;
        seen.add(real);
        const s = await stat(full).catch(() => null);
        if (!s) continue;
        const hasParams = existsSync(path.join(current, 'params', 'agent.yaml'));
        found.push({
          name: entry.name,
          path: full,
          size: s.size,
          mtime: s.mtimeMs,
          iteration: parseIteration(entry.name),
          runDir: current,
          runName: path.basename(current),
          hasParams,
          source: 'server',
          label: `${path.basename(current)} / ${entry.name}`,
        });
        if (found.length >= 600) return;
      } else if (entry.isDirectory() && depth > 0 && found.length < 600) {
        await walk(full, depth - 1);
      }
    }
  };
  await walk(root, 2);
  found.sort((a, b) => b.mtime - a.mtime);
  return { dir: root, checkpoints: found };
}

async function listUploads() {
  const entries = await readdir(UPLOADS, { withFileTypes: true }).catch(() => []);
  const out = [];
  for (const entry of entries) {
    if (!entry.isFile() || !/\.pt$/i.test(entry.name)) continue;
    const full = path.join(UPLOADS, entry.name);
    const s = await stat(full).catch(() => null);
    if (!s) continue;
    const displayName = entry.name.replace(/^\d+_/, '');
    out.push({
      name: entry.name,
      displayName,
      path: full,
      size: s.size,
      mtime: s.mtimeMs,
      iteration: parseIteration(entry.name),
      runDir: null,
      runName: 'upload',
      hasParams: false,
      source: 'upload',
      label: `upload / ${displayName}`,
    });
  }
  out.sort((a, b) => b.mtime - a.mtime);
  const deduped = [];
  const seen = new Set();
  for (const item of out) {
    const key = `${item.displayName}|${item.size}`;
    if (seen.has(key)) continue;
    seen.add(key);
    deduped.push(item);
  }
  return deduped;
}

async function listExports() {
  const entries = await readdir(EXPORTS, { withFileTypes: true }).catch(() => []);
  const out = [];
  for (const entry of entries) {
    if (!entry.isDirectory() || entry.name.startsWith('.')) continue;
    const configPath = path.join(EXPORTS, entry.name, 'config.json');
    if (!existsSync(configPath)) continue;
    try {
      const config = JSON.parse(await readFile(configPath, 'utf8'));
      const s = await stat(path.join(EXPORTS, entry.name)).catch(() => null);
      out.push({
        id: entry.name,
        base: `/exports/${entry.name}/`,
        checkpoint: config.checkpoint,
        observation_dim: config.observation_dim,
        mtime: s?.mtimeMs ?? 0,
      });
    } catch (_) { /* ignore broken exports */ }
  }
  out.sort((a, b) => b.mtime - a.mtime);
  return out;
}

async function sha256File(file) {
  return new Promise((resolve, reject) => {
    const hash = createHash('sha256');
    const stream = createReadStream(file);
    stream.on('data', (chunk) => hash.update(chunk));
    stream.on('error', reject);
    stream.on('end', () => resolve(hash.digest('hex')));
  });
}

function tailFile(file, maxBytes = 8192) {
  try {
    const data = readFileSync(file);
    return data.slice(Math.max(0, data.length - maxBytes)).toString('utf8');
  } catch (_) {
    return '';
  }
}

async function handleUpload(req, res) {
  const rawName = decodeURIComponent(String(req.headers['x-filename'] || 'checkpoint.pt'));
  const name = path.basename(rawName).replace(/[^A-Za-z0-9._-]/g, '_');
  if (!/\.pt$/i.test(name)) return sendJson(res, 400, { error: 'only .pt files can be uploaded' });
  const target = path.join(UPLOADS, `${Date.now()}_${name}`);
  const handle = await open(target, 'w');
  let size = 0;
  let tooBig = false;
  try {
    for await (const chunk of req) {
      size += chunk.length;
      if (size > MAX_UPLOAD) { tooBig = true; break; }
      await handle.write(chunk);
    }
  } finally {
    await handle.close();
  }
  if (tooBig) {
    await rm(target, { force: true });
    return sendJson(res, 413, { error: `upload exceeds ${Math.round(MAX_UPLOAD / 2 ** 20)} MB` });
  }
  if (size < 1024) {
    await rm(target, { force: true });
    return sendJson(res, 400, { error: 'upload is empty or too small' });
  }
  const head = Buffer.alloc(4);
  const fh = await open(target, 'r');
  await fh.read(head, 0, 4, 0);
  await fh.close();
  if (head[0] !== 0x50 || head[1] !== 0x4b) {
    await rm(target, { force: true });
    return sendJson(res, 400, { error: 'file does not look like a PyTorch .pt zip checkpoint' });
  }
  const s = await stat(target);
  const storedName = path.basename(target);
  // Keep only the newest copy of identical (name, size) uploads.
  const entries = await readdir(UPLOADS, { withFileTypes: true }).catch(() => []);
  for (const entry of entries) {
    if (!entry.isFile() || entry.name === storedName) continue;
    const display = entry.name.replace(/^\d+_/, '');
    if (display !== name) continue;
    const full = path.join(UPLOADS, entry.name);
    const old = await stat(full).catch(() => null);
    if (old && old.size === s.size) await rm(full, { force: true }).catch(() => {});
  }
  return sendJson(res, 200, {
    name: storedName, originalName: name, path: target, size: s.size, mtime: s.mtimeMs,
    source: 'upload', label: `upload / ${name}`, iteration: parseIteration(name),
  });
}

async function startExport(body) {
  if (activeJob) return { error: 'another export is already running', activeJob };
  let checkpoint = body.checkpoint ? path.resolve(String(body.checkpoint)) : null;
  if (body.uploadId) {
    const name = path.basename(String(body.uploadId));
    checkpoint = path.join(UPLOADS, name);
  }
  if (!checkpoint) return { error: 'checkpoint or uploadId is required' };
  if (!/\.pt$/i.test(checkpoint)) return { error: 'checkpoint must be a .pt file' };
  const info = await stat(checkpoint).catch(() => null);
  if (!info || !info.isFile()) return { error: 'checkpoint file not found' };
  if (info.size > MAX_UPLOAD) return { error: 'checkpoint is too large' };

  let runDir = body.runDir ? assertDirectory(String(body.runDir)) : path.dirname(checkpoint);
  if (!existsSync(path.join(runDir, 'params', 'agent.yaml'))) {
    runDir = activeState().actual_run;
  }
  const sha = await sha256File(checkpoint);
  const id = `ckpt_${sha.slice(0, 12)}`;
  const finalDir = path.join(EXPORTS, id);
  if (existsSync(path.join(finalDir, 'config.json'))) {
    const config = JSON.parse(await readFile(path.join(finalDir, 'config.json'), 'utf8'));
    return { id, cached: true, state: 'done', base: `/exports/${id}/`, checkpoint: config.checkpoint };
  }
  const tmpDir = path.join(EXPORTS, `.tmp_${id}_${Date.now()}`);
  await mkdir(tmpDir, { recursive: true });
  const logPath = path.join(EXPORT_LOGS, `${id}.log`);
  const log = createWriteStream(logPath, { flags: 'w' });
  const args = [
    path.join(ROOT, 'prepare.py'),
    '--checkpoint', checkpoint,
    '--out-dir', tmpDir,
    '--run-dir', runDir,
    '--label', String(body.label || path.basename(checkpoint, '.pt')),
  ];
  const child = spawn(PYTHON, args, {
    cwd: ROOT,
    env: { ...process.env, PYTHONUNBUFFERED: '1' },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  const job = {
    id, state: 'running', startedAt: Date.now(), logPath, tmpDir, finalDir,
    checkpoint, runDir, label: String(body.label || path.basename(checkpoint, '.pt')),
    sha256: sha,
  };
  jobs.set(id, job);
  activeJob = job;
  child.stdout.pipe(log);
  child.stderr.pipe(log);
  child.on('error', (err) => {
    job.state = 'error';
    job.error = String(err);
    activeJob = null;
    log.end(`\nspawn error: ${err}\n`);
  });
  child.on('exit', async (code) => {
    log.end(`\nexit ${code}\n`);
    if (code === 0 && existsSync(path.join(tmpDir, 'config.json'))) {
      try {
        await rename(tmpDir, finalDir);
        const config = JSON.parse(await readFile(path.join(finalDir, 'config.json'), 'utf8'));
        job.state = 'done';
        job.base = `/exports/${id}/`;
        job.checkpointInfo = config.checkpoint;
        job.result = config;
      } catch (err) {
        job.state = 'error';
        job.error = `commit failed: ${err.message}`;
      }
    } else {
      job.state = 'error';
      job.error = tailFile(logPath, 4096) || `exporter exited with ${code}`;
      await rm(tmpDir, { recursive: true, force: true }).catch(() => {});
    }
    activeJob = null;
  });
  return { id, state: 'running', base: `/exports/${id}/` };
}

async function handleApi(req, res, url) {
  try {
    if (url.pathname === '/api/info' && req.method === 'GET') {
      const state = activeState();
      return sendJson(res, 200, {
        activeRun: state.actual_run,
        sourceSha: state.source_sha,
        defaultDir: state.actual_run,
        exports: await listExports(),
        uploads: await listUploads(),
      });
    }
    if (url.pathname === '/api/checkpoints' && req.method === 'GET') {
      const dir = url.searchParams.get('dir') || activeState().actual_run;
      const result = await listDirectory(dir);
      return sendJson(res, 200, { ...result, uploads: await listUploads(), exports: await listExports() });
    }
    if (url.pathname === '/api/uploads' && req.method === 'GET') {
      return sendJson(res, 200, { uploads: await listUploads() });
    }
    if (url.pathname === '/api/upload' && req.method === 'POST') {
      return await handleUpload(req, res);
    }
    if (url.pathname === '/api/export' && req.method === 'POST') {
      const body = await readJson(req);
      const result = await startExport(body);
      const status = result.error ? (result.error.includes('already running') ? 409 : 400) : 200;
      return sendJson(res, status, result);
    }
    if (url.pathname.startsWith('/api/export/') && req.method === 'GET') {
      const id = decodeURIComponent(url.pathname.slice('/api/export/'.length));
      if (!/^[A-Za-z0-9._-]+$/.test(id)) return sendJson(res, 400, { error: 'bad id' });
      const job = jobs.get(id);
      if (job) {
        return sendJson(res, 200, {
          id: job.id, state: job.state, error: job.error || null,
          log: tailFile(job.logPath), base: job.base || null,
          checkpoint: job.checkpointInfo || null,
          elapsed: Date.now() - job.startedAt,
        });
      }
      const configPath = path.join(EXPORTS, id, 'config.json');
      if (existsSync(configPath)) {
        const config = JSON.parse(await readFile(configPath, 'utf8'));
        return sendJson(res, 200, { id, state: 'done', base: `/exports/${id}/`, checkpoint: config.checkpoint, log: '', elapsed: 0 });
      }
      return sendJson(res, 404, { error: 'unknown export' });
    }
    if (url.pathname === '/api/exports' && req.method === 'GET') {
      return sendJson(res, 200, { exports: await listExports() });
    }
    return sendJson(res, 404, { error: 'unknown API' });
  } catch (err) {
    return sendJson(res, 400, { error: err.message });
  }
}

async function resolveFile(urlPath) {
  if (urlPath === '/' || urlPath === '') return path.join(PUBLIC, 'index.html');
  if (urlPath.startsWith('/node_modules/')) return safeJoin(ROOT, urlPath);
  return safeJoin(PUBLIC, urlPath);
}

const server = http.createServer(async (req, res) => {
  const started = performance.now();
  let status = 500;
  let bytes = 0;
  try {
    const url = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
    if (url.pathname.startsWith('/api/')) {
      await handleApi(req, res, url);
      status = res.statusCode;
      return;
    }
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      status = 405;
      res.writeHead(status, { Allow: 'GET, HEAD' });
      res.end('method not allowed');
      return;
    }
    let file = await resolveFile(url.pathname);
    if (!file) {
      status = 403;
      res.writeHead(status);
      res.end('forbidden');
      return;
    }
    let info;
    try {
      info = await stat(file);
      if (info.isDirectory()) {
        file = path.join(file, 'index.html');
        info = await stat(file);
      }
    } catch {
      status = 404;
      res.writeHead(status, { 'Content-Type': 'text/plain; charset=utf-8' });
      res.end(`not found: ${url.pathname}`);
      return;
    }

    const ext = path.extname(file).toLowerCase();
    const cacheable = file.startsWith(NODE_MODULES + path.sep) || CACHEABLE_EXT.has(ext);
    const etag = `"${info.size.toString(16)}-${Math.floor(info.mtimeMs).toString(16)}"`;
    if (cacheable && req.headers['if-none-match'] === etag) {
      status = 304;
      res.writeHead(304, { ETag: etag, 'Cache-Control': 'public, max-age=86400' });
      res.end();
      return;
    }
    const headers = {
      'Content-Type': MIME[ext] || 'application/octet-stream',
      'Content-Length': String(info.size),
      'ETag': etag,
      'Cache-Control': cacheable ? 'public, max-age=86400' : 'no-store',
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Resource-Policy': 'cross-origin',
      'Access-Control-Allow-Origin': '*',
    };
    status = 200;
    res.writeHead(status, headers);
    if (req.method === 'HEAD') res.end();
    else {
      bytes = info.size;
      createReadStream(file).pipe(res);
    }
  } catch (err) {
    status = 500;
    if (!res.headersSent) res.writeHead(status, { 'Content-Type': 'text/plain; charset=utf-8' });
    res.end(`server error: ${err.message}`);
  } finally {
    const ms = (performance.now() - started).toFixed(1);
    console.log(`${req.method} ${req.url} ${status} ${bytes}B ${ms}ms`);
  }
});

server.listen(PORT, HOST, () => {
  console.log(`MuJoCo WASM preview: http://${HOST}:${PORT}/`);
  console.log(`  public:      ${PUBLIC}`);
  console.log(`  exports:     ${EXPORTS}`);
  console.log(`  uploads:     ${UPLOADS}`);
  console.log(`  exporter:    ${PYTHON}`);
});

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => {
    console.log(`\n${signal} received, shutting down`);
    server.close(() => process.exit(0));
  });
}
