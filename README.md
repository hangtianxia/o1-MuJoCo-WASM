# O1 · MuJoCo WASM preview

Browser-side MuJoCo preview of the O1 29-DoF parkour checkpoint. Physics runs in
the MuJoCo WASM build; the policy runs in ONNX Runtime Web; rendering is three.js
with a WebGPU/WebGL2 liquid-glass compositor.

## Run

```bash
npm install
npm run serve
```

Open <http://127.0.0.1:8099/>. The first load downloads the MuJoCo WASM and the
default checkpoint policy; wait for `ready`.

## Included

- `public/config.json`, `public/robot_template.xml`, `public/policy.onnx`
- `public/meshes/`, `public/meshes_low/` (default O1 checkpoint visuals)
- `server.mjs`, `start.sh`, `public/app.js`, `public/terrain.js`
- `public/index.html`, `public/ambient.js`, `public/glassgl.js`, `public/glassgpu.js`
- `browser-check.mjs`, `sim2real-check.mjs`

`public/exports/` (per-checkpoint bundles), `uploads/`, screenshots and logs are
intentionally not committed. To add another checkpoint, export it into
`public/exports/<id>/` with `config.json`, `robot_template.xml`, `policy.onnx`
and the matching meshes, then select it from the checkpoint control.

## Controls

- `W/S` forward/back, `A/D` turn, `Q/E` strafe
- Arrow keys are aliases for `W/S/A/D`
- `T` toggles the theme
- Glass backend can be forced with `?glass=webgpu`, `?glass=gl`, or `?glass=css`

## Checks

```bash
npm run check
```

The browser checks use `playwright-core` with the installed Google Chrome; the
flat/stairs physics checks are expected to pass for the default O1 checkpoint.
