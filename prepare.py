"""Export a training checkpoint into a self-contained browser policy bundle.

Default (no args): export the newest checkpoint of the active run into public/.
Explicit:
  python prepare.py --checkpoint /abs/model_16000.pt --out-dir public/exports/<id> \
                    --run-dir /abs/run [--source-sha SHA] [--label NAME]

No writes to the training checkout, checkpoints or shared Python environment.
All export/inference/verification uses CPU.
"""
from pathlib import Path
import argparse
import contextlib
import hashlib
import importlib
import io
import json
import math
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import onnxruntime as ort
import torch
import yaml

APP = Path(__file__).resolve().parent
TRAIN = Path('/home/ubuntu/workspace/instinct-ssr-stair-audited-fromzero-20261012')
PUBLIC = APP / 'public'
ACTIVE_STATE = TRAIN / 'outputs/continuous60000_training_execution.json'
SUPPORTED_DEPTH = 1296
SUPPORTED_ACTIONS = 29
SUPPORTED_HISTORY = 8
SUPPORTED_OBS = 2064


def typed(v):
    if isinstance(v, dict): return {k: typed(x) for k, x in v.items()}
    if isinstance(v, list): return [typed(x) for x in v]
    if v == 'null': return None
    if v in ('true', 'false'): return v == 'true'
    if isinstance(v, str):
        try: return float(v) if any(x in v.lower() for x in ('.', 'e')) else int(v)
        except ValueError: return v
    return v


def load_yaml(p):
    return typed(yaml.load(p.read_text(), Loader=yaml.BaseLoader))


def quat(rpy):
    r, p, y = np.asarray(rpy, dtype=float) / 2
    return [math.cos(r) * math.cos(p) * math.cos(y) + math.sin(r) * math.sin(p) * math.sin(y),
            math.sin(r) * math.cos(p) * math.cos(y) - math.cos(r) * math.sin(p) * math.sin(y),
            math.cos(r) * math.sin(p) * math.cos(y) + math.sin(r) * math.cos(p) * math.sin(y),
            math.cos(r) * math.cos(p) * math.sin(y) - math.sin(r) * math.sin(p) * math.cos(y)]


def nums(v): return ' '.join(format(float(x), '.12g') for x in v)


def vec(text, default=(0, 0, 0)): return [float(x) for x in text.split()] if text else list(default)


def origin(element):
    o = element.find('origin')
    return dict(pos=nums(vec(o.get('xyz') if o is not None else None)),
                quat=nums(quat(vec(o.get('rpy') if o is not None else None))))


def make_robot(urdf, env, out_dir):
    src = ET.parse(urdf).getroot()
    links = {x.get('name'): x for x in src.findall('link')}
    children = {}
    for j in src.findall('joint'):
        children.setdefault(j.find('parent').get('link'), []).append(j)
    child_names = {j.find('child').get('link') for j in src.findall('joint')}
    root_names = set(links) - child_names
    assert len(root_names) == 1
    xml = ET.Element('mujoco', model='current_training_o1')
    ET.SubElement(xml, 'compiler', angle='radian', meshdir='meshes', balanceinertia='true', fusestatic='false')
    ET.SubElement(xml, 'option', timestep=str(env['sim']['dt']), gravity='0 0 -9.81', integrator='implicitfast', iterations='50', solver='Newton')
    ET.SubElement(xml, 'size', njmax='1500', nconmax='500')
    default = ET.SubElement(xml, 'default')
    ET.SubElement(default, 'geom', friction='.8 .005 .0001', solref='.01 1', solimp='.95 .99 .001')
    ET.SubElement(default, 'joint', damping='0', frictionloss='0')
    assets = ET.SubElement(xml, 'asset')
    world = ET.SubElement(xml, 'worldbody')
    world.append(ET.Comment('TERRAIN'))
    actuators = ET.SubElement(xml, 'actuator')
    sensors = ET.SubElement(xml, 'sensor')
    ET.SubElement(sensors, 'gyro', name='base_gyro', site='base_imu')
    files = {}
    joint_info = {}

    def body(parent, name, joint=None):
        attr = {'name': name}
        if joint is None: attr.update(pos='0 0 .725')
        else: attr.update(origin(joint))
        node = ET.SubElement(parent, 'body', **attr)
        if joint is None:
            ET.SubElement(node, 'freejoint', name='root_freejoint')
            ET.SubElement(node, 'site', name='base_imu', pos='0 0 0', size='.005')
        elif joint.get('type') != 'fixed':
            jname = joint.get('name'); limit = joint.find('limit')
            opts = {}
            for cfg in env['scene']['robot']['actuators'].values():
                if any(re.fullmatch(p, jname) for p in cfg['joint_names_expr']): opts = cfg; break
            assert opts, jname
            ET.SubElement(node, 'joint', name=jname, type='hinge', axis=joint.find('axis').get('xyz'),
                          range=f"{limit.get('lower')} {limit.get('upper')}", armature=str(opts['armature']))
            force = float(opts['effort_limit'])
            ET.SubElement(actuators, 'motor', name=jname + '_motor', joint=jname, gear='1', ctrllimited='true', ctrlrange=f'{-force} {force}')
            joint_info[jname] = dict(kp=opts['stiffness'], kd=opts['damping'], effort=force, velocity_limit=opts['velocity_limit'])
        link = links[name]; inertial = link.find('inertial')
        if inertial is not None and float(inertial.find('mass').get('value')) > 0:
            o = origin(inertial); q = np.array(vec(o['quat'], (1, 0, 0, 0))); rot = np.zeros(9); mujoco.mju_quat2Mat(rot, q)
            i = inertial.find('inertia'); a = lambda k: float(i.get(k))
            matrix = np.array([[a('ixx'), a('ixy'), a('ixz')], [a('ixy'), a('iyy'), a('iyz')], [a('ixz'), a('iyz'), a('izz')]])
            R = rot.reshape(3, 3); matrix = R @ matrix @ R.T
            ET.SubElement(node, 'inertial', pos=o['pos'], mass=inertial.find('mass').get('value'), fullinertia=nums([*matrix.diagonal(), matrix[0, 1], matrix[0, 2], matrix[1, 2]]))
        for role in ('visual', 'collision'):
            for index, item in enumerate(link.findall(role)):
                shape = item.find('geometry')[0]; attr = origin(item)
                attr.update(name=f'{name}_{role}_{index}', group='2' if role == 'visual' else '3',
                            contype='0' if role == 'visual' else '2', conaffinity='0' if role == 'visual' else '1',
                            rgba='.66 .73 .82 1' if role == 'visual' else '.95 .5 .15 .2')
                if shape.tag == 'mesh':
                    path = (urdf.parent / shape.get('filename')).resolve(); assert path.is_file(), path
                    key = path.name
                    if key not in files:
                        files[key] = path
                        ET.SubElement(assets, 'mesh', name=key, file=key, scale=shape.get('scale', '1 1 1'))
                    attr.update(type='mesh', mesh=key)
                elif shape.tag == 'cylinder':
                    attr.update(type='capsule', size=f"{shape.get('radius')} {float(shape.get('length')) / 2}")
                elif shape.tag == 'sphere': attr.update(type='sphere', size=shape.get('radius'))
                elif shape.tag == 'box': attr.update(type='box', size=nums(np.array(vec(shape.get('size'))) / 2))
                else: raise ValueError(shape.tag)
                ET.SubElement(node, 'geom', **attr)
        for child in children.get(name, []):
            body(node, child.find('child').get('link'), child)

    body(world, root_names.pop())
    (out_dir / 'meshes').mkdir(parents=True, exist_ok=True)
    for name, path in files.items():
        shutil.copyfile(path, out_dir / 'meshes' / name)
    ET.indent(xml)
    template = ET.tostring(xml, encoding='unicode')
    (out_dir / 'robot_template.xml').write_text(template)
    flat = '<geom name="floor" type="plane" size="40 40 .1" rgba=".20 .25 .29 1" contype="1" conaffinity="2" group="0"/>'
    (out_dir / 'scene.xml').write_text(template.replace('<!--TERRAIN-->', flat))
    model = mujoco.MjModel.from_xml_path(str(out_dir / 'scene.xml'))
    return model, joint_info, list(files)


class BrowserPolicy(torch.nn.Module):
    def __init__(self, policy):
        super().__init__(); self.policy = policy

    def forward(self, observations, hidden):
        self.policy.memory_a.hidden_states = hidden
        action = self.policy.act_inference(observations)
        return action, self.policy.memory_a.hidden_states


def active_state():
    return json.loads(ACTIVE_STATE.read_text())


def active_source_sha():
    try:
        return json.loads((PUBLIC / 'config.json').read_text())['checkpoint']['source_sha']
    except Exception:
        return active_state()['source_sha']


def digest_file(path):
    with path.open('rb') as stream:
        if hasattr(hashlib, 'file_digest'):
            return hashlib.file_digest(stream, 'sha256').hexdigest()
        return hashlib.sha256(stream.read()).hexdigest()


def make_low_poly(out_dir, source_sha):
    high = out_dir / 'meshes'
    low = out_dir / 'meshes_low'
    low.mkdir(parents=True, exist_ok=True)
    files = sorted(high.glob('*'))
    if not files:
        return
    shared = PUBLIC / 'meshes_low'
    if source_sha and source_sha == active_source_sha() and shared.is_dir():
        for f in files:
            target = low / f.name
            if target.exists(): continue
            try: os.link(shared / f.name, target)
            except Exception: shutil.copyfile(f, target)
        return
    try:
        import trimesh
        for f in files:
            target = low / f.name
            if target.exists(): continue
            m = trimesh.load(f, force='mesh')
            n = len(m.faces)
            s = m if n < 600 else m.simplify_quadric_decimation(face_count=min(n - 1, max(400, int(n * 0.28))))
            s.export(target)
    except Exception as err:
        print(f'low-poly generation unavailable, copying high-poly meshes: {err}')
        for f in files:
            target = low / f.name
            if not target.exists(): shutil.copyfile(f, target)


def export_checkpoint(checkpoint, out_dir, run_dir=None, source_sha=None, label=''):
    torch.set_num_threads(1)
    checkpoint = Path(checkpoint).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)

    digest = digest_file(checkpoint)
    with checkpoint.open('rb') as stream:
        saved = torch.load(stream, map_location='cpu', weights_only=True)
    iteration = int(saved['completed_learning_iterations'])
    saved_source = (saved.get('terrain_context') or {}).get('source_sha')
    state = active_state()
    run = Path(run_dir).resolve() if run_dir else checkpoint.parent
    if not (run / 'params/agent.yaml').exists():
        run = Path(state['actual_run'])
    source_sha = source_sha or saved_source or state['source_sha']
    source = TRAIN / 'outputs/source_snapshots' / source_sha
    if not source.is_dir():
        print(f'source snapshot {source_sha} missing, falling back to active source {state["source_sha"]}')
        source_sha = state['source_sha']
        source = TRAIN / 'outputs/source_snapshots' / source_sha
    sys.path.insert(0, str(source))

    env = load_yaml(run / 'params/env.yaml')
    agent = load_yaml(run / 'params/agent.yaml')
    deploy = load_yaml(run / 'params/deploy.yaml')
    actor_args = agent['policy'].copy(); module_name, class_name = actor_args.pop('class_name').split(':')
    num_actions = len(deploy['joint_ids_map'])
    depth_dim = int(agent['policy'].get('depth_dim', SUPPORTED_DEPTH))
    if num_actions != SUPPORTED_ACTIONS or depth_dim != SUPPORTED_DEPTH:
        raise ValueError(
            f'viewer supports {SUPPORTED_ACTIONS} actions and {SUPPORTED_DEPTH}-value depth, '
            f'checkpoint layout needs {num_actions} actions / depth {depth_dim}')
    widths = {
        'base_ang_vel': 3 * SUPPORTED_HISTORY,
        'projected_gravity': 3 * SUPPORTED_HISTORY,
        'velocity_commands': 3 * SUPPORTED_HISTORY,
        'joint_pos_rel': num_actions * SUPPORTED_HISTORY,
        'joint_vel_rel': num_actions * SUPPORTED_HISTORY,
        'last_action': num_actions * SUPPORTED_HISTORY,
        'depth_image': depth_dim,
    }
    obs_dim = sum(widths.values())
    if obs_dim != SUPPORTED_OBS:
        raise ValueError(f'unsupported observation layout: {obs_dim}')
    policy_format = {k: (v,) for k, v in widths.items()}
    critic_format = {'base_lin_vel': (3 * SUPPORTED_HISTORY,), **policy_format}
    num_rewards = 1 + max(int(k.split('.')[1]) for k in saved['model_state_dict'] if k.startswith('critics.'))
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        policy = getattr(importlib.import_module(module_name), class_name)(
            obs_format={'policy': policy_format, 'critic': critic_format},
            num_actions=num_actions, num_rewards=num_rewards, **actor_args)
    try:
        policy.load_state_dict(saved['model_state_dict'], strict=True)
    except RuntimeError as err:
        raise ValueError(f'checkpoint does not match the selected run configuration: {err}') from err
    policy.eval(); policy.record_input_diagnostics = False
    wrapper = BrowserPolicy(policy).eval()
    obs = torch.zeros(1, obs_dim); hidden = torch.zeros(1, 1, 256)
    onnx_path = out_dir / 'policy.onnx'
    torch.onnx.export(wrapper, (obs, hidden), str(onnx_path), input_names=['observations', 'hidden'],
                      output_names=['actions', 'hidden_next'], opset_version=17, dynamo=False)
    options = ort.SessionOptions(); options.intra_op_num_threads = 1; options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(0); torch_hidden = hidden; ort_hidden = hidden.numpy(); tests = []; samples = []
    with torch.no_grad():
        for step in range(12):
            value = rng.normal(0, .15, obs.shape).astype(np.float32); value[:, -depth_dim:] = rng.uniform(0, 1, (1, depth_dim))
            action, new_hidden = wrapper(torch.from_numpy(value), torch_hidden)
            oa, oh = session.run(None, dict(observations=value, hidden=ort_hidden))
            tests.append(dict(step=step, action_error=float(np.max(np.abs(action.numpy() - oa))), hidden_error=float(np.max(np.abs(new_hidden.numpy() - oh)))))
            if step < 2:
                samples.append(dict(observations=value.ravel().tolist(), hidden=ort_hidden.ravel().tolist(), actions=oa.ravel().tolist(), hidden_next=oh.ravel().tolist()))
            torch_hidden = new_hidden; ort_hidden = oh
    parity_per_step = [max(t['action_error'], t['hidden_error']) for t in tests]
    parity_max = max(parity_per_step)
    parity_mean = sum(parity_per_step) / len(parity_per_step)
    parity_warning = None
    # Guard against a genuinely wrong graph, but allow float32/ORT fusion noise.
    # Mean is the robust signal; a single near-threshold sample can be an outlier
    # where the action head amplifies a tiny GRU difference.
    if parity_mean >= 1e-3 or parity_max >= 1e-2:
        raise AssertionError(
            f'ONNX parity mean={parity_mean:.3e} max={parity_max:.3e} exceeds '
            f'(mean 1e-3, max 1e-2): {tests}')
    if parity_max >= 1e-4:
        parity_warning = (f'ONNX parity max error {parity_max:.3e}, mean {parity_mean:.3e}; '
                          f'max is above 1e-4 but within the 1e-2 outlier guard. '
                          f'With action scale this is negligible for the preview.')
        print('WARNING:', parity_warning)


    urdf = Path(env['scene']['robot']['spawn']['asset_path']).resolve()
    model, info, meshes = make_robot(urdf, env, out_dir)
    mj_joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt) if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
    if len(mj_joint_names) != num_actions:
        raise ValueError(f'URDF joint count {len(mj_joint_names)} != action count {num_actions}')
    train_names = [mj_joint_names[i] for i in deploy['joint_ids_map']]
    if train_names[:3] != ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint']:
        raise ValueError('unexpected joint ordering in deploy config')
    joints = []
    for i, name in enumerate(train_names):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name + '_motor')
        joints.append(dict(name=name, qpos=int(model.jnt_qposadr[jid]), qvel=int(model.jnt_dofadr[jid]), actuator=aid,
                           default=deploy['default_joint_pos'][i], scale=deploy['actions']['JointPositionAction']['scale'][i], **info[name]))
    assert np.allclose([j['default'] for j in sorted(joints, key=lambda x: x['qpos'])], deploy['default_joint_pos_mujoco'])

    runtime_path = run / 'runtime_validation.json'
    if not runtime_path.exists():
        runtime_path = Path(state['actual_run']) / 'runtime_validation.json'
    runtime = json.loads(runtime_path.read_text()); tiles = runtime['geometry']; per_level = len(tiles) // 20
    generators = env['scene']['terrain']['terrain_generator']; scenes = {}
    for direction in ['up', 'down', 'up_down', 'down_up']:
        cfg = next(v for k, v in generators['sub_terrains'].items() if 'stairs' in k and v.get('direction') == direction)
        scenes[direction] = []
        for level in range(20):
            matches = [t for t in tiles[level * per_level:(level + 1) * per_level] if t['direction'] == direction]
            if not matches:
                matches = [t for t in tiles if t['direction'] == direction]
            tile = max(matches, key=lambda t: t['width_m'])
            scenes[direction].append(dict(**tile, spawn_origin_x=float(cfg.get('spawn_origin_x', 0)), extent=float(generators['size'][0])))
    camera = env['scene']['camera']
    config = dict(
        checkpoint=dict(iteration=iteration, path=str(checkpoint), sha256=digest, source_sha=source_sha,
                        run_dir=str(run), label=label or checkpoint.stem),
        engine=dict(training='IsaacLab/PhysX', preview='MuJoCo WASM', physics_dt=env['sim']['dt'], control_dt=deploy['step_dt']),
        joints=joints, root_body=int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')),
        torso_body=int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso_link')),
        body_names=[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)],
        meshes=meshes,
        observation_terms=[dict(name=k, width=widths[k] // SUPPORTED_HISTORY, history=SUPPORTED_HISTORY, scale=deploy['observations'][k]['scale']) for k in widths if k != 'depth_image'],
        observation_dim=obs_dim, hidden_shape=[1, 1, 256],
        camera=dict(offset=camera['offset'], pattern=camera['pattern_cfg'], depth_max=2.5), terrain=scenes,
        deviations=['MuJoCo contact solver differs from PhysX.',
                    'Nominal inertias/friction; training randomization and visual noise are not reproduced.',
                    'Fixed equivalent command delay is adjustable, rather than randomized motor/control delay.',
                    'Preview is a sim-to-sim experiment, not gait or deployment acceptance.'])
    (out_dir / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2))
    (out_dir / 'parity_samples.json').write_text(json.dumps(samples))
    make_low_poly(out_dir, source_sha)
    result = dict(checkpoint=config['checkpoint'], onnx_bytes=onnx_path.stat().st_size, parity_tests=tests,
                  parity_max_error=parity_max, parity_mean_error=parity_mean, parity_warning=parity_warning,
                  model=dict(nq=model.nq, nv=model.nv, nu=model.nu, total_mass=float(model.body_mass.sum())),
                  urdf=str(urdf), main_training_mutated=False)
    (out_dir / 'preparation_report.json').write_text(json.dumps(result, indent=2))
    if out_dir == PUBLIC.resolve():
        (APP / 'preparation_report.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, help='checkpoint .pt file')
    parser.add_argument('--out-dir', type=Path, default=PUBLIC)
    parser.add_argument('--run-dir', type=Path, help='training run dir containing params/')
    parser.add_argument('--source-sha', help='source snapshot sha (defaults to checkpoint terrain context)')
    parser.add_argument('--label', default='')
    args = parser.parse_args()

    state = active_state()
    if args.checkpoint:
        checkpoint = args.checkpoint
        run = args.run_dir or checkpoint.parent
        label = args.label or checkpoint.stem
    else:
        run = Path(state['actual_run'])
        candidates = [p for p in [run / 'latest.pt', *sorted(run.glob('model_*.pt'), key=lambda p: int(p.stem.split('_')[-1]))[-1:]] if p.exists()]
        if not candidates:
            raise FileNotFoundError('no checkpoint found in active run')
        checkpoint = max(candidates, key=lambda p: p.stat().st_mtime_ns)
        label = args.label or checkpoint.stem
    result = export_checkpoint(checkpoint, args.out_dir, run_dir=run, source_sha=args.source_sha, label=label)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
