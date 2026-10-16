"""Quick self-check of table.xml in plain MuJoCo (no mjlab needed).

    pip install mujoco numpy
    python validate.py            # from inside the table_urdf folder

Loads table.xml, adds a ground plane, and checks that
  1. the puck and paddle rest on the playing surface,
  2. the puck fits through the goal slot,
  3. 264 shots at 1-8 m/s from 11 lateral positions stay on the table (0 escapes),
  4. the paddle strikes the puck cleanly.
Prints a short report. Exit code 1 if anything fails.
"""
import math, os, sys
import numpy as np, mujoco
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
ZSURF = 0.739775            # playing surface height (floor frame)
XRAIL, YRAIL = 1.0448, 0.539   # inner faces of the end / side rails
XEND, YSLOT = 1.0828, 0.1238   # goal slot: exits the table at |x| = XEND, half-width YSLOT
ZSLOT = ZSURF + 0.017465       # slot ceiling

root = ET.parse(os.path.join(HERE, 'table.xml')).getroot()
root.find('worldbody').insert(0, ET.fromstring('<geom name="floor" type="plane" size="4 4 0.1"/>'))
assets = {f: open(os.path.join(HERE, 'meshes', f), 'rb').read() for f in os.listdir(os.path.join(HERE, 'meshes'))}
m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'), assets)
d = mujoco.MjData(m)
puck = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'puck'); pad = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'paddle')
qp, vp = m.jnt_qposadr[m.body_jntadr[puck]], m.jnt_dofadr[m.body_jntadr[puck]]
qd, vd = m.jnt_qposadr[m.body_jntadr[pad]], m.jnt_dofadr[m.body_jntadr[pad]]

def reset(puck_pos=(0, 0, ZSURF), puck_vel=(0, 0, 0), pad_pos=(-0.70, 0, ZSURF), pad_vel=(0, 0, 0)):
    mujoco.mj_resetData(m, d)
    d.qpos[qp:qp+3] = puck_pos; d.qpos[qp+3:qp+7] = [1, 0, 0, 0]; d.qvel[vp:vp+3] = puck_vel
    d.qpos[qd:qd+3] = pad_pos; d.qpos[qd+3:qd+7] = [1, 0, 0, 0]; d.qvel[vd:vd+3] = pad_vel
    mujoco.mj_forward(m, d)

ok = True
def check(name, cond, detail):
    global ok; ok &= bool(cond); print(f'[{"PASS" if cond else "FAIL"}] {name}: {detail}')

# 1. rest
reset()
for _ in range(int(1.0 / m.opt.timestep)): mujoco.mj_step(m, d)
check('puck rests on surface', abs(d.qpos[qp+2] - ZSURF) < 1e-3, f'bottom z = {d.qpos[qp+2]:.6f} (surface {ZSURF})')
check('paddle rests upright', abs(d.qpos[qd+2] - ZSURF) < 1e-3 and d.xmat[pad].reshape(3, 3)[2, 2] > 0.999, f'bottom z = {d.qpos[qd+2]:.6f}')

# 2. goal fit
reset(puck_vel=(3, 0, 0), pad_pos=(-1.0, 0.45, ZSURF))
for _ in range(2000):
    mujoco.mj_step(m, d)
    if d.qpos[qp] > XEND + 0.05: break
check('puck passes through goal slot', d.qpos[qp] > XEND + 0.05, f'x = {d.qpos[qp]:.3f} after straight 3 m/s shot')

# 3. shots
escapes = 0; n = 0; zmax = -9
for y0 in np.linspace(-0.45, 0.45, 11):
    for speed in range(1, 9):
        aims = [(XEND + 0.5, 0 - y0), (math.cos(math.radians(30)), math.sin(math.radians(30 if y0 <= 0 else -30))),
                (math.cos(math.radians(160)), math.sin(math.radians(160 if y0 <= 0 else -160)))]
        for dx, dy in aims:
            nrm = math.hypot(dx, dy); reset(puck_pos=(-0.5, y0, ZSURF), puck_vel=(speed*dx/nrm, speed*dy/nrm, 0), pad_pos=(-1.0, 0.45, ZSURF)); n += 1
            for _ in range(int(3.0 / m.opt.timestep)):
                mujoco.mj_step(m, d); x, y, z = d.qpos[qp:qp+3]; zmax = max(zmax, z)
                if abs(x) > XEND and abs(y) < YSLOT: break          # goal
                inside = abs(x) < XRAIL + 0.02 and abs(y) < YRAIL + 0.002
                in_slot = abs(y) < YSLOT and abs(x) > XRAIL - 0.05
                if (not inside and not in_slot) or z < ZSURF - 0.005 or z > ZSLOT:
                    escapes += 1; break
check('shot sweep escape-free', escapes == 0, f'{n} shots at 1-8 m/s, {escapes} escapes, max puck lift {(zmax-ZSURF)*1e3:.1f} mm')

# 4. paddle strike
reset(puck_pos=(0, 0, ZSURF), pad_pos=(-0.25, 0, ZSURF), pad_vel=(1.5, 0, 0))
for _ in range(int(0.4 / m.opt.timestep)): mujoco.mj_step(m, d)
check('paddle strike', d.qvel[vp] > 0.5 and d.xmat[pad].reshape(3, 3)[2, 2] > 0.99, f'puck vx = {d.qvel[vp]:.2f} m/s after a 1.5 m/s paddle hit')

print('ALL PASS' if ok else 'SOMETHING FAILED'); sys.exit(0 if ok else 1)
