# hockey-bot air-hockey table, puck and paddle (URDF + MJCF package)

Built 2026-10-15 from the Autodesk Fusion designs **Table v7**, **Puck v2** and **Paddle v3**
(Paddle v3 is the saved state of the design described as "Paddle v2" + the top divot; same part).
Everything you need to build the mjlab / MuJoCo simulation is in this folder. Nothing refers to files outside it.

Quick start (plain MuJoCo, from inside this folder):

```bash
pip install mujoco numpy
python validate.py          # loads table.xml, runs the rest / goal / 264-shot / paddle-strike checks, prints ALL PASS
```

## Files

| File | What it is |
|---|---|
| `table.xml` | **Use this one.** MJCF compiled by MuJoCo 3.8 from `table.urdf`, with the puck/paddle contact settings and the 1 ms timestep already applied. |
| `table.urdf` | Source URDF: fixed link `table` + floating links `puck` and `paddle`. Portable to ROS, Isaac etc., but it cannot carry MuJoCo contact settings (see below). |
| `meshes/table.stl`, `meshes/puck.stl`, `meshes/paddle.stl` | Exact Fusion meshes, binary STL, **metres**, already in the link frames below (table mesh is in the floor frame). |
| `validate.py` | Self-check script (needs only `mujoco` and `numpy`). Re-run it after any edit. |
| `validation_report.json` | Output of the validation run that shipped with this package (rest heights, 264 shots, goal fit, paddle strike, inertia check, collider list). |
| `inertia_check.json` | Mass/COM/inertia used in the URDF next to the same quantities recomputed from the meshes. |
| `fusion/export_*.json` | Raw Fusion mass properties and planar-face data per design (Fusion frame, SI units). |
| `preview_visual.png`, `preview_collision.png`, `preview_paddle_puck.png` | Renders of the shipped model: visual meshes, collision boxes, close-up. Compare with what you load. |

## Frame and units

* Metres, kilograms, **Z up**, X along the long axis of the table, Y across it.
* **Origin is on the floor**, at the centre of the table footprint. The table stands on `z = 0`; put a MuJoCo
  ground plane at `z = 0` and the legs rest on it. (`validate.py` adds one.) The previous package put the origin
  on the old slab underside; add 0.6747 m to any old z value to convert.

| Feature | z (m) |
|---|---|
| floor / bottom of legs | 0.0000 |
| apron bottom edge | 0.3781 |
| deck underside | 0.7287 |
| **playing surface** | **0.739775** |
| goal slot ceiling | 0.75724 |
| outer rim lip (underside / top) | 0.7591 / 0.7747 |
| bumper rail top | 0.79375 (31.25 in) |

Planar dimensions: outer rim footprint 2.286 x 1.270 m. Playing area between the rails: |x| < 1.0448, |y| < 0.539
(2.0896 x 1.078 m). Goal slot at each end: 0.2476 m wide (|y| < 0.1238), 17.5 mm high, running from x = ±1.0448 to
x = ±1.0828 where it **exits through the apron** (see "Goal behaviour"). Legs: 95 x 79 mm, centred at (±0.985, ±0.485).

## Names

Bodies / links: `table` (static, fused into the world by MuJoCo), `puck`, `paddle`.
Joints: `puck_free`, `paddle_free` (URDF `floating`; MuJoCo `free`). The compiled MJCF has `nq = 14`, `nv = 12`:
qpos = [puck xyz, puck quat wxyz, paddle xyz, paddle quat wxyz].

Table collision geoms (15 boxes): `deck_apron`, `rail_pos_y`, `rail_neg_y`, `endrail_{pos,neg}_x_{pos,neg}_y` (4),
`goal_lintel_{pos,neg}_x`, `rim_{pos,neg}_x`, `leg_{pos,neg}_x_{pos,neg}_y` (4).
Puck collision: `puck_col`. Paddle collision: `paddle_base_col`, `paddle_grip_col`. Visual mesh geoms are unnamed, group 1, no contact.

Puck and paddle link origins are at the **centre of the bottom face** (the face that touches the table). Both start
resting on the surface: puck at (0, 0, 0.739775) = centre ice, paddle at (-0.70, 0, 0.739775).

## Loading

MuJoCo (python):

```python
import mujoco
m = mujoco.MjModel.from_xml_path("table.xml")      # meshes/ must stay next to it
# or the URDF (then re-apply the contact settings below yourself):
m = mujoco.MjModel.from_xml_path("table.urdf")
```

mjlab: point the entity/asset config at `table.xml` (mjlab loads MJCF through `mujoco.MjSpec`; `MjSpec.from_file("table.xml")`
works, and so does attaching that spec into a scene spec). Keep `meshes/` beside the XML. If you only want the table and
you supply your own puck/paddle, delete the two `<body>` elements; if you only want the puck or paddle, copy its `<body>`
and its `<mesh>` asset line.

Timestep: `table.xml` sets `timestep="0.001"`. Keep it at 1 ms (or lower). See "Contact settings" for why.

## Attaching the paddle to the robot

The paddle body has its grip axis along its local +z and its origin on the bottom face. Grip: 45.7 mm diameter cylinder
from z = 0.015 to 0.057 above the bottom face (then a rounded top with a shallow divot). A reasonable grasp point is
(0, 0, 0.040) in the paddle frame, with the paddle's +z pointing away from the table.

* **MJCF**: delete `<joint name="paddle_free" .../>`, move the whole `<body name="paddle">` inside your end-effector body,
  and set its `pos`/`quat` so that the paddle bottom face sits where the tool touches the surface (paddle z axis = tool
  axis, pointing up). With `MjSpec`: `ee_body.attach(paddle_spec.body("paddle"))` style attachment, after removing the free joint.
* **URDF**: replace `paddle_free` with
  `<joint name="paddle_mount" type="fixed"><parent link="<ee_link>"/><child link="paddle"/><origin xyz="..." rpy="..."/></joint>`.
* For a quick test without a robot, make the paddle a mocap body (`mocap="true"`, no joint) and drive `mocap_pos`.

The puck/paddle contact settings below are on the paddle **geoms**, so they survive re-parenting.

## Contact settings (important) and why

MuJoCo's default contact is soft (`solref` time constant 0.02 s). A thin 34 g puck at 6 m/s then sinks tens of
millimetres into a wall and gets ejected. `table.xml` carries, on `puck_col`:

```
priority="1" friction="0.02 0.001 0.0001" solref="0.004 1"
```

and on `paddle_base_col` / `paddle_grip_col`:

```
priority="1" friction="0.3 0.005 0.0001" solref="0.004 1"
```

`priority="1"` makes these values win over the table's defaults for every puck-table / paddle-table / puck-paddle pair.
The low puck friction stands in for the air cushion. `solref="0.004 1"` is a 4 ms, critically damped contact. URDF cannot
express any of this, so if you rebuild from `table.urdf`, re-apply it (that is exactly what the build did to produce `table.xml`).

**Timestep must be 1 ms.** With `solref` 0.004 the 2 ms default is at the stability edge: in the sweep below the puck
hopped up to 29 mm off the surface on hard rail hits at 2 ms, and at most 2.3 mm at 1 ms, with the same escape-free result.
`multiccd` made no difference; a 1 mm `margin` on the puck blew up.

Rail restitution with these settings is low: a perpendicular hit on a side rail comes back at about 0.1-0.16 of the
incoming speed (2-6 m/s). If you want livelier rails, the knob is the puck `solref` damping ratio; measured at 1 ms:

| puck solref | restitution (2 / 4 / 6 m/s) | max hop | escapes / 99 |
|---|---|---|---|
| `0.004 1` (shipped) | 0.11 / 0.14 / 0.13 | 2.3 mm | 0 |
| `0.004 0.7` | 0.17 / 0.23 / 0.21 | 4.3 mm | 0 |
| `0.004 0.5` | 0.25 / 0.36 / 0.31 | 6.4 mm | 0 |
| `0.004 0.5` at 0.5 ms | 0.19 / 0.31 / 0.28 | 2.3 mm | 0 |
| `0.004 0.3` | 0.39 / 0.64 / 0.56 | 50 mm | 1 |

## Goal behaviour

The goal is a **through-slot**: the end rail is cut 17.5 mm high and 247.6 mm wide, the slot runs 38 mm deep and comes out
of the tapered apron on the outside. A scored puck leaves the deck at |x| = 1.081, drops off the end of the table and lands
on the floor. There is no catch tray or back wall in the design (Table v5 had one; the apron rebuild removed it). Detect a goal
with `|puck_x| > 1.083 and |puck_y| < 0.124`, then reset the puck. If you want the puck caught, add your own box under the slot exit.

## Collision model

MuJoCo replaces every mesh collider by its convex hull; a hull of the table fills the tray and leaves nothing to play on.
So the table's collision is 15 primitive boxes and the mesh is visual only. They match the mesh where the puck and paddle can
reach; outside the tray they are deliberately simplified:

* `deck_apron`: one box from the apron bottom (z 0.378) to the playing surface, |x| < 1.0813, |y| < 0.5737. The top face is
  the playing surface. The real apron is a hollow 5 deg tapered skirt; the box fills it and ignores the taper (up to 26 mm proud at the bottom).
* Side rails, end-rail blocks and goal lintels: inner faces exactly on the real rail faces (|y| = 0.539, |x| = 1.0448) and the
  slot opening (|y| = 0.1238, z 0.7398-0.7572). They extend outward to the rim edge / slot exit and 11 mm below the surface so a
  puck never meets a box edge at surface height. Rail height is the bumper height everywhere (the real outer rim is 19 mm lower).
* `rim_{pos,neg}_x`: the 16 mm thick rim lip beyond the slot exits. The side rim is covered by the rail boxes.
* Legs: exact footprint, floor to apron bottom.
* Not modelled: the 3 mm step and fillets on the rails, the rim's rounded edge. Visual only.

Puck collision is a cylinder of the measured size (r 40.41 mm, 6.35 mm). Paddle collision is two cylinders: the lip
(r 50.32 mm, 0-22.9 mm; the real outer wall is straight, the dish inside is ignored) and the grip (r 22.86 mm, 22.9-68.6 mm;
the real top is rounded with a divot). Both rest on flat bottoms; the real parts have a shallow underside recess and rest on a ring,
which does not matter in simulation.

## Puck and paddle: measured vs estimated

| | Puck v2 | Paddle v3 |
|---|---|---|
| Geometry | **Measured** from the real puck: 3.182 in dia x 0.25 in (80.82 x 6.35 mm), shallow 0.3 mm face recesses | **Measured** from the real paddle: 3.9625 in base, 1.8 in grip, 2.7 in tall (100.6 / 45.7 / 68.6 mm), dished base, tapered lip, top divot |
| Mass | **Measured ~34 g**; the URDF uses 0.034 kg. Fusion's ABS density (1.06) gives 32.1 g for the CAD volume, 6 % low | **Measured ~217 g**; Fusion ABS gives 217.1 g, used as is |
| Centre of mass | from CAD: 3.175 mm above the bottom (symmetry) | from CAD: 22.3 mm above the bottom |
| Inertia | from CAD geometry, scaled to 34 g: Ixx = Iyy = 1.42e-5, Izz = 2.82e-5 kg m^2 (thin disc m r^2/2 = 2.78e-5) | from CAD: Ixx = Iyy = 1.79e-4, Izz = 2.10e-4 kg m^2 |
| Material | ABS (uniform density assumed) | ABS (uniform density assumed) |

`inertia_check.json` recomputes COM and inertia from the shipped meshes; they agree with the URDF to 0.06 %.
The table's 803 kg is Fusion's "Steel" material and is irrelevant while the table is static.

## Validation that shipped with this package (MuJoCo 3.8, `validation_report.json`)

* Puck and paddle rest at z = 0.739775 (drop < 1 um, drift < 1 um in 1 s); paddle stays upright.
* Puck passes through the goal slot on a straight 3 m/s shot and falls off the end.
* 264 shots: 11 lateral start positions x 8 speeds (1-8 m/s) x 3 directions (at the goal, 30 deg into a side rail, 160 deg into
  the near end rail). 0 escapes, 127 goals, max puck lift 8.5 mm (on 8 m/s lintel hits), never below the surface by more than 1.2 mm.
* A free paddle sliding at 1.5 m/s hits a resting puck: puck leaves at 1.0 m/s, paddle stays upright.

## Re-exporting from Fusion

The Fusion script `ExportTableURDF` (Utilities > Add-Ins > Scripts and Add-Ins) exports every visible solid body of the
active document as binary STL (mm) plus `export.json` into `hockey-bot/fusion_export/<document name>/`. Run it once per
document. The package was then assembled by a build script (meshes scaled to metres and shifted to the floor frame, boxes
placed from mesh sections, URDF compiled by MuJoCo, contact settings re-applied, validation run). Collider positions are
listed in `validation_report.json` under `colliders` if you need to adjust them after a design change.
