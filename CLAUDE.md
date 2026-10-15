# Puckbot — autonomous air hockey player

## Code (current)

Everything vision-related lives in `src/puckbot/vision/` (code, settings, data, tests). Keep it there.
One job per file, in pipeline order:

- `settings.py`: dataclasses saying what each setting means; values in `configs/our_table.json`.
  `load_settings()` rejects typo'd keys.
- `table_calibration.py`: `TableCalibration.from_photo()` finds the blue surface's corners ->
  homography. Units are inches. `to_table(u, v)`, `to_pixel(x, y)`, `top_down(image)`.
- `puck_mallet_detector.py`: `PuckMalletDetector.find(image, table_outline)` finds all white shapes
  once, sorts each into puck / mallet / rejected (with reason). Pixels only.
- `pipeline.py`: `analyze(image, settings) -> Analysis` (table, puck, mallets in inches, problems).
- `drawing.py`: result picture, top-down view, and `draw_steps` = ONE grid of every step
  (user does not want separate step images).
- `cli.py`: `puckbot-analyze photo [--out x.png] [--steps y.png]`. The one command; keep it one.
- `tests/`: made-up photos (conftest.py) + whole pipeline on `data/samples/`.

Style the user asked for: clear names (no vague ones like "blob"), file docstrings that open with
a short numbered "Steps" list + usage, short sentences, sections marked `# ── Name ──`, detail
only where it's needed. Skimmable first, detailed second.

Facts:
- Our table: white/cream puck and mallets on blue; lines, logo and rails are also white.
  Outer table 50 x 90 in, aluminium rail 3.75 in wide -> playing surface 42.5 x 82.5 in (derived,
  not yet measured with a tape). Centre line lands at ~42.7 in instead of 41.25 -> expect ~1-2 in error
  from an angled photo (camera sees the inner rail walls). Puck measures ~3.6 in (real: measure it).
- Black corner caps were tried twice and dropped: in the sample photo one cap is out of frame, two
  merge with the dark shadow between table and wall, one is washed out by glare. No AprilTags for now.
- Scope right now: single images only. Live camera comes later; keep it behind one frame-source
  interface so the detector never cares where frames come from.

Student club project: a robot that plays air hockey on our own full-size table (~4 ft wide).
Budget: $500 total, ~$300 left at time of writing. Low-cost, DIY, club-friendly choices win.

## Current architecture (MVP)

```
USB camera (overhead, 60-120 fps) --image--> Mac (OpenCV tracking + policy) --x,y target--> ESP32
ESP32 --step/dir--> stepper drivers --> 2x NEMA 23 motors --> CoreXY gantry --> paddle
ESP32 --paddle x,y (encoder / step count)--> Mac   (policy input)
```

- **Mac** does all thinking: capture, calibration, puck tracking, prediction, policy inference.
- **ESP32** does all hard real-time timing: motion profile, CoreXY math, soft limits, safety stop,
  reports paddle position. Never generate step pulses from the Mac/Pi/Jetson (Linux jitter).
- **Training** happens offline on an NVIDIA GPU (A40 available) in mjlab; only the small policy
  network (ONNX) runs on the Mac via onnxruntime.

## Hardware decisions

| Part | Choice | Notes |
| --- | --- | --- |
| Mechanism | 2-motor CoreXY / H-bot gantry | Chosen over 4-cable robot: cable tension, edge overloads, thermal trips and calibration make cables much harder (see misprit7). |
| Camera | **USB (UVC) camera at 60–120 fps**, e.g. a USB Arducam global-shutter module | Works on the Mac like a webcam: `cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)`. Avoid MIPI/CSI ribbon-cable modules (Pi/Jetson only). iPhone Continuity Camera was tried and dropped as too slow; still fine for capturing test images. |
| Microcontroller | ESP32 (classic or S3) | FastAccelStepper (RMT driver, up to 200k steps/s) or FluidNC (CoreXY built in). Classic ESP32 has 8 PCNT counters, S3 has 4. 3 UARTs (one used for USB). |
| Motors/drivers — option A | StepperOnline closed-loop kit, 3.0 Nm NEMA 23 + CL57Y-V20 driver, ~$70 each on Amazon (B0FVX917GG), 1000-line encoder | 24–50 V (run 48 V). Driver inputs need 5 V (level shifter, e.g. 74HCT245). CL57Y does NOT output encoder position (only alarm / in-position); to read the encoder on the ESP32 you must tap EA±/EB± through a 3.3 V line receiver (AM26LV32) — or just use the ESP32 step count. |
| Motors/drivers — option B | MKS SERVO57C closed-loop board ($25.99) + StepperOnline NEMA 23 1.24 Nm ($25.99) | Board mounts on motor back, magnetic encoder reads a magnet on the REAR shaft → buy the dual-shaft motor (23HS22-2804D), confirm magnet included. 12–24 V only. Inputs accept 3.3 V (no level shifter). Position readable over UART: `FA 01 30 CRC` → 14-bit angle (16384 counts/rev). Calibrate unloaded (menu CAL). V1.0 UART = 1 board per port, cable <30 cm; RS-485 supports multiple. Manual: github.com/makerbase-mks/MKS-SERVO57C |
| Power | 48 V ~10 A PSU (option A) or 24 V (option B), behind an e-stop that cuts motor power | ESP32 powered from Mac USB. Common ground ESP32 ↔ driver signal ground. |
| Markers | Retroreflective/bright tape on puck, LED light, 4 ArUco corner tags outside rails | |

**Motor decision still open (A vs B).** Plan: move with step/dir, read position over UART (B) or
from step count (A). Test early whether MKS read commands work in CR_vFOC (step/dir) mode — the
manual only restricts motion commands to CR_UART.

## Software stack

Mac (Python): Capture (OpenCV) → Calibrate (cv2.aruco homography px→mm) → Detect (threshold +
blobs) → Track (least-squares slope over last ~6 frames, coast through dropouts) → Predict (lossy
wall bounces, arrival time at defence line) → Decide (scripted goalie first, ONNX policy later;
pause if puck lost ~0.5 s) → Send (pyserial, 50–100 Hz, target + accel cap) → Log every tick (CSV,
replayable into sim).

ESP32 (Arduino C++): Parse serial targets, clamp to soft limits → accel-limited motion profile that
can retarget mid-move → CoreXY (A = x + y, B = x − y) → step pulses. Safety: e-stop pin, homing
switches, halt if commands stop, stop on large position error. Report paddle x, y back to the Mac.

Keep ONE shared settings file (table size, goal width, soft limits, steps/mm, accel caps) read by
sim, Mac code and firmware build. misprit7 found caps duplicated in 7 places with 3 values.

## RL plan

- **Sim:** mjlab (MuJoCo Warp + RSL-RL PPO) on Linux + NVIDIA GPU (Turing or newer, driver 580+;
  A40 fine; RTX 40-series recommended). macOS = eval only. Neither prior team used MuJoCo — both
  wrote NumPy sims — so we must tune MuJoCo to our measurements.
- **Starting model:** `air_hockey_challenge/environments/data/table.xml` from
  github.com/AirHockeyChallenge/air_hockey_challenge (MIT). Puck on x/y slide joints + yaw hinge,
  puck–table contact excluded (air cushion), tuned rail contacts. Replace KUKA arm with a paddle on
  2 slide joints + position actuators; resize to our table.
- **Observations:** puck x,y at last 5 frames (e.g. 1st, 2nd, 3rd, 6th, 12th back), own paddle x,y,vx,vy,
  opponent mallet x,y history, previous action, time puck on our side. Normalize.
- **Actions:** target x, target y in [−1, 1], accel cap in [0, 1]; 50 Hz to the ESP32.
- **Reward:** +goal / −conceded, +shot on target scaled by speed, −acceleration (motor heat),
  −jitter (action change). Avoid per-step incomes the wanted behaviour switches off.
- **Domain randomization:** drag, rail restitution/tangential, paddle restitution, camera delay
  (mjlab observation delay), position noise + dropouts, paddle accel/speed, table size.
- **Opponents:** past selves + scripted bots (goalie, sniper, delayed defender). Mirror augmentation.
- **Algorithm:** PPO first; switch to SAC + shared replay buffer with old specialist policies if the
  agent collapses to one shot type (what Hudson did).
- **Rules:** sim must not do anything the table can't (no referee restarts of stuck pucks); sim wall
  margin = real soft limit; episodes end only on goals; judge checkpoints with fixed drills
  (block rate, shots on target), not training reward.
- **Deploy:** export ONNX, onnxruntime on Mac, same obs layout and control rate as training, gentle
  caps on first run. Test that the deployed obs encoder matches the training env.

## Parameters to measure

Record 5–10 varied minutes of puck motion with the tracker (or iPhone Slo-mo / Blackmagic Camera
app at 120–240 fps with locked fast shutter), save raw video + timestamped positions, fit offline.

| # | Parameter | How | Prior values |
| --- | --- | --- | --- |
| 1 | Playing area, goal width, rail height | Tape measure | misprit7 goal 380 mm |
| 2 | Puck/mallet diameter + mass | Calipers, 0.1 g scale | misprit7 puck 81.4 mm, mallet 100.8 mm |
| 3 | Puck slowdown a + b·v² | ~20 straight glides, slow→fast | misprit7 b ≈ 0.0348/m, μ ≈ 0.0015 |
| 4 | Rail restitution (normal) + along-rail retention | Square-on and 20–40° hits per rail; exclude goal mouth | misprit7 0.785 (speed-dependent), tangential 0.90 side / 0.66 end |
| 5 | Mallet restitution | Puck into a STILL mallet | misprit7 never measured (guess 0.9) |
| 6 | Paddle max speed/accel, tracking error, heat | Commanded moves at rising accel; return to mark | misprit7 tracked 4–8 mm at 20–60 m/s²; drives tripped after 20–30 s flat-out |
| 7 | Camera delay | ESP32 LED in view; time until Mac sees it; ~50 trials | Hudson 13.4 ms, misprit7 7.7 ms (iPhone will be far more) |
| 8 | Command delay | Timestamp echo through ESP32 | Hudson 6.0 ms |
| 9 | Vision noise / offset / dropouts | Puck still on a grid; known positions | Hudson 1.5 mm mean error |

MuJoCo note: joint damping gives drag ∝ v, not v²; if glides don't match, add a per-step drag force
on the puck. Rails: tune `solref`/`solimp` until replayed real shots match.

Our own quick court sim (NumPy, nominal 8×4 ft) showed: mirror bounces miss ~118 mm median after
one bounce → use the lossy wall model in the scripted goalie; cutting camera lag 100 → 30 ms raises
the fastest blockable straight shot as much as doubling gantry accel (2.85 → 3.55 m/s at 20 m/s²).

## Milestones

1. Camera + puck tracking + ArUco calibration on the Mac (~1–1.5 weeks part-time).
2. Record shots, fit parameters 1–5.
3. Build gantry + ESP32 firmware; homing, soft limits, e-stop; measure 6–8.
4. Scripted goalie (lossy bounces, arrival-time gating) on the real table.
5. mjlab scene + task (~1–2 weeks to first training), replay-check vs recordings, then PPO on A40.
6. ONNX on Mac, gentle first run, log everything, refit constants, retrain.

## Lessons from prior projects

- **HudsonNock/Air-Hockey-Sim** (UBC, CoreXY, DC motors + Blue Pill, 120 fps camera, NumPy sim,
  PPO year 1 → SAC year 2, beat all humans tested). Key lessons: measure delays with an LED; feed
  raw position history not velocity; retroreflective tape + coaxial LED; belt centred on a stiff
  crossbeam; supercapacitor for supply sag; shielded encoder cables; match sim/real wall margins
  (1 cm vs 5 cm caused own goals); firmware stop on position error; "hold" action reduces jitter;
  train on slow pucks. No license — read only.
- **misprit7/airhockey** (4-cable robot, ClearPath servos, Teensy, 200 Hz FLIR, NumPy sim; SAC →
  TD-MPC2 → PPO). Key lessons: lossy wall model (specular off ~67 mm), quadratic drag, 4 markers on
  the puck, gate goalie on arrival time not speed, no sim referee, thermal limits not in sim,
  one canonical geometry file, judge with drills. No license — read only.
- **Skild AI physical self-play:** self-play vs recent snapshots works but goals-only reward needs
  huge compute; keep shaped rewards at club scale.

## References

- github.com/HudsonNock/Air-Hockey-Sim (+ docs/2509 and docs/2550 final reports)
- github.com/misprit7/airhockey (CLAUDE.md, SIM2REAL.md, ai/RETRAIN.md)
- github.com/AirHockeyChallenge/air_hockey_challenge (MIT; MuJoCo table.xml)
- github.com/mujocolab/mjlab (docs/source/faq.rst for platform support)
- github.com/makerbase-mks/MKS-SERVO57C (manual V1.0.2)
- arxiv.org/abs/2405.03113 (Robot Air Hockey testbed)
- FastAccelStepper (ESP32), FluidNC kinematics docs
