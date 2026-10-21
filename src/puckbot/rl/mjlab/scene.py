"""Scene: our table, pucks and paddles as one mjlab entity.

The model is the same one the CPU sim uses (sim/model.py `build_spec`): same CAD, same contacts, same
paddle springs. mjlab copies it into every parallel world. Facts about the table (size, paddle limits,
homes) are read once from a CPU build, so nothing is typed in twice.

Usage:
    facts = TableFacts.load()                    # one robot paddle, one puck; Layout.of(2) for two
    scene_cfg = make_scene_cfg(facts, num_envs=4096, camera=train_settings.camera)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from puckbot.rl.sim.env import AirHockeyEnv
from puckbot.rl.sim.layout import Layout
from puckbot.rl.sim.model import TableGeometry, build_spec, paddle_geom, puck_geom
from puckbot.rl.sim.settings import Settings, load_settings
from puckbot.rl.train.settings import CameraSettings

ENTITY = "table"  # the one entity: table + pucks + paddles
TOUCH_SENSOR = "paddle_{paddle}_touches_puck_{puck}"
CAMERA = "camera"


@dataclass(frozen=True)
class TableFacts:
    settings: Settings
    layout: Layout
    geometry: TableGeometry
    paddle_low: np.ndarray  # (paddles, 2)
    paddle_high: np.ndarray
    home: np.ndarray
    physics_dt: float
    decimation: int  # physics ticks per control tick

    @classmethod
    def load(cls, layout: Layout | None = None, settings: Settings | None = None) -> TableFacts:
        cpu = AirHockeyEnv(settings or load_settings(), layout or Layout())
        return cls(
            settings=cpu.settings,
            layout=cpu.layout,
            geometry=cpu.geometry,
            paddle_low=cpu.paddle_low,
            paddle_high=cpu.paddle_high,
            home=cpu.home,
            physics_dt=cpu.physics_dt,
            decimation=cpu.physics_ticks_per_step,
        )

    @property
    def n_paddles(self) -> int:
        return len(self.layout.paddles)

    @property
    def n_pucks(self) -> int:
        return self.layout.pucks


def make_scene_cfg(facts: TableFacts, num_envs: int, camera: CameraSettings) -> SceneCfg:
    from puckbot.rl.mjlab.camera import PuckCameraCfg  # camera.py imports this module

    home_joints = {
        f"paddle_{k}_{axis}": float(facts.home[k, i])
        for k in range(facts.n_paddles)
        for i, axis in enumerate("xy")
    }
    table = EntityCfg(
        spec_fn=lambda: build_spec(facts.settings, facts.layout),
        articulation=EntityArticulationInfoCfg(actuators=(XmlActuatorCfg(target_names_expr=("paddle_.*",)),)),
        # mjlab takes the FIRST matching pattern, so no ".*" catch-all here: unlisted joints default to 0
        init_state=EntityCfg.InitialStateCfg(joint_pos=home_joints, joint_vel={".*": 0.0}),
    )
    touch_sensors = tuple(
        ContactSensorCfg(
            name=TOUCH_SENSOR.format(paddle=k, puck=p),
            primary=ContactMatch(
                mode="geom", pattern=(paddle_geom(k, "base"), paddle_geom(k, "grip")), entity=ENTITY
            ),
            secondary=ContactMatch(mode="geom", pattern=puck_geom(p), entity=ENTITY),
            fields=("found", "force"),
            reduce="netforce",
            history_length=facts.decimation,  # every physics tick of the control tick: no hit is missed
        )
        for k in range(facts.n_paddles)
        for p in range(facts.n_pucks)
    )
    sensors = (*touch_sensors, PuckCameraCfg(name=CAMERA, facts=facts, camera=camera))
    return SceneCfg(num_envs=num_envs, env_spacing=3.0, entities={ENTITY: table}, sensors=sensors)
