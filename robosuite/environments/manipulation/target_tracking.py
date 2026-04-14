from collections import OrderedDict
import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
class TargetTracking(ManipulationEnv):
    """
    A task where a single robot arm must track a moving (or randomly placed) target.
    The end-effector must reach and stay close to the target position.
    """
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=True,
        target_range=0.15,       # how far target can spawn from center
        success_threshold=0.02,  # distance threshold to count as "reached"
        moving_target=False,     # whether target moves during episode
        target_speed=0.01,       # speed of target movement
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs
        self.target_range = target_range
        self.success_threshold = success_threshold
        self.moving_target = moving_target
        self.target_speed = target_speed
        self.placement_initializer = placement_initializer
        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )
    def reward(self, action=None):
        reward = 0.0
        if self._check_success():
            reward = 1.0
        elif self.reward_shaping:
            dist = np.linalg.norm(self._eef_to_target())
            reward = 1 - np.tanh(10.0 * dist)
        if self.reward_scale is not None:
            reward *= self.reward_scale
        return reward
    def _eef_to_target(self):
        """Vector from end-effector to current target position."""
        return self._gripper_to_target(
            gripper=self.robots[0].gripper,
            target=self.target.root_body,
            target_type="body",
            return_distance=False,
        )
    def _load_model(self):
        super()._load_model()
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])
        # Target is a small visual sphere
        self.target = BallObject(
            name="target",
            size=[0.02],
            rgba=[0, 1, 0, 1],
            obj_type="visual",       # visual only, no physics interaction
            joints=None,             # no free joint — position set manually
        )
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.target)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="TargetSampler",
                mujoco_objects=self.target,
                x_range=[-self.target_range, self.target_range],
                y_range=[-self.target_range, self.target_range],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.15,  # target floats above table
                rng=self.rng,
            )
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.target,
        )
    def _setup_references(self):
        super()._setup_references()
        self.target_body_id = self.sim.model.body_name2id(self.target.root_body)
    def _setup_observables(self):
        observables = super()._setup_observables()
        if self.use_object_obs:
            modality = "object"
            @sensor(modality=modality)
            def target_pos(obs_cache):
                try:
                    return np.array(self.sim.data.body_xpos[self.target_body_id])
                except Exception:
                    return np.zeros(3)

            @sensor(modality=modality)
            def eef_to_target(obs_cache):
                try:
                    return self._eef_to_target()
                except Exception:
                    return np.zeros(3)
            sensors = [target_pos, eef_to_target]
            names = [s.__name__ for s in sensors]
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )
        return observables
    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                # For objects without joints, set body position directly
                body_id = self.sim.model.body_name2id(obj.root_body)
                self.sim.model.body_pos[body_id] = obj_pos
        # If using a moving target, initialize velocity direction
        if self.moving_target:
            self._target_velocity = self.rng.uniform(-1, 1, size=3)
            self._target_velocity[2] = 0  # keep it in the horizontal plane
            self._target_velocity = (
                self._target_velocity / np.linalg.norm(self._target_velocity) * self.target_speed
            )
    def _post_action(self, action):
        """Optionally move the target each step."""
        if self.moving_target:
            body_id = self.target_body_id
            pos = np.array(self.sim.model.body_pos[body_id])
            pos[:2] += self._target_velocity[:2]
            # Bounce off workspace boundaries
            for i in range(2):
                if abs(pos[i] - self.table_offset[i]) > self.target_range:
                    self._target_velocity[i] *= -1
                    pos[i] = np.clip(
                        pos[i],
                        self.table_offset[i] - self.target_range,
                        self.table_offset[i] + self.target_range,
                    )
            self.sim.model.body_pos[body_id] = pos
        return super()._post_action(action)
    def _check_success(self):
        dist = np.linalg.norm(self._eef_to_target())
        return dist < self.success_threshold
    def _check_robot_configuration(self, robots):
        """Only allow single-arm robots."""
        if type(robots) is str:
            robots = [robots]
        assert len(robots) == 1, "TargetTracking requires exactly one robot!"