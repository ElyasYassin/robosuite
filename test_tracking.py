import numpy as np
import robosuite
env = robosuite.make(
    "TargetTracking",
    robots="UR5e",
    has_renderer=True,
    reward_shaping=True,
    moving_target=False,
    success_threshold=0.02,
    control_freq=20,
    horizon=500,
)
obs = env.reset()
for _ in range(500):
    low, high = env.action_spec
    action = low + (high - low) * np.random.rand(env.action_dim)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        break
env.close()