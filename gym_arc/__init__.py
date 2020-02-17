import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='ARC-v0',
    entry_point='gym_arc.envs:ARCEnv'
    # timestep_limit=10000,
    # reward_threshold=1.0
    # nondeterministic=False
)