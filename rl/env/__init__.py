"""env/ — the training environment: observations, actions, rewards, randomisation.

`Walk` is the only thing here that trains. `rewards` and `randomize` are split out
because they are the two places a change is a POLICY decision rather than a
modelling one, and they should be readable without the env around them.
"""
from env.walk import (Walk, Commands, assemble_obs, stack_obs, init_hist,  # noqa: F401
                      rotate_inv, OBS_HIST, OBS_SIZE, check_obs_width)
from env.rewards import Weights              # noqa: F401
from env.randomize import domain_randomize   # noqa: F401
