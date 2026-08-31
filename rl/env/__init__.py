"""env/ — the training environment: observations, actions, rewards, randomisation.

`Walk` is the only thing here that trains. `rewards` and `randomize` are split out
because they are the two places a change is a POLICY decision rather than a
modelling one, and they should be readable without the env around them.
"""
from env.walk import Walk, Commands, assemble_obs, rotate_inv   # noqa: F401
from env.rewards import Weights              # noqa: F401
from env.randomize import domain_randomize   # noqa: F401
