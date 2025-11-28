# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import warnings
from typing import Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from chex import PRNGKey
from flax.linen import FrozenDict
from jumanji.env import Environment
from omegaconf import DictConfig

from mava.evaluator import ActorState, COMPASSEvalActFn, EvalActFn
from mava.utils.compass_utils import (
    get_compass_latent,
)
from mava.utils.logger import MavaLogger, MultiLogger, NeptuneLogger


def render(
    env: Environment,
    params: FrozenDict,
    actor_state: ActorState,
    key: PRNGKey,
    act_fn: Union[EvalActFn, COMPASSEvalActFn],
    logger: MavaLogger,
    compass_system: bool = False,
    config: Optional[DictConfig] = None,
) -> None:
    # TODO(Ruan): For now, we only support random latents.
    # Add best latent support later on.
    if compass_system:
        states = _rollout_compass_system(env, params, actor_state, key, act_fn, config)  # type: ignore[arg-type]
    else:
        states = _rollout_mava_system(env, params, actor_state, key, act_fn)  # type: ignore[arg-type]

    for i, state_set in enumerate(states):
        env.animate(state_set, save_path=f"env_{i}.gif")  # type: ignore

    _upload_gif_to_neptune(logger, num_gifs=len(states))


def _rollout_compass_system(
    env: Environment,
    params: FrozenDict,
    actor_state: ActorState,
    key: PRNGKey,
    act_fn: COMPASSEvalActFn,
    config: DictConfig,
    num_random_latents: int = 5,
) -> list:
    # NOTE: Always let the env reset to the same key and generate a random state once.
    # This is so we only look at latent differentiation.
    env_key = jax.random.PRNGKey(np.random.randint(0, 1e6))

    init_actor_state = copy.deepcopy(actor_state)
    env_step = jax.jit(env.step)
    jit_act = jax.jit(act_fn)

    all_states = []

    for _ in range(num_random_latents):
        episode_actor_state = copy.deepcopy(init_actor_state)
        state, ts = env.reset(env_key)

        key, latent_key = jax.random.split(key, 2)
        # Only 1 env for rendering.
        latent = get_compass_latent(latent_key, config, (1,))

        # Select a random latent
        latent_idx = np.random.randint(0, config.arch.num_latents_per_env)
        latent = latent[:, latent_idx, ...]

        states = []

        # Only loop once
        while not ts.last():
            # Eval env is wrapped in the record metrics wrapper. We just store the true env state
            # for jumanji to be able to render.
            states.append(state.env_state)
            ts = jax.tree_map(lambda x: x[jnp.newaxis, ...], ts)
            key, act_key = jax.random.split(key, 2)
            action, episode_actor_state = jit_act(params, ts, act_key, episode_actor_state, latent)

            # note: dangerous squeeze, but we don't want a batch or time dim here
            state, ts = env_step(state, action.squeeze())

        all_states.append(states)

    return all_states


def _rollout_mava_system(
    env: Environment, params: FrozenDict, actor_state: ActorState, key: PRNGKey, act_fn: EvalActFn
) -> list:
    env_step = jax.jit(env.step)
    jit_act = jax.jit(act_fn)

    key, env_key = jax.random.split(key)
    state, ts = env.reset(env_key)

    states = []

    # Only loop once
    while not ts.last():
        # Eval env is wrapped in the record metrics wrapper. We just store the true env state
        # for jumanji to be able to render.
        states.append(state.env_state)
        ts = jax.tree_map(lambda x: x[jnp.newaxis, ...], ts)
        key, act_key = jax.random.split(key, 2)
        action, actor_state = jit_act(params, ts, act_key, actor_state)

        # note: dangerous squeeze, but we don't want a batch or time dim here
        state, ts = env_step(state, action.squeeze())

    return [states]


def _upload_gif_to_neptune(logger: MavaLogger, num_gifs: int = 1) -> None:
    # get neptune logger:
    multi_logger: MultiLogger = logger.logger  # type: ignore
    neptune_logger = [
        logger for logger in multi_logger.loggers if isinstance(logger, NeptuneLogger)
    ]
    if neptune_logger:
        run = neptune_logger[0].logger
        for i in range(num_gifs):
            run[f"gif_{i}"].upload(f"env_{i}.gif")
    else:
        warnings.warn("No neptune logger found, could not upload env gif.")  # noqa: B028
