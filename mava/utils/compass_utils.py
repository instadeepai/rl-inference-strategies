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

from copy import deepcopy
from typing import Sequence

import chex
import jax.numpy as jnp
import jax.random as jax_random
from jax import tree
from jax.lax import cond
from omegaconf import DictConfig

from mava.types import MavaObservation


def duplicate_over_latent_dim(
    pytree: chex.ArrayTree, duplication_axis: int, num_latents: int
) -> chex.ArrayTree:
    """Duplicates the input pytree num_latent times.
    Generally the duplication axis will be the axis corrspoding to the environment dimension.
    For example, if the input array has shape (rollout_length, num_envs, num_agents, ...), this
    function will be called as `duplicate_over_latent_dim(pytree, 1, num_latents)` and will return
    a new array of shape (rollout_length, num_envs, num_latents, num_agents, ...).
    """

    def duplicate_fn(x: chex.Array) -> chex.Array:
        return jnp.expand_dims(x, axis=duplication_axis + 1).repeat(
            num_latents, axis=duplication_axis + 1
        )

    return tree.map(duplicate_fn, pytree)


def concatenate_latent_to_obs(observation: MavaObservation, latent: chex.Array) -> MavaObservation:
    return observation._replace(
        agents_view=jnp.concatenate([observation.agents_view, latent], axis=-1)
    )


def get_compass_latent(
    latent_key: chex.PRNGKey,
    config: DictConfig,
    batch_dim: Sequence[int],
    eval_diff_latent_num: bool = False,
) -> chex.Array:
    if eval_diff_latent_num:
        latent = jax_random.uniform(
            key=latent_key,
            shape=(
                *batch_dim,
                config.arch.eval_num_latents_per_env,
                config.arch.compass_latent_dim,
            ),
            minval=-1.0,
            maxval=1.0,
        )
    else:
        latent = jax_random.uniform(
            key=latent_key,
            shape=(
                *batch_dim,
                config.arch.num_latents_per_env,
                config.arch.compass_latent_dim,
            ),
            minval=-1.0,
            maxval=1.0,
        )

    # duplicate latent over agents
    # NOTE: we assume agents will always share the same latent and be on the second last axis
    latent = jnp.expand_dims(latent, axis=-2).repeat(config.system.num_agents, axis=-2)

    return latent * config.arch.latent_amplifier


def get_best_latent_idx(rollout_rewards: chex.Array, done_mask: chex.Array) -> chex.Array:
    """Returns the indices of the best latent for each environment based on the rollout rewards.
    This will take an input of shape (rollout_length, num_envs, num_latents_per_env, num_agents)
    and return an array of shape (num_envs, ) containing the indices of the best latent for each
    environment. It is important that the done_mask is True where we want data to be summed and
    not where we want it to be ignored.
    """

    # NOTE: we assume agents share the same reward hence the zero slice on the agent dimension.
    # NOTE when not masking, pass in all -1s.
    rollout_returns = cond(
        jnp.sum(done_mask) >= 0,
        lambda: tree.map(lambda x: x.sum(axis=(0, -1), where=done_mask), rollout_rewards),
        lambda: tree.map(lambda x: x.sum(axis=0), rollout_rewards).sum(axis=-1),
    )
    best_latent_idxs = jnp.argmax(rollout_returns, axis=-1)

    return best_latent_idxs


def get_mean_return_over_latents(rollout_rewards: chex.Array, done_mask: chex.Array) -> chex.Array:
    """Do a masked cumulative sum over time of the reward to get the return per env, per latent,
    per agent. Then do a mean over the latents to get a mean cumulative return per env.
    Ie input shape is `(rollout_length, num_envs, num_latents_per_env, num_agents)` and the output
    shape is `(rollout_length, num_envs, num_agents)`.
    """
    rollout_rewards = cond(
        jnp.sum(done_mask) >= 0,
        lambda: rollout_rewards * done_mask,
        lambda: rollout_rewards,
    )

    mean_return = rollout_rewards.cumsum(axis=0)
    # We need to mask again because cumsum will make all values after the first done
    # non-zero and the same.
    mean_return = mean_return * done_mask
    mean_return = mean_return.mean(axis=(-2), where=done_mask)
    return mean_return


def select_best_latent_data(
    traj_batch: chex.ArrayTree, best_latent_idxs: chex.Array, num_envs: int
) -> chex.ArrayTree:
    """Selects the data corresponding to the best latent for each environment in the traj_batch.
    The original shapes of all pytrees in traj_batch is
    (rollout_lenght, num_envs, num_latents_per_env, ...) and the best_latent_idxs is of shape
    (num_envs, ). This function will return a new pytree with shape (rollout_length, num_envs, ...).
    """
    # Select only the best latent's data for each environment in the traj_batch
    traj_batch = tree.map(lambda x: x[:, jnp.arange(num_envs), best_latent_idxs], traj_batch)

    return traj_batch


def pad_weights(
    key: chex.PRNGKey,
    params: chex.ArrayTree,
    path_to_kernel: Sequence[str],
    pad_dim: int,
    kernel: bool = True,
    random_weights: bool = True,
    noise: float = 0.01,
) -> chex.ArrayTree:
    """
    Pads weights or biases of a specified layer in a pretrained model (mostly the first layers)

    Args:
        key : PRNG key for generating random weights.
        params : Parameter dictionary containing the model's weights.
        path_to_kernel : List of strings representing the path to the target layer.
        pad_dim : Number of additional rows to pad.
        kernel : If True pad the kernel layer, else pad the bias
        random_weights : If True, pads with random weights; otherwise, pads with zeros.
        noise : Noise scale for random weights sampled from uniform distribution (-1, 1)

    Returns:
        chex.ArrayTree: Updated parameter dictionary with the specified layer padded.
    """
    updated_params = deepcopy(params)

    # Navigate to the target parameters
    current_dict = updated_params
    for dict_key in path_to_kernel[:-1]:
        current_dict = current_dict[dict_key]

    def create_padding(shape: Sequence[int]) -> chex.Array:
        if random_weights:
            return jax_random.uniform(key, shape=shape, minval=-1, maxval=1) * noise
        return jnp.zeros(shape=shape)

    # Get and pad the layer weights
    layer = current_dict[path_to_kernel[-1]]

    pad_shape = (pad_dim, layer.shape[-1]) if kernel else (pad_dim,)
    padding = create_padding(pad_shape)

    # Update the weights
    current_dict[path_to_kernel[-1]] = jnp.concatenate([layer, padding], axis=0)

    return updated_params
