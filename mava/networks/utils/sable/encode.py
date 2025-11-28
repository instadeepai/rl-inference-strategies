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

from typing import Optional, Tuple

import chex
from einops import rearrange
from flax import linen as nn

# General shapes legend:
# B: batch size
# S: sequence length
# C: number of agents per chunk of sequence


def train_encoder_fn(
    encoder: nn.Module,
    obs: chex.Array,
    hstate: chex.Array,
    dones: chex.Array,
    step_count: chex.Array,
    chunk_size: int,
    latent: Optional[chex.Array] = None,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Chunkwise encoding for discrete action spaces."""
    B, S = obs.shape[:2]

    # Apply the encoder per chunk
    num_chunks = S // chunk_size

    # Reshape inputs for scanning over chunks
    obs = rearrange(obs, "b (nc cs) d -> nc b cs d", nc=num_chunks, cs=chunk_size)
    dones = rearrange(dones, "b (nc cs) -> nc b cs", nc=num_chunks, cs=chunk_size)
    step_count = rearrange(step_count, "b (nc cs) -> nc b cs", nc=num_chunks, cs=chunk_size)
    if latent is not None:
        latent = rearrange(latent, "b (nc cs) d -> nc b cs d", nc=num_chunks, cs=chunk_size)

    def single_chunk_encoder_fn(
        encoder: nn.Module,
        hstate: chex.Array,
        chunked_inputs: Tuple[chex.Array, chex.Array, chex.Array, Optional[chex.Array]],
    ) -> Tuple[chex.Array, Tuple[chex.Array, chex.Array]]:
        obs, dones, step_count, latent = chunked_inputs
        v_loc, obs_rep, hstate = encoder(obs, hstate, dones, step_count, latent)
        return hstate, (v_loc, obs_rep)

    encode_obs = nn.scan(
        nn.remat(single_chunk_encoder_fn, prevent_cse=False),
        variable_broadcast="params",
        split_rngs={"params": False},
        unroll=1,
    )

    hstate, (v_loc, obs_rep) = encode_obs(encoder, hstate, (obs, dones, step_count, latent))

    # Reshape outputs
    v_loc = rearrange(v_loc, "nc b cs d -> b (nc cs) d", nc=num_chunks, cs=chunk_size)
    obs_rep = rearrange(obs_rep, "nc b cs d -> b (nc cs) d", nc=num_chunks, cs=chunk_size)

    return v_loc, obs_rep, hstate


def act_encoder_fn(
    encoder: nn.Module,
    obs: chex.Array,
    decayed_hstate: chex.Array,
    step_count: chex.Array,
    chunk_size: int,
    latent: Optional[chex.Array] = None,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Chunkwise encoding for ff-Sable and for discrete action spaces."""
    B, C = obs.shape[:2]

    # Apply the encoder per chunk
    num_chunks = C // chunk_size

    # Reshape inputs for scanning over chunks
    obs = rearrange(obs, "b (nc cs) d -> nc b cs d", nc=num_chunks, cs=chunk_size)
    step_count = rearrange(step_count, "b (nc cs) -> nc b cs", nc=num_chunks, cs=chunk_size)
    if latent is not None:
        latent = rearrange(latent, "b (nc cs) d -> nc b cs d", nc=num_chunks, cs=chunk_size)

    def single_chunk_encoder_fn(
        encoder: nn.Module,
        hstate: chex.Array,
        chunked_inputs: Tuple[chex.Array, chex.Array, Optional[chex.Array]],
    ) -> Tuple[chex.Array, Tuple[chex.Array, chex.Array]]:
        obs, step_count, latent = chunked_inputs
        v_loc, obs_rep, hstate = encoder.recurrent(obs, hstate, step_count, latent)
        return hstate, (v_loc, obs_rep)

    encode_obs = nn.scan(
        nn.remat(single_chunk_encoder_fn, prevent_cse=False),
        variable_broadcast="params",
        split_rngs={"params": False},
        unroll=1,
    )

    decayed_hstate, (v_loc, obs_rep) = encode_obs(
        encoder, decayed_hstate, (obs, step_count, latent)
    )

    # Reshape outputs
    v_loc = rearrange(v_loc, "nc b cs d -> b (nc cs) d", nc=num_chunks, cs=chunk_size)
    obs_rep = rearrange(obs_rep, "nc b cs d -> b (nc cs) d", nc=num_chunks, cs=chunk_size)

    return v_loc, obs_rep, decayed_hstate
