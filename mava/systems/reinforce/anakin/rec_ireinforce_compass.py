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
import time
from typing import Any, Dict, Tuple

import chex
import flax
import hydra
import jax
import jax.numpy as jnp
import optax
import tensorflow_probability.substrates.jax.distributions as tfd
from colorama import Fore, Style
from flax.core.frozen_dict import FrozenDict
from jax import tree
from jumanji.types import TimeStep
from omegaconf import DictConfig, OmegaConf
from optax._src.base import OptState
from rich.pretty import pprint

from mava.evaluator import ActorState, COMPASSEvalActFn, get_num_eval_envs
from mava.evaluator import get_compass_eval_fn as get_eval_fn
from mava.networks import RecurrentCOMPASSActor as Actor
from mava.networks import ScannedRNN
from mava.networks.distributions import IdentityTransformation
from mava.networks.heads import DiscreteLogitHead
from mava.systems.ppo.types import (
    HiddenStates,
    RNNLearnerState,
    RNNPPOTransition,
)
from mava.types import (
    Action,
    CompassRecActorApply,
    ExperimentOutput,
    LearnerFn,
    MarlEnv,
)
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.compass_utils import (
    duplicate_over_latent_dim,
    get_best_latent_idx,
    get_compass_latent,
    pad_weights,
    select_best_latent_data,
)
from mava.utils.config import check_total_timesteps
from mava.utils.jax_utils import unreplicate_batch_dim, unreplicate_n_dims
from mava.utils.logger import LogEvent, MavaLogger
from mava.utils.model_downloads import delete_local_checkpoints, unzip_local_checkpoints
from mava.utils.network_utils import _DISCRETE, get_action_head
from mava.utils.render_env import render
from mava.utils.training import make_learning_rate
from mava.wrappers.episode_metrics import get_final_step_metrics


def get_learner_fn(
    env: MarlEnv,
    actor_apply_fn: CompassRecActorApply,
    actor_update_fn: optax.TransformUpdateFn,
    config: DictConfig,
) -> LearnerFn[RNNLearnerState]:
    """Get the learner function."""

    def _update_step(learner_state: RNNLearnerState, _: Any) -> Tuple[RNNLearnerState, Tuple]:
        """A single update of the network.

        This function steps the environment and records the trajectory batch for
        training. It then calculates advantages and targets based on the recorded
        trajectory and updates the actor and critic networks based on the calculated
        losses.

        Args:
        ----
            learner_state (NamedTuple):
                - params (Params): The current model parameters.
                - opt_states (OptStates): The current optimizer states.
                - key (PRNGKey): The random number generator state.
                - env_state (State): The environment state.
                - last_timestep (TimeStep): The last timestep in the current trajectory.
                - dones (bool): Whether the last timestep was a terminal state.
                - hstates (HiddenStates): The current hidden states of the RNN.
            _ (Any): The current metrics info.

        """

        def _env_step(
            learner_state_latent: Tuple[RNNLearnerState, chex.Array], _: Any
        ) -> Tuple[Tuple[RNNLearnerState, chex.Array], RNNPPOTransition]:
            """Step the environment."""

            learner_state, latent = learner_state_latent
            (
                params,
                opt_states,
                key,
                env_state,
                last_timestep,
                last_done,
                last_policy_hidden_state,
            ) = learner_state

            key, policy_key = jax.random.split(key)

            # Add a time dimension to the observation.
            batched_observation = tree.map(lambda x: x[jnp.newaxis, :], last_timestep.observation)
            ac_in = (
                batched_observation,
                last_done[jnp.newaxis, ...],
            )

            # apply takes in (params, hstates, (obs, dones), latent)
            vmap_actor_apply_fn = jax.vmap(
                actor_apply_fn,
                in_axes=(
                    None,
                    1,
                    (2, 2),
                    1,
                ),
                out_axes=(1, 2),
            )

            # Run the network.
            policy_hidden_state, actor_logits = vmap_actor_apply_fn(
                params,
                last_policy_hidden_state,
                ac_in,
                latent,
            )
            actor_policy = IdentityTransformation(distribution=tfd.Categorical(logits=actor_logits))

            # Sample action from the policy and squeeze out the batch dimension.
            action = actor_policy.sample(seed=policy_key)
            log_prob = actor_policy.log_prob(action)
            action, log_prob = action.squeeze(0), log_prob.squeeze(0)

            # Step the environment.
            env_state, timestep = jax.vmap(jax.vmap(env.step, in_axes=(0, 0)), in_axes=(0, 0))(
                env_state, action
            )

            # log episode return and length
            done = tree.map(
                lambda x: jnp.repeat(x, config.system.num_agents).reshape(
                    config.arch.num_envs, config.arch.num_latents_per_env, -1
                ),
                timestep.last(),
            )
            info = timestep.extras["episode_metrics"]

            # NOTE: Discounts are 1 when not done and 0 when done, so we flip them here.
            transition = RNNPPOTransition(
                last_done,
                action,
                None,
                timestep.reward,
                log_prob,
                last_timestep.observation,
                last_policy_hidden_state,
                info,
                (1.0 - last_timestep.discount).astype(bool),
            )
            learner_state = RNNLearnerState(
                params, opt_states, key, env_state, timestep, done, policy_hidden_state
            )
            return (learner_state, latent), transition

        key, latent_key = jax.random.split(learner_state.key)
        learner_state = learner_state._replace(key=key)

        if config.arch.latent_sampling_same:
            latent = get_compass_latent(latent_key, config, (1,))
            latent = jnp.repeat(latent, config.arch.num_envs, axis=0)

        else:
            latent = get_compass_latent(latent_key, config, (config.arch.num_envs,))

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        (learner_state, _), traj_batch = jax.lax.scan(
            _env_step, (learner_state, latent), None, config.system.rollout_length
        )

        # CALCULATE ADVANTAGE
        (
            params,
            opt_states,
            key,
            _,
            _,
            last_done,
            policy_hstate,
        ) = learner_state

        # Reset the environment for the next rollout since this is reinforce and not PPO.
        key, env_key = jax.random.split(key)
        env_keys = jax.random.split(env_key, config.arch.num_envs)
        env_state, last_timestep = jax.vmap(env.reset)(env_keys)
        (env_state, last_timestep) = tree.map(
            lambda x: duplicate_over_latent_dim(x, 0, config.arch.num_latents_per_env),
            (env_state, last_timestep),
        )

        # Reset the policy hidden state.
        policy_hstate = jnp.zeros_like(policy_hstate)
        # Reset all the dones since the environment is reset.
        last_done = jnp.zeros_like(last_done, dtype=last_done.dtype)

        full_rewards = traj_batch.reward
        full_done_mask = traj_batch.done_mask

        # Get the index for the best latent for each environment and select the related data.
        # We need to say ~traj_batch.done_mask since we want to train on data before
        # the episode is done ie. when done_mask is False in the transition.
        best_latent_idxs = get_best_latent_idx(traj_batch.reward, ~traj_batch.done_mask)
        traj_batch = select_best_latent_data(traj_batch, best_latent_idxs, config.arch.num_envs)
        best_latent = latent[jnp.arange(config.arch.num_envs), best_latent_idxs]

        def _calculate_reward_to_go(
            traj_batch: RNNPPOTransition,
        ) -> Tuple[chex.Array, chex.Array]:
            """Calculate the reward-to-go and advantages."""

            def _get_reward_to_go(
                accumulated_reward: chex.Array, transition: RNNPPOTransition
            ) -> Tuple[chex.Array, chex.Array]:
                """Calculate the reward-to-go for a single transition."""
                reward, done = transition.reward, transition.done_mask
                gamma = config.system.gamma
                accumulated_reward = (reward + gamma * accumulated_reward) * (1 - done)
                return accumulated_reward, accumulated_reward

            # Initialize accumulated_reward with zeros of the same shape as the rewards
            initial_accumulated_reward = jnp.zeros_like(traj_batch.reward[0])

            # Compute the reward-to-go using a reverse scan over the trajectory
            _, rewards_to_go = jax.lax.scan(
                _get_reward_to_go,
                initial_accumulated_reward,
                traj_batch,
                reverse=True,
                unroll=16,
            )

            return rewards_to_go

        reward_to_go = _calculate_reward_to_go(traj_batch)

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _update_minibatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
                """Update the network for a single minibatch."""
                # UNPACK TRAIN STATE AND BATCH INFO
                params, opt_states, key = train_state
                traj_batch, reward_to_go, train_done_mask, best_latent = batch_info

                def _actor_loss_fn(
                    actor_params: FrozenDict,
                    actor_opt_state: OptState,
                    traj_batch: RNNPPOTransition,
                    reward_to_go: chex.Array,
                    train_done_mask: chex.Array,
                    key: chex.PRNGKey,
                    best_latent: chex.Array,
                ) -> Tuple:
                    """Calculate the actor loss."""
                    # RERUN NETWORK

                    obs_and_done = (traj_batch.obs, train_done_mask)
                    _, actor_logits = actor_apply_fn(
                        actor_params, traj_batch.hstates[0], obs_and_done, best_latent
                    )
                    actor_policy = IdentityTransformation(
                        distribution=tfd.Categorical(logits=actor_logits)
                    )
                    log_prob = actor_policy.log_prob(traj_batch.action)

                    # Negative because we are doing gradient ascent.
                    loss_actor = reward_to_go * -log_prob
                    # NOTE: Pretty sure it should be the inverse of the done mask since it is
                    # true when done but we want to train on data before that.
                    loss_actor = loss_actor.mean(where=~traj_batch.done_mask)

                    # Just a dummy, not used.
                    entropy = actor_policy.entropy(seed=key).mean(where=~traj_batch.done_mask)

                    total_loss_actor = loss_actor
                    return total_loss_actor, (loss_actor, entropy)

                # CALCULATE ACTOR LOSS
                key, entropy_key = jax.random.split(key)
                actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                actor_loss_info, actor_grads = actor_grad_fn(
                    params,
                    opt_states,
                    traj_batch,
                    reward_to_go,
                    train_done_mask,
                    entropy_key,
                    best_latent,
                )

                # Compute the parallel mean (pmean) over the batch.
                # This calculation is inspired by the Anakin architecture demo notebook.
                # available at https://tinyurl.com/26tdzs5x
                # This pmean could be a regular mean as the batch axis is on the same device.
                actor_grads, actor_loss_info = jax.lax.pmean(
                    (actor_grads, actor_loss_info), axis_name="batch"
                )
                # pmean over devices.
                actor_grads, actor_loss_info = jax.lax.pmean(
                    (actor_grads, actor_loss_info), axis_name="device"
                )

                # UPDATE ACTOR PARAMS AND OPTIMISER STATE
                actor_updates, actor_new_opt_state = actor_update_fn(actor_grads, opt_states)
                actor_new_params = optax.apply_updates(params, actor_updates)

                # PACK LOSS INFO
                total_loss = actor_loss_info[0]
                actor_loss = actor_loss_info[1][0]
                entropy = actor_loss_info[1][1]
                loss_info = {
                    "total_loss": total_loss,
                    "actor_loss": actor_loss,
                    "entropy": entropy,
                }

                return (actor_new_params, actor_new_opt_state, entropy_key), loss_info

            params, opt_states, traj_batch, reward_to_go, key, best_latent = update_state
            key, shuffle_key, entropy_key = jax.random.split(key, 3)

            # We need a mask for resetting the hidden states during training that is False up until
            # when all agents are done and then True afterwards.
            all_true_mask = jnp.all(traj_batch.done_mask, axis=-1, keepdims=True)
            train_done_mask = traj_batch.done_mask & all_true_mask

            # SHUFFLE MINIBATCHES
            batch = (traj_batch, reward_to_go, train_done_mask)
            num_recurrent_chunks = (
                config.system.rollout_length // config.system.recurrent_chunk_size
            )
            batch = tree.map(
                lambda x: x.reshape(
                    config.system.recurrent_chunk_size,
                    config.arch.num_envs * num_recurrent_chunks,
                    *x.shape[2:],
                ),
                batch,
            )
            permutation = jax.random.permutation(
                shuffle_key, config.arch.num_envs * num_recurrent_chunks
            )
            shuffled_batch = tree.map(lambda x: jnp.take(x, permutation, axis=1), batch)
            shuffled_best_latent = tree.map(lambda x: jnp.take(x, permutation, axis=0), best_latent)
            reshaped_batch = tree.map(
                lambda x: jnp.reshape(
                    x, (x.shape[0], config.system.num_minibatches, -1, *x.shape[2:])
                ),
                shuffled_batch,
            )
            minibatches = tree.map(lambda x: jnp.swapaxes(x, 1, 0), reshaped_batch)
            best_latent_minibatch = jnp.reshape(
                shuffled_best_latent,
                (config.system.num_minibatches, -1, *shuffled_best_latent.shape[1:]),
            )

            # UPDATE MINIBATCHES
            (params, opt_states, entropy_key), loss_info = jax.lax.scan(
                _update_minibatch,
                (params, opt_states, entropy_key),
                (*minibatches, best_latent_minibatch),
            )

            update_state = (
                params,
                opt_states,
                traj_batch,
                reward_to_go,
                key,
                best_latent,
            )
            return update_state, loss_info

        update_state = (
            params,
            opt_states,
            traj_batch,
            reward_to_go,
            key,
            best_latent,
        )

        # UPDATE EPOCHS
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.ppo_epochs
        )

        params, opt_states, traj_batch, reward_to_go, key, _ = update_state
        learner_state = RNNLearnerState(
            params,
            opt_states,
            key,
            env_state,
            last_timestep,
            last_done,
            policy_hstate,
        )
        metric = traj_batch.info
        metric["full_rewards"] = full_rewards
        metric["full_done_mask"] = full_done_mask
        return learner_state, (metric, loss_info)

    def learner_fn(learner_state: RNNLearnerState) -> ExperimentOutput[RNNLearnerState]:
        """Learner function.

        This function represents the learner, it updates the network parameters
        by iteratively applying the `_update_step` function for a fixed number of
        updates. The `_update_step` function is vectorized over a batch of inputs.

        Args:
        ----
            learner_state (NamedTuple):
                - params (Params): The initial model parameters.
                - opt_states (OptStates): The initial optimizer states.
                - key (chex.PRNGKey): The random number generator state.
                - env_state (LogEnvState): The environment state.
                - timesteps (TimeStep): The initial timestep in the initial trajectory.
                - dones (bool): Whether the initial timestep was a terminal state.
                - hstateS (HiddenStates): The initial hidden states of the RNN.

        """
        batched_update_step = jax.vmap(_update_step, in_axes=(0, None), axis_name="batch")

        learner_state, (episode_info, loss_info) = jax.lax.scan(
            batched_update_step, learner_state, None, config.system.num_updates_per_eval
        )
        return ExperimentOutput(
            learner_state=learner_state,
            episode_metrics=episode_info,
            train_metrics=loss_info,
        )

    return learner_fn


def learner_setup(
    env: MarlEnv, keys: chex.Array, config: DictConfig
) -> Tuple[LearnerFn[RNNLearnerState], Actor, RNNLearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())

    # Get number of agents.
    num_agents = env.num_agents
    config.system.num_agents = num_agents

    # PRNG keys.
    key, actor_net_key, _ = keys

    # Define network and optimisers.
    actor_pre_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    actor_post_torso = hydra.utils.instantiate(config.network.actor_network.post_torso)
    _, action_space_type = get_action_head(env.action_spec())

    if action_space_type != _DISCRETE:
        raise NotImplementedError("COMPASS REINFORCE systems only support discrete action spaces")

    policy_head = DiscreteLogitHead(env.action_dim)

    actor_network = Actor(
        pre_torso=actor_pre_torso,
        post_torso=actor_post_torso,
        hidden_state_dim=config.network.hidden_state_dim,
        action_head=policy_head,
    )
    actor_lr = make_learning_rate(config.system.actor_lr, config)
    actor_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(actor_lr, eps=1e-5),
    )
    # Initialise observation with obs of all agents.
    # We don;t
    init_obs = env.observation_spec().generate_value()
    init_obs = tree.map(
        lambda x: jnp.repeat(x[jnp.newaxis, ...], config.arch.num_envs, axis=0),
        init_obs,
    )
    # (time, batch, agents, ...)
    init_obs = tree.map(lambda x: x[jnp.newaxis, ...], init_obs)
    # (time, batch, agents)
    init_done = jnp.zeros((1, config.arch.num_envs, num_agents), dtype=bool)
    init_x = (init_obs, init_done)
    # Latent vectors do not need a time axis since we put the same latent at each timestep in a
    # trajectory.
    # (batch, agent, latent_dim)
    init_latent = jnp.ones(
        (config.arch.num_envs, config.system.num_agents, config.arch.compass_latent_dim)
    )

    # Initialise hidden states.
    # (batch, agents, ...)
    init_policy_hstate = ScannedRNN.initialize_carry(
        (config.arch.num_envs, num_agents), config.network.hidden_state_dim
    )
    # initialise params and optimiser state.
    actor_params = actor_network.init(actor_net_key, init_policy_hstate, init_x, init_latent)

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(env, actor_network.apply, actor_optim.update, config)
    learn = jax.pmap(learn, axis_name="device")

    # Initialise environment states and timesteps: across devices and batches.
    key, *env_keys = jax.random.split(
        key, n_devices * config.system.update_batch_size * config.arch.num_envs + 1
    )
    env_states, timesteps = jax.vmap(env.reset, in_axes=(0))(
        jnp.stack(env_keys),
    )
    reshape_states = lambda x: x.reshape(
        (n_devices, config.system.update_batch_size, config.arch.num_envs) + x.shape[1:]
    )
    # (devices, update batch size, num_envs, ...)
    env_states = tree.map(reshape_states, env_states)
    timesteps = tree.map(reshape_states, timesteps)

    (env_states, timesteps) = tree.map(
        lambda x: duplicate_over_latent_dim(x, 2, config.arch.num_latents_per_env),
        (env_states, timesteps),
    )

    # Create an hstate for each latent.
    init_policy_hstate = tree.map(
        lambda x: duplicate_over_latent_dim(x, 0, config.arch.num_latents_per_env),
        init_policy_hstate,
    )

    # Download checkpoint from Neptune if specified.
    if config.logger.checkpointing.unzip_local_model:
        unzip_local_checkpoints(
            checkpoint_rel_dir="checkpoints",
            model_name=config.logger.system_name,
            run_id=config.logger.checkpointing.download_args.neptune_run_name,
        )

    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.logger.system_name,
            **config.logger.checkpointing.load_args,  # Other checkpoint args
        )
        # Restore the learner state from the checkpoint
        restored_params, _ = loaded_checkpoint.restore_params(
            input_params=actor_params, restore_hstates=True, THiddenState=HiddenStates
        )

        # Update the params and hstates
        params = restored_params["actor_params"]

        key, kernel_key = jax.random.split(key, num=2)

        path_to_kernel = ["params", "pre_torso", "Dense_0", "kernel"]
        params = pad_weights(
            kernel_key,
            params,
            path_to_kernel,
            config.arch.compass_latent_dim,
            random_weights=config.arch.padding_with_random_weights,
            noise=config.arch.weights_noise,
        )
        actor_params = params

    actor_opt_state = actor_optim.init(actor_params)

    # Define params to be replicated across devices and batches.
    dones = jnp.zeros(
        (config.arch.num_envs, config.arch.num_latents_per_env, num_agents),
        dtype=bool,
    )
    key, step_keys = jax.random.split(key)
    replicate_learner = (actor_params, actor_opt_state, init_policy_hstate, step_keys, dones)

    # Duplicate learner for update_batch_size.
    broadcast = lambda x: jnp.broadcast_to(x, (config.system.update_batch_size, *x.shape))
    replicate_learner = tree.map(broadcast, replicate_learner)

    # Duplicate learner across devices.
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=jax.devices())

    # Initialise learner state.
    actor_params, actor_opt_state, init_policy_hstate, step_keys, dones = replicate_learner
    init_learner_state = RNNLearnerState(
        params=actor_params,
        opt_states=actor_opt_state,
        key=step_keys,
        env_state=env_states,
        timestep=timesteps,
        dones=dones,
        hstates=init_policy_hstate,
    )
    return learn, actor_network, init_learner_state


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    if "base_system_name" in _config.logger.checkpointing:
        system_name = f"{_config.logger.checkpointing.base_system_name}_compass"
    else:
        system_name = "rec_ireinforce_compass"

    _config.logger.system_name = system_name
    config = copy.deepcopy(_config)

    n_devices = len(jax.devices())

    if config.system.recurrent_chunk_size is not None:
        raise NotImplementedError("Compass systems do not support recurrent rollout chunking.")

    # Set recurrent chunk size.
    if config.system.recurrent_chunk_size is None:
        config.system.recurrent_chunk_size = config.system.rollout_length
    else:
        assert (
            config.system.rollout_length % config.system.recurrent_chunk_size == 0
        ), "Rollout length must be divisible by recurrent chunk size."

        assert (
            config.arch.num_envs % config.system.num_minibatches == 0
        ), "Number of envs must be divisibile by number of minibatches."

    # Create the enviroments for train and eval.
    env, eval_env = environments.make(config, fixed_reset=True)

    # PRNG keys.
    key, key_e, actor_net_key, critic_net_key = jax.random.split(
        jax.random.PRNGKey(config.system.seed), num=4
    )

    # Setup learner.
    learn, actor_network, learner_state = learner_setup(
        env, (key, actor_net_key, critic_net_key), config
    )

    # Setup evaluator.
    # One key per device for evaluation.
    eval_keys = jax.random.split(key_e, n_devices)

    def make_rec_eval_act_fn(
        actor_apply_fn: CompassRecActorApply,
        config: DictConfig,
        vmap_apply_over_latents: bool = True,
    ) -> COMPASSEvalActFn:
        """Makes an act function that conforms to the evaluator API given a standard
        recurrent mava actor network."""

        _hidden_state = "hidden_state"

        def eval_act_fn(
            params: FrozenDict,
            timestep: TimeStep,
            key: chex.PRNGKey,
            actor_state: ActorState,
            latent: chex.Array,
        ) -> Tuple[Action, Dict]:
            hidden_state = actor_state[_hidden_state]

            n_agents = timestep.observation.agents_view.shape[2]
            last_done = timestep.last()[..., jnp.newaxis].repeat(n_agents, axis=-1)
            ac_in = (timestep.observation, last_done)
            ac_in = tree.map(lambda x: x[jnp.newaxis], ac_in)  # add batch dim to obs

            if vmap_apply_over_latents:
                _actor_apply_fn = jax.vmap(
                    actor_apply_fn,
                    in_axes=(
                        None,
                        1,
                        (2, 2),
                        1,
                    ),
                    out_axes=(1, 2),
                )

            else:
                _actor_apply_fn = actor_apply_fn

            hidden_state, actor_logits = _actor_apply_fn(params, hidden_state, ac_in, latent)
            pi = IdentityTransformation(distribution=tfd.Categorical(logits=actor_logits))
            action = pi.mode() if config.arch.evaluation_greedy else pi.sample(seed=key)
            return action.squeeze(0), {_hidden_state: hidden_state}

        return eval_act_fn

    eval_act_fn = make_rec_eval_act_fn(actor_network.apply, config)
    evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=False)

    # Calculate total timesteps.
    config = check_total_timesteps(config)
    assert (
        config.system.num_updates > config.arch.num_evaluation
    ), "Number of updates per evaluation must be less than total number of updates."

    # Calculate number of updates per evaluation.
    config.system.num_updates_per_eval = config.system.num_updates // config.arch.num_evaluation
    steps_per_rollout = (
        n_devices
        * config.system.num_updates_per_eval
        * config.system.rollout_length
        * config.system.update_batch_size
        * config.arch.num_envs
    )

    # Logger setup
    logger = MavaLogger(config)
    cfg: Dict = OmegaConf.to_container(config, resolve=True)
    cfg["arch"]["devices"] = jax.devices()
    pprint(cfg)

    # Set up checkpointer
    save_checkpoint = config.logger.checkpointing.save_model
    if save_checkpoint:
        checkpointer = Checkpointer(
            metadata=config,  # Save all config as metadata in the checkpoint
            model_name=config.logger.system_name,
            **config.logger.checkpointing.save_args,  # Checkpoint args
        )

    # Create an initial hidden state used for resetting memory for evaluation
    eval_batch_size = get_num_eval_envs(config, absolute_metric=False)
    eval_hs = ScannedRNN.initialize_carry(
        (n_devices, eval_batch_size, config.system.num_agents),
        config.network.hidden_state_dim,
    )
    if config.arch.eval_diff_latent_num:
        eval_hs = tree.map(
            lambda x: duplicate_over_latent_dim(x, 1, config.arch.eval_num_latents_per_env), eval_hs
        )
    else:
        eval_hs = tree.map(
            lambda x: duplicate_over_latent_dim(x, 1, config.arch.num_latents_per_env), eval_hs
        )

    # Run experiment for a total number of evaluations.
    max_episode_return = -jnp.inf
    best_params = None
    for eval_step in range(config.arch.num_evaluation):
        # Train.
        start_time = time.time()
        learner_output = learn(learner_state)
        jax.block_until_ready(learner_output)

        # Log the results of the training.
        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))
        episode_metrics, ep_completed = get_final_step_metrics(learner_output.episode_metrics)
        episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time

        # Separately log timesteps, actoring metrics and training metrics.
        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        if ep_completed:  # only log episode metrics if an episode was completed in the rollout.
            logger.log(episode_metrics, t, eval_step, LogEvent.ACT)
        logger.log(learner_output.train_metrics, t, eval_step, LogEvent.TRAIN)

        # Prepare for evaluation.
        trained_params = unreplicate_batch_dim(learner_state.params)
        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)
        # Evaluate.
        eval_metrics = evaluator(trained_params, eval_keys, {"hidden_state": eval_hs})
        logger.log(eval_metrics, t, eval_step, LogEvent.EVAL)
        # Use the mean episode return from the best latent here.
        eval_metrics["episode_return"] = eval_metrics["latent_max_episode_return"]
        if "latent_max_win_rate" in eval_metrics:
            eval_metrics["win_rate"] = eval_metrics["latent_max_win_rate"]
        episode_return = jnp.mean(eval_metrics["episode_return"])

        if save_checkpoint:
            # Save checkpoint of learner state
            checkpointer.save(
                timestep=steps_per_rollout * (eval_step + 1),
                unreplicated_learner_state=unreplicate_n_dims(learner_output.learner_state),
                episode_return=episode_return,
            )

        if config.arch.absolute_metric and max_episode_return <= episode_return:
            best_params = copy.deepcopy(trained_params)
            max_episode_return = episode_return

        # Update runner state to continue training.
        learner_state = learner_output.learner_state

    # Record the performance for the final evaluation run.
    eval_performance = float(jnp.mean(eval_metrics[config.env.eval_metric]))

    render_hs = ScannedRNN.initialize_carry(
        (1, config.system.num_agents),
        config.network.hidden_state_dim,
    )

    if config.arch.render:
        render(
            eval_env,
            tree.map(lambda x: x[0], trained_params),
            {"hidden_state": render_hs},
            key,
            eval_act_fn,
            logger,
        )

    # Measure absolute metric.
    if config.arch.absolute_metric:
        eval_batch_size = get_num_eval_envs(config, absolute_metric=True)
        eval_hs = ScannedRNN.initialize_carry(
            (n_devices, eval_batch_size, config.system.num_agents),
            config.network.hidden_state_dim,
        )
        if config.arch.eval_diff_latent_num:
            eval_hs = tree.map(
                lambda x: duplicate_over_latent_dim(x, 1, config.arch.eval_num_latents_per_env),
                eval_hs,
            )
        else:
            eval_hs = tree.map(
                lambda x: duplicate_over_latent_dim(x, 1, config.arch.num_latents_per_env), eval_hs
            )
        abs_metric_evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=True)
        eval_keys = jax.random.split(key, n_devices)

        eval_metrics = abs_metric_evaluator(best_params, eval_keys, {"hidden_state": eval_hs})

        t = int(steps_per_rollout * (eval_step + 1))
        logger.log(eval_metrics, t, eval_step, LogEvent.ABSOLUTE)

    # Stop the logger.
    logger.stop()

    # Remove the local model checkpoint when training is complete
    if config.logger.checkpointing.delete_local_checkpoints:
        delete_local_checkpoints(checkpoint_folder_dir="checkpoints")

    return eval_performance


@hydra.main(
    config_path="../../../configs/default",
    config_name="rec_ireinforce.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    eval_performance = run_experiment(cfg)
    print(
        f"{Fore.CYAN}{Style.BRIGHT}Recurrent IREINFORCE COMPASS "
        f"experiment completed{Style.RESET_ALL}"
    )
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
