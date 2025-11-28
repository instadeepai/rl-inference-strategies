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
from functools import partial
from typing import Any, Callable, Dict, Tuple

import chex
import flax
import hydra
import jax
import jax.numpy as jnp
import optax
from colorama import Fore, Style
from flax.core.frozen_dict import FrozenDict as Params
from jax import tree
from jumanji.env import Environment
from jumanji.types import TimeStep
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from mava.evaluator import ActorState, COMPASSEvalActFn, get_num_eval_envs
from mava.evaluator import get_compass_eval_fn as get_eval_fn
from mava.networks import SableNetwork
from mava.networks.utils.sable import get_init_hidden_state
from mava.systems.sable.types import (
    ActorApply,
    HiddenStates,
    LearnerApply,
    Transition,
)
from mava.systems.sable.types import RecLearnerState as LearnerState
from mava.types import Action, ExperimentOutput, LearnerFn, MarlEnv, Observation
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.compass_utils import (
    duplicate_over_latent_dim,
    get_best_latent_idx,
    get_compass_latent,
    get_mean_return_over_latents,
    pad_weights,
    select_best_latent_data,
)
from mava.utils.config import check_total_timesteps
from mava.utils.jax_utils import concat_time_and_agents, unreplicate_batch_dim, unreplicate_n_dims
from mava.utils.logger import LogEvent, MavaLogger
from mava.utils.model_downloads import delete_local_checkpoints, unzip_local_checkpoints
from mava.utils.network_utils import get_action_head
from mava.utils.render_env import render
from mava.utils.training import adjust_config_for_gradient_accumulation, make_learning_rate
from mava.wrappers.episode_metrics import get_final_step_metrics


def get_learner_fn(
    env: Environment,
    apply_fns: Tuple[ActorApply, LearnerApply],
    update_fn: optax.TransformUpdateFn,
    config: DictConfig,
) -> LearnerFn[LearnerState]:
    """Get the learner function."""

    # Get apply functions for executing and training the network.
    sable_action_select_fn, sable_apply_fn = apply_fns

    def _update_step(learner_state: LearnerState, _: Any) -> Tuple[LearnerState, Tuple]:
        """A single update of the network.

        This function steps the environment and records the trajectory batch for
        training. It then calculates advantages and targets based on the recorded
        trajectory and updates the actor and critic networks based on the calculated
        losses.

        Args:
        ----
            learner_state (NamedTuple):
                - params (FrozenDict): The current model parameters.
                - opt_states (OptState): The current optimizer states.
                - key (PRNGKey): The random number generator state.
                - env_state (State): The environment state.
                - last_timestep (TimeStep): The last timestep in the current trajectory.
                - hstates (HiddenStates): The hidden state of the network.
            _ (Any): The current metrics info.

        """

        def _env_step(
            learner_state_latent: Tuple[LearnerState, chex.Array], _: int
        ) -> Tuple[Tuple[LearnerState, chex.Array], Transition]:
            """Step the environment."""
            learner_state, latent = learner_state_latent
            params, opt_states, key, env_state, last_timestep, hstates = learner_state

            # SELECT ACTION
            key, policy_key = jax.random.split(key)

            # Apply the actor network to get the action, log_prob, value and updated hstates.
            last_obs = last_timestep.observation
            vmap_action_select_fn = jax.vmap(
                sable_action_select_fn,
                in_axes=(
                    None,
                    Observation(1, 1, 1),
                    HiddenStates(1, 1, 1),
                    None,
                    1,
                ),
                out_axes=(1, 1, 1, HiddenStates(1, 1, 1)),
            )
            action, log_prob, _, hstates = vmap_action_select_fn(  # type: ignore
                params,
                last_obs,
                hstates,
                policy_key,
                latent,
            )

            # STEP ENVIRONMENT
            env_state, timestep = jax.vmap(jax.vmap(env.step, in_axes=(0, 0)), in_axes=(0, 0))(
                env_state, action
            )

            # LOG EPISODE METRICS
            info = tree.map(
                lambda x: jnp.repeat(x[..., jnp.newaxis], config.system.num_agents, axis=-1),
                timestep.extras["episode_metrics"],
            )

            # Reset hidden state if done.
            done = timestep.last()
            done = jnp.expand_dims(done, (2, 3, 4, 5))
            hstates = tree.map(lambda hs: jnp.where(done, jnp.zeros_like(hs), hs), hstates)

            # SET TRANSITION
            prev_done = last_timestep.last()[..., jnp.newaxis].repeat(
                config.system.num_agents, axis=-1
            )
            transition = Transition(
                prev_done,
                action,
                None,
                timestep.reward,
                log_prob,
                last_timestep.observation,
                info,
                (1.0 - last_timestep.discount).astype(bool),
            )
            learner_state = LearnerState(params, opt_states, key, env_state, timestep, hstates)
            return (learner_state, latent), transition

        # COPY OLD HIDDEN STATES: TO BE USED IN THE TRAINING LOOP
        prev_hstates = tree.map(lambda x: jnp.copy(x), learner_state.hstates)

        key, latent_key = jax.random.split(learner_state.key)
        learner_state = learner_state._replace(key=key)

        if config.arch.latent_sampling_same:
            latent = get_compass_latent(latent_key, config, (1,))
            latent = jnp.repeat(latent, config.arch.num_envs, axis=0)

        else:
            latent = get_compass_latent(latent_key, config, (config.arch.num_envs,))

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        (learner_state, _), traj_batch = jax.lax.scan(
            _env_step,
            (learner_state, latent),
            jnp.arange(config.system.rollout_length),
            config.system.rollout_length,
        )

        # CALCULATE ADVANTAGE
        params, opt_states, key, _, _, updated_hstates = learner_state

        # Reset the environment for the next rollout.
        key, env_key = jax.random.split(key)
        env_keys = jax.random.split(env_key, config.arch.num_envs)
        env_state, last_timestep = jax.vmap(env.reset)(env_keys)
        (env_state, last_timestep) = tree.map(
            lambda x: duplicate_over_latent_dim(x, 0, config.arch.num_latents_per_env),
            (env_state, last_timestep),
        )

        # Reset the hidden states for the next rollout.
        updated_hstates = tree.map(lambda x: jnp.zeros_like(x), updated_hstates)

        # Before getting the best latent, we also compute the mean episode return for each latent.
        # We will log this and use it as the reinforce baseline.
        mean_return_over_latents = get_mean_return_over_latents(
            traj_batch.reward, ~traj_batch.done_mask
        )

        full_rewards = traj_batch.reward
        full_done_mask = traj_batch.done_mask

        # Get the index for the best latent for each environment and select the related data.
        # We need to say ~traj_batch.done_mask since we want to train on data before
        # the episode is done ie. when done_mask is False in the transition.
        best_latent_idxs = get_best_latent_idx(traj_batch.reward, ~traj_batch.done_mask)
        traj_batch = select_best_latent_data(traj_batch, best_latent_idxs, config.arch.num_envs)
        prev_hstates = tree.map(
            lambda x: x[jnp.arange(config.arch.num_envs), best_latent_idxs], prev_hstates
        )
        best_latent = latent[jnp.arange(config.arch.num_envs), best_latent_idxs]

        def _calculate_reward_to_go(
            traj_batch: Transition,
        ) -> Tuple[chex.Array, chex.Array]:
            """Calculate the reward-to-go and advantages."""

            def _get_reward_to_go(
                accumulated_reward: chex.Array, transition: Transition
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

        # We won't really use these advantages.
        reward_to_go = _calculate_reward_to_go(traj_batch)

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _update_minibatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
                """Update the network for a single minibatch."""
                # UNPACK TRAIN STATE AND BATCH INFO
                params, opt_state = train_state
                (
                    traj_batch,
                    reward_to_go,
                    mean_return_over_latents,
                    train_done_mask,
                    prev_hstates,
                    best_latent,
                ) = batch_info

                def _loss_fn(
                    params: Params,
                    traj_batch: Transition,
                    reward_to_go: chex.Array,
                    mean_return_over_latents: chex.Array,
                    train_done_mask: chex.Array,
                    prev_hstates: HiddenStates,
                    best_latent: chex.Array,
                ) -> Tuple:
                    """Calculate Sable loss."""

                    # RERUN NETWORK
                    _, log_prob, entropy = sable_apply_fn(  # type: ignore
                        params,
                        traj_batch.obs,
                        traj_batch.action,
                        prev_hstates,
                        train_done_mask,
                        None,
                        best_latent,
                    )

                    # Negative because we are doing gradient ascent.
                    loss_actor = (reward_to_go) * -log_prob
                    # NOTE: Pretty sure it should be the inverse of the done mask since it is
                    # true when done but we want to train on data before that.
                    loss_actor = loss_actor.mean(where=~traj_batch.done_mask)

                    # Just a dummy, not used.
                    entropy = entropy.mean(where=~traj_batch.done_mask)

                    # TOTAL LOSS
                    total_loss = loss_actor
                    return total_loss, (loss_actor, entropy)

                # CALCULATE ACTOR LOSS
                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                loss_info, grads = grad_fn(
                    params,
                    traj_batch,
                    reward_to_go,
                    mean_return_over_latents,
                    train_done_mask,
                    prev_hstates,
                    best_latent,
                )

                # Compute the parallel mean (pmean) over the batch.
                # This calculation is inspired by the Anakin architecture demo notebook.
                # available at https://tinyurl.com/26tdzs5x
                # This pmean could be a regular mean as the batch axis is on the same device.
                grads, loss_info = jax.lax.pmean((grads, loss_info), axis_name="batch")
                # pmean over devices.
                grads, loss_info = jax.lax.pmean((grads, loss_info), axis_name="device")

                # UPDATE PARAMS AND OPTIMISER STATE
                updates, new_opt_state = update_fn(grads, opt_state)
                new_params = optax.apply_updates(params, updates)

                # PACK LOSS INFO
                total_loss = loss_info[0]
                actor_loss = loss_info[1][0]
                entropy = loss_info[1][1]
                loss_info = {
                    "total_loss": total_loss,
                    "actor_loss": actor_loss,
                    "entropy": entropy,
                }

                return (new_params, new_opt_state), loss_info

            (
                params,
                opt_states,
                traj_batch,
                reward_to_go,
                mean_return_over_latents,
                key,
                prev_hstates,
                best_latent,
            ) = update_state

            # SHUFFLE MINIBATCHES
            key, batch_shuffle_key, agent_shuffle_key = jax.random.split(key, 3)

            # We need a mask for resetting the sable hidden states inside the chunkwise
            # training that is False up until when all agents are done and then
            # True afterwards.
            all_true_mask = jnp.all(traj_batch.done_mask, axis=-1, keepdims=True)
            train_done_mask = traj_batch.done_mask & all_true_mask

            # Shuffle batch
            batch_size = config.arch.num_envs
            batch_perm = jax.random.permutation(batch_shuffle_key, batch_size)
            batch = (traj_batch, reward_to_go, mean_return_over_latents, train_done_mask)
            batch = tree.map(lambda x: jnp.take(x, batch_perm, axis=1), batch)
            best_latent = jnp.take(best_latent, batch_perm, axis=0)

            # Shuffle hidden states
            prev_hstates = tree.map(lambda x: jnp.take(x, batch_perm, axis=0), prev_hstates)

            # Shuffle agents
            agent_perm = jax.random.permutation(agent_shuffle_key, config.system.num_agents)
            batch = tree.map(lambda x: jnp.take(x, agent_perm, axis=2), batch)
            best_latent = jnp.take(best_latent, agent_perm, axis=1)

            # CONCATENATE TIME AND AGENTS
            batch = tree.map(concat_time_and_agents, batch)

            # NOTE (Ruan): This can definitely be optimised. We do it because,
            # best latent has shape (batch, agents, latent_dim). We need to duplicate
            # rollout_length times along the agent axis to match the shape of the obs which
            # has shape (batch, agents * rollout_length, obs_dim).
            best_latent_minibatch = jnp.tile(best_latent, (1, config.system.rollout_length, 1))

            # SPLIT INTO MINIBATCHES
            minibatches = tree.map(
                lambda x: jnp.reshape(x, (config.system.num_minibatches, -1, *x.shape[1:])),
                batch,
            )
            prev_hs_minibatch = tree.map(
                lambda x: jnp.reshape(x, (config.system.num_minibatches, -1, *x.shape[1:])),
                prev_hstates,
            )
            best_latent_minibatch = jnp.reshape(
                best_latent_minibatch,
                (config.system.num_minibatches, -1, *best_latent_minibatch.shape[1:]),
            )

            # UPDATE MINIBATCHES
            (params, opt_states), loss_info = jax.lax.scan(
                _update_minibatch,
                (params, opt_states),
                (*minibatches, prev_hs_minibatch, best_latent_minibatch),
            )

            update_state = (
                params,
                opt_states,
                traj_batch,
                reward_to_go,
                mean_return_over_latents,
                key,
                prev_hstates,
                best_latent,
            )
            return update_state, loss_info

        update_state = (
            params,
            opt_states,
            traj_batch,
            reward_to_go,
            mean_return_over_latents,
            key,
            prev_hstates,
            best_latent,
        )

        # UPDATE EPOCHS
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.ppo_epochs
        )

        params, opt_states, traj_batch, reward_to_go, mean_return_over_latents, key, _, _ = (
            update_state
        )
        learner_state = LearnerState(
            params,
            opt_states,
            key,
            env_state,
            last_timestep,
            updated_hstates,
        )
        metric = traj_batch.info
        metric["full_rewards"] = full_rewards
        metric["full_done_mask"] = full_done_mask
        return learner_state, (metric, loss_info)

    def learner_fn(learner_state: LearnerState) -> ExperimentOutput[LearnerState]:
        """Learner function.

        This function represents the learner, it updates the network parameters
        by iteratively applying the `_update_step` function for a fixed number of
        updates. The `_update_step` function is vectorized over a batch of inputs.

        Args:
        ----
            learner_state (NamedTuple):
                - params (FrozenDict): The initial model parameters.
                - opt_state (OptState): The initial optimizer state.
                - key (chex.PRNGKey): The random number generator state.
                - env_state (LogEnvState): The environment state.
                - timesteps (TimeStep): The initial timestep in the initial trajectory.
                - hstates (HiddenStates): The initial hidden states of the network.

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
) -> Tuple[LearnerFn[LearnerState], Callable, LearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())

    # Get number of agents.
    config.system.num_agents = env.num_agents

    # PRNG keys.
    key, net_key = keys

    # Get number of agents and actions.
    action_dim = int(env.action_spec().num_values[0])
    n_agents = env.action_spec().shape[0]
    config.system.num_agents = n_agents
    config.system.num_actions = action_dim

    # Setting the chunksize - smaller chunks save memory at the cost of speed
    if config.network.memory_config.timestep_chunk_size:
        config.network.memory_config.chunk_size = (
            config.network.memory_config.timestep_chunk_size * n_agents
        )
    else:
        config.network.memory_config.chunk_size = config.system.rollout_length * n_agents

    _, action_space_type = get_action_head(env.action_spec())

    # Define network.
    sable_network = SableNetwork(
        n_agents=n_agents,
        n_agents_per_chunk=n_agents,
        action_dim=action_dim,
        net_config=config.network.net_config,
        memory_config=config.network.memory_config,
        action_space_type=action_space_type,
    )

    # Define optimiser.
    lr = make_learning_rate(config.system.actor_lr, config)
    optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(lr, eps=1e-5),
    )
    config = adjust_config_for_gradient_accumulation(config)
    optim = optax.MultiSteps(optim, every_k_schedule=config.arch.grad_accumulation_steps)

    # Get mock inputs to initialise network.
    init_obs = env.observation_spec().generate_value()
    init_latent = jnp.ones((1, config.system.num_agents, config.arch.compass_latent_dim))
    init_obs = tree.map(lambda x: x[jnp.newaxis, ...], init_obs)  # Add batch dim
    init_hs = get_init_hidden_state(config.network.net_config, config.arch.num_envs)
    init_hs = tree.map(lambda x: x[0, jnp.newaxis], init_hs)

    # Initialise params and optimiser state.
    params = sable_network.init(
        net_key,
        init_obs,
        init_hs,
        net_key,
        init_latent,
        method="get_actions",
    )

    # Pack apply and update functions.
    apply_fns = (
        partial(sable_network.apply, method="get_actions"),  # Execution function
        sable_network.apply,  # Training function
    )

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(env, apply_fns, optim.update, config)
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

    # Initialise hidden state.
    init_hstates = get_init_hidden_state(config.network.net_config, config.arch.num_envs)

    # Create an hstate for each latent.
    init_hstates = tree.map(
        lambda x: duplicate_over_latent_dim(x, 0, config.arch.num_latents_per_env), init_hstates
    )

    # Download checkpoint from Neptune if specified.
    if config.logger.checkpointing.unzip_local_model:
        unzip_local_checkpoints(
            checkpoint_rel_dir="checkpoints",
            model_name=config.logger.system_name,
            run_id=config.logger.checkpointing.download_args.neptune_run_name,
        )

    # Load model from checkpoint if specified.
    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.logger.system_name,
            **config.logger.checkpointing.load_args,  # Other checkpoint args
        )
        # Restore the learner state from the checkpoint
        # We don't restore hidden states since we are restarting training.
        restored_params, _ = loaded_checkpoint.restore_params(
            input_params=params, restore_hstates=True, THiddenState=HiddenStates
        )
        # Update the params and hidden states
        params = restored_params

        key, kernel_key = jax.random.split(key, num=2)

        path_to_kernel = ["params", "encoder", "obs_encoder", "layers_1", "kernel"]
        params = pad_weights(
            kernel_key,
            params,
            path_to_kernel,
            config.arch.compass_latent_dim,
            random_weights=config.arch.padding_with_random_weights,
            noise=config.arch.weights_noise,
        )

    # We actually need to set the opt state after we have augemented the params.
    opt_state = optim.init(params)

    # Define params to be replicated across devices and batches.
    key, step_keys = jax.random.split(key)
    replicate_learner = (params, opt_state, step_keys)

    # Duplicate learner for update_batch_size.
    broadcast = lambda x: jnp.broadcast_to(x, (config.system.update_batch_size, *x.shape))
    replicate_learner = tree.map(broadcast, replicate_learner)
    init_hstates = tree.map(broadcast, init_hstates)

    # Duplicate learner across devices.
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=jax.devices())
    init_hstates = flax.jax_utils.replicate(init_hstates, devices=jax.devices())

    # Initialise learner state.
    params, opt_state, step_keys = replicate_learner

    init_learner_state = LearnerState(
        params=params,
        opt_states=opt_state,
        key=step_keys,
        env_state=env_states,
        timestep=timesteps,
        hstates=init_hstates,
    )

    return learn, apply_fns[0], init_learner_state


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    _config.logger.system_name = "rec_sable_compass_post_ln"
    config = copy.deepcopy(_config)

    n_devices = len(jax.devices())

    # Create the enviroments for train and eval.
    env, eval_env = environments.make(config, fixed_reset=True)

    # PRNG keys.
    key, key_e, net_key = jax.random.split(jax.random.PRNGKey(config.system.seed), num=3)

    # Setup learner.
    learn, sable_execution_fn, learner_state = learner_setup(env, (key, net_key), config)

    # Setup evaluator.
    def make_rec_sable_act_fn(
        actor_apply_fn: ActorApply, vmap_apply_over_latents: bool = True
    ) -> COMPASSEvalActFn:
        _hidden_state = "hidden_state"

        def eval_act_fn(
            params: Params,
            timestep: TimeStep,
            key: chex.PRNGKey,
            actor_state: ActorState,
            latent: chex.Array,
        ) -> Tuple[Action, Dict]:
            hidden_state = actor_state[_hidden_state]

            if vmap_apply_over_latents:
                _actor_apply_fn = jax.vmap(
                    actor_apply_fn,
                    in_axes=(
                        None,
                        Observation(1, 1, 1),
                        HiddenStates(1, 1, 1),
                        None,
                        1,
                    ),
                    out_axes=(1, 1, 1, HiddenStates(1, 1, 1)),
                )
            else:
                _actor_apply_fn = actor_apply_fn

            output_action, _, _, hidden_state = _actor_apply_fn(  # type: ignore
                params,
                timestep.observation,
                hidden_state,
                key,
                latent,
            )
            return output_action, {_hidden_state: hidden_state}

        return eval_act_fn

    # One key per device for evaluation.
    eval_keys = jax.random.split(key_e, n_devices)
    eval_act_fn = make_rec_sable_act_fn(sable_execution_fn)
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
    eval_hs = get_init_hidden_state(config.network.net_config, eval_batch_size)
    if config.arch.eval_diff_latent_num:
        eval_hs = tree.map(
            lambda x: duplicate_over_latent_dim(x, 0, config.arch.eval_num_latents_per_env), eval_hs
        )
    else:
        eval_hs = tree.map(
            lambda x: duplicate_over_latent_dim(x, 0, config.arch.num_latents_per_env), eval_hs
        )
    eval_hs = flax.jax_utils.replicate(eval_hs, devices=jax.devices())

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

    if config.arch.render:
        render_apply_fn = make_rec_sable_act_fn(sable_execution_fn, vmap_apply_over_latents=False)
        render_hs = get_init_hidden_state(config.network.net_config, 1)
        render(
            eval_env,
            tree.map(lambda x: x[0], trained_params),
            {"hidden_state": render_hs},
            key,
            render_apply_fn,
            logger,
            compass_system=True,
            config=config,
        )

    # Measure absolute metric.
    if config.arch.absolute_metric:
        eval_batch_size = get_num_eval_envs(config, absolute_metric=True)
        abs_hs = get_init_hidden_state(config.network.net_config, eval_batch_size)
        if config.arch.eval_diff_latent_num:
            abs_hs = tree.map(
                lambda x: duplicate_over_latent_dim(x, 0, config.arch.eval_num_latents_per_env),
                abs_hs,
            )
        else:
            abs_hs = tree.map(
                lambda x: duplicate_over_latent_dim(x, 0, config.arch.num_latents_per_env), abs_hs
            )
        abs_hs = tree.map(lambda x: x[jnp.newaxis], abs_hs)
        abs_metric_evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=True)
        eval_keys = jax.random.split(key, n_devices)

        eval_metrics = abs_metric_evaluator(best_params, eval_keys, {"hidden_state": abs_hs})

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
    config_name="rec_sable.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    eval_performance = run_experiment(cfg)
    print(f"{Fore.CYAN}{Style.BRIGHT}Rec Sable COMPASS experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
