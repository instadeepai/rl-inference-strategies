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
from typing import Any, Callable, Dict, List, Tuple, Union

import chex
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from colorama import Fore, Style
from flax.core.frozen_dict import FrozenDict as Params
from jax import tree
from jumanji.env import Environment
from jumanji.types import TimeStep
from omegaconf import DictConfig, OmegaConf
from optax._src.base import OptState
from rich.pretty import pprint
from typing_extensions import NamedTuple, TypeAlias

from mava.networks import SableNetwork
from mava.networks.utils.sable import get_init_hidden_state
from mava.systems.sable.types import (
    ActorApply,
    HiddenStates,
    LearnerApply,
    Transition,
)
from mava.types import MarlEnv
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.jax_utils import concat_time_and_agents
from mava.utils.logger import LogEvent, MavaLogger
from mava.utils.model_downloads import delete_local_checkpoints, unzip_local_checkpoints
from mava.utils.network_utils import get_action_head
from mava.utils.training import make_learning_rate


class InferenceState(NamedTuple):
    """State of the learner for Memory Sable"""

    params: Params
    opt_states: OptState
    key: chex.PRNGKey
    env_state: chex.Array
    timestep: TimeStep
    hstates: HiddenStates
    fixed_reset_key: chex.PRNGKey


Metrics: TypeAlias = Dict[str, chex.Array]
LearnerFn = Callable[[InferenceState], Tuple[InferenceState, Metrics]]


class InferenceTimeLogger(MavaLogger):
    def log(self, metrics: Dict, t: int, t_eval: int, event: LogEvent) -> None:
        metrics = jax.tree.map(np.mean, metrics)
        self.logger.log_dict(metrics, t, t_eval, event)
        return

    def calc_winrate(self, _episode_metrics: Dict, _event: LogEvent) -> Dict:
        raise NotImplementedError


def get_learner_fn(
    env: Environment,
    apply_fns: Tuple[ActorApply, LearnerApply],
    update_fn: optax.TransformUpdateFn,
    config: DictConfig,
) -> LearnerFn:
    """Get the learner function."""

    # Get apply functions for executing and training the network.
    sable_action_select_fn, sable_apply_fn = apply_fns

    def evaluate_fn(
        inference_state: InferenceState,
    ) -> Tuple[InferenceState, Transition, Any, chex.Array]:
        """Evaluation of the environment"""

        def _env_step(inference_state: InferenceState, _: int) -> Tuple[InferenceState, Transition]:
            """Step the environment."""
            params, opt_states, key, env_state, last_timestep, hstates, fixed_reset_key = (
                inference_state
            )

            # SELECT ACTION
            key, policy_key = jax.random.split(key)

            # Apply the actor network to get the action, log_prob, value and updated hstates.
            last_obs = last_timestep.observation
            action, log_prob, _, hstates = sable_action_select_fn(  # type: ignore
                params,
                last_obs,
                hstates,
                policy_key,
            )

            # STEP ENVIRONMENT
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

            # LOG EPISODE METRICS
            info = tree.map(
                lambda x: jnp.repeat(x[..., jnp.newaxis], config.system.num_agents, axis=-1),
                timestep.extras,
            )

            # Reset hidden state if done.
            done = timestep.last()
            done = jnp.expand_dims(done, (1, 2, 3, 4))
            hstates = tree.map(lambda hs: jnp.where(done, jnp.zeros_like(hs), hs), hstates)

            # SET TRANSITION
            done = timestep.last()[..., jnp.newaxis].repeat(config.system.num_agents, axis=-1)

            # NOTE: Discounts are 1 when not done and 0 when done, so we flip them here.
            transition = Transition(
                done,
                action,
                None,
                timestep.reward,
                log_prob,
                last_timestep.observation,
                info,
                (1.0 - last_timestep.discount).astype(bool),
            )
            inference_state = InferenceState(
                params, opt_states, key, env_state, timestep, hstates, fixed_reset_key
            )
            return inference_state, transition

        # COPY OLD HIDDEN STATES: TO BE USED IN THE TRAINING LOOP
        prev_hstates = tree.map(lambda x: x, inference_state.hstates)

        params, opt_states, key, env_state, timestep, hstates, fixed_reset_key = inference_state

        # Reset environment here using the fixed reset key
        env_keys = jnp.stack([fixed_reset_key] * config.inference.n_attempts)

        env_state, timestep = jax.vmap(env.reset, in_axes=(0))(
            env_keys,
        )

        # Replace the env_state and timestep
        inference_state = InferenceState(
            params, opt_states, key, env_state, timestep, hstates, fixed_reset_key
        )

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        inference_state, traj_batch = jax.lax.scan(
            _env_step,
            inference_state,
            jnp.arange(config.system.rollout_length),
            config.system.rollout_length,
        )

        # compute metrics here
        metrics = traj_batch.info["episode_metrics"]
        if config.env.log_win_rate:
            metrics["won_episode"] = traj_batch.info["won_episode"]

        # remove the agent dimension
        metrics = jax.tree.map(lambda x: x[..., 0], metrics)

        # find the first instance of done to get the metrics at that timestep, we don't
        # care about subsequent steps because we only the results from the first episode
        done_idx = jnp.argmax(traj_batch.done, axis=0)
        done_idx = jax.tree.map(lambda x: x[..., 0], done_idx)
        metrics = tree.map(lambda m: m[done_idx, jnp.arange(config.inference.n_attempts)], metrics)
        del metrics["is_terminal_step"]  # uneeded for logging

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

        reward_to_go = _calculate_reward_to_go(traj_batch)

        # Replace the env_state and timestep
        inference_state = InferenceState(
            params,
            opt_states,
            inference_state.key,
            env_state,
            timestep,
            prev_hstates,
            fixed_reset_key,
        )

        return inference_state, traj_batch, reward_to_go, metrics

    def update_params_fn(_: Any, update_state: Any) -> Tuple[Any, Tuple[Params, OptState]]:
        params, opt_states, traj_batch, key, reward_to_go, prev_hstates = update_state

        def _loss_fn(
            params: Params,
            traj_batch: Transition,
            reward_to_go: chex.Array,
            train_done_mask: chex.Array,
            prev_hstates: HiddenStates,
        ) -> Tuple:
            """Calculate Sable loss."""
            _, log_prob, entropy = sable_apply_fn(  # type: ignore
                params,
                traj_batch.obs,
                traj_batch.action,
                prev_hstates,
                train_done_mask,
            )

            # Negative because we are doing gradient ascent.
            loss_actor = reward_to_go * -log_prob
            # NOTE: Pretty sure it should be the inverse of the done mask since it is
            # true when done but we want to train on data before that.
            loss_actor = loss_actor.mean(where=~traj_batch.done_mask)

            # Just a dummy, not used.
            entropy = entropy.mean(where=~traj_batch.done_mask)

            # TOTAL LOSS
            total_loss = loss_actor - config.inference.entropy_coef * entropy
            return total_loss, (loss_actor, entropy)

        train_done_mask = jnp.all(traj_batch.done_mask, axis=-1, keepdims=True)
        train_done_mask = traj_batch.done_mask & train_done_mask

        # shuffle data
        key, batch_shuffle_key, agent_shuffle_key = jax.random.split(key, 3)

        batch_size = config.inference.n_attempts
        batch_perm = jax.random.permutation(batch_shuffle_key, batch_size)

        # select a subset of the data to update the network for effiscient training
        batch_size = min(config.inference.max_trajectory_size, batch_size)
        batch_perm = batch_perm[:batch_size]

        batch = (traj_batch, reward_to_go, train_done_mask)
        batch = tree.map(lambda x: jnp.take(x, batch_perm, axis=1), batch)

        # Shuffle hidden states
        prev_hstates = tree.map(lambda x: jnp.take(x, batch_perm, axis=0), prev_hstates)

        # Shuffle agents
        agent_perm = jax.random.permutation(agent_shuffle_key, config.system.num_agents)
        batch = tree.map(lambda x: jnp.take(x, agent_perm, axis=2), batch)

        # CONCATENATE TIME AND AGENTS
        batch = tree.map(concat_time_and_agents, batch)

        traj_batch, reward_to_go, train_done_mask = batch

        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
        _, grads = grad_fn(
            params,
            traj_batch,
            reward_to_go,
            train_done_mask,
            prev_hstates,
        )

        # Update parameters
        updates, new_opt_states = update_fn(grads, opt_states)
        new_params = optax.apply_updates(params, updates)

        return None, (new_params, new_opt_states)

    def learner_fn(inference_state: InferenceState) -> Tuple[InferenceState, Metrics]:
        inference_state, traj_batch, reward_to_go, metrics = jax.vmap(evaluate_fn, in_axes=0)(
            inference_state
        )

        # unpack the inference  state
        params, opt_states, key, env_state, timestep, hstates, fixed_reset_key = inference_state

        # build the update state
        update_state = params, opt_states, traj_batch, key, reward_to_go, hstates

        # reshape update state for batch processing
        batched_update_state = tree.map(
            lambda x: jnp.reshape(x, (config.inference.num_params_updates, -1, *x.shape[1:])),
            update_state,
        )
        # vectorize update function for batch update
        batched_update = jax.vmap(update_params_fn, in_axes=(None, 0))

        # perform batched parameter updates
        _, (new_params, new_opt_states) = jax.lax.scan(batched_update, None, batched_update_state)

        # reshape parameters and optimizer states back to expected dimensions
        new_params = tree.map(
            lambda x: jnp.reshape(x, (config.inference.n_envs_per_batch, *x.shape[2:])),
            new_params,
        )
        new_opt_states = tree.map(
            lambda x: jnp.reshape(x, (config.inference.n_envs_per_batch, *x.shape[2:])),
            new_opt_states,
        )

        # return the new inference state with the new params and new opt states
        inference_state = InferenceState(
            new_params, new_opt_states, key, env_state, timestep, hstates, fixed_reset_key
        )

        return inference_state, metrics

    return learner_fn


def learner_setup(
    env: MarlEnv, keys: chex.Array, config: DictConfig
) -> Tuple[LearnerFn, InferenceState]:
    """Initialise learner_fn, network, optimiser, environment and states."""

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

    # Get mock inputs to initialise network.
    init_obs = env.observation_spec().generate_value()
    init_obs = tree.map(lambda x: x[jnp.newaxis, ...], init_obs)  # Add batch dim
    init_hs = get_init_hidden_state(config.network.net_config, config.inference.n_attempts)
    init_hs = tree.map(lambda x: x[0, jnp.newaxis], init_hs)

    # Initialise params and optimiser state.
    params = sable_network.init(
        net_key,
        init_obs,
        init_hs,
        net_key,
        method="get_actions",
    )
    opt_state = optim.init(params)

    # Pack apply and update functions.
    apply_fns = (
        partial(sable_network.apply, method="get_actions"),  # Execution function
        sable_network.apply,  # Training function
    )

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(env, apply_fns, optim.update, config)
    learn = jax.pmap(learn)

    # Initialise hidden state.
    init_hstates = get_init_hidden_state(config.network.net_config, config.inference.n_attempts)

    # Unzip local model checkpoint
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
        restored_params, _ = loaded_checkpoint.restore_params(
            input_params=params, restore_hstates=True, THiddenState=HiddenStates
        )
        # Update the params and hidden states
        params = restored_params

    # get step keys
    key, step_keys = jax.random.split(key)

    init_inference_state = InferenceState(
        params=params,
        opt_states=opt_state,
        key=step_keys,
        env_state=None,
        timestep=None,
        hstates=init_hstates,
        fixed_reset_key=None,
    )

    return learn, init_inference_state


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""

    exp_start_time = time.time()

    _config.logger.system_name = "rec_sable_reinforce"
    config = copy.deepcopy(_config)

    # Create the enviroments for train and eval.
    env, _ = environments.make(config, fixed_reset=True)

    n_devices = len(jax.devices())
    budget = config.inference.budget
    n_attempts = config.inference.n_attempts
    n_envs = config.inference.n_envs
    time_constraint = config.inference.time_constraint
    n_envs_per_batch = config.inference.n_envs_per_batch

    # PRNG keys.
    key, net_key = jax.random.split(jax.random.PRNGKey(config.system.seed), num=2)

    # Setup learner.
    learn, inference_state = learner_setup(env, (key, net_key), config)

    # Copy the initial inference_state
    # This will allow us to start from the the same learner state everytime
    init_inference_state = copy.deepcopy(inference_state)

    # Logger setup
    logger = InferenceTimeLogger(config)
    cfg: Dict = OmegaConf.to_container(config, resolve=True)
    cfg["arch"]["devices"] = jax.devices()
    pprint(cfg)

    # prepare the budget
    sequential_budget = budget // n_attempts

    # get the env-keys
    env_key = jax.random.PRNGKey(config.inference.env_seed)
    fixed_reset_keys = jax.random.split(env_key, num=n_envs)

    # to save every metrics
    eval_metrics_all_batch: Dict[str, List[Metrics]] = {}

    # the number of batch
    number_of_batch = n_envs // n_envs_per_batch
    time_constraint_per_batch = time_constraint // number_of_batch
    number_of_step: Union[int, None] = None  # to keep the number of step for each batch

    for batch in range(number_of_batch):
        exp_start_time = time.time()
        # repeat the initial state for each problem for vmapping
        inference_state = jax.tree.map(
            lambda x: jnp.repeat(jnp.expand_dims(x, axis=0), n_envs_per_batch, axis=0),
            init_inference_state,
        )

        # get batch keys
        minibacth_fixed_keys = fixed_reset_keys[
            batch * n_envs_per_batch : (batch + 1) * n_envs_per_batch
        ]

        # replace the fixed keys
        inference_state = inference_state._replace(fixed_reset_key=minibacth_fixed_keys)

        # reshape for pmap
        inference_state = jax.tree.map(
            lambda x: jnp.reshape(x, (n_devices, n_envs_per_batch, *x.shape[1:])),
            inference_state,
        )

        # save metrics for batch
        eval_metrics_all_batch[str(batch)] = []

        for eval_step in range(sequential_budget):
            # remove the time for compilation with jit
            if eval_step <= 2 and number_of_step is None:
                exp_start_time = time.time()

            total_exp_time = time.time() - exp_start_time

            # Since we are doing minibatching, the time sould be divided
            # evenly, across all the minibatches and eval step should be the same for
            # each minibatch
            # So get  the number step in the first minibatch and then
            # make the break condition at this number of step for other minibatches
            break_condition = (
                (time_constraint is not None and (total_exp_time > time_constraint_per_batch))
                if number_of_step is None
                else (eval_step == number_of_step)
            )

            if break_condition:
                number_of_step = eval_step
                print(
                    f"Breaking due to time constraint per batch: \
                    {total_exp_time} > {time_constraint_per_batch}"
                )
                break

            # Update.
            update_output = learn(inference_state)
            jax.tree_util.tree_map(jax.block_until_ready, update_output)
            inference_state, eval_metrics = update_output

            eval_metrics_all_batch[str(batch)].append(eval_metrics)

        print(f"eval step: {eval_step}")

    # Aggregate across bacth
    aggregated_metrics = jax.tree.map(
        lambda *x: jnp.stack(x, axis=0),
        *[eval_metrics_all_batch[str(batch)] for batch in range(number_of_batch)],
    )

    # log only here at each evaluation (the mean across all envs)
    print(f"{Fore.CYAN} Start logging statitic across all envs {Style.RESET_ALL}")

    # initialise accumulated metrics
    max_episode_return = -jnp.inf * jnp.ones(n_envs)
    max_episode_win = jnp.zeros(n_envs)

    for eval_step in range(0, len(aggregated_metrics)):
        t = n_attempts * (eval_step + 1)
        # Take the mean metrics for the current eval_step
        eval_metrics = aggregated_metrics[eval_step]

        # reshape into (n_envs, n_attempts)
        eval_metrics = tree.map(
            lambda x: jnp.reshape(
                x,
                (
                    number_of_batch * n_envs_per_batch,
                    n_attempts,
                ),
            ),
            eval_metrics,
        )

        # get max per episode over the last batch of attempts
        max_latest_batch = np.max(eval_metrics["episode_return"], axis=-1)

        # keep track of the best ever per environment
        max_episode_return = np.maximum(max_episode_return, max_latest_batch)

        # now get min, mean, max, std over the attempts
        eval_metrics["return(mean)"] = np.mean(eval_metrics["episode_return"], axis=-1)
        eval_metrics["return(min)"] = np.min(eval_metrics["episode_return"], axis=-1)
        eval_metrics["return(max)"] = np.max(eval_metrics["episode_return"], axis=-1)
        eval_metrics["return(std)"] = np.std(eval_metrics["episode_return"], axis=-1)

        eval_metrics["return(accumulated_max)"] = max_episode_return

        # get the win rate
        if "won_episode" in eval_metrics:
            # was the episode won over the last batch
            latest_batch_win = np.max(eval_metrics["won_episode"], axis=-1)
            max_episode_win = np.maximum(max_episode_win, latest_batch_win)
            eval_metrics["win_rate(accumulated)"] = (jnp.sum(max_episode_win) / n_envs) * 100

            n_won_episodes: int = np.sum(eval_metrics["won_episode"])
            # print(eval_metrics['won_episode'])
            n_episodes: int = np.size(eval_metrics["won_episode"])  # handle
            win_rate = (n_won_episodes / n_episodes) * 100

            eval_metrics["win_rate"] = win_rate
            eval_metrics.pop("won_episode")
        # Log eval metrics here
        logger.log(eval_metrics, t, eval_step, LogEvent.EVAL)

    # Record the performance for the final evaluation run.
    eval_performance = float(jnp.mean(eval_metrics[config.env.eval_metric]))

    # Stop the logger.
    logger.stop()

    # Remove the local model checkpoint when training is complete
    if config.logger.checkpointing.delete_local_checkpoints:
        delete_local_checkpoints(checkpoint_folder_dir="checkpoints")

    return eval_performance


@hydra.main(
    config_path="../../../configs/default",
    config_name="rec_sable_eval_finetuning.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    eval_performance = run_experiment(cfg)
    print(f"{Fore.CYAN}{Style.BRIGHT}Rec Sable REINFORCE experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
