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
import functools
import time
from typing import Any, Callable, Dict, Tuple

import chex
import flax
import hydra
import jax
import jax.numpy as jnp
import numpy as np
from chex import PRNGKey
from colorama import Fore, Style
from flax.core.frozen_dict import FrozenDict
from jumanji.types import TimeStep
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from mava.evaluator import ActorState, _EvalEnvStepState
from mava.networks import SableNetwork
from mava.networks.utils.sable import get_init_hidden_state
from mava.systems.sable.types import HiddenStates
from mava.types import Action, MarlEnv, Metrics
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.cmaes_utils import CMAPoolEmitter
from mava.utils.logger import LogEvent, MavaLogger
from mava.utils.model_downloads import delete_local_checkpoints, unzip_local_checkpoints
from mava.utils.network_utils import get_action_head


class InferenceTimeLogger(MavaLogger):
    def log(self, metrics: Dict, t: int, t_eval: int, event: LogEvent) -> None:
        metrics = jax.tree.map(np.mean, metrics)
        self.logger.log_dict(metrics, t, t_eval, event)
        return

    def calc_winrate(self, _episode_metrics: Dict, _event: LogEvent) -> Dict:
        raise NotImplementedError


def make_sable_execution_fn(
    env: MarlEnv, key: chex.Array, config: DictConfig
) -> Tuple[Callable, FrozenDict]:
    """Initialise learner_fn, network, optimiser, environment and states."""

    # Get number of agents.
    config.system.num_agents = env.num_agents

    # Get number of agents and actions.
    action_dim = env.action_dim
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

    # define network
    sable_network = SableNetwork(
        n_agents=n_agents,
        n_agents_per_chunk=n_agents,
        action_dim=action_dim,
        net_config=config.network.net_config,
        memory_config=config.network.memory_config,
        action_space_type=action_space_type,
    )

    # Get mock inputs to initialise network.
    init_obs = env.observation_spec().generate_value()
    init_obs = jax.tree.map(lambda x: x[jnp.newaxis, ...], init_obs)  # Add batch dim

    # get mock hidden state
    init_hs = get_init_hidden_state(config.network.net_config, config.inference.n_envs)
    init_hs = jax.tree.map(lambda x: x[0, jnp.newaxis], init_hs)

    # get mock latent vectors, needs to be batched
    init_latent = jnp.zeros((1, config.system.num_agents, config.arch.compass_latent_dim))

    # Initialise params and optimiser state.
    params = sable_network.init(
        key,
        init_obs,
        init_hs,
        key,
        init_latent,
        method="get_actions",
    )

    # Execution function
    exec_apply_fn = functools.partial(sable_network.apply, method="get_actions")

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
        restored_params, _restored_hstates = loaded_checkpoint.restore_params(
            input_params=params, restore_hstates=False, THiddenState=HiddenStates
        )
        # Update the params and hidden states
        params = restored_params

    return exec_apply_fn, params


def get_eval_fn(
    env: MarlEnv,
    act_fn: Any,
    n_envs: int = 1,
    log_win_rate: bool = False,
) -> Callable:
    """Creates a function that can be used to evaluate agents on a given environment.

    Args:
    ----
        env: an environment that conforms to the mava environment spec.
        act_fn: a function that takes in params, timestep, key and optionally a state
                and returns actions and optionally a state (see `EvalActFn`).
        n_envs: the number of environments to run in parallel.
        log_win_rate: whether to log the win rate of the agents.
    """

    def eval_fn(
        params: FrozenDict,
        env_key: PRNGKey,
        acting_key: PRNGKey,
        init_act_state: ActorState,
        latents: chex.Array,
    ) -> Metrics:
        """Evaluates the given params on an environment and returns relevant
        metrics.

        Returns: Dict[str, Array] - dictionary of metric name to metric values
        for each episode.
        """

        def _env_step(eval_state: _EvalEnvStepState, _: Any) -> Tuple[_EvalEnvStepState, TimeStep]:
            """Performs a single environment step"""
            env_state, ts, key, actor_state = eval_state

            print("ts.observation.shape: ", ts.observation)
            print("latents.shape: ", latents.shape)

            key, act_key = jax.random.split(key)
            action, actor_state = act_fn(params, ts, act_key, actor_state, latents)

            env_state, ts = jax.vmap(env.step)(env_state, action)

            return (env_state, ts, key, actor_state), ts

        def _episode(env_key: PRNGKey) -> Metrics:
            """Simulates `n_envs` episodes."""
            # split key
            reset_keys = jax.random.split(env_key, n_envs)

            # get initial env states
            env_state, ts = jax.vmap(env.reset)(reset_keys)

            # Note: the key of the step state must be handled outside
            step_state = env_state, ts, acting_key, init_act_state
            _, timesteps = jax.lax.scan(_env_step, step_state, jnp.arange(env.time_limit + 1))

            metrics = timesteps.extras["episode_metrics"]
            if log_win_rate:
                metrics["won_episode"] = timesteps.extras["won_episode"]

            # find the first instance of done to get the metrics at that timestep
            # we don't care about subsequent steps
            done_idx = jnp.argmax(timesteps.last(), axis=0)
            metrics = jax.tree.map(
                lambda m: m[done_idx, jnp.arange(n_envs)],
                metrics,
            )
            del metrics["is_terminal_step"]  # not needed for logging

            return metrics

        metrics = _episode(env_key)

        return metrics

    return eval_fn


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    exp_start_time = time.time()

    _config.logger.system_name = "rec_sable_compass"
    config = copy.deepcopy(_config)

    print("Devices : ", jax.devices())

    n_devices = len(jax.devices())

    print("Num devices: ", n_devices)

    # create the enviroments for eval
    _env, eval_env = environments.make(config)

    # setup the evaluator
    def make_rec_sable_act_fn(actor_apply_fn: Any) -> Any:
        _hidden_state = "hidden_state"

        def eval_act_fn(
            params: FrozenDict,
            timestep: TimeStep,
            key: chex.PRNGKey,
            actor_state: ActorState,
            latent: chex.Array,
        ) -> Tuple[Action, Dict]:
            hidden_state = actor_state[_hidden_state]

            output_action, _, _, hidden_state = actor_apply_fn(  # type: ignore
                params,
                timestep.observation,
                hidden_state,
                key,
                latent,
            )
            return output_action, {_hidden_state: hidden_state}

        return eval_act_fn

    # PRNG keys.
    key = jax.random.PRNGKey(config.system.seed)
    key, net_key, emitter_key, eval_key = jax.random.split(key, num=4)

    # create the sable execution function
    sable_execution_fn, params = make_sable_execution_fn(eval_env, net_key, config)

    # duplicate params across devices
    params = flax.jax_utils.replicate(params, devices=jax.devices())

    # create the "action decision" function used at inference time
    eval_act_fn = make_rec_sable_act_fn(sable_execution_fn)

    # get the main "inference time" parameters here
    n_devices = jax.device_count()
    n_envs = config.inference.n_envs  # typically 32

    budget = config.inference.budget  # typically 6400
    n_attempts = config.inference.n_attempts

    # prepare the budget
    sequential_budget = budget // n_attempts

    if n_envs % n_devices != 0:
        raise ValueError(f"Num eval episodes must be divisible by num devices ({n_devices})")

    n_envs_per_device = n_envs // n_devices

    # define the evaluator
    evaluator = get_eval_fn(
        eval_env,
        eval_act_fn,
        n_envs_per_device,
        config.env.log_win_rate,
    )

    # inputs are params, env_key, acting_key, init_act_state, latents
    # with dims going as deep as (n_devices, n_attempts, ...)
    # deepest being acting_key
    # evaluator is already vmapped over parallel envs
    # hence we only vmap over the attempts
    # latent vectors: (n_devices, n_attempts, n_envs, latent_dim)
    evaluator = jax.vmap(evaluator, in_axes=(None, None, 0, None, 0))
    evaluator = jax.pmap(evaluator)

    # Logger setup
    logger = InferenceTimeLogger(config)
    cfg: Dict = OmegaConf.to_container(config, resolve=True)
    cfg["arch"]["devices"] = jax.devices()
    pprint(cfg)

    # create an initial hidden state used for resetting memory for evaluation
    eval_hs = get_init_hidden_state(config.network.net_config, n_envs_per_device)
    eval_hs = flax.jax_utils.replicate(eval_hs, devices=jax.devices())

    # initialise accumulated metrics
    max_episode_return = -jnp.inf * jnp.ones(n_envs)
    max_episode_win = jnp.zeros(n_envs)

    # create the env keys (fixed envs for the entire evaluation)
    env_key = jax.random.PRNGKey(config.inference.env_seed)
    env_keys = jax.random.split(env_key, n_devices).reshape(n_devices, -1)

    emitter = None
    if config.inference.search_strategy.name == "cmaes":
        e_kwargs = config.inference.search_strategy.kwargs
        e_kwargs.num_best = n_attempts // 2
        emitter = CMAPoolEmitter(
            num_states=e_kwargs.num_states,
            population_size=n_attempts,
            num_best=e_kwargs.num_best,
            search_dim=config.arch.compass_latent_dim,
            init_sigma=e_kwargs.init_sigma,
            delay_eigen_decomposition=e_kwargs.delay_eigen_decomposition,
            init_minval=-config.arch.latent_amplifier * jnp.ones((config.arch.compass_latent_dim,)),
            init_maxval=config.arch.latent_amplifier * jnp.ones((config.arch.compass_latent_dim,)),
            random_key=emitter_key,
        )

        cmaes_state = jax.tree.map(
            lambda x: jnp.repeat(jnp.expand_dims(x, axis=0), repeats=n_envs, axis=0),
            emitter.init(),
        )

    # main loop
    for eval_step in range(sequential_budget):
        if eval_step <= 2:
            exp_start_time = time.time()

        total_exp_time = time.time() - exp_start_time

        break_condition = config.inference.time_constraint is not None and (
            total_exp_time > config.inference.time_constraint
        )
        if break_condition:
            print(
                f"""Breaking due to time constraint:
                {total_exp_time} > {config.inference.time_constraint}"""
            )
            break

        # prepare acting_keys: (n_devices, n_attempts, ...)
        eval_key, acting_key, latent_key = jax.random.split(eval_key, num=3)
        acting_keys = jax.random.split(acting_key, n_devices * n_attempts).reshape(
            n_devices, n_attempts, -1
        )

        if config.inference.search_strategy.name == "cmaes":
            subkeys = jax.random.split(latent_key, num=n_envs)
            latents, _ = jax.vmap(emitter.sample)(cmaes_state, subkeys)  # type: ignore

            # switch attempts & envs dimensions
            latents = latents.transpose((1, 0, 2))

        elif config.inference.search_strategy.name == "uniform_sampling":
            # prepare the latent vectors - for now, uniform sampling
            latents = jax.random.uniform(
                latent_key,
                (
                    n_attempts,
                    n_envs,
                    config.arch.compass_latent_dim,
                ),
                minval=config.inference.search_strategy.kwargs.minval,
                maxval=config.inference.search_strategy.kwargs.maxval,
            )
        else:
            raise ValueError("Unknown search strategy")

        # split over the devices
        latents = latents.reshape(
            n_devices, n_attempts, n_envs_per_device, config.arch.compass_latent_dim
        )

        # repeat the latents for agents (TODO: hide it in the network)
        latents = jnp.repeat(jnp.expand_dims(latents, axis=-2), config.system.num_agents, axis=-2)

        start_time = time.time()

        # evaluate & get metrics
        eval_metrics = evaluator(params, env_keys, acting_keys, {"hidden_state": eval_hs}, latents)

        eval_metrics = jax.block_until_ready(eval_metrics)

        end_time = time.time()

        # remove the device dimension for all metrics
        # i.e. merge dim_devices and dim_envs_per_device
        eval_metrics = jax.tree.map(lambda x: x.reshape(n_attempts, -1, *x.shape[3:]), eval_metrics)

        # update of the emitter
        if config.inference.search_strategy.name == "cmaes":
            # remove the agents dimension
            latents = latents[:, :, :, 0, :]

            # remove the device dimension again for the latents
            latents = latents.reshape(n_attempts, -1, *latents.shape[3:])

            # re-switch attempts and envs dimensions - (n_envs, n_attempts, #dim)
            latents = latents.transpose((1, 0, 2))

            # Calculate fitness for CMA-ES
            fitness = eval_metrics["episode_return"]

            # also need to reshape the fitness (n_envs, n_attempts)
            fitness = fitness.transpose((1, 0))

            # Sort latents based on fitness
            sorted_indices = jnp.argsort(-fitness, axis=1)
            sorted_latents = jnp.take_along_axis(latents, sorted_indices[:, :, jnp.newaxis], axis=1)

            # Update the CMA-ES emitter state
            cmaes_state = jax.vmap(
                emitter.update_state  # type: ignore
            )(  # type: ignore
                cmaes_state,
                sorted_candidates=sorted_latents[:, : e_kwargs.num_best, :],
            )

        # get max per episode over the last batch of attempts
        max_latest_batch = np.max(eval_metrics["episode_return"], axis=0)

        # keep track of the best ever per environment
        max_episode_return = np.maximum(max_episode_return, max_latest_batch)

        # metrics post-processing and logging
        total_timesteps = jnp.sum(eval_metrics["episode_length"])
        eval_metrics["steps_per_second"] = total_timesteps / (end_time - start_time)

        # now get min, mean, max, std over the attempts
        eval_metrics["return(mean)"] = np.mean(eval_metrics["episode_return"], axis=0)
        eval_metrics["return(min)"] = np.min(eval_metrics["episode_return"], axis=0)
        eval_metrics["return(max)"] = np.max(eval_metrics["episode_return"], axis=0)
        eval_metrics["return(std)"] = np.std(eval_metrics["episode_return"], axis=0)

        eval_metrics["return(accumulated_max)"] = max_episode_return

        # was the episode won over the last batch
        if "won_episode" in eval_metrics:
            latest_batch_win = np.max(eval_metrics["won_episode"], axis=0)
            max_episode_win = np.maximum(max_episode_win, latest_batch_win)
            eval_metrics["win_rate(accumulated)"] = (jnp.sum(max_episode_win) / n_envs) * 100

            # get the win rate
            n_won_episodes: int = np.sum(eval_metrics["won_episode"])
            n_episodes: int = np.size(eval_metrics["won_episode"])
            win_rate = (n_won_episodes / n_episodes) * 100

            eval_metrics["win_rate"] = win_rate
            eval_metrics.pop("won_episode")

        # log
        logger.log(eval_metrics, (eval_step + 1) * n_attempts, eval_step, LogEvent.EVAL)

    # record the performance for the final evaluation run
    eval_performance = float(np.mean(max_episode_return))

    # stop logging
    logger.stop()

    # Remove the local model checkpoint when training is complete
    if config.logger.checkpointing.delete_local_checkpoints:
        delete_local_checkpoints(checkpoint_folder_dir="checkpoints")

    return eval_performance


@hydra.main(
    config_path="../../../configs/default",
    # config_name="rec_sable_compass_eval_local.yaml",
    config_name="rec_sable_compass_eval_cmaes.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    eval_performance = run_experiment(cfg)
    print(f"{Fore.CYAN}{Style.BRIGHT}Rec Sable experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
