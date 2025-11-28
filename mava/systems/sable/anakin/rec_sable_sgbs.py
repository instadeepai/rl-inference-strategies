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
from typing import Any, Callable, Dict, Protocol, Tuple, Union

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
from mava.systems.sable.types import ActorApply, HiddenStates
from mava.types import Action, MarlEnv, Metrics, Observation, ObservationGlobalState
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
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


class EvalActFn(Protocol):
    """The API for the acting function that is passed to the `EvalFn`.

    A get_action function must conform to this API in order to be used with Mava's evaluator.
    See `make_ff_eval_act_fn` and `make_rec_eval_act_fn` as examples.
    """

    def __call__(
        self,
        params: FrozenDict,
        timestep: TimeStep[Union[Observation, ObservationGlobalState]],
        key: PRNGKey,
        actor_state: ActorState,
        select_greedy_action: bool,
    ) -> Tuple[chex.Array, ActorState]: ...


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

    # Define network.
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

    # initialise the hidden state
    init_hs = get_init_hidden_state(config.network.net_config, config.inference.n_envs)
    init_hs = jax.tree.map(lambda x: x[0, jnp.newaxis], init_hs)

    # Initialise params and optimiser state.
    params = sable_network.init(
        key,
        init_obs,
        init_hs,
        key,
        method="get_actions",
    )

    # Execution function
    exec_apply_fn = functools.partial(sable_network.apply, method="get_actions")

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
            input_params=params, restore_hstates=False, THiddenState=HiddenStates
        )
        # Update the params and hidden states
        params = restored_params

    return exec_apply_fn, params


def get_eval_fn(
    env: MarlEnv,
    act_fn: EvalActFn,
    n_envs: int = 1,
    beam_width: int = 1,
    top_k: int = 1,
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
    ) -> Metrics:
        """Evaluates the given params on an environment and returns relevant
        metrics.

        Returns: Dict[str, Array] - dictionary of metric name to metric values
        for each episode.
        """

        def _env_step(eval_state: _EvalEnvStepState, _: Any) -> Tuple[_EvalEnvStepState, TimeStep]:
            """Performs a single environment step"""

            def simulate_while(
                eval_state_cum_return: Tuple[_EvalEnvStepState, jnp.ndarray],
            ) -> Tuple[_EvalEnvStepState, TimeStep]:
                eval_state, cum_return = eval_state_cum_return
                env_state, ts, key, actor_state = eval_state
                action, actor_state = act_fn(
                    params, ts, act_key, actor_state, True
                )  # always greedy simulation
                env_state, ts = jax.vmap(env.step)(env_state, action)

                cum_return += jnp.mean(ts.reward, axis=-1)

                return (env_state, ts, key, actor_state), cum_return

            def while_cond_fn(
                eval_state_cum_return: Tuple[_EvalEnvStepState, jnp.ndarray],
            ) -> bool:
                eval_state, _ = eval_state_cum_return
                _, ts, _, _ = eval_state
                all_done = jnp.all(ts.last())
                return ~all_done

            # repeat the env_state, ts
            env_state, ts, key, actor_state = eval_state

            # NOTE: keep reusing the same key here?
            _, act_key = jax.random.split(key)

            # Sample beam width action
            act_keys = jax.random.split(act_key, num=beam_width)

            action, actor_state = jax.vmap(act_fn, in_axes=(None, None, 0, None, None))(
                params, ts, act_keys, actor_state, False
            )

            env_state, ts = jax.tree.map(
                lambda x: jnp.repeat(jnp.expand_dims(x, axis=0), beam_width, axis=0),
                (env_state, ts),
            )

            # flatten topK and beam width to falicitate the simulation and  selection
            reshape_states = lambda x: jnp.reshape(x, shape=((beam_width * top_k,) + x.shape[2:]))

            env_state, ts, actor_state, action = jax.tree.map(
                reshape_states, (env_state, ts, actor_state, action)
            )

            env_state, ts = jax.vmap(env.step)(env_state, action)

            eval_state = env_state, ts, key, actor_state

            _, cum_return = jax.lax.while_loop(
                while_cond_fn,
                simulate_while,
                (eval_state, jnp.zeros_like(jnp.mean(ts.reward, axis=-1))),
            )

            _, top_k_indices = jax.lax.top_k(cum_return, k=top_k)
            env_state, ts, actor_state = jax.tree.map(
                lambda m: m[top_k_indices], (env_state, ts, actor_state)
            )
            # simulate and return only the top k (env_state, ts)

            return (env_state, ts, key, actor_state), ts

        def _episode(reset_key: PRNGKey, init_actor_state: Any) -> Metrics:
            """Simulates `n_envs` episodes."""
            # Repeat the initial state for the top k sampling
            # NOTE: should this duplicate or split uniquely?
            top_k_reset_keys = jnp.tile(reset_key[None, :], (top_k, 1))

            # get initial env states
            env_state, ts = jax.vmap(env.reset)(top_k_reset_keys)

            # repeat the actor state for the top k sampling
            init_actor_state = jax.tree.map(
                lambda x: jnp.repeat(jnp.expand_dims(x, axis=0), top_k, axis=0),
                init_actor_state,
            )

            # Note: the key of the step state must be handled outside
            step_state = env_state, ts, acting_key, init_actor_state
            _, timesteps = jax.lax.scan(_env_step, step_state, jnp.arange(env.time_limit + 1))

            metrics = timesteps.extras["episode_metrics"]
            if log_win_rate:
                metrics["won_episode"] = timesteps.extras["won_episode"]

            # find the first instance of done to get the metrics at that timestep
            # we don't care about subsequent steps
            done_idx = jnp.argmax(timesteps.last(), axis=0)
            metrics = jax.tree.map(
                lambda m: m[done_idx, jnp.arange(top_k)],
                metrics,
            )
            del metrics["is_terminal_step"]  # not needed for logging

            return metrics

        # split key
        reset_keys = jax.random.split(env_key, n_envs)

        metrics = jax.vmap(_episode)(reset_key=reset_keys, init_actor_state=init_act_state)

        return metrics

    return eval_fn


def run_experiment(_config: DictConfig) -> float:
    """Sable stochastic sampling evaluation."""
    exp_start_time = time.time()

    _config.logger.system_name = "rec_sable_sgbs_greedy"
    config = copy.deepcopy(_config)

    print("Devices : ", jax.devices())

    n_devices = len(jax.devices())

    print("Num devices: ", n_devices)

    # Create the enviroments for eval.
    _, eval_env = environments.make(config)

    # initialise PRNG keys.
    key = jax.random.PRNGKey(config.system.seed)
    key, net_key, eval_key = jax.random.split(key, 3)

    # setup the evaluator
    def make_rec_sable_act_fn(actor_apply_fn: ActorApply) -> EvalActFn:
        _hidden_state = "hidden_state"

        def eval_act_fn(
            params: FrozenDict,
            timestep: TimeStep,
            key: chex.PRNGKey,
            actor_state: ActorState,
            select_greedy_action: bool = False,
        ) -> Tuple[Action, Dict]:
            hidden_state = actor_state[_hidden_state]
            output_action, _, _, hidden_state = actor_apply_fn(  # type: ignore
                params,
                timestep.observation,
                hidden_state,
                key,
                None,  # no latent here
                select_greedy_action,
            )
            return output_action, {_hidden_state: hidden_state}

        return eval_act_fn

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
    n_attempts = config.inference.beam_width * config.inference.top_k
    config.inference.n_attempts = n_attempts

    # prepare the budget
    sequential_budget = budget // n_attempts

    if n_envs % n_devices != 0:
        raise ValueError(f"Num eval episodes must be divisible by num devices ({n_devices})")

    if config.inference.beam_width is None:
        beam_width_value = int(n_attempts**0.5)
        config.inference.top_k = int(n_attempts / beam_width_value)
        config.inference.beam_width = beam_width_value

    n_envs_per_device = config.inference.n_envs // n_devices

    # define the evaluator
    evaluator = get_eval_fn(
        eval_env,
        eval_act_fn,
        n_envs_per_device,
        config.inference.beam_width,
        config.inference.top_k,
        config.env.log_win_rate,
    )

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
    # only keep the single best beam.
    max_episode_return = -jnp.inf
    max_episode_win = jnp.array(0)

    # create the env keys (fixed envs for the entire evaluation)
    env_key = jax.random.PRNGKey(config.inference.env_seed)
    env_keys = jax.random.split(env_key, n_devices).reshape(n_devices, -1)

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
        eval_key, acting_key = jax.random.split(eval_key)
        acting_keys = jax.random.split(acting_key, n_devices).reshape(n_devices, -1)

        start_time = time.time()

        # evaluate & get metrics
        eval_metrics = evaluator(params, env_keys, acting_keys, {"hidden_state": eval_hs})

        eval_metrics = jax.block_until_ready(eval_metrics)

        end_time = time.time()

        # get max per episode over the last batch of attempts
        # always max over beams (axis = -1), and then mean over devices and envs (axis = (0, 1))
        max_latest_batch = eval_metrics["episode_return"].max(axis=-1).mean(axis=(0, 1))

        # keep track of the best ever per environment
        max_episode_return = np.maximum(max_episode_return, max_latest_batch)

        # metrics post-processing and logging
        total_timesteps = jnp.sum(eval_metrics["episode_length"])
        eval_metrics["steps_per_second"] = total_timesteps / (end_time - start_time)

        # now get min, mean, max, std over the attempts
        eval_metrics["return(mean)"] = np.mean(eval_metrics["episode_return"], axis=-1).mean(
            axis=(0, 1)
        )
        eval_metrics["return(min)"] = np.min(eval_metrics["episode_return"], axis=-1).mean(
            axis=(0, 1)
        )
        eval_metrics["return(max)"] = np.max(eval_metrics["episode_return"], axis=-1).mean(
            axis=(0, 1)
        )
        eval_metrics["return(std)"] = np.std(eval_metrics["episode_return"], axis=-1).mean(
            axis=(0, 1)
        )

        eval_metrics["return(accumulated_max)"] = max_episode_return

        # was the episode won over the last batch
        if "won_episode" in eval_metrics:
            latest_batch_win = np.max(eval_metrics["won_episode"], axis=-1).mean(axis=(0, 1))
            max_episode_win = np.maximum(max_episode_win, latest_batch_win)
            eval_metrics["win_rate(accumulated)"] = max_episode_win * 100

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
    # config_name="rec_sable.yaml",
    # config_name="rec_sable_stoch_sampling_local.yaml",
    config_name="rec_sable_sgbs.yaml",
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
