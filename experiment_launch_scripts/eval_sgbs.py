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

# ruff: noqa: E501

import logging
import subprocess
import time

import pandas as pd

from inference_configurations.base_policy_run_ids import run_ids
from inference_configurations.sgbs_task_budgets import task_budget

scenarios_to_run = [
    "con-10x10x10a",
    "con-15x15x23a",
    "large-4ag",
    "large-4ag-hard",
    "large-8ag",
    "large-8ag-hard",
    "medium-4ag",
    "medium-4ag-hard",
    "medium-6ag",
    "small-4ag",
    "small-4ag-hard",
    "tiny-2ag-hard",
    "tiny-4ag-hard",
    "xlarge-4ag",
    "xlarge-4ag-hard",
    "smacv2_10_units",
    "smacv2_20_units",
]
systems_to_run = [
    "rec_ippo",
    "rec_mappo",
    "rec_sable",
]
top_ks = [
    "1",
    "2",
    "4",
    "8",
    "16",
    "32",
    "64",
]
beam_width = "4"

# Starting with 100M steps
env_run_details = {
    "rware": {  # correct
        "rollout_length": 500,
        "num_updates": 1560,
        "num_evaluation": 156,
    },
    "smax": {  # correct
        "rollout_length": 100,
        "num_updates": 7810,
        "num_evaluation": 781,
    },
}

task_run_details = {
    "con-10x10x10a": {  # correct.
        "rollout_length": 100,
        "num_updates": 7810,
        "num_evaluation": 781,
    },
    "con-15x15x23a": {  # correct.
        "rollout_length": 225,
        "num_updates": 3470,
        "num_evaluation": 347,
    },
}
task_timestep_chunksize = {
    "con-15x15x23a": 75,
    "medium-6ag": 250,
    "xlarge-4ag-hard": 250,
    "large-8ag-hard": 250,
    "smacv2_20_units": 50,
}

NEPTUNE_TAG = "to-delete"


def system_name_to_run_file(system_name: str) -> str:
    if system_name == "rec_sable":
        return "mava/systems/sable/anakin/rec_sable_sgbs.py"
    elif system_name == "rec_mappo":
        return "mava/systems/reinforce/anakin/rec_ireinforce_sgbs.py"
    elif system_name == "rec_ippo":
        return "mava/systems/reinforce/anakin/rec_ireinforce_sgbs.py"


def get_system_script(hyperparams: pd.Series) -> str:
    script = ""
    script += f"system.max_grad_norm={float(hyperparams['max_grad_norm'])} \\\n"
    script += f"system.clip_eps={float(hyperparams['clip_eps'])} \\\n"
    script += f"system.ent_coef={float(hyperparams['ent_coef'])} \\\n"
    script += f"system.actor_lr={float(hyperparams['actor_lr'])} \\\n"

    if pd.notna(hyperparams["critic_lr"]):
        script += f"system.critic_lr={float(hyperparams['critic_lr'])} \\\n"

    return script


def get_env_script(env: str, scenario: str) -> str:
    env_script = f"env={env} \\\n"
    if env == "smax":
        scenario_script = f"env.scenario.task_name={scenario} \\\n"
    else:
        scenario_script = f"env/scenario={scenario} \\\n"

    return env_script + scenario_script


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # each system has it's own dataframe
    dfs = {
        "on_policy": pd.read_csv("base_policy_hyperparameters/onpolicy_params.csv"),
    }

    # filter for systems and scenarios we want to run for each system's dataframe
    for system_type, df in dfs.items():
        mask = (df["system_name"].isin(systems_to_run)) & (df["task"].isin(scenarios_to_run))
        dfs[system_type] = df[mask]

    for df in dfs.values():
        for _, row in df.iterrows():
            system_name: str = row["system_name"]  # type: ignore
            scenario: str = row["task"]  # type: ignore
            env: str = row["env_name"]  # type: ignore
            # drop non-system params
            hyperparams = row.drop(["system_name", "task", "env_name", "compute"])

            if env != "vector-connector":
                rollout_length = env_run_details[env]["rollout_length"]
                num_updates = env_run_details[env]["num_updates"]
                num_evaluation = env_run_details[env]["num_evaluation"]

            else:
                rollout_length = task_run_details[scenario]["rollout_length"]
                num_updates = task_run_details[scenario]["num_updates"]
                num_evaluation = task_run_details[scenario]["num_evaluation"]

            for top_k in top_ks:
                attempt = str(int(top_k) * int(beam_width))
                budget_filename = "ppo" if "ppo" in system_name else system_name
                budget = task_budget[budget_filename][scenario][attempt]

                init_script = (
                    "#!/bin/bash\n"
                    f"# Script auto-generated by experiment_dispatcher.py on {timestamp}\n"
                    f"python {system_name_to_run_file(system_name)} -m \\\n"
                    f"system.seed=0 \\\n"
                    f"logger.use_neptune=False \\\n"
                    f"logger.kwargs.neptune_tag=[{NEPTUNE_TAG}] \\\n"
                    f"arch.num_envs=128 \\\n"
                    f"system.num_updates={num_updates} \\\n"
                    f"arch.num_evaluation={num_evaluation} \\\n"
                    f"system.rollout_length={rollout_length} \\\n"
                    f"system.update_batch_size=1 \\\n"
                    f"system.num_minibatches=1 \\\n"
                    f"system.ppo_epochs=1 \\\n"
                    f"inference.n_attempts={attempt} \\\n"
                    f"inference.budget={budget} \\\n"
                    f"inference.n_envs=128 \\\n"
                    f"inference.env_seed=0 \\\n"
                    f"inference.time_constraint=3000000000 \\\n"
                    f"inference.beam_width={beam_width} \\\n"
                    f"inference.top_k={top_k} \\\n"
                )

                system_script = get_system_script(hyperparams)
                env_script = get_env_script(env, scenario)

                inference_script = (
                    "arch.num_latents_per_env=64 \\\n"
                    "arch.eval_diff_latent_num=False \\\n"
                    "arch.eval_num_latents_per_env=64 \\\n"
                    "arch.compass_latent_dim=16 \\\n"
                    "arch.latent_amplifier=1 \\\n"
                    "arch.latent_sampling_same=False \\\n"
                    "arch.padding_with_random_weights=True \\\n"
                    "arch.weights_noise=0.01 \\\n"
                    "arch.grad_accumulation_steps=1 \\\n"
                )

                script = init_script + inference_script + system_script + env_script

                if system_name == "rec_sable":
                    script += f"network.memory_config.decay_scaling_factor={float(hyperparams['decay_scaling_factor'])} \\\n"
                    script += f"network.net_config.n_head={int(hyperparams['n_head'])} \\\n"
                    script += f"network.net_config.n_block={int(hyperparams['n_block'])} \\\n"
                    script += f"network.net_config.embed_dim={int(hyperparams['embed_dim'])} \\\n"

                if (scenario in task_timestep_chunksize) and (system_name == "rec_sable"):
                    script += f"network.memory_config.timestep_chunk_size={task_timestep_chunksize[scenario]} \\\n"

                run_id = run_ids[system_name][scenario]
                # Set model neptune run name for downloading
                script += f"logger.checkpointing.download_args.neptune_run_name={run_id} \\\n"
                script += "logger.checkpointing.delete_local_checkpoints=True \\\n"
                script += "logger.checkpointing.load_args.checkpoint_uid=paper-baselines \\\n"
                script += "logger.checkpointing.save_model=False \\\n"
                script += "logger.checkpointing.upload_model=False \\\n"
                script += "logger.checkpointing.load_model=True \\\n"
                script += "logger.checkpointing.download_model=False \\\n"
                script += "logger.checkpointing.unzip_local_model=True \\\n"

                # Sets the base system name correctly so that mappo and ippo are not called reinforce.
                if system_name in ["rec_mappo", "rec_ippo"]:
                    script += f"+logger.checkpointing.base_system_name={system_name} \\\n"

                with open("run.sh", mode="w+") as f:
                    f.write(script)

                subprocess.run(["chmod", "+x", "run.sh"])
                subprocess.run(["./run.sh"])
