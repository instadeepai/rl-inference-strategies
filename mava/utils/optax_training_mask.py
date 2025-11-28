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

from __future__ import annotations

from typing import Any, List, Tuple

import jax.tree_util as jtu
import optax


def create_optax_partial_freezing_mask(
    model_params: optax.Params,
    path_to_trainable_subtree: List[str],
) -> Tuple[optax.GradientTransformation, optax.OptState]:
    """
    Sets up an Optax optimizer where only a specified subtree of parameters is trained,
    and all other parameters are frozen using optax.masked.

    Args:
        model_params: The complete PyTree of model parameters.
        path_to_trainable_subtree: A list of string keys representing the path from the
                                   root of `model_params` to the subtree that should be
                                   trained. For example, to train only parameters within
                                   `model_params['layer_group1']['sub_layerA']`, this
                                   would be `['layer_group1', 'sub_layerA']`.

    Returns:
        A mask for freezing parameters while training with optax.
    """

    # Convert the list of keys for the trainable path to a tuple for efficient comparison.
    trainable_path_tuple = tuple(path_to_trainable_subtree)

    def get_freeze_mask_value(
        path_entries: Tuple[jtu.PathEntry, ...], leaf_param_value: Any
    ) -> bool:
        """
        Determines if a parameter should be frozen based on its path.
        Returns True if the parameter should be frozen, False if it should be trained.
        `path_entries` is a tuple of jax.tree_util.PathEntry objects.
        """
        current_param_key_path = []
        for entry in path_entries:
            if isinstance(entry, jtu.DictKey):
                current_param_key_path.append(entry.key)
            elif isinstance(entry, jtu.SequenceKey):
                current_param_key_path.append(entry.idx)
            else:
                current_param_key_path.append(entry)
        current_param_key_path_tuple = tuple(current_param_key_path)

        is_trainable = False
        if len(current_param_key_path_tuple) >= len(trainable_path_tuple):
            if current_param_key_path_tuple[: len(trainable_path_tuple)] == trainable_path_tuple:
                is_trainable = True

        # Mask should be True to freeze (apply set_to_zero), False to train.
        return not is_trainable

    # Create the boolean mask PyTree. It will have the same structure as model_params.
    # Leaves will be True for frozen parameters, False for trainable parameters.
    freeze_mask = jtu.tree_map_with_path(get_freeze_mask_value, model_params)

    return freeze_mask
