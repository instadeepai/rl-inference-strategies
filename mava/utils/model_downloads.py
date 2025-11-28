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

import shutil
import zipfile
from pathlib import Path


def unzip_local_checkpoints(
    checkpoint_rel_dir: str,
    model_name: str,
    run_id: str,
    *,  # keyword-only from here
    source_dir: str = "all_checkpoints",
) -> None:
    """
    Equivalent to `download_checkpoint_from_neptune` but grabs the
    archive from a *local* folder instead of the network.

    Parameters
    ----------
    checkpoint_rel_dir : str
        Destination parent directory (e.g. "checkpoints").
    model_name : str
        Sub-folder name (e.g. "rec_sable").
    run_id : str
        Checkpoint run ID, e.g. "MAR-3159".
    source_dir : str, default "all_checkpoints"
        Folder that already contains <run_id>.zip.
    """

    # --- 1. paths ---------------------------------------------------------
    src_zip = Path.cwd() / source_dir / f"{run_id}.zip"
    if not src_zip.exists():
        raise FileNotFoundError(f"Checkpoint archive not found: {src_zip}")

    dst_dir = Path.cwd() / checkpoint_rel_dir / model_name
    dst_dir.mkdir(parents=True, exist_ok=True)

    dst_zip = dst_dir / src_zip.name

    # --- 2. copy (so we keep the master cache intact) ---------------------
    if not dst_zip.exists():
        shutil.copy2(src_zip, dst_zip)
        print(f"✓ copied {src_zip}  →  {dst_zip}")

    # --- 3. unzip ---------------------------------------------------------
    print(f"⇢ extracting to {dst_dir}")
    with zipfile.ZipFile(dst_zip) as zf:
        zf.extractall(dst_dir)

    # --- 4. clean up copied archive --------------------------------------
    dst_zip.unlink()  # matches behaviour of download helpers
    print("✔ done")


def delete_local_checkpoints(
    checkpoint_folder_dir: str,
) -> None:
    """Delete a folder and all its contents"""
    shutil.rmtree(checkpoint_folder_dir)
