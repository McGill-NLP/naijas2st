"""Quick integrity check for SeamlessM4T fine-tuned safetensors shards.

Loads each model shard via ``safetensors.torch.load_file`` to catch
truncated or corrupted checkpoint files before launching inference.
"""

import os
from safetensors.torch import load_file

CHECKPOINT_DIR = "./models/seamless-stt-finetuned/checkpoint-12500"


def main():
    """Try to load both fine-tuned SeamlessM4T safetensors shards.

    Workflow:
        1. Construct the path to ``checkpoint-12500`` under
           ``CHECKPOINT_DIR``.
        2. Call ``safetensors.torch.load_file`` on each shard
           (``model-00001-of-00002.safetensors`` then
           ``model-00002-of-00002.safetensors``) in turn so that any
           corruption raises immediately.

    Outputs:
        Nothing if both shards load cleanly; an exception is raised
        with a clear pointer to the corrupted shard otherwise. Useful
        as a fast pre-flight check before launching inference.

    Returns:
        None.
    """
    load_file(os.path.join(CHECKPOINT_DIR, "model-00001-of-00002.safetensors"))
    load_file(os.path.join(CHECKPOINT_DIR, "model-00002-of-00002.safetensors"))


if __name__ == "__main__":
    main()
