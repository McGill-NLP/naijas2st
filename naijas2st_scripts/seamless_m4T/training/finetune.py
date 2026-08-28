# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.
"""SeamlessM4T (UnitY) fine-tuning CLI.

Local copy of Meta's reference fine-tuning script. Loads the UnitY
model, text tokenizer, and unit tokenizer, builds train and eval
``UnitYDataLoader`` instances from JSONL manifests, and runs
``UnitYFinetune`` in S2T/T2U/S2S mode according to ``--mode``.
"""

import argparse
import logging
import os
from pathlib import Path

import torch

from meta_seamless.seamless_communication.src.seamless_communication.cli.m4t.finetune import dataloader, dist_utils, trainer
from meta_seamless.seamless_communication.src.seamless_communication.models.unity import (
    load_unity_model,
    load_unity_text_tokenizer,
    load_unity_unit_tokenizer,
)

logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s %(levelname)s -- %(name)s.{os.getpid()}: %(message)s",
)

logger = logging.getLogger("finetune")


def init_parser() -> argparse.ArgumentParser:
    """Construct the CLI ``argparse.ArgumentParser`` for fine-tuning options.

    Returns:
        argparse.ArgumentParser: A parser with the dataset / model /
        save / batch / patience / epoch / LR / warmup / eval / log /
        max-source-tokens / mode / freeze / device flags installed.
    """
    parser = argparse.ArgumentParser(
        description="Example finetuning script for M4T models"
    )
    parser.add_argument(
        "--train_dataset",
        type=Path,
        required=True,
        help="Path to manifest with train samples",
    )
    parser.add_argument(
        "--eval_dataset",
        type=Path,
        required=True,
        help="Path to manifest with eval samples",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="seamless-m4t-v2-large",
        help="Base model name (`seamlessM4T_medium`, `seamlessM4T_large`)",
    )
    parser.add_argument(
        "--save_model_to",
        type=Path,
        required=True,
        help="Path to save best finetuned model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2343,
        help="Randomizer seed value",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help=(
            "Set early termination after `patience` number of evaluations "
            "without eval loss improvements"
        ),
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10,
        help=("Max number of training epochs"),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-7,
        help=("Finetuning learning rate"),
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help=("Number of steps with linearly increasing learning rate"),
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=50,
        help=("Get eval loss after each `eval_steps` training steps "),
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=10,
        help=("Log inner loss after each `log_steps` training steps"),
    )
    parser.add_argument(
        "--max_src_tokens",
        type=int,
        default=7000,
        help=("Maximum number of src_tokens per batch, used to avoid GPU OOM and maximize the effective batch size"),
    )
    parser.add_argument(
        "--mode",
        type=trainer.FinetuneMode,
        choices=list(trainer.FinetuneMode),
        default=trainer.FinetuneMode.SPEECH_TO_TEXT,
        help=(
            "* `SPEECH_TO_SPEECH` -- finetune S2T and T2U parts of the model; "
            "* `TEXT_TO_SPEECH` -- finetune only T2U; "
            "* `SPEECH_TO_TEXT` -- finetune only S2T"
        ),
    )
    parser.add_argument(
        "--freeze_layers",
        nargs="*",
        required=False,
        default=None,
        # TODO: better description
        help=("A list of modules to freeze in the model. If empty, everything will be trained."),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help=("Device to fine-tune on. See `torch.device`."),
    )
    return parser


def main() -> None:
    """Parse CLI args and run a UnitY fine-tuning loop end-to-end.

    Workflow:
        1. Parse CLI args via :func:`init_parser` (covers manifests,
           model name, save path, batch size, patience, epochs,
           learning rate, warmup steps, eval/log steps, max source
           tokens, finetune mode and frozen modules).
        2. Initialise distributed training with
           ``dist_utils.init_distributed`` so the logger is shared
           between this script and the trainer module.
        3. Choose ``float_dtype`` (``float16`` on GPU, ``bfloat16`` on
           CPU) and load the UnitY text and unit tokenizers.
        4. Build ``trainer.FinetuneParams`` from the CLI args.
        5. Load ``UnitYModel`` on CPU/float32 and assert that its
           ``target_vocab_info`` matches the text tokenizer to catch
           tokenizer/model mismatches early.
        6. Strip unused components based on the chosen finetune mode:
            - ``SPEECH_TO_TEXT``: drop ``t2u_model`` entirely.
            - Always: drop ``text_encoder`` (unused in S2T/S2S).
        7. Move the trimmed model to the chosen device and build the
           train and eval ``UnitYDataLoader``s with
           ``batching_config`` parameters from the args.
        8. Instantiate ``trainer.UnitYFinetune`` with the model, loaders
           and ``freeze_layers`` list, then call ``.run()`` to start
           training.

    Returns:
        None.
    """
    args = init_parser().parse_args()
    
    dist_utils.init_distributed([logger, trainer.logger])
    float_dtype = torch.float16 if torch.device(args.device).type != "cpu" else torch.bfloat16
    
    text_tokenizer = load_unity_text_tokenizer(args.model_name)
    unit_tokenizer = load_unity_unit_tokenizer(args.model_name)
    
    # # Add the Hausa token if it's missing
    # new_tokens = ["hau"]
    # processor.tokenizer.add_tokens(new_tokens)
    # model.resize_token_embeddings(len(processor.tokenizer))

    finetune_params = trainer.FinetuneParams(
        model_name=args.model_name,
        finetune_mode=args.mode,
        save_model_path=args.save_model_to,
        device=torch.device(args.device),
        float_dtype=float_dtype,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        patience=args.patience,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        eval_steps=args.eval_steps,
        log_steps=args.log_steps,
    )
    
    logger.info(f"Finetune Params: {finetune_params}")
    
    model = load_unity_model(args.model_name, device=torch.device("cpu"), dtype=torch.float32)
    assert model.target_vocab_info == text_tokenizer.vocab_info
    
    if (
        finetune_params.finetune_mode == trainer.FinetuneMode.SPEECH_TO_TEXT
        and model.t2u_model is not None
    ):
        model.t2u_model = None
    
    if model.text_encoder is not None:
        model.text_encoder = None
    
    # Put model on selected device
    model = model.to(finetune_params.device)

    # TODO: delete unused params to reduce GPU memory consumption
    print("starting data loader initialization")
    train_dataloader = dataloader.UnitYDataLoader(
        text_tokenizer=text_tokenizer,
        unit_tokenizer=unit_tokenizer,
        batching_config=dataloader.BatchingConfig(
            batch_size=finetune_params.train_batch_size,
            rank=dist_utils.get_rank(),
            world_size=dist_utils.get_world_size(),
            max_audio_length_sec=90.0,
            float_dtype=finetune_params.float_dtype,
        ),
        dataset_manifest_path=args.train_dataset,
        max_src_tokens_per_batch=args.max_src_tokens)

    eval_dataloader = dataloader.UnitYDataLoader(
        text_tokenizer=text_tokenizer,
        unit_tokenizer=unit_tokenizer,
        batching_config=dataloader.BatchingConfig(
            batch_size=finetune_params.eval_batch_size,
            rank=dist_utils.get_rank(),
            world_size=dist_utils.get_world_size(),
            max_audio_length_sec=90.0,
            float_dtype=finetune_params.float_dtype,
        ),
        dataset_manifest_path=args.eval_dataset)
    
    finetune = trainer.UnitYFinetune(
        model=model,
        params=finetune_params,
        train_data_loader=train_dataloader,
        eval_data_loader=eval_dataloader,
        freeze_modules=args.freeze_layers)
    
    finetune.run()


if __name__ == "__main__":
    main()
