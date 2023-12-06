#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import ast
import logging
import math
import os
import sys
from argparse import Namespace
from itertools import chain

import numpy as np
import torch
from omegaconf import DictConfig

from fairseq import checkpoint_utils, options, scoring, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter

def get_encoder_representations(model, sample, prefix_tokens=None, constraints=None):
    with torch.no_grad():
        encoder_input = {"src_tokens": sample["net_input"]["src_tokens"], "src_lengths": sample["net_input"]["src_lengths"]}
        encoder_out = model[0].encoder(**encoder_input, return_all_hiddens=True)
        hidden_states_layers = encoder_out["encoder_states"] # List[T x B x C]
        encoder_padding_mask = encoder_out["encoder_padding_mask"] # B x T
        return hidden_states_layers, encoder_padding_mask[0]


def main(cfg: DictConfig):

    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for generation!"
    assert (
        not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        cfg.generation.replace_unk is None or cfg.dataset.dataset_impl == "raw"
    ), "--replace-unk requires a raw text dataset (--dataset-impl=raw)"

    if cfg.common_eval.results_path is not None:
        os.makedirs(cfg.common_eval.results_path, exist_ok=True)
        output_path = os.path.join(
            cfg.common_eval.results_path,
            "generate-{}.txt".format(cfg.dataset.gen_subset),
        )
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            return _main(cfg, h)
    else:
        return _main(cfg, sys.stdout)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}


def _main(cfg: DictConfig, output_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("fairseq_cli.generate")

    utils.import_user_module(cfg.common)

    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.max_tokens = 12000
    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # Load dataset splits
    task = tasks.setup_task(cfg.task)

    # Set dictionaries
    try:
        src_dict = getattr(task, "source_dictionary", None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    overrides = ast.literal_eval(cfg.common_eval.model_overrides)

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, saved_cfg = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )

    # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)

    if cfg.generation.lm_path is not None:
        overrides["data"] = cfg.task.data

        try:
            lms, _ = checkpoint_utils.load_model_ensemble(
                [cfg.generation.lm_path], arg_overrides=overrides, task=None
            )
        except:
            logger.warning(
                f"Failed to load language model! Please make sure that the language model dict is the same "
                f"as target dict and is located in the data dir ({cfg.task.data})"
            )
            raise

        assert len(lms) == 1
    else:
        lms = [None]

    # Optimize ensemble for generation
    for model in chain(models, lms):
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(cfg.generation.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[m.max_positions() for m in models]
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    # Initialize generator
    gen_timer = StopwatchMeter()

    extra_gen_cls_kwargs = {"lm_model": lms[0], "lm_weight": cfg.generation.lm_weight}
    generator = task.build_generator(
        models, cfg.generation, extra_gen_cls_kwargs=extra_gen_cls_kwargs
    )

    # Handle tokenization and BPE
    tokenizer = task.build_tokenizer(cfg.tokenizer)
    bpe = task.build_bpe(cfg.bpe)

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    scorer = scoring.build_scorer(cfg.scoring, tgt_dict)

    num_sentences = 0
    has_target = True
    wps_meter = TimeMeter()
    # 12 layers dict
    all_hidden_states = {
        "layer_0": None,
        "layer_1": None,
        "layer_2": None,
        "layer_3": None,
        "layer_4": None,
        "layer_5": None,
        "layer_6": None,
        "layer_7": None,
        "layer_8": None,
        "layer_9": None,
        "layer_10": None,
        "layer_11": None,
    }
    all_sample_id = None
    test_target_str = []

    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if "net_input" not in sample:
            continue

        prefix_tokens = None
        if cfg.generation.prefix_size > 0:
            prefix_tokens = sample["target"][:, : cfg.generation.prefix_size]

        constraints = None
        if "constraints" in sample:
            constraints = sample["constraints"]

        gen_timer.start()
        hidden_states, encoder_mask = get_encoder_representations(models, sample)

        # if all_hidden_states is None:
        #     all_hidden_states = np.mean(hidden_states[-1].cpu().numpy(), axis=0)
        # else:
        #     all_hidden_states = np.concatenate(
        #         (all_hidden_states, np.mean(hidden_states[-1].cpu().numpy(), axis=0)), axis=0
        #     )
        for i in [11]:
            hidden_states[i] = hidden_states[i].cpu().numpy().reshape(-1, hidden_states[-1].shape[-1])
            hidden_states[i] = hidden_states[i][encoder_mask.cpu().numpy().flatten() == 0]
            # full_states = None
            # for idx in range(hidden_states[i].shape[1]):
            #     mask = encoder_mask[idx].cpu().numpy().reshape(-1)
            #     res = hidden_states[i][:, idx, :]
            #     res = res.reshape(res.shape[0], -1)
            #     # remove padding where encoder_mask is True
            #     res = res[mask == 0]
            #     res = np.mean(res, axis=0)
            #     res = res.reshape(1, -1)
            #     if full_states is None:
            #         full_states = res
            #     else:
            #         full_states = np.concatenate((full_states, res), axis=0)
            # concat to all_hidden_states
            if all_hidden_states["layer_" + str(i)] is None:
                all_hidden_states["layer_" + str(i)] = hidden_states[i]
            else:
                all_hidden_states["layer_" + str(i)] = np.concatenate(
                    (all_hidden_states["layer_" + str(i)], hidden_states[i]), axis=0
                )
        curr_sample_id = sample["id"].tolist()
        if all_sample_id is None:
            all_sample_id = curr_sample_id
        else:
            all_sample_id.extend(curr_sample_id)

        for i, sample_id in enumerate(sample["id"].tolist()):
            has_target = sample["target"] is not None

            target_tokens = None
            if has_target:
                target_tokens = (
                    utils.strip_pad(sample["target"][i, :], tgt_dict.pad()).int().cpu()
                )

            # Either retrieve the original sentences or regenerate them from tokens.
            if align_dict is not None:
                target_str = task.dataset(cfg.dataset.gen_subset).tgt.get_original_text(
                    sample_id
                )
            else:
                if has_target:
                    target_str = tgt_dict.string(
                        target_tokens,
                        cfg.common_eval.post_process,
                        escape_unk=True,
                        extra_symbols_to_ignore=get_symbols_to_strip_from_output(
                            generator
                        ),
                    )

            if has_target:
                target_str = decode_fn(target_str)
                test_target_str.append(target_str)

            # if not cfg.common_eval.quiet:
            #     if has_target:
            #         print("T-{}\t{}".format(sample_id, target_str), file=output_file)

    # save all_hidden_states as numpy
    for i in range(len(hidden_states)):
        all_hidden_states["layer_" + str(i)] = np.array(all_hidden_states["layer_" + str(i)])

    sorted_order = np.argsort(all_sample_id)
    test_target_str = np.array(test_target_str)
    test_target_str = test_target_str[sorted_order]

    # for i in range(len(all_hidden_states)):
    i = 11
    # all_hidden_states["layer_" + str(i)] = all_hidden_states["layer_" + str(i)][sorted_order]
    np.save(f"{cfg.task.data}/test/{cfg.task.lang_pair}_{cfg.task.language}_layer_{str(i)}_token.npy", all_hidden_states["layer_" + str(i)])

    return scorer


def cli_main():
    parser = options.get_generation_parser()
    # TODO: replace this workaround with refactoring of `AudioPretraining`
    parser.add_argument(
        "--arch",
        "-a",
        metavar="ARCH",
        default="wav2vec2",
        help="Model architecture. For constructing tasks that rely on "
        "model args (e.g. `AudioPretraining`)",
    )
    parser.add_argument(
        "--language",
        type=str,
        metavar="LANG",
        help="target language for translation (default: en)",
    )
    parser.add_argument(
        "--lang-pair",
        type=str,
        metavar="PAIR",
        help="language pair for translation, e.g., 'fr-en' (default: None)",
    )

    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
