#!/usr/bin/env python
"""
Training and Evaluation script for VRD IMP models.
Trains models, evaluates them, and saves results to a structured output.

Usage:
    # Train single config
    python train_eval_vrd.py --config-file configs/e2e_relIMP_vrd_bert_pce.yaml

    # Train with custom iterations
    python train_eval_vrd.py --config-file configs/e2e_relIMP_vrd_bert_pce.yaml --max-iter 50000

    # Multi-GPU training (2 GPUs)
    torchrun --nproc_per_node=2 train_eval_vrd.py --config-file configs/e2e_relIMP_vrd_bert_pce.yaml

    # Train all Phase 1 configs
    python train_eval_vrd.py --all --max-iter 30000

    # Resume training
    python train_eval_vrd.py --config-file configs/e2e_relIMP_vrd_bert_pce.yaml --resume
"""

import argparse
import datetime
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from pysgg.config import cfg
from pysgg.data import make_data_loader
from pysgg.engine.inference import inference
from pysgg.engine.trainer import reduce_loss_dict
from pysgg.modeling.detector import build_detection_model
from pysgg.solver import make_lr_scheduler, make_optimizer
from pysgg.utils.checkpoint import DetectronCheckpointer, clip_grad_norm
from pysgg.utils.comm import synchronize, get_rank, get_world_size, all_gather
from pysgg.utils.logger import setup_logger, debug_print, TFBoardHandler_LEVEL
from pysgg.utils.metric_logger import MetricLogger
from pysgg.utils.miscellaneous import mkdir, save_config

# ============================================================================
# Default configurations for VRD training
# ============================================================================
DEFAULTS = {
    "max_iter": 30000,           # Total training iterations
    "checkpoint_period": 5000,   # Save checkpoint every N iterations
    "val_period": 5000,          # Validate every N iterations
    "batch_size": 8,             # Images per GPU (will be multiplied by num_gpus)
    "base_lr": 0.008,            # Base learning rate
    "warmup_iters": 500,         # Warmup iterations
    "lr_steps": [20000],         # LR decay steps
    "lr_gamma": 0.5,             # LR decay factor
    "grad_clip": 5.0,            # Gradient clipping norm
    "num_workers": 4,            # DataLoader workers per GPU
    "pce_pretrain_iter": 2000,   # PCE module pretraining iterations
}

# All VRD IMP configs from Phase 1
ALL_CONFIGS = [
    # Baselines (no PCE)
    "configs/e2e_relIMP_vrd_no_semantics.yaml",
    "configs/e2e_relIMP_vrd_glove.yaml",
    "configs/e2e_relIMP_vrd_word2vec.yaml",
    "configs/e2e_relIMP_vrd_bert.yaml",
    "configs/e2e_relIMP_vrd_minilm.yaml",
    # With PCE (learnable gating)
    "configs/e2e_relIMP_vrd_no_semantics_pce.yaml",
    "configs/e2e_relIMP_vrd_glove_pce.yaml",
    "configs/e2e_relIMP_vrd_word2vec_pce.yaml",
    "configs/e2e_relIMP_vrd_bert_pce.yaml",
    "configs/e2e_relIMP_vrd_minilm_pce.yaml",
    # Alternative gating
    "configs/e2e_relIMP_vrd_bert_pce_sigmoid.yaml",
    "configs/e2e_relIMP_vrd_bert_pce_poly.yaml",
    "configs/e2e_relIMP_vrd_word2vec_pce_sigmoid.yaml",
    "configs/e2e_relIMP_vrd_word2vec_pce_poly.yaml",
]

# Seeding
SEED = 666
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def fix_eval_modules(eval_modules):
    """Freeze modules that should not be trained."""
    for module in eval_modules:
        if module is None:
            continue
        for _, param in module.named_parameters():
            param.requires_grad = False


def get_config_name(config_file):
    """Extract config name from path."""
    return Path(config_file).stem


def train_one_config(
    config_file,
    args,
    local_rank,
    distributed,
    results_dir,
):
    """Train and evaluate a single config."""

    config_name = get_config_name(config_file)

    # Load and merge config
    cfg.defrost()
    cfg.merge_from_file(config_file)

    # Apply command-line overrides
    if args.max_iter:
        cfg.SOLVER.MAX_ITER = args.max_iter
    if args.batch_size:
        cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    if args.base_lr:
        cfg.SOLVER.BASE_LR = args.base_lr
    if args.checkpoint_period:
        cfg.SOLVER.CHECKPOINT_PERIOD = args.checkpoint_period
    if args.val_period:
        cfg.SOLVER.VAL_PERIOD = args.val_period

    # Set output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.OUTPUT_DIR = os.path.join(
        args.output_dir,
        config_name,
        timestamp if not args.resume else "resumed"
    )

    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if get_rank() == 0:
        mkdir(output_dir)
    synchronize()

    # Setup logger
    logger = setup_logger("pysgg", output_dir, get_rank())
    logger.info("=" * 70)
    logger.info(f"Training: {config_name}")
    logger.info("=" * 70)
    logger.info(f"Config file: {config_file}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Max iterations: {cfg.SOLVER.MAX_ITER}")
    logger.info(f"Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
    logger.info(f"Base LR: {cfg.SOLVER.BASE_LR}")

    # Save config
    if get_rank() == 0:
        save_config(cfg, os.path.join(output_dir, "config.yml"))

    # Build model
    logger.info("Building model...")
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE if cfg.MODEL.DEVICE != "auto" else "cuda")
    model.to(device)

    # Freeze detector modules
    eval_modules = (
        model.rpn,
        model.backbone,
        model.roi_heads.box,
    )
    fix_eval_modules(eval_modules)

    # Load pretrained detector
    load_mapping = {
        "roi_heads.relation.box_feature_extractor": "roi_heads.box.feature_extractor",
        "roi_heads.relation.rel_pair_box_feature_extractor": "roi_heads.box.feature_extractor",
        "roi_heads.relation.union_feature_extractor.feature_extractor": "roi_heads.box.feature_extractor",
    }

    # Build optimizer and scheduler
    optimizer = make_optimizer(cfg, model, logger, rl_factor=float(cfg.SOLVER.IMS_PER_BATCH))
    scheduler = make_lr_scheduler(cfg, optimizer, logger)

    # Setup checkpointer
    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk, custom_scheduler=True
    )

    # Load checkpoint or pretrained weights
    arguments = {"iteration": 0}
    if args.resume:
        extra_checkpoint_data = checkpointer.load(checkpointer.get_checkpoint_file())
        arguments.update(extra_checkpoint_data)
    elif cfg.MODEL.PRETRAINED_DETECTOR_CKPT:
        checkpointer.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT, with_optim=False, load_mapping=load_mapping)

    # Setup DDP
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    # Create data loaders
    logger.info("Creating data loaders...")
    train_data_loader = make_data_loader(
        cfg,
        mode="train",
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )
    val_data_loaders = make_data_loader(
        cfg,
        mode="val",
        is_distributed=distributed,
    )

    # Initialize PCE pretraining if needed
    pre_clser_pretrain_on = False
    m2opt = model.module if distributed else model
    if (
        cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.PRETRAIN_RELNESS_MODULE
        and cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.SET_ON
    ):
        m2opt.roi_heads.relation.predictor.start_preclser_relpn_pretrain()
        logger.info("Started PCE pretraining mode")
        pre_clser_pretrain_on = True
        stop_iter = cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.PRETRAIN_ITER_RELNESS_MODULE

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    # Training loop
    logger.info("Starting training...")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(train_data_loader)
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()

    model.train()

    for iteration, (images, targets, _) in enumerate(train_data_loader, start_iter):
        if any(len(target) < 1 for target in targets):
            logger.error(f"Empty targets at iteration {iteration + 1}")
            continue

        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        model.train()
        fix_eval_modules(eval_modules)
        optimizer.zero_grad()

        # Forward pass with AMP
        if scaler is not None:
            with torch.cuda.amp.autocast():
                images = images.to(device)
                targets = [t.to(device) for t in targets]
                loss_dict = model(images, targets, logger=logger)
                losses = sum(loss for loss in loss_dict.values())
        else:
            images = images.to(device)
            targets = [t.to(device) for t in targets]
            loss_dict = model(images, targets, logger=logger)
            losses = sum(loss for loss in loss_dict.values())

        # Reduce losses for logging
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        # Backward pass
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
        else:
            losses.backward()

        # Gradient clipping
        clip_grad_norm(
            [(n, p) for n, p in model.named_parameters() if p.requires_grad],
            max_norm=cfg.SOLVER.GRAD_NORM_CLIP,
            logger=logger,
            verbose=(iteration % 4000 == 0),
            clip=True,
        )

        # Optimizer step
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        # End PCE pretraining if needed
        if pre_clser_pretrain_on and iteration == stop_iter:
            logger.info("PCE pretraining ended.")
            m2opt.roi_heads.relation.predictor.end_preclser_relpn_pretrain()
            pre_clser_pretrain_on = False

        # Logging
        if iteration % 100 == 0 or iteration == max_iter:
            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(
                f"iter: {iteration}/{max_iter}  "
                f"eta: {eta_string}  "
                f"loss: {losses_reduced:.4f}  "
                f"lr: {optimizer.param_groups[0]['lr']:.6f}  "
                f"mem: {torch.cuda.max_memory_allocated() / 1024**2:.0f}MB"
            )

        # Checkpoint
        if iteration % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
            checkpointer.save(f"model_{iteration:07d}", **arguments)

        # Validation
        if cfg.SOLVER.TO_VAL and iteration % cfg.SOLVER.VAL_PERIOD == 0:
            logger.info("Running validation...")
            val_results = run_validation(cfg, model, val_data_loaders, distributed, logger)
            logger.info(f"Validation results: {val_results}")

    # Save final model
    checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    logger.info(f"Total training time: {datetime.timedelta(seconds=int(total_training_time))}")

    # Final evaluation
    logger.info("=" * 70)
    logger.info("Running final evaluation...")
    logger.info("=" * 70)

    eval_results = run_evaluation(cfg, model, distributed, logger)

    # Save results
    results = {
        "config_name": config_name,
        "config_file": config_file,
        "output_dir": output_dir,
        "max_iter": cfg.SOLVER.MAX_ITER,
        "training_time_seconds": total_training_time,
        "eval_results": eval_results,
    }

    if get_rank() == 0:
        results_file = os.path.join(output_dir, "results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {results_file}")

        # Also save to central results directory
        central_results_file = os.path.join(results_dir, f"{config_name}.json")
        with open(central_results_file, "w") as f:
            json.dump(results, f, indent=2)

    return results


def run_validation(cfg, model, val_data_loaders, distributed, logger):
    """Run validation and return results."""
    if distributed:
        model_to_eval = model.module
    else:
        model_to_eval = model

    model_to_eval.eval()

    iou_types = ("bbox",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations",)

    results = {}
    for dataset_name, val_data_loader in zip(cfg.DATASETS.VAL, val_data_loaders):
        dataset_result = inference(
            cfg,
            model_to_eval,
            val_data_loader,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=None,
            logger=logger,
        )
        synchronize()
        results[dataset_name] = dataset_result

    model.train()
    return results


def run_evaluation(cfg, model, distributed, logger):
    """Run final test evaluation."""
    if distributed:
        model_to_eval = model.module
    else:
        model_to_eval = model

    model_to_eval.eval()

    iou_types = ("bbox",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations",)

    output_folders = []
    for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        if get_rank() == 0:
            mkdir(output_folder)
        output_folders.append(output_folder)

    synchronize()

    test_data_loaders = make_data_loader(cfg, mode="test", is_distributed=distributed)

    all_results = {}
    for output_folder, dataset_name, data_loader in zip(
        output_folders, cfg.DATASETS.TEST, test_data_loaders
    ):
        result = inference(
            cfg,
            model_to_eval,
            data_loader,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
            logger=logger,
        )
        synchronize()

        # Extract metrics if available
        if isinstance(result, tuple) and len(result) >= 2:
            all_results[dataset_name] = {
                "recall": float(result[0]) if result[0] is not None else None,
                "detailed": result[1] if len(result) > 1 else None,
            }
        else:
            all_results[dataset_name] = result

    return all_results


def aggregate_results(results_dir):
    """Aggregate all results into a summary file."""
    results_files = list(Path(results_dir).glob("*.json"))

    all_results = []
    for rf in results_files:
        if rf.name == "summary.json":
            continue
        with open(rf) as f:
            all_results.append(json.load(f))

    # Sort by config name
    all_results.sort(key=lambda x: x["config_name"])

    # Create summary
    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "num_configs": len(all_results),
        "results": all_results,
    }

    summary_file = os.path.join(results_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate VRD IMP models")

    # Config selection
    parser.add_argument("--config-file", type=str, help="Path to config file")
    parser.add_argument("--all", action="store_true", help="Train all Phase 1 configs")

    # Training parameters
    parser.add_argument("--max-iter", type=int, default=DEFAULTS["max_iter"],
                        help=f"Max training iterations (default: {DEFAULTS['max_iter']})")
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"],
                        help=f"Batch size per GPU (default: {DEFAULTS['batch_size']})")
    parser.add_argument("--base-lr", type=float, default=DEFAULTS["base_lr"],
                        help=f"Base learning rate (default: {DEFAULTS['base_lr']})")
    parser.add_argument("--checkpoint-period", type=int, default=DEFAULTS["checkpoint_period"],
                        help=f"Checkpoint period (default: {DEFAULTS['checkpoint_period']})")
    parser.add_argument("--val-period", type=int, default=DEFAULTS["val_period"],
                        help=f"Validation period (default: {DEFAULTS['val_period']})")

    # Output
    parser.add_argument("--output-dir", type=str, default="checkpoints/vrd_experiments",
                        help="Base output directory")
    parser.add_argument("--results-dir", type=str, default="results/vrd_imp_phase1",
                        help="Directory to save aggregated results")

    # Misc
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # Distributed setup
    num_gpus = int(os.environ.get("WORLD_SIZE", 1))
    distributed = num_gpus > 1
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))

    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    # Create results directory
    if get_rank() == 0:
        mkdir(args.results_dir)
    synchronize()

    # Determine which configs to train
    if args.all:
        configs_to_train = [c for c in ALL_CONFIGS if os.path.exists(c)]
        print(f"Training {len(configs_to_train)} configs")
    elif args.config_file:
        configs_to_train = [args.config_file]
    else:
        parser.error("Must specify --config-file or --all")

    # Train each config
    all_results = []
    for config_file in configs_to_train:
        if not os.path.exists(config_file):
            print(f"WARNING: Config not found, skipping: {config_file}")
            continue

        try:
            result = train_one_config(
                config_file=config_file,
                args=args,
                local_rank=local_rank,
                distributed=distributed,
                results_dir=args.results_dir,
            )
            all_results.append(result)
        except Exception as e:
            print(f"ERROR training {config_file}: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Reset config for next iteration
        cfg.defrost()

    # Aggregate results
    if get_rank() == 0 and len(all_results) > 0:
        summary = aggregate_results(args.results_dir)
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Trained {len(all_results)} configs")
        print(f"Results saved to: {args.results_dir}")
        print(f"Summary file: {args.results_dir}/summary.json")

    if distributed:
        dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    sys.exit(main())
