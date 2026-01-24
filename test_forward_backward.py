#!/usr/bin/env python
"""
Test script for running 1 forward + backward pass on VRD IMP models.
Validates that model can train on 128 samples across 2 GPUs.

Usage:
    # Single GPU training test (128 samples on 1 GPU)
    python test_forward_backward.py --config-file configs/e2e_relIMP_vrd_word2vec_pce.yaml

    # Multi-GPU training test (64 samples per GPU, 2 GPUs)
    torchrun --nproc_per_node=2 test_forward_backward.py --config-file configs/e2e_relIMP_vrd_word2vec_pce.yaml

    # Evaluation test (inference mode)
    python test_forward_backward.py --config-file configs/e2e_relIMP_vrd_word2vec_pce.yaml --mode eval
"""

import argparse
import os
import sys
import time
import random
import numpy as np
import torch
import torch.distributed as dist

from pysgg.config import cfg
from pysgg.data import make_data_loader
from pysgg.modeling.detector import build_detection_model
from pysgg.solver import make_optimizer
from pysgg.utils.checkpoint import DetectronCheckpointer, clip_grad_norm
from pysgg.utils.comm import synchronize, get_rank, get_world_size
from pysgg.utils.logger import setup_logger
from pysgg.utils.miscellaneous import mkdir
from pysgg.engine.trainer import reduce_loss_dict
from pysgg.engine.inference import inference

# Seeding for reproducibility
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fix_eval_modules(eval_modules):
    """Freeze modules that should not be trained."""
    for module in eval_modules:
        if module is None:
            continue
        for _, param in module.named_parameters():
            param.requires_grad = False


def count_parameters(model):
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def run_eval_test(cfg, model, device, distributed, local_rank, num_gpus, logger, args):
    """Run evaluation/inference test on a few samples."""
    from pysgg.utils.comm import synchronize

    logger.info("-" * 40)
    logger.info("Running EVALUATION test...")
    logger.info("-" * 40)

    # Wrap model with DDP if distributed
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    # Set model to eval mode
    model.eval()

    # Create test data loader
    logger.info("Creating test data loader...")
    test_data_loaders = make_data_loader(
        cfg,
        mode="test",
        is_distributed=distributed,
    )

    # Run inference on a limited number of samples
    torch.cuda.reset_peak_memory_stats(device)
    eval_start = time.time()

    with torch.no_grad():
        for data_loader in test_data_loaders:
            sample_count = 0
            max_samples = args.total_samples

            for batch_idx, (images, targets, image_ids) in enumerate(data_loader):
                images = images.to(device)
                targets = [t.to(device) for t in targets]

                # Run forward pass (inference)
                with torch.cuda.amp.autocast():
                    output = model(images)

                sample_count += len(targets)
                logger.info(f"Processed batch {batch_idx + 1}: {sample_count} samples")

                if sample_count >= max_samples:
                    break

    eval_time = time.time() - eval_start

    # Memory stats
    max_mem = 0.0
    if torch.cuda.is_available():
        max_mem = torch.cuda.max_memory_allocated(device) / 1024**3
        logger.info(f"Peak GPU memory: {max_mem:.2f} GB")

    # Summary
    logger.info("=" * 60)
    logger.info("EVAL TEST PASSED!")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config_file}")
    logger.info(f"Samples evaluated: {sample_count}")
    logger.info(f"Evaluation time: {eval_time:.3f}s")
    logger.info(f"Peak memory: {max_mem:.2f} GB")
    logger.info("=" * 60)

    # Print memory for shell script parsing
    print(f"GPU_MEMORY_GB:{max_mem:.2f}")

    # Cleanup
    if distributed:
        import torch.distributed as dist
        dist.destroy_process_group()

    return 0


def main():
    parser = argparse.ArgumentParser(description="Test forward/backward pass for VRD IMP models")
    parser.add_argument(
        "--config-file",
        required=True,
        metavar="FILE",
        help="path to config file (e.g., configs/e2e_relIMP_vrd_word2vec_pce.yaml)",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--total-samples",
        type=int,
        default=128,
        help="Total number of samples to process (default: 128)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval"],
        default="train",
        help="Test mode: 'train' for forward/backward pass, 'eval' for inference (default: train)",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    # Distributed setup
    num_gpus = int(os.environ.get("WORLD_SIZE", 1))
    distributed = num_gpus > 1
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))

    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    # Calculate batch size per GPU
    # For 128 samples on 2 GPUs: 64 per GPU
    samples_per_gpu = args.total_samples // num_gpus

    # Load config
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Override batch size for this test
    cfg.defrost()
    cfg.SOLVER.IMS_PER_BATCH = samples_per_gpu
    cfg.OUTPUT_DIR = "checkpoints/test_forward_backward"
    # Disable multiprocessing in data loader (Python 3.13 compatibility)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()

    # Setup output and logger
    output_dir = cfg.OUTPUT_DIR
    if get_rank() == 0:
        mkdir(output_dir)

    logger = setup_logger("test_forward_backward", output_dir, get_rank())

    logger.info("=" * 60)
    logger.info("Test Forward/Backward Pass")
    logger.info("=" * 60)
    logger.info(f"Config file: {args.config_file}")
    logger.info(f"Number of GPUs: {num_gpus}")
    logger.info(f"Total samples: {args.total_samples}")
    logger.info(f"Samples per GPU: {samples_per_gpu}")
    logger.info(f"Distributed: {distributed}")
    logger.info("=" * 60)

    # Build model
    logger.info("Building model...")
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE if cfg.MODEL.DEVICE != "auto" else "cuda")
    model.to(device)

    # Freeze detector modules (only train relation head)
    eval_modules = (
        model.rpn,
        model.backbone,
        model.roi_heads.box,
    )
    fix_eval_modules(eval_modules)

    trainable, total = count_parameters(model)
    logger.info(f"Model parameters: {trainable/1e6:.2f}M trainable / {total/1e6:.2f}M total")

    # Load pretrained detector weights
    load_mapping = {
        "roi_heads.relation.box_feature_extractor": "roi_heads.box.feature_extractor",
        "roi_heads.relation.rel_pair_box_feature_extractor": "roi_heads.box.feature_extractor",
        "roi_heads.relation.union_feature_extractor.feature_extractor": "roi_heads.box.feature_extractor",
    }

    if cfg.MODEL.PRETRAINED_DETECTOR_CKPT:
        logger.info(f"Loading pretrained detector from: {cfg.MODEL.PRETRAINED_DETECTOR_CKPT}")
        checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
        checkpointer.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT, with_optim=False, load_mapping=load_mapping)

    # Branch based on mode
    if args.mode == "eval":
        return run_eval_test(cfg, model, device, distributed, local_rank, num_gpus, logger, args)

    # Build optimizer
    logger.info("Building optimizer...")
    optimizer = make_optimizer(cfg, model, logger, rl_factor=float(samples_per_gpu))

    # Wrap model with DDP if distributed
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    # Create data loader
    logger.info("Creating data loader...")
    train_data_loader = make_data_loader(
        cfg,
        mode="train",
        is_distributed=distributed,
        start_iter=0,
    )

    # Get one batch (this will be samples_per_gpu images)
    logger.info("Loading batch...")
    data_iter = iter(train_data_loader)
    images, targets, _ = next(data_iter)

    actual_batch_size = len(targets)
    logger.info(f"Batch size on this GPU: {actual_batch_size}")

    # Move data to device
    images = images.to(device)
    targets = [t.to(device) for t in targets]

    # Enable gradient scaler for mixed precision (A100 benefits from this)
    scaler = torch.cuda.amp.GradScaler()

    # Set model to training mode
    model.train()
    fix_eval_modules(eval_modules)  # Re-freeze after .train()

    # Initialize PCE pretraining if needed
    m2opt = model.module if distributed else model
    if hasattr(m2opt.roi_heads.relation, 'predictor'):
        if hasattr(m2opt.roi_heads.relation.predictor, 'start_preclser_relpn_pretrain'):
            m2opt.roi_heads.relation.predictor.start_preclser_relpn_pretrain()
            logger.info("Started PCE pretraining mode")

    # ========== FORWARD PASS ==========
    logger.info("-" * 40)
    logger.info("Starting FORWARD pass...")
    torch.cuda.synchronize()
    forward_start = time.time()

    optimizer.zero_grad()

    with torch.cuda.amp.autocast():
        loss_dict = model(images, targets, logger=logger)
        losses = sum(loss for loss in loss_dict.values())

    torch.cuda.synchronize()
    forward_time = time.time() - forward_start

    # Reduce losses for logging
    loss_dict_reduced = reduce_loss_dict(loss_dict)
    losses_reduced = sum(loss for loss in loss_dict_reduced.values())

    logger.info(f"Forward pass completed in {forward_time:.3f}s")
    logger.info(f"Total loss: {losses_reduced.item():.4f}")
    for name, value in loss_dict_reduced.items():
        logger.info(f"  {name}: {value.item():.4f}")

    # ========== BACKWARD PASS ==========
    logger.info("-" * 40)
    logger.info("Starting BACKWARD pass...")
    torch.cuda.synchronize()
    backward_start = time.time()

    scaler.scale(losses).backward()

    torch.cuda.synchronize()
    backward_time = time.time() - backward_start
    logger.info(f"Backward pass completed in {backward_time:.3f}s")

    # ========== GRADIENT CLIPPING & OPTIMIZER STEP ==========
    logger.info("-" * 40)
    logger.info("Applying gradients...")
    torch.cuda.synchronize()
    optim_start = time.time()

    # Unscale gradients for clipping
    scaler.unscale_(optimizer)

    # Clip gradients
    clip_grad_norm(
        [(n, p) for n, p in model.named_parameters() if p.requires_grad],
        max_norm=cfg.SOLVER.GRAD_NORM_CLIP,
        logger=logger,
        verbose=True,
        clip=True,
    )

    # Apply gradients
    scaler.step(optimizer)
    scaler.update()

    torch.cuda.synchronize()
    optim_time = time.time() - optim_start
    logger.info(f"Gradient application completed in {optim_time:.3f}s")

    # ========== MEMORY STATS ==========
    logger.info("-" * 40)
    logger.info("Memory Statistics:")
    if torch.cuda.is_available():
        max_mem = torch.cuda.max_memory_allocated(device) / 1024**3
        current_mem = torch.cuda.memory_allocated(device) / 1024**3
        logger.info(f"  Peak GPU memory: {max_mem:.2f} GB")
        logger.info(f"  Current GPU memory: {current_mem:.2f} GB")

    # ========== SUMMARY ==========
    logger.info("=" * 60)
    logger.info("TEST PASSED!")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config_file}")
    logger.info(f"GPUs: {num_gpus}")
    logger.info(f"Total samples: {actual_batch_size * num_gpus}")
    logger.info(f"Forward time: {forward_time:.3f}s")
    logger.info(f"Backward time: {backward_time:.3f}s")
    logger.info(f"Optimizer step time: {optim_time:.3f}s")
    logger.info(f"Total time: {forward_time + backward_time + optim_time:.3f}s")
    logger.info(f"Total loss: {losses_reduced.item():.4f}")
    if torch.cuda.is_available():
        logger.info(f"Peak memory per GPU: {max_mem:.2f} GB")
    logger.info("=" * 60)

    # Print memory for shell script parsing
    print(f"GPU_MEMORY_GB:{max_mem:.2f}")

    # Cleanup
    if distributed:
        dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    sys.exit(main())
