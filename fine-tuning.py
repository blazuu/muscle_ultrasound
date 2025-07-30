#!/usr/bin/env python
"""
Easy Fine-tuning Script for Muscle Segmentation Models

This script allows users to easily fine-tune the pre-trained muscle segmentation models
on their own data with minimal setup and configuration.

Usage:
    # Prepare your data in this structure:
    data/
    ├── images/
    │   ├── image1.png
    │   └── image2.png
    └── masks/
        ├── image1.png  (same name as image)
        └── image2.png

    # Run fine-tuning:
    python finetune.py --model Multi --pretrained Multi_weights.pth --data data/ --epochs 50

    # Advanced usage:
    python finetune.py --model BB --pretrained BB_weights.pth --data my_data/ --epochs 100 --lr 0.0001 --batch-size 4 --val-split 0.2

Requirements:
    - PyTorch
    - MMSegmentation
    - Your labeled training data (images + masks)
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import argparse
import json
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
from collections import OrderedDict
from datetime import datetime
import random
from sklearn.model_selection import train_test_split

# MMSegmentation imports
from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.registry import MODELS
import mmseg.models
import mmseg.datasets
import mmseg.datasets.transforms
from mmseg.utils import register_all_modules

import warnings
warnings.filterwarnings('ignore')

class EasyFineTuner:
    def __init__(self, model_name, pretrained_weights, data_dir, output_dir="finetuned_models"):
        """
        Initialize the fine-tuning pipeline
        
        Args:
            model_name: Name of the model (Multi, BB, D, etc.)
            pretrained_weights: Path to pretrained weights
            data_dir: Directory containing images/ and masks/ folders
            output_dir: Directory to save fine-tuned models
        """
        # Register all MMSegmentation modules
        register_all_modules()
        
        self.model_name = model_name
        self.pretrained_weights = pretrained_weights
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Initializing fine-tuning for model: {model_name}")
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Device: {self.device}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate data
        self.validate_data_structure()

    def validate_data_structure(self):
        """Validate that the data directory has the correct structure"""
        images_dir = self.data_dir / "images"
        masks_dir = self.data_dir / "masks"
        
        if not images_dir.exists():
            raise ValueError(f"Images directory not found: {images_dir}")
        if not masks_dir.exists():
            raise ValueError(f"Masks directory not found: {masks_dir}")
        
        # Find image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        images = []
        for ext in image_extensions:
            images.extend(list(images_dir.glob(f"*{ext}")))
            images.extend(list(images_dir.glob(f"*{ext.upper()}")))
        
        if len(images) == 0:
            raise ValueError(f"No images found in {images_dir}")
        
        # Check for corresponding masks
        valid_pairs = 0
        for img_path in images:
            mask_path = masks_dir / img_path.name
            if mask_path.exists():
                valid_pairs += 1
        
        print(f"Found {len(images)} images, {valid_pairs} with corresponding masks")
        
        if valid_pairs == 0:
            raise ValueError("No valid image-mask pairs found. Ensure masks have the same names as images.")
        
        if valid_pairs < len(images):
            print(f"Warning: {len(images) - valid_pairs} images don't have corresponding masks")
        
        self.num_samples = valid_pairs
        return True

    def analyze_dataset(self):
        """Analyze the dataset to understand class distribution"""
        print("Analyzing dataset...")
        
        images_dir = self.data_dir / "images"
        masks_dir = self.data_dir / "masks"
        
        # Find valid pairs
        image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        images = []
        for ext in image_extensions:
            images.extend(list(images_dir.glob(f"*{ext}")))
            images.extend(list(images_dir.glob(f"*{ext.upper()}")))
        
        class_counts = {}
        total_pixels = 0
        
        sample_size = min(50, len(images))  # Analyze subset for speed
        analyzed_images = random.sample(images, sample_size)
        
        for img_path in analyzed_images:
            mask_path = masks_dir / img_path.name
            if not mask_path.exists():
                continue
                
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            
            unique_values, counts = np.unique(mask, return_counts=True)
            total_pixels += mask.size
            
            for val, count in zip(unique_values, counts):
                if val in class_counts:
                    class_counts[val] += count
                else:
                    class_counts[val] = count
        
        print(f"Dataset analysis (from {sample_size} samples):")
        for class_id, count in sorted(class_counts.items()):
            percentage = (count / total_pixels) * 100
            print(f"   Class {class_id}: {count:,} pixels ({percentage:.1f}%)")
        
        # Detect if binary masks need conversion
        if set(class_counts.keys()) == {0, 255}:
            print("Binary masks detected (0, 255) - will convert to (0, 1)")
            self.needs_binary_conversion = True
        elif max(class_counts.keys()) <= 1:
            print("Masks already in correct format (0, 1)")
            self.needs_binary_conversion = False
        else:
            print(f"Multi-class masks detected (classes: {sorted(class_counts.keys())})")
            self.needs_binary_conversion = False
        
        self.class_counts = class_counts
        return class_counts

    def create_dataset_split(self, val_split=0.2, test_split=0.1):
        """Create train/val/test splits and organize data for MMSegmentation"""
        print(f"Creating dataset splits (train: {1-val_split-test_split:.1f}, val: {val_split:.1f}, test: {test_split:.1f})")
        
        images_dir = self.data_dir / "images"
        masks_dir = self.data_dir / "masks"
        
        # Find valid pairs
        image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        valid_pairs = []
        
        for ext in image_extensions:
            for img_path in images_dir.glob(f"*{ext}"):
                mask_path = masks_dir / img_path.name
                if mask_path.exists():
                    valid_pairs.append(img_path.name)
        
        # Create splits
        train_files, temp_files = train_test_split(valid_pairs, test_size=val_split+test_split, random_state=42)
        
        if test_split > 0:
            val_files, test_files = train_test_split(temp_files, test_size=test_split/(val_split+test_split), random_state=42)
        else:
            val_files = temp_files
            test_files = []
        
        print(f"Split sizes: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
        
        # Create MMSegmentation-style dataset structure
        dataset_dir = self.output_dir / "dataset"
        dataset_dir.mkdir(exist_ok=True)
        
        for split, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            if not files:
                continue
                
            split_img_dir = dataset_dir / f"img_dir/{split}"
            split_ann_dir = dataset_dir / f"ann_dir/{split}"
            split_img_dir.mkdir(parents=True, exist_ok=True)
            split_ann_dir.mkdir(parents=True, exist_ok=True)
            
            for filename in files:
                # Copy and potentially convert images and masks
                src_img = images_dir / filename
                src_mask = masks_dir / filename
                dst_img = split_img_dir / filename
                dst_mask = split_ann_dir / filename
                
                # Copy image
                shutil.copy2(src_img, dst_img)
                
                # Copy and potentially convert mask
                mask = cv2.imread(str(src_mask), cv2.IMREAD_GRAYSCALE)
                if self.needs_binary_conversion:
                    mask = (mask > 127).astype(np.uint8)  # Convert 255 to 1
                cv2.imwrite(str(dst_mask), mask)
        
        self.dataset_dir = dataset_dir
        self.splits = {"train": train_files, "val": val_files, "test": test_files}
        return dataset_dir

    def create_training_config(self, epochs=50, learning_rate=0.0001, batch_size=4):
        """Create training configuration based on the model type"""
        print(f"Creating training config (epochs={epochs}, lr={learning_rate}, batch_size={batch_size})")
        
        # Determine number of classes
        if self.model_name == "Multi":
            num_classes = 17
            classes = ['background', 'Biceps_brachii', 'Deltoideus', 'Depressor_anguli_oris', 
                      'Digastricus', 'Gastrocnemius_medial_head', 'Geniohyoideus', 'Masseter',
                      'Mentalis', 'Orbicularis_oris', 'Rectus_abdominis', 'Rectus_femoris',
                      'Temporalis', 'Tibialis_anterior', 'Trapezius', 'Vastus_lateralis', 'Zygomaticus']
        else:
            num_classes = 2
            muscle_names = {
                'BB': 'Biceps_brachii', 'D': 'Deltoideus', 'FCR': 'Flexor_carpi_radialis',
                'FCU': 'Flexor_carpi_ulnaris', 'VL': 'Vastus_lateralis', 'RF': 'Rectus_femoris',
                'GCMM': 'Gastrocnemius_medial_head', 'TA': 'Tibialis_anterior'
            }
            muscle_name = muscle_names.get(self.model_name, 'muscle')
            classes = ['background', muscle_name]
        
        # Save config as a proper Python file using string formatting
        config_path = self.output_dir / f"{self.model_name}_finetune_config.py"
        
        # Create the config file content as a string
        config_content = f'''# Fine-tuning configuration for {self.model_name}
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# Model configuration
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        size=(512, 512),
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255
    ),
    backbone=dict(
        type='SwinTransformer',
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3)
    ),
    decode_head=dict(
        type='IterativeDecodeHead',
        num_stages=3,
        kernel_update_head=[
            dict(
                type='KernelUpdateHead',
                num_classes={num_classes},
                num_ffn_fcs=2,
                num_heads=8,
                num_mask_fcs=1,
                feedforward_channels=2048,
                in_channels=512,
                out_channels=512,
                dropout=0.0,
                conv_kernel_size=1,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                with_ffn=True,
                feat_transform_cfg=dict(conv_cfg=dict(type='Conv2d'), act_cfg=None),
                kernel_updator_cfg=dict(
                    type='KernelUpdator',
                    in_channels=256,
                    feat_channels=256,
                    out_channels=256,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')
                )
            ),
            dict(
                type='KernelUpdateHead',
                num_classes={num_classes},
                num_ffn_fcs=2,
                num_heads=8,
                num_mask_fcs=1,
                feedforward_channels=2048,
                in_channels=512,
                out_channels=512,
                dropout=0.0,
                conv_kernel_size=1,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                with_ffn=True,
                feat_transform_cfg=dict(conv_cfg=dict(type='Conv2d'), act_cfg=None),
                kernel_updator_cfg=dict(
                    type='KernelUpdator',
                    in_channels=256,
                    feat_channels=256,
                    out_channels=256,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')
                )
            ),
            dict(
                type='KernelUpdateHead',
                num_classes={num_classes},
                num_ffn_fcs=2,
                num_heads=8,
                num_mask_fcs=1,
                feedforward_channels=2048,
                in_channels=512,
                out_channels=512,
                dropout=0.0,
                conv_kernel_size=1,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                with_ffn=True,
                feat_transform_cfg=dict(conv_cfg=dict(type='Conv2d'), act_cfg=None),
                kernel_updator_cfg=dict(
                    type='KernelUpdator',
                    in_channels=256,
                    feat_channels=256,
                    out_channels=256,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')
                )
            )
        ],
        kernel_generate_head=dict(
            type='UPerHead',
            in_channels=[192, 384, 768, 1536],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes={num_classes},
            norm_cfg=dict(type='BN', requires_grad=True),
            align_corners=False,
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
        )
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes={num_classes},
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# Dataset configuration
dataset_type = 'BaseSegDataset'
data_root = '{str(self.dataset_dir).replace(chr(92), "/")}'

# Pipeline configuration
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

# Dataloader configuration
train_dataloader = dict(
    batch_size={batch_size},
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='BaseSegDataset',
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),
        img_suffix='.png',         
        seg_map_suffix='.png',     
        pipeline=train_pipeline,
        metainfo=dict(classes={classes})
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BaseSegDataset',
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        img_suffix='.png',         
        seg_map_suffix='.png', 
        pipeline=test_pipeline,
        metainfo=dict(classes={classes})
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BaseSegDataset',
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/test', seg_map_path='ann_dir/test'),
        img_suffix='.png',         
        seg_map_suffix='.png', 
        pipeline=test_pipeline,
        metainfo=dict(classes={classes})
    )
)

# Evaluation configuration
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'])
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'])

# Optimization configuration
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr={learning_rate},
        betas=(0.9, 0.999),
        weight_decay=0.01
    ),
    paramwise_cfg=dict(
        custom_keys=dict(backbone=dict(lr_mult=0.1))  # Lower LR for backbone
    )
)

# Learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end={max(1, epochs // 10)}
    ),
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=0,
        end={epochs}
    )
]

# Training configuration
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs={epochs},
    val_interval={max(1, epochs // 10)}
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Hook configuration
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval={max(1, epochs // 5)},
        max_keep_ckpts=3,
        save_best='mIoU',
        save_last=True
    ),
    sampler_seed=dict(type='DistSamplerSeedHook')
)

# Load pretrained weights
load_from = r'{str(self.pretrained_weights)}'

# Work directory
work_dir = r'{str(self.output_dir / "training_logs")}'
'''
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        print(f"Config saved to: {config_path}")
        return config_path

    def run_training(self, config_path):
        """Run the training process"""
        print("Starting training...")
        
        # Load config
        cfg = Config.fromfile(config_path)
        
        # Create work directory
        work_dir = Path(cfg.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # Create runner and start training
        runner = Runner.from_cfg(cfg)
        runner.train()
        
        print("Training completed!")
        
        # Find best model
        best_model_path = work_dir / "best_mIoU.pth"
        if best_model_path.exists():
            # Copy to main output directory with clear name
            final_model_path = self.output_dir / f"{self.model_name}_finetuned_best.pth"
            shutil.copy2(best_model_path, final_model_path)
            print(f"Best model saved to: {final_model_path}")
            self.run_testing(config_path, str(best_model_path))
            return final_model_path
        else:
            print("Best model not found, check training logs")
            return None

    def run_training(self, config_path):
        """Run the training process"""
        print("Starting training...")
        
        # Load config
        cfg = Config.fromfile(config_path)
        
        # Create work directory
        work_dir = Path(cfg.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # Create runner and start training
        runner = Runner.from_cfg(cfg)
        runner.train()
        
        print("Training completed!")
        
        # Find best model - try different naming patterns
        best_model_path = None
        
        # Try epoch-based naming first
        best_models = list(work_dir.glob("best_mIoU_epoch_*.pth"))
        if best_models:
            # Get the latest epoch
            best_model_path = max(best_models, key=lambda x: int(x.stem.split('_')[-1]))
        else:
            # Try iteration-based naming
            best_models = list(work_dir.glob("best_mIoU_iter_*.pth"))
            if best_models:
                best_model_path = max(best_models, key=lambda x: int(x.stem.split('_')[-1]))
            else:
                # Try simple naming
                simple_best = work_dir / "best_mIoU.pth"
                if simple_best.exists():
                    best_model_path = simple_best
        
        if best_model_path:
            # Copy to main output directory with clear name
            final_model_path = self.output_dir / f"{self.model_name}_finetuned_best.pth"
            shutil.copy2(best_model_path, final_model_path)
            print(f"Best model saved to: {final_model_path}")
            
            # Run testing on held-out test set
            self.run_testing(config_path, str(best_model_path))
            
            return final_model_path
        else:
            print("Best model not found, check training logs")
            # List available checkpoints for debugging
            checkpoints = list(work_dir.glob("*.pth"))
            if checkpoints:
                print(f"   Available checkpoints: {[c.name for c in checkpoints]}")
            return None

    def run_testing(self, config_path, model_path):
        """Run testing on held-out test set"""
        print("\nRunning testing on held-out test set...")
        
        try:
            # Load config for testing
            cfg = Config.fromfile(config_path)
            
            # Create test work directory
            test_work_dir = self.output_dir / "test_results"
            test_work_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if test dataloader exists and has data
            if not hasattr(cfg, 'test_dataloader') or not cfg.test_dataloader:
                print("No test dataloader found in config, skipping testing")
                return
            
            # Check if test dataset directory exists
            test_data_root = cfg.test_dataloader.dataset.data_root
            test_img_path = Path(test_data_root) / cfg.test_dataloader.dataset.data_prefix.img_path
            
            if not test_img_path.exists() or not any(test_img_path.glob("*.png")):
                print("No test images found, skipping testing")
                return
            
            print(f"Test data found: {test_img_path}")
            
            # Create test runner
            cfg.work_dir = str(test_work_dir)
            cfg.load_from = model_path
            
            # Create runner for testing
            runner = Runner.from_cfg(cfg)
            
            # Load the trained model
            runner.load_checkpoint(model_path)
            
            # Run testing
            test_metrics = runner.test()
            
            print("Testing completed!")
            
            # Save test results summary
            if test_metrics:
                self.save_test_results(test_metrics, test_work_dir)
            
        except Exception as e:
            print(f"Error during testing: {str(e)}")
            print("   Training completed successfully, but testing failed")

    def save_test_results(self, test_metrics, test_work_dir):
        """Save test results to file"""
        results_file = test_work_dir / "test_results_summary.txt"
        
        with open(results_file, 'w') as f:
            f.write(f"Test Results Summary - {self.model_name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Test date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if 'aAcc' in test_metrics:
                f.write(f"Overall Accuracy: {test_metrics['aAcc']:.4f}\n")
            if 'mIoU' in test_metrics:
                f.write(f"Mean IoU: {test_metrics['mIoU']:.4f}\n")
            if 'mAcc' in test_metrics:
                f.write(f"Mean Accuracy: {test_metrics['mAcc']:.4f}\n")
            if 'mDice' in test_metrics:
                f.write(f"Mean Dice: {test_metrics['mDice']:.4f}\n")
            
            f.write(f"\nFull metrics:\n")
            for key, value in test_metrics.items():
                if isinstance(value, (int, float)):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        print(f"Test results saved to: {results_file}")
        
        # Print key metrics to console
        print(f"\nTest Results:")
        if 'mIoU' in test_metrics:
            print(f"   Mean IoU: {test_metrics['mIoU']:.4f}")
        if 'aAcc' in test_metrics:
            print(f"   Overall Accuracy: {test_metrics['aAcc']:.4f}")
        if 'mDice' in test_metrics:
            print(f"   Mean Dice: {test_metrics['mDice']:.4f}")

    def create_summary_report(self, final_model_path=None):
        """Create a summary report of the fine-tuning process"""
        report_path = self.output_dir / "finetuning_summary.md"
        
        with open(report_path, 'w') as f:
            f.write(f"# Fine-tuning Summary: {self.model_name}\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Original Model:** {self.model_name}\n")
            f.write(f"**Pretrained Weights:** {self.pretrained_weights}\n")
            f.write(f"**Data Directory:** {self.data_dir}\n\n")
            
            f.write("## Dataset Information\n\n")
            f.write(f"- **Total Samples:** {self.num_samples}\n")
            f.write(f"- **Train Split:** {len(self.splits['train'])} samples\n")
            f.write(f"- **Validation Split:** {len(self.splits['val'])} samples\n")
            f.write(f"- **Test Split:** {len(self.splits['test'])} samples\n\n")
            
            if hasattr(self, 'class_counts'):
                f.write("## Class Distribution\n\n")
                total_pixels = sum(self.class_counts.values())
                for class_id, count in sorted(self.class_counts.items()):
                    percentage = (count / total_pixels) * 100
                    f.write(f"- Class {class_id}: {count:,} pixels ({percentage:.1f}%)\n")
                f.write("\n")
            
            if final_model_path:
                f.write(f"## Output\n\n")
                f.write(f"- **Fine-tuned Model:** {final_model_path}\n")
                f.write(f"- **Training Logs:** {self.output_dir / 'training_logs'}\n")
                f.write(f"- **Dataset:** {self.dataset_dir}\n\n")
            
            f.write("## Usage\n\n")
            f.write("To use your fine-tuned model:\n\n")
            f.write("```bash\n")
            f.write(f"python inference.py --model {self.model_name} --weights {final_model_path.name if final_model_path else 'your_model.pth'} --image your_image.png\n")
            f.write("```\n\n")
        
        print(f"Summary report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Easy Fine-tuning for Muscle Segmentation Models")
    parser.add_argument("--model", required=True,
                       choices=['Multi', 'Binary', 'BB', 'D', 'FCR', 'FCU', 'VL', 'RF', 'GCMM', 'TA'],
                       help="Model to fine-tune")
    parser.add_argument("--pretrained", required=True, help="Path to pretrained weights")
    parser.add_argument("--data", required=True, help="Data directory (should contain images/ and masks/ folders)")
    parser.add_argument("--output", default="finetuned_models", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--test-split", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze dataset without training")
    
    args = parser.parse_args()
    
    print("Easy Fine-tuning for Muscle Segmentation Models")
    print("=" * 60)
    
    try:
        # Initialize fine-tuner
        finetuner = EasyFineTuner(args.model, args.pretrained, args.data, args.output)
        
        # Analyze dataset
        finetuner.analyze_dataset()
        
        if args.analyze_only:
            print("Dataset analysis complete. Use --analyze-only=False to start training.")
            return
        
        # Create dataset splits
        finetuner.create_dataset_split(args.val_split, args.test_split)
        
        # Create training config
        config_path = finetuner.create_training_config(args.epochs, args.lr, args.batch_size)
        
        # Run training
        final_model_path = finetuner.run_training(config_path)
        
        # Create summary
        finetuner.create_summary_report(final_model_path)
        
        print("\nFine-tuning completed successfully!")
        print(f"Check output directory: {args.output}")
        
    except Exception as e:
        print(f"Error during fine-tuning: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()