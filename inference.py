#!/usr/bin/env python
"""
Standalone Inference Script for Marzola et al. 2025 Muscle Segmentation Models fine-tuned by Michał Błaż

This script allows users to run inference on muscle ultrasound images using 
pre-trained models from the paper "Machine learning-driven Heckmatt grading 
in facioscapulohumeral muscular dystrophy"

Usage:
    python inference.py --model Multi --image path/to/image.png
    python inference.py --model BB --image path/to/image.png --output results/
    python inference.py --model Multi --batch path/to/images/ --output results/

Requirements:
    - PyTorch
    - MMSegmentation
    - OpenCV
    - NumPy
    - Matplotlib
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from collections import OrderedDict

# MMSegmentation imports
from mmengine.config import Config
from mmseg.registry import MODELS
import mmseg.models
import mmseg.datasets
import mmseg.datasets.transforms
from mmseg.utils import register_all_modules

import warnings
warnings.filterwarnings('ignore')

class MuscleSegmentationInference:
    def __init__(self, model_name, weights_path, config_path=None):
        """
        Initialize the muscle segmentation inference pipeline
        
        Args:
            model_name: Name of the model (Multi, Binary, BB, D, FCR, etc.)
            weights_path: Path to the model weights (.pth file)
            config_path: Path to config file (optional, will use default if not provided)
        """
        # Register all MMSegmentation modules
        register_all_modules()
        
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.model, self.cfg = self.load_model(weights_path, config_path)
        
        # Define muscle information
        self.muscle_info = {
            'Multi': {
                'type': 'multiclass',
                'classes': ['background', 'Biceps_brachii', 'Deltoideus', 'Depressor_anguli_oris', 
                           'Digastricus', 'Gastrocnemius_medial_head', 'Geniohyoideus', 'Masseter',
                           'Mentalis', 'Orbicularis_oris', 'Rectus_abdominis', 'Rectus_femoris',
                           'Temporalis', 'Tibialis_anterior', 'Trapezius', 'Vastus_lateralis', 'Zygomaticus'],
                'description': 'Multi-class segmentation for 16 different muscles'
            },
            'Binary': {
                'type': 'binary',
                'classes': ['background', 'muscle'],
                'description': 'Binary segmentation (muscle vs background)'
            },
            'BB': {
                'type': 'muscle_specific',
                'classes': ['background', 'Biceps_brachii'],
                'description': 'Biceps brachii specific segmentation'
            },
            'D': {
                'type': 'muscle_specific',
                'classes': ['background', 'Deltoideus'],
                'description': 'Deltoideus specific segmentation'
            },
            'FCR': {
                'type': 'muscle_specific',
                'classes': ['background', 'Flexor_carpi_radialis'],
                'description': 'Flexor carpi radialis specific segmentation'
            },
            'FCU': {
                'type': 'muscle_specific',
                'classes': ['background', 'Flexor_carpi_ulnaris'],
                'description': 'Flexor carpi ulnaris specific segmentation'
            },
            'VL': {
                'type': 'muscle_specific',
                'classes': ['background', 'Vastus_lateralis'],
                'description': 'Vastus lateralis specific segmentation'
            },
            'RF': {
                'type': 'muscle_specific',
                'classes': ['background', 'Rectus_femoris'],
                'description': 'Rectus femoris specific segmentation'
            },
            'GCMM': {
                'type': 'muscle_specific',
                'classes': ['background', 'Gastrocnemius_medial_head'],
                'description': 'Gastrocnemius medial head specific segmentation'
            },
            'TA': {
                'type': 'muscle_specific',
                'classes': ['background', 'Tibialis_anterior'],
                'description': 'Tibialis anterior specific segmentation'
            }
        }
        
        if model_name in self.muscle_info:
            print(f"Model: {model_name} - {self.muscle_info[model_name]['description']}")
        else:
            print(f"Warning: Unknown model {model_name}, using default settings")

    def create_default_config(self, model_name):
        """Create a default configuration for the model"""
        # This is a minimal config that should work for most models
        # Users should ideally provide the full config file
        config = {
            'model': {
                'type': 'EncoderDecoder',
                'data_preprocessor': {
                    'type': 'SegDataPreProcessor',
                    'mean': [127.5, 127.5, 127.5],
                    'std': [127.5, 127.5, 127.5],
                    'size': (512, 512),
                    'bgr_to_rgb': True,
                    'pad_val': 0,
                    'seg_pad_val': 255
                },
                'backbone': {
                    'type': 'SwinTransformer',
                    'embed_dims': 192,
                    'depths': [2, 2, 18, 2],
                    'num_heads': [6, 12, 24, 48],
                    'window_size': 7,
                    'mlp_ratio': 4,
                    'qkv_bias': True,
                    'qk_scale': None,
                    'drop_rate': 0.,
                    'attn_drop_rate': 0.,
                    'drop_path_rate': 0.3,
                    'use_abs_pos_embed': False,
                    'patch_norm': True,
                    'out_indices': (0, 1, 2, 3)
                },
                'decode_head': {
                    'type': 'IterativeDecodeHead',
                    'num_stages': 3,
                    'kernel_update_head': [
                        {
                            'type': 'KernelUpdateHead',
                            'num_classes': 17 if model_name == 'Multi' else 2,
                            'num_ffn_fcs': 2,
                            'num_heads': 8,
                            'num_mask_fcs': 1,
                            'feedforward_channels': 2048,
                            'in_channels': 512,
                            'out_channels': 512,
                            'dropout': 0.0,
                            'conv_kernel_size': 1,
                            'ffn_act_cfg': {'type': 'ReLU', 'inplace': True},
                            'with_ffn': True,
                            'feat_transform_cfg': {'conv_cfg': {'type': 'Conv2d'}, 'act_cfg': None},
                            'kernel_updator_cfg': {
                                'type': 'KernelUpdator',
                                'in_channels': 256,
                                'feat_channels': 256,
                                'out_channels': 256,
                                'act_cfg': {'type': 'ReLU', 'inplace': True},
                                'norm_cfg': {'type': 'LN'}
                            }
                        } for _ in range(3)
                    ],
                    'kernel_generate_head': {
                        'type': 'UPerHead',
                        'in_channels': [192, 384, 768, 1536],
                        'in_index': [0, 1, 2, 3],
                        'pool_scales': (1, 2, 3, 6),
                        'channels': 512,
                        'dropout_ratio': 0.1,
                        'num_classes': 17 if model_name == 'Multi' else 2,
                        'norm_cfg': {'type': 'BN', 'requires_grad': True},
                        'align_corners': False,
                        'loss_decode': {'type': 'CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 1.0}
                    }
                },
                'auxiliary_head': {
                    'type': 'FCNHead',
                    'in_channels': 768,
                    'in_index': 2,
                    'channels': 256,
                    'num_convs': 1,
                    'concat_input': False,
                    'dropout_ratio': 0.1,
                    'num_classes': 17 if model_name == 'Multi' else 2,
                    'norm_cfg': {'type': 'BN', 'requires_grad': True},
                    'align_corners': False,
                    'loss_decode': {'type': 'CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 0.4}
                },
                'train_cfg': {},
                'test_cfg': {'mode': 'whole'}
            }
        }
        return config

    def load_model(self, weights_path, config_path=None):
        """Load model with weights"""
        try:
            print(f"Loading model: {self.model_name}")
            
            # Load config
            if config_path and os.path.exists(config_path):
                print(f"Using provided config: {config_path}")
                cfg = Config.fromfile(config_path)
            else:
                print("Using default config (recommended to provide full config file)")
                cfg_dict = self.create_default_config(self.model_name)
                cfg = Config(cfg_dict)
            
            # Build model
            model = MODELS.build(cfg.model)
            
            # Load weights
            print(f"Loading weights from: {weights_path}")
            weights = torch.load(weights_path, map_location='cpu')
            
            # Handle different weight formats
            if isinstance(weights, dict):
                if 'state_dict' in weights:
                    weights = weights['state_dict']
                elif 'model' in weights:
                    weights = weights['model']
            
            # Remove module prefix if present
            clean_weights = OrderedDict()
            for k, v in weights.items():
                if k.startswith('module.'):
                    clean_weights[k[7:]] = v
                else:
                    clean_weights[k] = v
            
            # Load weights into model
            missing_keys, unexpected_keys = model.load_state_dict(clean_weights, strict=False)
            
            if missing_keys:
                print(f"Warning: {len(missing_keys)} missing keys")
            if unexpected_keys:
                print(f"Warning: {len(unexpected_keys)} unexpected keys")
            
            model.to(self.device)
            model.eval()
            
            print(f"Model loaded successfully")
            return model, cfg
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise e

    def preprocess_image(self, image_path):
        """Load and preprocess image for inference"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_size = image.shape[:2]
            
            # Resize to 512x512
            image = cv2.resize(image, (512, 512))
            
            # Convert to tensor and normalize
            image_tensor = torch.from_numpy(image).float()
            image_tensor = image_tensor.permute(2, 0, 1)  # HWC to CHW
            
            # Normalize to [-1, 1] range
            image_tensor = (image_tensor / 127.5) - 1.0
            
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)
            
            return image_tensor.to(self.device), image, original_size
            
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {str(e)}")
            return None, None, None

    def run_inference(self, image_tensor):
        """Run model inference"""
        try:
            with torch.no_grad():
                # Run inference
                outputs = self.model(image_tensor)
                
                # Handle different output formats
                if isinstance(outputs, dict):
                    if 'pred_sem_seg' in outputs:
                        pred = outputs['pred_sem_seg']
                    elif 'decode_head' in outputs:
                        pred = outputs['decode_head']
                    else:
                        pred = list(outputs.values())[0]
                elif isinstance(outputs, (list, tuple)):
                    pred = outputs[0]
                else:
                    pred = outputs
                
                # Get prediction mask
                if pred.dim() == 4:  # [B, C, H, W]
                    pred = pred.squeeze(0)  # Remove batch dimension
                
                if pred.dim() == 3:  # [C, H, W]
                    pred_mask = torch.argmax(pred, dim=0)  # Get class predictions
                else:  # [H, W]
                    pred_mask = pred
                
                # Resize to 512x512 if needed
                if pred_mask.shape != (512, 512):
                    pred_mask_tensor = pred_mask.unsqueeze(0).unsqueeze(0).float()
                    pred_mask_tensor = F.interpolate(pred_mask_tensor, size=(512, 512), mode='nearest')
                    pred_mask = pred_mask_tensor.squeeze(0).squeeze(0)
                
                pred_mask = pred_mask.cpu().numpy()
                
                return pred_mask
                
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            return None

    def postprocess_mask(self, pred_mask, original_size):
        """Resize mask back to original image size if needed"""
        if original_size != (512, 512):
            pred_mask = cv2.resize(pred_mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
        return pred_mask

    def visualize_result(self, image, pred_mask, save_path=None, show=True):
        """Visualize segmentation result"""
        try:
            # Ensure mask is integer type
            pred_mask = pred_mask.astype(np.int32)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(image)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Prediction mask
            axes[1].imshow(pred_mask, cmap='tab10', vmin=0, vmax=max(pred_mask.max(), 1))
            axes[1].set_title(f'Segmentation - {self.model_name}')
            axes[1].axis('off')
            
            # Overlay
            overlay = image.copy().astype(np.uint8)
            if pred_mask.max() > 0:
                # Create colored mask - normalize prediction mask for colormap
                normalized_mask = pred_mask.astype(np.float32) / max(pred_mask.max(), 1)
                colored_mask = plt.cm.tab10(normalized_mask)[:, :, :3]
                colored_mask = (colored_mask * 255).astype(np.uint8)
                
                # Only overlay non-background pixels
                mask_overlay = np.zeros_like(overlay)
                non_bg = pred_mask > 0
                mask_overlay[non_bg] = colored_mask[non_bg]
                
                overlay = cv2.addWeighted(overlay, 0.7, mask_overlay, 0.3, 0)
            
            axes[2].imshow(overlay)
            axes[2].set_title('Overlay')
            axes[2].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Result saved: {save_path}")
            
            if show:
                plt.show()
            else:
                plt.close()
            
            # Print prediction info
            unique_classes = np.unique(pred_mask)
            print(f"Predicted classes: {unique_classes}")
            
            if self.model_name in self.muscle_info:
                classes = self.muscle_info[self.model_name]['classes']
                for cls_id in unique_classes:
                    cls_id_int = int(cls_id)  # Ensure integer indexing
                    if cls_id_int < len(classes):
                        print(f"  Class {cls_id_int}: {classes[cls_id_int]}")
            
            return True
            
        except Exception as e:
            print(f"Error visualizing result: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def predict_single(self, image_path, output_path=None, show=True):
        """Run inference on a single image"""
        print(f"\nProcessing: {image_path}")
        
        # Preprocess
        image_tensor, image, original_size = self.preprocess_image(image_path)
        if image_tensor is None:
            return None
        
        # Run inference
        pred_mask = self.run_inference(image_tensor)
        if pred_mask is None:
            return None
        
        # Postprocess
        pred_mask = self.postprocess_mask(pred_mask, original_size)
        
        # Visualize
        if output_path:
            self.visualize_result(image, pred_mask, output_path, show)
        else:
            self.visualize_result(image, pred_mask, show=show)
        
        return pred_mask

    def predict_batch(self, input_dir, output_dir=None, show=False):
        """Run inference on a directory of images"""
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"Error: Input directory does not exist: {input_dir}")
            return
        
        # Find images
        image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        images = []
        for ext in image_extensions:
            images.extend(input_path.glob(f"*{ext}"))
            images.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not images:
            print(f"No images found in {input_dir}")
            return
        
        print(f"Found {len(images)} images to process")
        
        # Create output directory
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Process each image
        for i, img_path in enumerate(images):
            print(f"\nProcessing {i+1}/{len(images)}: {img_path.name}")
            
            output_file = None
            if output_dir:
                output_file = output_path / f"{img_path.stem}_{self.model_name}_result.png"
            
            self.predict_single(str(img_path), str(output_file) if output_file else None, show)

def main():
    parser = argparse.ArgumentParser(description="Muscle Segmentation Inference")
    parser.add_argument("--model", required=True, 
                       choices=['Multi', 'Binary', 'BB', 'D', 'FCR', 'FCU', 'VL', 'RF', 'GCMM', 'TA'],
                       help="Model to use for inference")
    parser.add_argument("--weights", required=True, help="Path to model weights (.pth file)")
    parser.add_argument("--config", help="Path to model config file (optional)")
    parser.add_argument("--image", help="Path to single image for inference")
    parser.add_argument("--batch", help="Path to directory containing images")
    parser.add_argument("--output", help="Output directory for results")
    parser.add_argument("--no-show", action="store_true", help="Don't display results")
    
    args = parser.parse_args()
    
    if not args.image and not args.batch:
        print("Error: Must specify either --image or --batch")
        return
    
    # Initialize inference pipeline
    try:
        pipeline = MuscleSegmentationInference(args.model, args.weights, args.config)
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        return
    
    # Run inference
    if args.image:
        pipeline.predict_single(args.image, args.output, not args.no_show)
    elif args.batch:
        pipeline.predict_batch(args.batch, args.output, not args.no_show)

if __name__ == "__main__":
    main()