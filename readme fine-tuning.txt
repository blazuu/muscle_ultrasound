# Easy Fine-tuning Guide

This guide shows you how to fine-tune the muscle segmentation models on your own data.

## 📋 Quick Start

## Installation

### Requirements
```bash
pip install torch torchvision
pip install openmim
mim install mmengine
mim install "mmcv>=2.0.0"  
mim install "mmsegmentation>=1.0.0"
pip install opencv-python matplotlib numpy
```

### Quick Setup
1. Download the model weights and this script
2. Install dependencies (see above)

### 1. Prepare Your Data
Organize your data in this structure:
```
your_data/
├── images/
│   ├── ultrasound001.png
│   ├── ultrasound002.png
│   └── ultrasound003.png
└── masks/
    ├── ultrasound001.png  (same filename as image)
    ├── ultrasound002.png
    └── ultrasound003.png
```

**Important:**
- ✅ **Image names must match exactly** between images/ and masks/ folders
- ✅ **Supported formats**: PNG, JPG, JPEG, TIFF, BMP
- ✅ **Mask values**: 
  - Binary models: 0 (background), 1 (muscle) or 0 (background), 255 (muscle)
  - Multi-class: 0 (background), 1-16 (different muscles)

### 2. Run Fine-tuning
```bash
# Basic fine-tuning (recommended)
python finetune.py --model Multi --pretrained Multi_weights.pth --data your_data/ --epochs 50 --output Multi

# Muscle-specific fine-tuning
python finetune.py --model BB --pretrained BB_weights.pth --data your_data/ --epochs 50 --output BB

# Advanced fine-tuning with custom parameters
python finetune.py --model Multi --pretrained Multi_weights.pth --data your_data/ \
    --epochs 100 --lr 0.0001 --batch-size 4 --val-split 0.2 --output my_finetuned_models/
```

### 3. Use Your Fine-tuned Model
```bash
# After fine-tuning completes, use your new model:
python inference.py --model Multi --weights finetuned_models/Multi_finetuned_best.pth --image test.png
```

## 🔧 Parameters

| Parameter | Description | Default | Examples |
|-----------|-------------|---------|----------|
| `--model` | Model type to fine-tune | Required | Multi, BB, D, FCR, VL, RF, TA |
| `--pretrained` | Path to pretrained weights | Required | Multi_weights.pth |
| `--data` | Your data directory | Required | my_data/, ./ultrasound_data/ |
| `--epochs` | Number of training epochs | 50 | 10 (quick), 100 (thorough) |
| `--lr` | Learning rate | 0.0001 | 0.001 (faster), 0.00001 (slower) |
| `--batch-size` | Training batch size | 4 | 2 (less GPU memory), 8 (more GPU memory) |
| `--val-split` | Validation split ratio | 0.2 | 0.1 (more training), 0.3 (more validation) |
| `--test-split` | Test split ratio | 0.1 | 0.0 (no test set), 0.2 (larger test set) |
| `--output` | Output directory | finetuned_models | my_models/, ./results/ |

## 🎯 Use Cases

### Case 1: Different Ultrasound Machine
You have data from a different ultrasound machine with different image characteristics.

```bash
# Fine-tune with moderate epochs to adapt to new machine settings
python finetune.py --model Multi --pretrained Multi_weights.pth --data new_machine_data/ --epochs 75
```

### Case 2: New Patient Population
Your patient population has different characteristics (age, BMI, pathology).

```bash
# Fine-tune with more epochs for better adaptation
python finetune.py --model Multi --pretrained Multi_weights.pth --data patient_data/ --epochs 100 --lr 0.00005
```

### Case 3: Small Dataset
You have limited data (< 100 images).

```bash
# Use lower learning rate and fewer epochs to avoid overfitting
python finetune.py --model BB --pretrained BB_weights.pth --data small_dataset/ \
    --epochs 30 --lr 0.00001 --val-split 0.15
```

### Case 4: Large Dataset
You have lots of data (> 1000 images).

```bash
# Can use larger batch size and more aggressive training
python finetune.py --model Multi --pretrained Multi_weights.pth --data large_dataset/ \
    --epochs 100 --batch-size 8 --lr 0.0002
```

### Case 5: Specific Muscle Focus
You only care about one specific muscle.

```bash
# Use muscle-specific model for better accuracy
python finetune.py --model BB --pretrained BB_weights.pth --data biceps_data/ --epochs 80
```

## 📊 Understanding the Process

### What Happens During Fine-tuning:

1. **📂 Data Analysis**: Script analyzes your dataset and reports class distribution
2. **🔄 Data Splitting**: Automatically splits your data into train/validation/test sets
3. **⚙️ Config Creation**: Creates optimized training configuration
4. **🚀 Training**: Runs the fine-tuning process with progress monitoring
5. **💾 Model Saving**: Saves the best model based on validation performance
6. **📋 Summary Report**: Generates a detailed summary of the process

### Output Files:
```
finetuned_models/
├── Multi_finetuned_best.pth           # Your fine-tuned model (use this!)
├── Multi_finetune_config.py           # Training configuration used
├── finetuning_summary.md              # Detailed report
├── dataset/                           # Organized dataset for training
│   ├── img_dir/
│   └── ann_dir/
└── training_logs/                     # Full training logs and checkpoints
    ├── vis_data/
    └── 20241201_120000/
```


### Common Issues and Solutions

#### "No valid image-mask pairs found"
- **Problem**: Filenames don't match between images/ and masks/ folders
- **Solution**: Ensure exact filename matching (including extensions)

#### "CUDA out of memory"
- **Problem**: Batch size too large for your GPU
- **Solution**: Reduce `--batch-size` to 2 or 1

#### "Loss not decreasing"
- **Problem**: Learning rate might be too high or too low
- **Solution**: Try different learning rates (0.001, 0.0001, 0.00001)

#### "Model overfitting"
- **Problem**: Training accuracy high but validation accuracy low
- **Solution**: Reduce epochs, increase validation split, or add more data

#### "Poor performance after fine-tuning"
- **Problem**: Too aggressive fine-tuning or domain mismatch
- **Solution**: Use lower learning rate or try starting from original model

## 🧪 Advanced Usage

### Dataset Analysis Only
First check your data quality without training:
```bash
python finetune.py --model Multi --pretrained Multi_weights.pth --data your_data/ --analyze-only
```

### Custom Data Splits
Control exactly how data is split:
```bash
python finetune.py --model Multi --pretrained Multi_weights.pth --data your_data/ \
    --val-split 0.15 --test-split 0.15  # 70% train, 15% val, 15% test
```

### Multiple Models
Fine-tune different models for comparison:
```bash
# Fine-tune multi-class model
python finetune.py --model Multi --pretrained Multi_weights.pth --data your_data/ --output multi_results/

# Fine-tune muscle-specific model
python finetune.py --model BB --pretrained BB_weights.pth --data your_data/ --output bb_results/
```

## 📈 Monitoring Training

The script provides real-time feedback:
- **📊 Dataset analysis**: Class distribution and data quality checks
- **🔄 Training progress**: Loss and accuracy metrics
- **💾 Automatic saving**: Best model saved based on validation performance
- **📋 Final report**: Comprehensive summary of results

Check the `training_logs/` directory for detailed logs and intermediate checkpoints.

## 🚀 Next Steps

After successful fine-tuning:

1. **Test your model**: Use the inference script with your fine-tuned weights
2. **Compare performance**: Test on held-out data to verify improvement
3. **Share results**: Document what worked for your specific domain
4. **Iterate**: Fine-tune further if needed with different parameters
