# Fine-Tuned Muscle Segmentation Models

Easy-to-use scripts for fine-tuning and running inference on muscle ultrasound segmentation models, based on the research from Marzola et al. 2025 - "Machine learning-driven Heckmatt grading in facioscapulohumeral muscular dystrophy.", adapted by Błaż et al. (2025) - "Evaluating machine learning muscle ultrasound models on external data – is this the way to propagate forward?"

## 🚀 Quick Start

### 1. Installation

```bash
# Install PyTorch (follow official instructions for your system)
pip install torch torchvision

# Install MMSegmentation
pip install openmim
mim install mmengine
mim install "mmcv>=2.0.0"  
mim install "mmsegmentation>=1.0.0"

# Install other dependencies
pip install opencv-python matplotlib numpy scikit-learn
```

If there is trouble with versions of the packages, I recommend these exact versions:

```bash
conda create --name mmseg_finetune python=3.9 -y
conda activate mmseg_finetune

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu

pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0,<2.2.0"
pip install "mmsegmentation>=1.0.0"
pip install mmpretrain

pip install ftfy xgboost shap pandas seaborn scikit-learn opencv-python Pillow openpyxl xlrd tqdm matplotlib numpy
```

### 2. Download Model Weights

Download the pre-trained model weights from the [Mendeley Dataset](Błaż, Michał (2025), “Evaluating machine learning muscle ultrasound models on external data”, Mendeley Data, V2, doi: 10.17632/7bcsvh6nfv.2):
- `Multi_weights.pth` - Multi-class model (17 muscle classes)
- `BB_weights.pth` - Biceps Brachii specific model
- `VL_weights.pth` - Vastus Lateralis specific model
- ... etc. - other muscle-specific models (D, FCR, FCU, RF, GCMM, TA)
- Muscle names: BB - biceps brachii, D - deltoid, FCR - flexor carpi radialis, FCU - flexor carpi ulnaris, RF - rectus femoris, GCMM - gastrocnemius medial head, TA - tibialis anterior, VL - vastus lateralis

### 3. Quick Usage

#### Run Inference on Ultrasound Images
```bash
# Single image inference
python inference.py --model Multi --weights Multi_weights.pth --image your_ultrasound.png

# Batch processing
python inference.py --model Multi --weights Multi_weights.pth --batch ./images/ --output ./results/
```

#### Fine-tune on Your Own Data
```bash
# Prepare your data structure:
# your_data/
# ├── images/
# │   ├── ultrasound001.png
# │   └── ultrasound002.png
# └── masks/
#     ├── ultrasound001.png  (same filename as image)
#     └── ultrasound002.png

# Run fine-tuning
python fine-tuning.py --model Multi --pretrained Multi_weights.pth --data your_data/ --epochs 50
```

## 📁 Repository Contents

### Core Scripts
- **`inference.py`** - Standalone inference script for processing ultrasound images
- **`fine-tuning.py`** - Easy fine-tuning pipeline for adapting models to your data

### Documentation
- **`inference.txt`** - Detailed inference guide and troubleshooting
- **`fine-tuning.txt`** - Comprehensive fine-tuning documentation

## 🔧 Fine-Tuning Features

### Automated Pipeline
- **📊 Dataset Analysis**: Analyzes class distribution and data quality
- **🔄 Data Splitting**: Automatic train/validation/test splits
- **⚙️ Config Creation**: Generates optimized training configurations
- **🚀 Training**: Progress monitoring with real-time metrics
- **💾 Model Saving**: Automatically saves best performing model
- **📋 Reporting**: Detailed training summary and statistics

### Supported Models
- **Multi** - 17-class muscle segmentation (Background + 16 muscles)
- **Binary** - Binary muscle/background segmentation
- **Muscle-specific models**: BB, D, FCR, FCU, VL, RF, GCMM, TA

### Example Commands

```bash
# Basic fine-tuning
python fine-tuning.py --model Multi --pretrained Multi_weights.pth --data your_data/ --epochs 50

# Advanced fine-tuning with custom parameters
python fine-tuning.py --model Multi --pretrained Multi_weights.pth --data your_data/ \
    --epochs 100 --lr 0.0001 --batch-size 4 --val-split 0.2 --output my_models/

# Dataset analysis only (no training)
python fine-tuning.py --model Multi --pretrained Multi_weights.pth --data your_data/ --analyze-only
```

## 🔍 Inference Features

### Multi-Model Support
- **Multi Model**: 17 classes including Biceps brachii, Rectus femoris, Tibialis anterior, etc. (note - this model works best on muscles present in the dataset used to fine-tune the original model ie. BB, D, VL, RF, TA, GCMM). If your dataset contains 16 muscles present in the original dataset, I recomennd reproducing and using the original model).
- **Binary/Muscle-specific**: Target muscle vs background classification

### Processing Options
- **Single Image**: Process individual ultrasound images
- **Batch Processing**: Analyze entire directories of images
- **Visualization**: Generated overlays showing segmentation results

### Example Commands

```bash
# Quick inference
python inference.py --model Multi --weights Multi_weights.pth --image ultrasound.png

# Batch processing with custom output
python inference.py --model BB --weights BB_finetuned_best.pth --batch ./images/ --output ./results/

# High-quality inference with config
python inference.py --model Multi --weights Multi_weights.pth --config Multi_config.py --image test.png
```

## 📊 Output Structure

### Fine-tuning Outputs
```
finetuned_models/
├── Multi_finetuned_best.pth           # Fine-tuned model (ready to use!)
├── Multi_finetune_config.py           # Training configuration
├── finetuning_summary.md              # Detailed training report
├── dataset/                           # Organized training data
└── training_logs/                     # Complete training history
```

### Inference Outputs
- **Segmentation masks**: Pixel-wise muscle classifications
- **Visualizations**: Original image, mask, and overlay
- **Class predictions**: Lists of detected muscle types

## 🎯 Use Cases

### Research Use Only

## 🛠️ Parameters Reference

### Fine-tuning Parameters
| Parameter | Description | Default | Examples |
|-----------|-------------|---------|----------|
| `--model` | Model type | Required | Multi, BB, VL, RF |
| `--pretrained` | Pre-trained weights path | Required | Multi_weights.pth |
| `--data` | Data directory | Required | ./my_data/ |
| `--epochs` | Training epochs | 50 | 10 (quick), 100 (thorough) |
| `--lr` | Learning rate | 0.0001 | 0.001, 0.00001 |
| `--batch-size` | Batch size | 4 | 2 (less memory), 8 (more memory) |

### Inference Parameters
| Parameter | Description | Examples |
|-----------|-------------|----------|
| `--model` | Model type | Multi, BB, VL |
| `--weights` | Model weights path | Multi_weights.pth |
| `--image` | Single image path | ultrasound.png |
| `--batch` | Image directory | ./images/ |
| `--output` | Output directory | ./results/ |

## 📚 Detailed Documentation

For comprehensive guides, troubleshooting, and advanced usage:
- Read `fine-tuning.txt` for detailed fine-tuning instructions
- Read `inference.txt` for complete inference documentation

## 🔗 Related Resources

- **Original Paper**: [Marzola et al. 2025 - Clinical Neurophysiology](https://doi.org/10.1016/j.clinph.2025.01.016)
- **Letter to the Editor** that describes the idea behind the adaptations of the original work, by the Author of this repository: [Błaż et al. 2025 - Clinical Neurophysiology](https://doi.org/10.1016/j.clinph.2025.2110968)
- **Model Weights & Configs**: [Mendeley Dataset](https://doi.org/10.17632/7bcsvh6nfv.2)
- **MMSegmentation Framework**: [GitHub Repository](https://github.com/open-mmlab/mmsegmentation)

## 📄 Citation

If you use these models or scripts, please cite:

```bibtex
@article{MARZOLA2025,
title = {Machine learning-driven Heckmatt grading in facioscapulohumeral muscular dystrophy: A novel pathway for musculoskeletal ultrasound analysis},
journal = {Clinical Neurophysiology},
year = {2025},
issn = {1388-2457},
doi = {https://doi.org/10.1016/j.clinph.2025.01.016},
url = {https://www.sciencedirect.com/science/article/pii/S1388245725000367},
author = {Francesco Marzola and Nens {van Alfen} and Jonne Doorduin and Kristen Mariko Meiburger}
}
```
```bibtex
@article{BLAZ2025,
title = {Evaluating machine learning muscle ultrasound models on external data – is this the way to propagate forward?},
journal = {Clinical Neurophysiology},
year = {2025},
pages = {2110968},
issn = {1388-2457},
doi = {https://doi.org/10.1016/j.clinph.2025.2110968},
url = {https://www.sciencedirect.com/science/article/pii/S138824572500820X},
author = {Michał Błaż and Monika Ostrowska and Agnieszka Kułaga and Ewa Maludzińska and Michał Michalski},
keywords = {Muscle ultrasound; Neuromuscular disorders; Image segmentation; Machine learning}
}
```
## ⚠️ System Requirements

- **Python**: ≥ 3.8
- **PyTorch**: ≥ 1.9.0
- **CUDA**: Optional (models work on CPU)
- **Memory**: 4GB+ RAM recommended
- **Storage**: 500MB+ for model weights

## 🆘 Common Issues

### Installation Problems
- **MMSegmentation errors**: Follow the official installation guide
- **CUDA issues**: Models automatically fall back to CPU

### Data Problems  
- **"No valid pairs found"**: Ensure image and mask filenames match exactly
- **"CUDA out of memory"**: Reduce batch size with `--batch-size 2`

### Performance Issues
- **Poor results**: Try lower learning rates or more training data
- **Overfitting**: Increase validation split or reduce epochs

For detailed troubleshooting, see the documentation files included in this repository.

---

**Ready to get started?** Download the model weights and try the quick start examples above!
