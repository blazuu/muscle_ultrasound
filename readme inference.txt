# Muscle Ultrasound Segmentation - Inference Tool

This tool provides easy-to-use inference for the muscle segmentation models from:

**"Machine learning-driven Heckmatt grading in facioscapulohumeral muscular dystrophy: A novel pathway for musculoskeletal ultrasound analysis"** by Marzola et al. (2025)

modified by Michał Błaż

## Available Models

- **Multi**: Multi-class segmentation for 16 different muscles
 
- **BB**: Biceps brachii specific segmentation
- **D**: Deltoideus specific segmentation
- **FCR**: Flexor carpi radialis specific segmentation
- **FCU**: Flexor carpi ulnaris specific segmentation
- **VL**: Vastus lateralis specific segmentation
- **RF**: Rectus femoris specific segmentation
- **GCMM**: Gastrocnemius medial head specific segmentation
- **TA**: Tibialis anterior specific segmentation

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
3. Run inference!

## Usage

### Single Image Inference
```bash
# Multi-class segmentation
python inference.py --model Multi --weights Multi_weights.pth --image ultrasound.png

# Muscle-specific segmentation
python inference.py --model BB --weights BB_weights.pth --image biceps_image.png --output results/
```

### Batch Processing
```bash
# Process all images in a directory
python inference.py --model Multi --weights Multi_weights.pth --batch images/ --output results/
```

### Advanced Usage
```bash
# Use custom config file (recommended for best results)
python inference.py --model Multi --weights Multi_weights.pth --config Multi_config.py --image test.png

# Process without showing results (for batch processing)
python inference.py --model Multi --weights Multi_weights.pth --batch images/ --output results/ --no-show
```

## Arguments

- `--model`: Model type (Multi, Binary, BB, D, FCR, FCU, VL, RF, GCMM, TA)
- `--weights`: Path to model weights file (.pth)
- `--config`: Path to model config file (.py) - optional but recommended
- `--image`: Single image path for inference
- `--batch`: Directory containing images for batch processing
- `--output`: Output directory for saving results
- `--no-show`: Don't display results (useful for batch processing)

## Input Requirements

- **Image format**: PNG, JPG, JPEG, TIFF, BMP
- **Recommended size**: 512×512 pixels (images will be automatically resized)
- **Image type**: Grayscale or RGB ultrasound images
- **Acquisition**: B-mode ultrasound images of muscles

## Output

The script generates:
1. **Visualization**: Original image, segmentation mask, and overlay
2. **Segmentation mask**: Pixel-wise class predictions
3. **Class information**: Lists of detected muscle classes

### Output Classes

**Multi Model (17 classes):**
- 0: Background
- 1: Biceps brachii
- 2: Deltoideus  
- 3: Depressor anguli oris
- 4: Digastricus
- 5: Gastrocnemius medial head
- 6: Geniohyoideus
- 7: Masseter
- 8: Mentalis
- 9: Orbicularis oris
- 10: Rectus abdominis
- 11: Rectus femoris
- 12: Temporalis
- 13: Tibialis anterior
- 14: Trapezius
- 15: Vastus lateralis
- 16: Zygomaticus

**Binary/Muscle-specific Models:**
- 0: Background
- 1: Target muscle

## Examples

### Example 1: Quick test with Multi model
```bash
python inference.py --model Multi --weights Multi_weights.pth --image sample_ultrasound.png
```

### Example 2: Batch processing with output
```bash
python inference.py --model BB --weights BB_weights.pth --batch ./ultrasound_images/ --output ./results/
```

### Example 3: High-quality inference with config
```bash
python inference.py --model Multi --weights Multi_weights.pth --config Multi_config.py --image test.png --output results/
```


## Troubleshooting

### Common Issues

1. **"Module not found" errors**: Install MMSegmentation properly
   ```bash
   pip install openmim
   mim install mmengine "mmcv>=2.0.0" "mmsegmentation>=1.0.0"
   ```

2. **CUDA memory issues**: The models work on CPU if CUDA is not available

3. **Config file errors**: The script includes default configs, but using the original config files gives better results

4. **Image loading errors**: Ensure images are valid and readable by OpenCV



## Citation

If you use these models, please cite:

```bibtex
@article{MARZOLA2025,
title = {Machine learning-driven Heckmatt grading in facioscapulohumeral muscular dystrophy: A novel pathway for musculoskeletal ultrasound analysis},
journal = {Clinical Neurophysiology},
year = {2025},
issn = {1388-2457},
doi = {https://doi.org/10.1016/j.clinph.2025.01.016},
url = {https://www.sciencedirect.com/science/article/pii/S1388245725000367},
author = {Francesco Marzola and Nens {van Alfen} and Jonne Doorduin and Kristen Mariko Meiburger},
keywords = {Muscle ultrasound, Machine learning, Muscle segmentation, Heckmatt grading, Neuromuscular disease diagnosis}}
```

## License

Please refer to the original paper and repository for license information.

