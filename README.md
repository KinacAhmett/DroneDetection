# DroneDetection Project - YOLOv11 Enhanced

## Overview
Advanced drone detection and tracking system enhanced with YOLOv11, IR-specific fine-tuning, and comprehensive dataset analysis tools.

## Recent Updates & Improvements

### 🔥 YOLOv11 Integration
- **Upgraded from YOLOv5 to YOLOv11** for improved detection accuracy
- Enhanced model architecture with better performance on both RGB and IR imagery
- Optimized inference pipeline for real-time detection

### 📊 Intelligent Dataset Analysis
- **Video Quality Assessment Tool** (`simple_video_analyzer.py`)
  - Analyzes 160+ folders with infrared.mp4 and visible.mp4 pairs
  - Calculates difficulty scores based on brightness, contrast, and background type
  - HSV-based background classification (land/sea/sky/mixed)
  - Automated frame sampling for efficient analysis
- **Difficulty Scoring System**
  - Brightness penalties for dark frames
  - Contrast penalties for blurry content
  - Background-specific challenge ratings
  - Identifies optimal videos for fine-tuning

### 🎯 IR-Specific Fine-Tuning Pipeline
- **Enhanced Dataset Creator** (`ir_enhanced_dataset_creator.py`)
  - Automatically selects best IR videos based on difficulty scores
  - Extracts frames from low-difficulty IR videos
  - Integrates with existing 36K subset for balanced training
  - Increases IR ratio from ~18% to ~35% in training data
- **IR-Optimized Hyperparameters** (`ir_finetune_config.yaml`)
  - Specialized HSV augmentation for grayscale/thermal imagery
  - Conservative learning rates for stable fine-tuning
  - Reduced geometric augmentations to preserve thermal signatures

### 🛠️ Advanced Frame & Label Processing
- **Improved Frame Extraction** (`framecut.py`)
  - Recursive video processing for complex directory structures
  - Automated frame naming and organization
  - Support for both RGB and IR video formats
- **Label Management System**
  - Automated label file generation for unlabeled IR frames
  - YOLO format compatibility maintained
  - Validation set creation and management

### 📈 Performance Monitoring & Metrics
- **Comprehensive Evaluation Pipeline**
  - Precision, Recall, F1-Score tracking
  - mAP (mean Average Precision) calculation
  - IoU (Intersection over Union) analysis
  - Confusion Matrix generation
- **Training Progress Visualization**
  - Real-time loss tracking
  - Validation metrics monitoring
  - Early stopping implementation

### 🔧 System Optimizations
- **Detection & Tracking Improvements**
  - Reduced center threshold from 60 to 40 pixels for better accuracy
  - Optimized FPS performance with selective detection frequency
  - Enhanced visualization with resolution-adaptive display
  - Improved debug logging and error handling
- **Memory & Processing Efficiency**
  - Optimized batch sizes for stable training
  - Efficient frame sampling strategies
  - Automated cleanup and resource management

### 📋 Configuration & Deployment
- **Complete Requirements File** (`requirements_v11_complete.txt`)
  - YOLOv11 compatible dependencies
  - Version-locked packages for reproducibility
- **Automated Pipeline Scripts**
  - End-to-end training pipeline
  - Validation and testing automation
  - Model deployment utilities

## Key Features

### Detection Capabilities
- **Dual-Mode Processing**: RGB and IR image support
- **Real-time Tracking**: Persistent object tracking across frames
- **Center-based Validation**: Focused detection on target areas
- **Multi-resolution Support**: Adaptive processing for different input sizes

### Training Enhancements
- **Smart Data Selection**: Automated selection of optimal training data
- **Balanced Dataset Creation**: IR/RGB ratio optimization
- **Conservative Fine-tuning**: Preserves existing knowledge while adapting to IR
- **Comprehensive Validation**: Multiple metrics and validation strategies

### Analysis Tools
- **Video Quality Assessment**: Automated difficulty scoring
- **Dataset Statistics**: Comprehensive data analysis and reporting
- **Performance Metrics**: Industry-standard evaluation metrics
- **Visualization Tools**: Real-time detection visualization

## Technical Specifications

### Model Architecture
- **Base Model**: YOLOv11
- **Input Resolution**: 640x640
- **Classes**: 1 (drone)
- **Framework**: PyTorch + Ultralytics

### Training Configuration
- **Learning Rate**: 0.0003 (IR-optimized)
- **Batch Size**: 14 (stability-focused)
- **Epochs**: 22-25 (fine-tuning optimized)
- **Patience**: 8 (early stopping)

### Dataset Composition
- **Original Subset**: ~36,000 frames
- **Enhanced Dataset**: ~44,000 frames
- **IR Ratio**: Improved from 18% to 35%
- **Video Sources**: 160+ folder structure with paired IR/RGB data

## Usage

### Quick Start
```bash
# Install dependencies
pip install -r requirements_v11_complete.txt

# Analyze dataset
python simple_video_analyzer.py

# Create enhanced dataset
python ir_enhanced_dataset_creator.py

# Fine-tune model
yolo train model=runs/detect/subset36_run22/weights/best.pt data=ir_enhanced_dataset/data.yaml epochs=22 batch=14 lr0=0.0003

# Run detection
python detect_tracking.py
```

### Configuration Files
- `ir_finetune_config.yaml`: IR-specific training parameters
- `data.yaml`: Dataset configuration
- `requirements_v11_complete.txt`: Complete dependency list

## Performance Improvements
- **Enhanced IR Detection**: Significant improvement in infrared image processing
- **Reduced False Positives**: Better precision through optimized thresholds
- **Stable Tracking**: Improved object persistence across frames
- **Faster Processing**: Optimized detection frequency for real-time performance

## Contributing
This project implements state-of-the-art drone detection with specialized enhancements for infrared imagery and robust tracking capabilities.

## Model Files
- Pre-trained YOLOv11 weights available in `runs/detect/subset36_run22/weights/`
- Fine-tuned IR models in respective training directories
- Complete model pipeline with validation metrics

---
*Enhanced with YOLOv11, intelligent dataset analysis, and IR-specific optimizations for superior drone detection performance.*
