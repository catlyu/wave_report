# Wave Feature Classification with Domain Knowledge

An advanced multi-task CNN that classifies wave features using domain-specific knowledge about surf conditions. The model incorporates attention mechanisms, color contrast analysis, texture consistency, and heuristic wind features to accurately predict wave quality, size, and wind conditions.

## Model Architecture

### Core Components

1. **Backbone**: ResNet18 pre-trained on ImageNet for robust feature extraction
2. **Wave Attention Module**: Fixed attention with sea detection (blue color, ripple patterns, horizontal bias, horizon suppression)
3. **Color Contrast Analyzer**: Multi-scale contrast and white water detection for wave size analysis
4. **Texture Consistency Analyzer**: Multi-directional texture analysis with smoothness detection for wind conditions
5. **Specialized Multi-task Heads**: Task-specific architectures optimized for quality (9-class), size (7-class), and wind (3-class) prediction
6. **Wind Heuristics**: White-cap ratio and choppiness features directly influence wind logits for more robust wind direction prediction

### Domain Knowledge Integration

- **Attention Mechanism**: Focuses on sea/wave regions, suppresses sky and beach
- **Color & Texture Analysis**: Detects white water, foam, and smoothness for size and wind
- **Wind Heuristics**: White-cap ratio (sparkle = choppy = onshore), choppiness (smooth = offshore)
- **Transfer Learning**: Leverages ImageNet pre-training for robust visual features

### Loss Functions

- **Focal Cross-Entropy Loss**: For all classification heads, with class weights for wind direction ([2, 1, 2])
- **Distance-Aware Regression**: L2 loss between expected value and true value for quality/size
- **Entropy Bonus**: Encourages diverse predictions
- **Label Smoothing**: 0.05 for all heads

## Dataset Overview

- **Training Data**: `train_imgs/` (128 images, 14 surf locations)
- **Testing Data**: `wave_imgs/` (16 images)
- **Labels**: `train_metadata.json`, `test_metadata.json`
- **Features**: Quality (1-9), Size (1-7), Wind Direction (-1, 0, 1)

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train.py --epochs 20 --patience 3 --batch_size 8
```
- Model checkpoints are saved as `model_YYYYMMDD_HHMMSS.pth`.

### 3. Evaluate the Model
```bash
python test.py --model model_YYYYMMDD_HHMMSS.pth --visualize
```
- Generates `evaluation_report.json` and `evaluation_plots.png`.

### 4. Interactive Web Interface
```bash
streamlit run interactive_wave_classifier.py
```
- Select your model in the sidebar and upload images for instant predictions and attention maps.

### 5. Predict on a Single Image
```bash
python predict_single_image.py path/to/your_image.png --model model_YYYYMMDD_HHMMSS.pth --save-attention
```

## Technical Details

- **Image Normalization**: Standard ImageNet mean/std
- **Augmentation**: Random crop, color jitter, rotation, erasing
- **Attention**: Suppresses sky, focuses on sea
- **Wind Heuristics**: White-cap ratio and choppiness features
- **Loss**: Focal loss (γ=1.5), class weights for wind, L2 regression, entropy bonus
- **Automatic device selection**: Uses GPU if available

## Data Availability

- The full training and test image datasets are **not included** in this repository due to size and copyright.
- The metadata files (`train_metadata.json`, `test_metadata.json`) are provided for reference.
- To use your own images, place them in a folder and update the metadata format accordingly.

## References

- Multi-task learning with attention mechanisms
- Surf forecasting and wave analysis principles
- ImageNet pre-trained ResNet18
- Streamlit for web interfaces
