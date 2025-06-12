#!/usr/bin/env python3
"""
🌊 Interactive Wave Classifier
Upload a wave image and get predictions for quality, size, and wind conditions!
"""

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import json
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io
import base64
import glob

# Import our model components
from model import WaveClassifier, get_transforms

class InteractiveWaveClassifier:
    """
    Interactive wave classifier with web interface
    """
    def __init__(self, model_path=None):
        # If no model_path provided, use latest model_*.pth or fallback
        if model_path is None:
            model_files = sorted(glob.glob('model_*.pth'), reverse=True)
            if model_files:
                model_path = model_files[0]
            elif os.path.exists('best_wave_model.pth'):
                model_path = 'best_wave_model.pth'
            else:
                model_path = None
        self.model_path = model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.transform = get_transforms(train=False)
        
        # Label mappings (from training)
        self.quality_min = 1  # Will be updated when model loads
        self.size_min = 1     # Will be updated when model loads
        
        self.load_model()
    
    def load_model(self):
        """
        Load the trained wave classification model
        """
        if self.model_path is None or not os.path.exists(self.model_path):
            st.error("No model file found. Please ensure you have a trained model.")
            return False
        try:
            model = WaveClassifier()
            checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            st.session_state['model_file'] = self.model_path
            self.model = model
            return True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def predict_wave_features(self, image, add_diversity=False, diversity_strength=0.2):
        """
        Predict wave features from an image with optional diversity
        """
        if self.model is None:
            return None
        
        # Preprocess image
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Use enhanced model with optional diversity
            predictions = self.model(input_tensor, add_diversity=add_diversity, diversity_strength=diversity_strength)
            
            # Get predicted classes and confidence scores
            quality_probs = F.softmax(predictions['quality'], dim=1)
            size_probs = F.softmax(predictions['size'], dim=1)
            wind_probs = F.softmax(predictions['wind_dir'], dim=1)
            
            quality_pred = torch.argmax(quality_probs, dim=1).item()
            size_pred = torch.argmax(size_probs, dim=1).item()
            wind_pred = torch.argmax(wind_probs, dim=1).item()
            
            quality_conf = quality_probs.max().item()
            size_conf = size_probs.max().item()
            wind_conf = wind_probs.max().item()
            
            # Convert back to original label space
            quality_original = quality_pred + self.quality_min
            size_original = size_pred + self.size_min
            wind_original = wind_pred - 1  # Convert 0,1,2 back to -1,0,1
            
            # Get attention map
            attention_map = predictions['attention_map'].cpu().numpy()[0, 0]
            
            return {
                'quality': {'prediction': quality_original, 'confidence': quality_conf, 'probabilities': quality_probs.cpu().numpy()[0]},
                'size': {'prediction': size_original, 'confidence': size_conf, 'probabilities': size_probs.cpu().numpy()[0]},
                'wind_dir': {'prediction': wind_original, 'confidence': wind_conf, 'probabilities': wind_probs.cpu().numpy()[0]},
                'attention_map': attention_map
            }
    
    def create_attention_visualization(self, original_image, attention_map):
        """
        Create attention map visualization
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image', fontsize=14)
        axes[0].axis('off')
        
        # Attention overlay
        attention_resized = np.array(Image.fromarray(attention_map).resize(original_image.size))
        
        axes[1].imshow(original_image)
        im = axes[1].imshow(attention_resized, alpha=0.6, cmap='jet')
        axes[1].set_title('Model Attention\n(Red = High Focus)', fontsize=14)
        axes[1].axis('off')
        
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        plt.tight_layout()
        
        # Convert to base64 for display
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        plt.close()
        
        return img_base64

def main():
    """
    Main Streamlit application
    """
    st.set_page_config(
        page_title="🌊 Wave Classifier",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🌊 Wave Classifier")
    st.markdown("Upload a wave image to get instant predictions for quality, size, and wind conditions!")
    
    # Sidebar with information
    with st.sidebar:
        st.header("🌊 About")
        st.markdown("""
        Upload a wave image to get predictions for:
        - **Quality**: 1-9 scale 
        - **Size**: 1-7 scale
        - **Wind**: Offshore/Onshore/Neutral
        
        The model uses enhanced attention to focus on wave regions and provides confidence scores.
        """)
        
        st.header("🎛️ Settings")
        add_diversity = st.checkbox("Enable Prediction Diversity", value=False, 
                                   help="Add controlled randomness for more varied predictions")
        diversity_strength = 0.2
        if add_diversity:
            diversity_strength = st.slider("Diversity Strength", 0.1, 0.5, 0.2, 0.1,
                                          help="Higher values = more diverse predictions")
        st.markdown("---")
        # Model selection
        model_files = sorted(glob.glob('model_*.pth'), reverse=True)
        model_options = model_files if model_files else ['best_wave_model.pth']
        selected_model = st.selectbox("Select Model Checkpoint", model_options)
        st.caption(f"Model: {selected_model} | Device: {('CUDA' if torch.cuda.is_available() else 'CPU').upper()}")
    
    # Initialize classifier with selected model
    classifier = InteractiveWaveClassifier(model_path=selected_model)
    if classifier.model is None:
        st.stop()
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a wave image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of waves/surf conditions"
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)
            # Skip sea detection - proceed directly with wave classification
            # st.header("Wave Analysis")
            with st.spinner("Analyzing wave features..."):
                predictions = classifier.predict_wave_features(image, add_diversity, diversity_strength)
            if predictions:
                with col2:
                    st.header("📊 Predictions")
                    # Quality prediction
                    st.subheader("Wave Quality")
                    quality_score = predictions['quality']['prediction']
                    quality_conf = predictions['quality']['confidence']
                    
                    if quality_score <= 3:
                        quality_desc = "Poor"
                        quality_color = "🔴"
                    elif quality_score <= 6:
                        quality_desc = "Fair to Good"
                        quality_color = "🟡"
                    else:
                        quality_desc = "Excellent"
                        quality_color = "🟢"
                    
                    st.metric(
                        label="Quality Score",
                        value=f"{quality_score}/9",
                        help=f"Confidence: {quality_conf:.3f}"
                    )
                    st.write(f"{quality_color} **{quality_desc}** conditions")
                    
                    # Size prediction
                    st.subheader("Wave Size")
                    size_score = predictions['size']['prediction']
                    size_conf = predictions['size']['confidence']
                    
                    if size_score <= 2:
                        size_desc = "Small (1-3 feet)"
                    elif size_score <= 4:
                        size_desc = "Medium (3-6 feet)"
                    elif size_score <= 6:
                        size_desc = "Large (6-10 feet)"
                    else:
                        size_desc = "Extra Large (10+ feet)"
                    
                    st.metric(
                        label="Size Score",
                        value=f"{size_score}/7",
                        help=f"Confidence: {size_conf:.3f}"
                    )
                    st.write(f"📐 **{size_desc}**")
                    
                    # Wind direction prediction
                    st.subheader("💨 Wind Direction")
                    wind_score = predictions['wind_dir']['prediction']
                    wind_conf = predictions['wind_dir']['confidence']
                    
                    if wind_score == -1:
                        wind_desc = "Onshore (Poor)"
                        wind_emoji = "💨➡️"
                        wind_color = "🔴"
                    elif wind_score == 0:
                        wind_desc = "Cross-shore (Neutral)"
                        wind_emoji = "💨↕️"
                        wind_color = "🟡"
                    else:
                        wind_desc = "Offshore (Good)"
                        wind_emoji = "💨⬅️"
                        wind_color = "🟢"
                    
                    st.metric(
                        label="Wind Effect",
                        value=wind_score,
                        help=f"Confidence: {wind_conf:.3f}"
                    )
                    st.write(f"{wind_color} {wind_emoji} **{wind_desc}**")
                    
                    # Overall assessment
                    st.subheader("Overall Assessment")
                    avg_conf = (quality_conf + size_conf + wind_conf) / 3
                    
                    if quality_score >= 6 and wind_score >= 0:
                        overall = "🟢 **Great surf conditions!**"
                    elif quality_score >= 4 and wind_score >= -1:
                        overall = "🟡 **Decent surf conditions**"
                    else:
                        overall = "🔴 **Poor surf conditions**"
                    
                    st.write(overall)
                
                # Attention visualization
                st.header("Model Attention Map")
                st.markdown("See where the model is focusing when making predictions:")
                attention_viz = classifier.create_attention_visualization(
                    image, predictions['attention_map']
                )
                st.markdown(
                    f'<img src="data:image/png;base64,{attention_viz}" style="width:100%">',
                    unsafe_allow_html=True
                )
            
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🌊 Built with Streamlit & PyTorch</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 