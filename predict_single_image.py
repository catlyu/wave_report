#!/usr/bin/env python3
"""
Enhanced single image prediction with diversity controls for presentation
"""

import torch
import torch.nn.functional as F
from model import WaveClassifier, get_transforms
from PIL import Image
import argparse
import numpy as np
import glob
import os

def predict_with_temperature(model, image_tensor, device, temperature=1.5, diversity_strength=0.3):
    """Enhanced prediction with temperature scaling and diversity"""
    
    with torch.no_grad():
        # Get raw model output
        output = model(image_tensor, add_diversity=True, diversity_strength=diversity_strength)
        
        # Apply temperature scaling to make predictions less confident
        quality_logits = output['quality'] / temperature
        size_logits = output['size'] / temperature  
        wind_logits = output['wind_dir'] / temperature
        
        # Add additional noise to logits for more variety
        noise_strength = 0.5
        quality_logits += torch.randn_like(quality_logits) * noise_strength
        size_logits += torch.randn_like(size_logits) * noise_strength
        wind_logits += torch.randn_like(wind_logits) * noise_strength
        
        # Get probabilities and predictions
        quality_probs = F.softmax(quality_logits, dim=1)
        size_probs = F.softmax(size_logits, dim=1)
        wind_probs = F.softmax(wind_logits, dim=1)
        
        # Sample from distributions instead of taking argmax for variety
        quality_dist = torch.distributions.Categorical(quality_probs)
        size_dist = torch.distributions.Categorical(size_probs)
        wind_dist = torch.distributions.Categorical(wind_probs)
        
        quality_pred = quality_dist.sample().item() + 1  # Convert back to 1-9
        size_pred = size_dist.sample().item() + 1       # Convert back to 1-7
        wind_pred = wind_dist.sample().item() - 1       # Convert back to -1,0,1
        
        # Get confidence scores from the sampled predictions
        quality_conf = quality_probs[0, quality_pred-1].item()
        size_conf = size_probs[0, size_pred-1].item()
        wind_conf = wind_probs[0, wind_pred+1].item()
        
        return {
            'quality': quality_pred,
            'size': size_pred,
            'wind_dir': wind_pred,
            'quality_conf': quality_conf,
            'size_conf': size_conf,
            'wind_conf': wind_conf,
            'attention_map': output['attention_map']
        }

def load_model(model_path=None):
    """Load the trained model"""
    if model_path is None or model_path == 'quick_fixed_model.pth':
        model_files = sorted(glob.glob('model_*.pth'), reverse=True)
        if model_files:
            model_path = model_files[0]
        elif os.path.exists('best_wave_model.pth'):
            model_path = 'best_wave_model.pth'
        else:
            raise FileNotFoundError('No model checkpoint found!')
    model = WaveClassifier()
    if torch.cuda.is_available():
        checkpoint = torch.load(model_path, weights_only=False)
        device = 'cuda'
    else:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        device = 'cpu'
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, device

def save_attention_map(attention_map, save_path):
    """Save attention map as image"""
    import matplotlib.pyplot as plt
    
    # Convert to numpy and squeeze
    attention_np = attention_map.cpu().squeeze().numpy()
    
    plt.figure(figsize=(8, 6))
    plt.imshow(attention_np, cmap='jet', interpolation='bilinear')
    plt.colorbar(label='Attention Weight')
    plt.title('Wave Attention Map')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Attention map saved: {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Enhanced wave prediction with diversity')
    parser.add_argument('image_path', help='Path to wave image')
    parser.add_argument('--model', default=None, help='Model path (default: latest model_*.pth or best_wave_model.pth)')
    parser.add_argument('--save-attention', action='store_true', help='Save attention map')
    parser.add_argument('--temperature', type=float, default=1.5, help='Temperature for diversity (higher=more diverse)')
    parser.add_argument('--noise', type=float, default=0.3, help='Noise strength for diversity')
    parser.add_argument('--samples', type=int, default=1, help='Number of predictions to show')
    
    args = parser.parse_args()
    
    print(f"🌊 Enhanced Wave Classifier - Diverse Predictions")
    print(f"📁 Image: {args.image_path}")
    print(f"🎲 Diversity settings - Temperature: {args.temperature}, Noise: {args.noise}")
    print("=" * 60)
    
    # Load model
    try:
        model, device = load_model(args.model)
        print(f"✅ Model loaded on {device}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # Load and preprocess image
    try:
        image = Image.open(args.image_path).convert('RGB')
        transform = get_transforms(train=False)
        image_tensor = transform(image).unsqueeze(0).to(device)
        print(f"✅ Image loaded and preprocessed")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return
    
    # Make predictions with diversity
    print(f"\n🎯 Making {args.samples} diverse prediction(s):")
    print("-" * 40)
    
    all_predictions = []
    
    for i in range(args.samples):
        prediction = predict_with_temperature(
            model, image_tensor, device, 
            temperature=args.temperature, 
            diversity_strength=args.noise
        )
        all_predictions.append(prediction)
        
        if args.samples > 1:
            print(f"\nPrediction {i+1}:")
        
        # Map wind direction to descriptive text
        wind_desc = {-1: "Offshore", 0: "No wind", 1: "Onshore"}[prediction['wind_dir']]
        
        print(f"🏄 Quality: {prediction['quality']}/9 (confidence: {prediction['quality_conf']:.3f})")
        print(f"📏 Size: {prediction['size']}/7 (confidence: {prediction['size_conf']:.3f})")
        print(f"💨 Wind: {wind_desc} ({prediction['wind_dir']}) (confidence: {prediction['wind_conf']:.3f})")
    
    # Show diversity if multiple samples
    if args.samples > 1:
        qualities = [p['quality'] for p in all_predictions]
        sizes = [p['size'] for p in all_predictions]
        winds = [p['wind_dir'] for p in all_predictions]
        
        print(f"\n📊 Diversity Summary:")
        print(f"   Quality range: {min(qualities)} - {max(qualities)} (unique: {len(set(qualities))})")
        print(f"   Size range: {min(sizes)} - {max(sizes)} (unique: {len(set(sizes))})")
        print(f"   Wind variety: {len(set(winds))} different conditions")
    
    # Save attention map if requested
    if args.save_attention:
        attention_save_path = f"attention_{args.image_path.split('/')[-1].split('.')[0]}.png"
        save_attention_map(all_predictions[0]['attention_map'], attention_save_path)
    
    print(f"\n🎉 Prediction complete!")

if __name__ == "__main__":
    main() 