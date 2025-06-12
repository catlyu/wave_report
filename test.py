import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
from PIL import Image
import argparse

# Try to import sklearn with fallback
try:
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    SKLEARN_AVAILABLE = True
    print("✅ sklearn imported successfully")
except ImportError as e:
    print(f"⚠️  sklearn import failed: {e}")
    print("📝 Will use basic evaluation metrics instead")
    SKLEARN_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from model import WaveClassifier, WaveDataset, get_transforms

class WaveEvaluator:
    def __init__(self, model_path, test_metadata, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.test_metadata = test_metadata
        
        # Load model
        print(f"Loading model from {model_path}...")
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        except TypeError:
            # Fallback for older PyTorch versions that don't have weights_only parameter
            checkpoint = torch.load(model_path, map_location=device)
        
        self.model = WaveClassifier(
            n_q=9,
            n_s=7,
            n_w=3
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        print(f"Model loaded successfully! Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
        
        # Load test dataset
        self.test_dataset = WaveDataset(
            test_metadata,
            transform=get_transforms(train=False)
        )
        
        # Label mappings for interpretation
        self.quality_min = self.test_dataset.q_min
        self.size_min = self.test_dataset.s_min
        
        print(f"Test dataset loaded: {len(self.test_dataset)} samples")
    
    def predict_single_image(self, image_path, visualize_attention=True):
        """Predict on a single image and optionally visualize attention"""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        original_image = np.array(image)
        
        transform = get_transforms(train=False)
        input_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(input_tensor)
            
            # Get predicted classes
            quality_pred = torch.argmax(predictions['quality'], dim=1).item()
            size_pred = torch.argmax(predictions['size'], dim=1).item()
            wind_pred = torch.argmax(predictions['wind_dir'], dim=1).item()
            
            # Get confidence scores
            quality_conf = F.softmax(predictions['quality'], dim=1).max().item()
            size_conf = F.softmax(predictions['size'], dim=1).max().item()
            wind_conf = F.softmax(predictions['wind_dir'], dim=1).max().item()
            
            # Convert back to original label space
            quality_original = quality_pred + self.quality_min
            size_original = size_pred + self.size_min
            wind_original = wind_pred - 1  # Convert 0,1,2 back to -1,0,1
            
            results = {
                'quality': {'prediction': quality_original, 'confidence': quality_conf},
                'size': {'prediction': size_original, 'confidence': size_conf},
                'wind_dir': {'prediction': wind_original, 'confidence': wind_conf},
                'attention_map': predictions['attention_map'].cpu().numpy()
            }
            
            if visualize_attention:
                self.visualize_prediction(original_image, results, image_path)
            
            return results
    
    def visualize_prediction(self, original_image, results, image_path):
        """Visualize prediction results with attention map"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Attention map
        attention = results['attention_map'][0, 0]  # First batch, first channel
        attention_resized = np.array(Image.fromarray(attention).resize(original_image.shape[:2][::-1]))
        
        axes[1].imshow(original_image)
        im = axes[1].imshow(attention_resized, alpha=0.6, cmap='jet')
        axes[1].set_title('Attention Map\n(Red = High Attention)')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Predictions
        axes[2].axis('off')
        pred_text = f"""
        Predictions for: {os.path.basename(image_path)}
        
        Quality: {results['quality']['prediction']} (conf: {results['quality']['confidence']:.3f})
        Size: {results['size']['prediction']} (conf: {results['size']['confidence']:.3f})
        Wind Dir: {results['wind_dir']['prediction']} (conf: {results['wind_dir']['confidence']:.3f})
        
        Wind Direction Legend:
        -1: Onshore (poor conditions)
         0: No wind/cross-shore
         1: Offshore (good conditions)
        """
        axes[2].text(0.1, 0.5, pred_text, fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        plt.tight_layout()
        
        # Save with a proper filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = f"attention_map_{base_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        print(f"Attention map saved to {save_path}")
    
    def evaluate_test_set(self):
        """Evaluate model on entire test set"""
        all_predictions = {'quality': [], 'size': [], 'wind_dir': []}
        all_targets = {'quality': [], 'size': [], 'wind_dir': []}
        all_confidences = {'quality': [], 'size': [], 'wind_dir': []}
        detailed_results = []
        
        print("Evaluating on test set...")
        
        for i in range(len(self.test_dataset)):
            sample = self.test_dataset[i]
            image = sample['image'].unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                predictions = self.model(image)
                
                # Get predictions and confidences
                for task in ['quality', 'size', 'wind_dir']:
                    pred_probs = F.softmax(predictions[task], dim=1)
                    pred_class = torch.argmax(pred_probs, dim=1).item()
                    confidence = pred_probs.max().item()
                    
                    all_predictions[task].append(pred_class)
                    all_targets[task].append(sample[task].item())
                    all_confidences[task].append(confidence)
                
                # Store detailed results, robust to missing 'filename'
                detailed_results.append({
                    'filename': sample.get('filename', f'image_{i}'),
                    'true_quality': sample['quality'].item() + self.quality_min,
                    'pred_quality': all_predictions['quality'][-1] + self.quality_min,
                    'quality_conf': all_confidences['quality'][-1],
                    'true_size': sample['size'].item() + self.size_min,
                    'pred_size': all_predictions['size'][-1] + self.size_min,
                    'size_conf': all_confidences['size'][-1],
                    'true_wind': sample['wind_dir'].item() - 1,
                    'pred_wind': all_predictions['wind_dir'][-1] - 1,
                    'wind_conf': all_confidences['wind_dir'][-1]
                })
        
        return all_predictions, all_targets, all_confidences, detailed_results
    
    def generate_evaluation_report(self, save_path='evaluation_report.json'):
        """Generate comprehensive evaluation report"""
        predictions, targets, confidences, detailed_results = self.evaluate_test_set()
        
        # Calculate metrics for each task
        task_names = ['quality', 'size', 'wind_dir']
        task_labels = ['Quality', 'Size', 'Wind Direction']
        num_classes_map = {'quality': 9, 'size': 7, 'wind_dir': 3}
        
        evaluation_results = {}
        
        for task, label in zip(task_names, task_labels):
            num_classes = num_classes_map[task]
            
            if SKLEARN_AVAILABLE:
                # Use sklearn functions
                try:
                    accuracy = accuracy_score(targets[task], predictions[task])
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        targets[task], predictions[task], average='weighted', zero_division=0
                    )
                    
                    # Classification report
                    class_report = classification_report(
                        targets[task], predictions[task], 
                        output_dict=True, zero_division=0
                    )
                    
                    # Confusion matrix
                    cm = confusion_matrix(targets[task], predictions[task])
                    
                except Exception as e:
                    print(f"⚠️  sklearn failed for {task}, using fallback: {e}")
                    # Fall back to basic functions
                    accuracy = basic_accuracy_score(targets[task], predictions[task])
                    precision, recall, f1 = basic_precision_recall_f1(targets[task], predictions[task], num_classes)
                    class_report = {"weighted avg": {"precision": precision, "recall": recall, "f1-score": f1}}
                    cm = basic_confusion_matrix(targets[task], predictions[task], num_classes)
            else:
                # Use basic fallback functions
                accuracy = basic_accuracy_score(targets[task], predictions[task])
                precision, recall, f1 = basic_precision_recall_f1(targets[task], predictions[task], num_classes)
                class_report = {"weighted avg": {"precision": precision, "recall": recall, "f1-score": f1}}
                cm = basic_confusion_matrix(targets[task], predictions[task], num_classes)
            
            evaluation_results[task] = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'avg_confidence': float(np.mean(confidences[task])),
                'classification_report': class_report,
                'confusion_matrix': cm.tolist()
            }
        
        # Overall summary
        overall_accuracy = np.mean([evaluation_results[task]['accuracy'] for task in task_names])
        overall_confidence = np.mean([evaluation_results[task]['avg_confidence'] for task in task_names])
        
        evaluation_results['overall'] = {
            'average_accuracy': float(overall_accuracy),
            'average_confidence': float(overall_confidence),
            'total_samples': len(detailed_results),
            'sklearn_available': SKLEARN_AVAILABLE
        }
        
        evaluation_results['detailed_predictions'] = detailed_results
        
        # Save results
        with open(save_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        print(f"Evaluation report saved to {save_path}")
        if not SKLEARN_AVAILABLE:
            print("📝 Note: Used basic evaluation metrics (sklearn not available)")
        
        return evaluation_results
    
    def plot_evaluation_results(self, evaluation_results, save_path='evaluation_plots.png'):
        """Create visualization plots for evaluation results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Evaluation Results', fontsize=16)
        
        task_names = ['quality', 'size', 'wind_dir']
        task_labels = ['Quality', 'Size', 'Wind Direction']
        
        # Plot confusion matrices
        for i, (task, label) in enumerate(zip(task_names, task_labels)):
            cm = np.array(evaluation_results[task]['confusion_matrix'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, i])
            axes[0, i].set_title(f'{label} Confusion Matrix')
            axes[0, i].set_xlabel('Predicted')
            axes[0, i].set_ylabel('True')
        
        # Plot metrics comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_values = {metric: [evaluation_results[task][metric] for task in task_names] for metric in metrics}
        
        x = np.arange(len(task_labels))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            axes[1, 0].bar(x + i*width, metric_values[metric], width, label=metric.title())
        
        axes[1, 0].set_xlabel('Tasks')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Performance Metrics by Task')
        axes[1, 0].set_xticks(x + width * 1.5)
        axes[1, 0].set_xticklabels(task_labels)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot confidence scores
        confidences = [evaluation_results[task]['avg_confidence'] for task in task_names]
        axes[1, 1].bar(task_labels, confidences, color=['green', 'orange', 'purple'])
        axes[1, 1].set_title('Average Confidence by Task')
        axes[1, 1].set_ylabel('Confidence')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Overall summary
        axes[1, 2].axis('off')
        summary_text = f"""
        Overall Performance Summary
        
        Average Accuracy: {evaluation_results['overall']['average_accuracy']:.3f}
        Average Confidence: {evaluation_results['overall']['average_confidence']:.3f}
        Total Test Samples: {evaluation_results['overall']['total_samples']}
        
        Individual Task Accuracies:
        Quality: {evaluation_results['quality']['accuracy']:.3f}
        Size: {evaluation_results['size']['accuracy']:.3f}
        Wind Direction: {evaluation_results['wind_dir']['accuracy']:.3f}
        """
        axes[1, 2].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        print(f"Evaluation plots saved to {save_path}")
    
    def test_all_images(self, visualize_attention=True):
        """Test all images in the test set with visualizations"""
        print(f"Testing all {len(self.test_dataset)} images...")
        
        for i in range(len(self.test_dataset)):
            sample = self.test_dataset[i]
            image_path = sample['filename']
            
            print(f"\nTesting image {i+1}/{len(self.test_dataset)}: {image_path}")
            
            # Get ground truth
            true_quality = sample['quality'].item() + self.quality_min
            true_size = sample['size'].item() + self.size_min
            true_wind = sample['wind_dir'].item() - 1
            
            print(f"Ground Truth - Quality: {true_quality}, Size: {true_size}, Wind: {true_wind}")
            
            # Get prediction
            results = self.predict_single_image(image_path, visualize_attention)
            
            print(f"Predictions - Quality: {results['quality']['prediction']} (conf: {results['quality']['confidence']:.3f})")
            print(f"             Size: {results['size']['prediction']} (conf: {results['size']['confidence']:.3f})")
            print(f"             Wind: {results['wind_dir']['prediction']} (conf: {results['wind_dir']['confidence']:.3f})")

# Fallback functions when sklearn is not available
def basic_accuracy_score(y_true, y_pred):
    """Basic accuracy calculation"""
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)

def basic_confusion_matrix(y_true, y_pred, num_classes):
    """Basic confusion matrix calculation"""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        if 0 <= true < num_classes and 0 <= pred < num_classes:
            cm[true, pred] += 1
    return cm

def basic_precision_recall_f1(y_true, y_pred, num_classes):
    """Basic precision, recall, F1 calculation"""
    cm = basic_confusion_matrix(y_true, y_pred, num_classes)
    
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
    
    # Weighted average
    weights = [cm[i, :].sum() for i in range(num_classes)]
    total_weight = sum(weights)
    
    if total_weight > 0:
        weighted_precision = sum(p * w for p, w in zip(precision_scores, weights)) / total_weight
        weighted_recall = sum(r * w for r, w in zip(recall_scores, weights)) / total_weight
        weighted_f1 = sum(f * w for f, w in zip(f1_scores, weights)) / total_weight
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0
    
    return weighted_precision, weighted_recall, weighted_f1

def main(visualize_individual=False):
    parser = argparse.ArgumentParser(description='Test Wave Classification Model')
    parser.add_argument('--model', type=str, default=None, help='Path to model checkpoint (.pth)')
    parser.add_argument('--visualize', action='store_true', help='Visualize individual image predictions with attention maps')
    args = parser.parse_args()

    # Find latest model if not specified
    model_path = args.model
    if model_path is None:
        # Find the most recent model_*.pth file
        import glob
        model_files = sorted(glob.glob('model_*.pth'), reverse=True)
        if model_files:
            model_path = model_files[0]
            print(f"No model specified. Using latest: {model_path}")
        else:
            model_path = 'best_wave_model.pth'
            print(f"No model specified and no model_*.pth found. Using default: {model_path}")

    # Initialize evaluator
    evaluator = WaveEvaluator(
        model_path=model_path,
        test_metadata='test_metadata.json'
    )
    
    # Generate comprehensive evaluation report
    print("Generating evaluation report...")
    evaluation_results = evaluator.generate_evaluation_report('evaluation_report.json')
    
    # Create evaluation plots
    evaluator.plot_evaluation_results(evaluation_results, 'evaluation_plots.png')
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Overall Average Accuracy: {evaluation_results['overall']['average_accuracy']:.3f}")
    print(f"Overall Average Confidence: {evaluation_results['overall']['average_confidence']:.3f}")
    print(f"Total Test Samples: {evaluation_results['overall']['total_samples']}")
    print("\nTask-specific Accuracies:")
    print(f"  Quality: {evaluation_results['quality']['accuracy']:.3f}")
    print(f"  Size: {evaluation_results['size']['accuracy']:.3f}")
    print(f"  Wind Direction: {evaluation_results['wind_dir']['accuracy']:.3f}")
    
    # Test individual images with visualizations
    if args.visualize:
        print("\n" + "="*60)
        print("INDIVIDUAL IMAGE TESTING")
        print("="*60)
        evaluator.test_all_images(visualize_attention=True)
    else:
        print("\n💡 To see individual image predictions with attention maps, use:")
        print("   python test.py --visualize")
    
    print("\nEvaluation completed!")
    print("Files generated:")
    print("- evaluation_report.json (detailed metrics)")
    print("- evaluation_plots.png (visualization plots)")

if __name__ == "__main__":
    main() 