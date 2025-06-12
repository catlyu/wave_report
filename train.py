import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from collections import defaultdict
import json
import argparse
import datetime

from model import WaveClassifier, create_data_loaders, FocalLoss

class WaveTrainer:
    def __init__(self, model, train_loader, val_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss function and optimizer
        self.quality_loss = FocalLoss(gamma=1.5, weight=None, label_smoothing=0.05).to(device)
        self.size_loss = FocalLoss(gamma=1.5, weight=None, label_smoothing=0.05).to(device)
        self.wind_loss = FocalLoss(gamma=1.5, weight=torch.tensor([2.0, 1.0, 2.0], device=device), label_smoothing=0.05).to(device)
        self.optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=8)
        
        # Gradient accumulation for effective larger batch size
        self.accumulation_steps = 2  # Effective batch size = batch_size * accumulation_steps
        
        # Training tracking
        self.train_history = defaultdict(list)
        self.val_history = defaultdict(list)
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0
        # Increased patience for small datasets
        self.early_stop_patience = 25
        
        print(f"Training on device: {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Early stopping patience: {self.early_stop_patience} epochs")
        print(f"Gradient accumulation steps: {self.accumulation_steps}")
        
    def train_epoch(self):
        self.model.train()
        epoch_losses = defaultdict(list)
        
        self.optimizer.zero_grad()  # Initialize gradients
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            images = batch['image'].to(self.device)
            targets = {
                'quality': batch['quality'].to(self.device),
                'size': batch['size'].to(self.device),
                'wind_dir': batch['wind_dir'].to(self.device),
                'quality_float': batch['quality_float'].to(self.device),
                'size_float': batch['size_float'].to(self.device)
            }
            
            # Forward pass
            predictions = self.model(images)
            
            # Focal loss for classification
            loss_q = self.quality_loss(predictions['quality'], targets['quality'])
            loss_s = self.size_loss(predictions['size'], targets['size'])
            loss_w = self.wind_loss(predictions['wind_dir'], targets['wind_dir'])
            
            # Distance-aware regression (L2)
            softmax = torch.nn.functional.softmax
            q_values = torch.arange(1, 10, device=self.device).float()
            s_values = torch.arange(1, 8, device=self.device).float()
            exp_q = (softmax(predictions['quality'], 1) * q_values).sum(1)
            exp_s = (softmax(predictions['size'], 1) * s_values).sum(1)
            mse_q = F.mse_loss(exp_q, targets['quality_float'])
            mse_s = F.mse_loss(exp_s, targets['size_float'])
            reg_loss = mse_q + mse_s
            
            # Entropy bonus
            ent_q = -(softmax(predictions['quality'], 1) * torch.log_softmax(predictions['quality'], 1)).sum(1).mean()
            ent_s = -(softmax(predictions['size'], 1) * torch.log_softmax(predictions['size'], 1)).sum(1).mean()
            ent_w = -(softmax(predictions['wind_dir'], 1) * torch.log_softmax(predictions['wind_dir'], 1)).sum(1).mean()
            entropy_bonus = 0.03 * (ent_q + ent_s + ent_w)
            
            # Total loss
            total_loss = loss_q + loss_s + loss_w + 0.2 * reg_loss - entropy_bonus
            total_loss = total_loss / self.accumulation_steps
            
            # Backward pass
            total_loss.backward()
            
            # Track losses (unscaled for logging)
            for key, value in zip(['quality_loss', 'size_loss', 'wind_loss', 'regression_loss', 'entropy_bonus', 'total_loss'],
                                  [loss_q, loss_s, loss_w, reg_loss, entropy_bonus, total_loss * self.accumulation_steps]):
                epoch_losses[key].append(value.item())
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % self.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {total_loss.item():.4f}, '
                      f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
        
        # Final update if there are remaining gradients
        if len(self.train_loader) % self.accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        # Calculate average losses
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        return avg_losses
    
    def validate_epoch(self):
        self.model.eval()
        epoch_losses = defaultdict(list)
        correct_predictions = defaultdict(int)
        total_predictions = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to device
                images = batch['image'].to(self.device)
                targets = {
                    'quality': batch['quality'].to(self.device),
                    'size': batch['size'].to(self.device),
                    'wind_dir': batch['wind_dir'].to(self.device),
                    'quality_float': batch['quality_float'].to(self.device),
                    'size_float': batch['size_float'].to(self.device)
                }
                
                # Forward pass
                predictions = self.model(images)
                
                # Focal loss for classification
                loss_q = self.quality_loss(predictions['quality'], targets['quality'])
                loss_s = self.size_loss(predictions['size'], targets['size'])
                loss_w = self.wind_loss(predictions['wind_dir'], targets['wind_dir'])
                
                # Distance-aware regression (L2)
                softmax = torch.nn.functional.softmax
                q_values = torch.arange(1, 10, device=self.device).float()
                s_values = torch.arange(1, 8, device=self.device).float()
                exp_q = (softmax(predictions['quality'], 1) * q_values).sum(1)
                exp_s = (softmax(predictions['size'], 1) * s_values).sum(1)
                mse_q = F.mse_loss(exp_q, targets['quality_float'])
                mse_s = F.mse_loss(exp_s, targets['size_float'])
                reg_loss = mse_q + mse_s
                
                # Entropy bonus
                ent_q = -(softmax(predictions['quality'], 1) * torch.log_softmax(predictions['quality'], 1)).sum(1).mean()
                ent_s = -(softmax(predictions['size'], 1) * torch.log_softmax(predictions['size'], 1)).sum(1).mean()
                ent_w = -(softmax(predictions['wind_dir'], 1) * torch.log_softmax(predictions['wind_dir'], 1)).sum(1).mean()
                entropy_bonus = 0.03 * (ent_q + ent_s + ent_w)
                
                # Total loss
                total_loss = loss_q + loss_s + loss_w + 0.2 * reg_loss - entropy_bonus
                
                # Track losses
                for key, value in zip(['quality_loss', 'size_loss', 'wind_loss', 'regression_loss', 'entropy_bonus', 'total_loss'],
                                      [loss_q, loss_s, loss_w, reg_loss, entropy_bonus, total_loss]):
                    epoch_losses[key].append(value.item())
                
                # Calculate accuracies
                for task in ['quality', 'size', 'wind_dir']:
                    pred_labels = torch.argmax(predictions[task], dim=1)
                    correct_predictions[task] += (pred_labels == targets[task]).sum().item()
                
                total_predictions += images.size(0)
        
        # Calculate average losses and accuracies
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        accuracies = {task: correct / total_predictions for task, correct in correct_predictions.items()}
        
        return avg_losses, accuracies
    
    def train(self, num_epochs=50, save_path='best_wave_model.pth'):
        print(f"Starting training for {num_epochs} epochs...")
        print("=" * 60)
        
        start_time = time.time()
        
        best_loss = float('inf')
        best_epoch = -1
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 40)
            
            # Training phase
            train_losses = self.train_epoch()
            
            # Validation phase
            val_losses, val_accuracies = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step(val_losses['total_loss'])
            
            # Save history
            for key, value in train_losses.items():
                self.train_history[key].append(value)
            for key, value in val_losses.items():
                self.val_history[key].append(value)
            for key, value in val_accuracies.items():
                self.val_history[f'{key}_acc'].append(value)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"Time: {epoch_time:.2f}s")
            print(f"Train Loss: {train_losses['total_loss']:.4f}")
            print(f"Val Loss: {val_losses['total_loss']:.4f}")
            print(f"Val Accuracies - Quality: {val_accuracies['quality']:.3f}, "
                  f"Size: {val_accuracies['size']:.3f}, "
                  f"Wind: {val_accuracies['wind_dir']:.3f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Improved checkpoint logic
            improved = val_losses['total_loss'] < best_loss * 0.997
            if improved or (epoch - best_epoch) >= 5:
                best_loss, best_epoch = val_losses['total_loss'], epoch
                self.best_model_state = self.model.state_dict().copy()
                torch.save({
                    'model_state_dict': self.best_model_state,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'best_val_loss': best_loss,
                    'train_history': dict(self.train_history),
                    'val_history': dict(self.val_history)
                }, save_path)
                print(f"✓ Model saved! Val Loss: {best_loss:.4f} (epoch {epoch+1})")
            else:
                print(f"No significant improvement. Last best at epoch {best_epoch+1} (Val Loss: {best_loss:.4f})")
            
            # Early stopping and model saving - IMPROVED for small datasets
            min_epochs = 5  # Always train at least 5 epochs
            improvement_margin = 0.05  # Require 5% improvement to reset patience
            
            if epoch >= min_epochs:
                # Check if there's been substantial improvement recently
                recent_losses = self.val_history['total_loss'][-min(5, len(self.val_history['total_loss'])):]
                current_loss = val_losses['total_loss']
                min_recent = min(recent_losses)
                
                # Only trigger early stopping if no improvement over margin
                if current_loss > min_recent * (1 + improvement_margin):
                    effective_patience = self.patience_counter
                else:
                    effective_patience = 0  # Reset if there's meaningful improvement
                    
                if effective_patience >= self.early_stop_patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    print(f"No significant improvement ({improvement_margin*100}%) in last {self.early_stop_patience} epochs")
                    break
            else:
                print(f"Minimum training: {epoch+1}/{min_epochs} epochs completed")
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.2f} minutes")
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Loaded best model with validation loss: {self.best_val_loss:.4f}")
        
        return self.train_history, self.val_history
    
    def plot_training_history(self, save_path='training_plots.png'):
        """Plot training and validation curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16)
        
        # Total loss
        axes[0, 0].plot(self.train_history['total_loss'], label='Train', color='blue')
        axes[0, 0].plot(self.val_history['total_loss'], label='Validation', color='red')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Individual losses
        axes[0, 1].plot(self.val_history['quality_loss'], label='Quality', color='green')
        axes[0, 1].plot(self.val_history['size_loss'], label='Size', color='orange')
        axes[0, 1].plot(self.val_history['wind_loss'], label='Wind', color='purple')
        axes[0, 1].set_title('Validation Losses by Task')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Accuracies
        axes[1, 0].plot(self.val_history['quality_acc'], label='Quality', color='green')
        axes[1, 0].plot(self.val_history['size_acc'], label='Size', color='orange')
        axes[1, 0].plot(self.val_history['wind_dir_acc'], label='Wind', color='purple')
        axes[1, 0].set_title('Validation Accuracies')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate
        if hasattr(self.scheduler, '_last_lr'):
            lr_history = [group['lr'] for group in self.optimizer.param_groups]
            axes[1, 1].plot(lr_history)
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        print(f"Training plots saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='🌊 Train Wave Classification Model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--patience', type=int, default=25, help='Early stopping patience')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--accumulation_steps', type=int, default=2, help='Gradient accumulation steps')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create data loaders
    print("Loading datasets...")
    train_loader, val_loader = create_data_loaders(
        'train_metadata.json',
        'test_metadata.json',  # Using test as validation for now
        batch_size=args.batch_size
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print("\nInitializing model...")
    model = WaveClassifier(
        n_q=9,  # 1-9 quality range
        n_s=7,  # 1-7 size range
        n_w=3   # -1,0,1 wind directions
    )
    
    # Create trainer with custom parameters
    trainer = WaveTrainer(model, train_loader, val_loader)
    trainer.early_stop_patience = args.patience
    trainer.accumulation_steps = args.accumulation_steps
    
    # Update optimizer with custom learning rate and weight decay
    trainer.optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer, 
        mode='min', 
        factor=0.5, 
        patience=10
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=trainer.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"Resumed from epoch {start_epoch}")
        else:
            print(f"Checkpoint {args.resume} not found, starting fresh")
    
    print(f"Training on device: {trainer.device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Early stopping patience: {trainer.early_stop_patience} epochs")
    print(f"Gradient accumulation steps: {trainer.accumulation_steps}")
    
    # Train model
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_name = f"model_{timestamp}.pth"
    train_history, val_history = trainer.train(
        num_epochs=args.epochs,
        save_path=model_save_name
    )
    
    # Plot results
    trainer.plot_training_history('training_plots.png')
    
    # Save training history
    history = {
        'train': dict(train_history),
        'validation': dict(val_history),
        'best_val_loss': trainer.best_val_loss,
        'best_epoch': len(train_history['total_loss']) - trainer.patience_counter - 1
    }
    
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\nTraining completed!")
    print("Files saved:")
    print(f"- {model_save_name} (best model weights)")
    print("- training_plots.png (training curves)")
    print("- training_history.json (detailed history)")

if __name__ == "__main__":
    main() 