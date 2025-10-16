"""
Model training utilities with comprehensive trainer classes for NAFLD pipeline
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from pathlib import Path
import logging
from ..config.parameters import model_params

logger = logging.getLogger(__name__)

class BaseTrainer:
    """Base trainer class with common training functionality"""
    
    def __init__(self, model, model_name: str, checkpoint_dir: Path):
        self.model = model
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(model_params.device)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        
        # Early stopping
        self.best_metric = float('inf')
        self.epochs_no_improve = 0
        self.early_stop = False
        
    def train_epoch(self, train_loader) -> float:
        """Train for one epoch - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement train_epoch")
    
    def validate(self, val_loader) -> Tuple[float, Optional[float]]:
        """Validate model - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement validate")
    
    def _is_best_metric(self, current_metric: float) -> bool:
        """Check if current metric is the best"""
        return current_metric < self.best_metric
    
    def _save_checkpoint(self, epoch: int, metric: float, additional_info: Dict = None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'metric': metric,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates
        }
        
        # Save optimizer state if available
        if hasattr(self.model, 'optimizer'):
            checkpoint['optimizer_state_dict'] = self.model.optimizer.state_dict()
        
        # Save scheduler state if available
        if hasattr(self.model, 'scheduler'):
            checkpoint['scheduler_state_dict'] = self.model.scheduler.state_dict()
        
        if additional_info:
            checkpoint.update(additional_info)
        
        checkpoint_path = self.checkpoint_dir / f"{self.model_name}_best.pth"
        torch.save(checkpoint, checkpoint_path)
        self.best_metric = metric
        
        logger.info(f"Saved checkpoint at epoch {epoch} with metric {metric:.4f}")
    
    def _load_best_checkpoint(self):
        """Load the best checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"{self.model_name}_best.pth"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state if available
            if hasattr(self.model, 'optimizer') and 'optimizer_state_dict' in checkpoint:
                self.model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state if available
            if hasattr(self.model, 'scheduler') and 'scheduler_state_dict' in checkpoint:
                self.model.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            logger.info(f"Loaded best checkpoint from epoch {checkpoint['epoch']}")
    
    def _log_metrics(self, epoch: int, train_loss: float, val_loss: float, 
                   train_acc: Optional[float] = None, val_acc: Optional[float] = None):
        """Log training metrics"""
        log_msg = f"Epoch {epoch:03d}: "
        log_msg += f"Train Loss: {train_loss:.4f} | "
        log_msg += f"Val Loss: {val_loss:.4f}"
        
        if train_acc is not None and val_acc is not None:
            log_msg += f" | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"
        
        # Log learning rate if available
        if hasattr(self.model, 'optimizer'):
            current_lr = self.model.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            log_msg += f" | LR: {current_lr:.2e}"
        
        logger.info(log_msg)
        
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        if train_acc is not None:
            self.train_accuracies.append(train_acc)
        if val_acc is not None:
            self.val_accuracies.append(val_acc)
    
    def _check_early_stopping(self, current_metric: float) -> bool:
        """Check if training should stop early"""
        if self._is_best_metric(current_metric):
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1
        
        if self.epochs_no_improve >= model_params.early_stopping_patience:
            logger.info(f"Early stopping triggered after {self.epochs_no_improve} epochs without improvement")
            return True
        return False

class ClassificationTrainer(BaseTrainer):
    """Trainer for classification models"""
    
    def __init__(self, model, model_name: str, checkpoint_dir: Path):
        super().__init__(model, model_name, checkpoint_dir)
        self.model.to(self.device)
    
    def train_epoch(self, train_loader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.model.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.model.criterion(outputs, targets)
            loss.backward()
            self.model.optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Log batch progress occasionally
            if batch_idx % 10 == 0:
                batch_acc = 100 * correct / total if total > 0 else 0
                logger.debug(f'Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {batch_acc:.2f}%')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total if total > 0 else 0
        
        return epoch_loss, epoch_accuracy
    
    def validate(self, val_loader) -> Tuple[float, float]:
        """Validate classification model"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.model.criterion(outputs, targets)
                
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total if total > 0 else 0
        
        return val_loss, val_accuracy
    
    def train(self, train_loader, val_loader, num_epochs: int):
        """Complete training loop for classification"""
        logger.info(f"Starting training for {self.model_name} for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            if self.early_stop:
                break
                
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader)
            
            # Update learning rate scheduler if available
            if hasattr(self.model, 'scheduler'):
                self.model.scheduler.step(val_loss)
            
            # Log metrics
            self._log_metrics(epoch, train_loss, val_loss, train_acc, val_acc)
            
            # Checkpoint based on validation loss
            if self._is_best_metric(val_loss):
                self._save_checkpoint(epoch, val_loss, {
                    'train_accuracy': train_acc,
                    'val_accuracy': val_acc
                })
            
            # Check early stopping
            self.early_stop = self._check_early_stopping(val_loss)
        
        # Load best model
        self._load_best_checkpoint()
        logger.info(f"Training completed for {self.model_name}")

class SegmentationTrainer(BaseTrainer):
    """Trainer for segmentation models"""
    
    def __init__(self, model, model_name: str, checkpoint_dir: Path):
        super().__init__(model, model_name, checkpoint_dir)
        self.model.to(self.device)
        self.best_iou = 0.0
    
    def train_epoch(self, train_loader) -> float:
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.model.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.model.criterion(outputs, targets)
            loss.backward()
            self.model.optimizer.step()
            
            epoch_loss += loss.item()
            
            # Log batch progress occasionally
            if batch_idx % 10 == 0:
                logger.debug(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        return epoch_loss / len(train_loader)
    
    def validate(self, val_loader) -> Tuple[float, float]:
        """Validate segmentation model"""
        self.model.eval()
        val_loss = 0
        ious = []
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.model.criterion(outputs, targets)
                val_loss += loss.item()
                
                # Calculate IoU
                iou = self.calculate_iou(outputs, targets)
                ious.append(iou)
        
        avg_val_loss = val_loss / len(val_loader)
        avg_iou = np.mean(ious) if ious else 0.0
        
        return avg_val_loss, avg_iou
    
    def calculate_iou(self, outputs, targets):
        """Calculate Intersection over Union for segmentation"""
        # Convert outputs to predictions
        preds = torch.argmax(outputs, dim=1)
        
        # Calculate IoU for each class
        ious = []
        num_classes = outputs.shape[1]
        
        for class_idx in range(num_classes):
            pred_mask = (preds == class_idx)
            target_mask = (targets == class_idx)
            
            intersection = (pred_mask & target_mask).float().sum()
            union = (pred_mask | target_mask).float().sum()
            
            if union > 0:
                ious.append((intersection / union).item())
            else:
                ious.append(float('nan'))
        
        # Return mean IoU, ignoring NaN values
        valid_ious = [iou for iou in ious if not np.isnan(iou)]
        return np.mean(valid_ious) if valid_ious else 0.0
    
    def calculate_dice_score(self, outputs, targets):
        """Calculate Dice coefficient for segmentation"""
        preds = torch.argmax(outputs, dim=1)
        
        dice_scores = []
        num_classes = outputs.shape[1]
        
        for class_idx in range(num_classes):
            pred_mask = (preds == class_idx)
            target_mask = (targets == class_idx)
            
            intersection = (pred_mask & target_mask).float().sum()
            total = pred_mask.float().sum() + target_mask.float().sum()
            
            if total > 0:
                dice = (2.0 * intersection) / total
                dice_scores.append(dice.item())
            else:
                dice_scores.append(float('nan'))
        
        valid_dice = [dice for dice in dice_scores if not np.isnan(dice)]
        return np.mean(valid_dice) if valid_dice else 0.0
    
    def train(self, train_loader, val_loader, num_epochs: int):
        """Complete training loop for segmentation"""
        logger.info(f"Starting training for {self.model_name} for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            if self.early_stop:
                break
                
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_iou = self.validate(val_loader)
            
            # Calculate Dice score for additional metrics
            val_dice = self.calculate_dice_score(
                next(iter(val_loader))[0].to(self.device),
                next(iter(val_loader))[1].to(self.device)
            )
            
            # Update learning rate scheduler if available
            if hasattr(self.model, 'scheduler'):
                self.model.scheduler.step(val_loss)
            
            # Log metrics
            self._log_metrics(epoch, train_loss, val_loss)
            logger.info(f"Epoch {epoch:03d}: Val IoU: {val_iou:.4f}, Val Dice: {val_dice:.4f}")
            
            # Checkpoint based on validation loss
            if self._is_best_metric(val_loss):
                self._save_checkpoint(epoch, val_loss, {
                    'val_iou': val_iou,
                    'val_dice': val_dice
                })
                self.best_iou = val_iou
            
            # Check early stopping
            self.early_stop = self._check_early_stopping(val_loss)
        
        # Load best model
        self._load_best_checkpoint()
        logger.info(f"Training completed for {self.model_name}. Best IoU: {self.best_iou:.4f}")

class DetectionTrainer(BaseTrainer):
    """Trainer for object detection models"""
    
    def __init__(self, model, model_name: str, checkpoint_dir: Path):
        super().__init__(model, model_name, checkpoint_dir)
        self.model.to(self.device)
    
    def train_epoch(self, train_loader) -> float:
        """Train for one epoch for detection"""
        self.model.train()
        epoch_loss = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            self.model.optimizer.zero_grad()
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            self.model.optimizer.step()
            
            epoch_loss += losses.item()
            
            # Log batch progress occasionally
            if batch_idx % 10 == 0:
                logger.debug(f'Batch {batch_idx}, Loss: {losses.item():.4f}')
                # Log individual losses
                for loss_name, loss_value in loss_dict.items():
                    logger.debug(f'  {loss_name}: {loss_value.item():.4f}')
        
        return epoch_loss / len(train_loader)
    
    def validate(self, val_loader) -> Tuple[float, Dict]:
        """Validate detection model"""
        self.model.eval()
        val_loss = 0
        loss_components = {}
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
                
                # Accumulate loss components
                for loss_name, loss_value in loss_dict.items():
                    if loss_name not in loss_components:
                        loss_components[loss_name] = 0
                    loss_components[loss_name] += loss_value.item()
        
        avg_loss = val_loss / len(val_loader)
        
        # Average loss components
        for loss_name in loss_components:
            loss_components[loss_name] /= len(val_loader)
        
        return avg_loss, loss_components
    
    def calculate_map(self, val_loader, iou_threshold: float = 0.5):
        """Calculate mAP for detection model (simplified version)"""
        # This is a simplified mAP calculation
        # In practice, you'd use a proper evaluation metric from torchvision
        self.model.eval()
        all_detections = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(self.device) for img in images]
                outputs = self.model(images)
                
                for i, output in enumerate(outputs):
                    boxes = output['boxes'].cpu().numpy()
                    scores = output['scores'].cpu().numpy()
                    labels = output['labels'].cpu().numpy()
                    
                    target_boxes = targets[i]['boxes'].cpu().numpy()
                    target_labels = targets[i]['labels'].cpu().numpy()
                    
                    # Simplified matching - in practice use proper IoU calculation
                    matches = self._match_detections(boxes, target_boxes, iou_threshold)
                    
                    all_detections.append({
                        'boxes': boxes,
                        'scores': scores,
                        'labels': labels,
                        'matches': matches
                    })
                    all_targets.append({
                        'boxes': target_boxes,
                        'labels': target_labels
                    })
        
        # Simplified mAP calculation
        precision, recall = self._calculate_precision_recall(all_detections, all_targets)
        ap = self._calculate_average_precision(precision, recall)
        
        return ap
    
    def _match_detections(self, pred_boxes, target_boxes, iou_threshold):
        """Match predictions to ground truth boxes"""
        # Simplified matching - in practice use proper IoU computation
        if len(pred_boxes) == 0 or len(target_boxes) == 0:
            return []
        
        # Calculate IoU between all pairs (simplified)
        matches = []
        for i, pred_box in enumerate(pred_boxes):
            max_iou = 0
            best_match = -1
            for j, target_box in enumerate(target_boxes):
                iou = self._calculate_iou(pred_box, target_box)
                if iou > max_iou and iou >= iou_threshold:
                    max_iou = iou
                    best_match = j
            matches.append(best_match)
        
        return matches
    
    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        # Simplified IoU calculation
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _calculate_precision_recall(self, detections, targets):
        """Calculate precision and recall"""
        # Simplified precision/recall calculation
        true_positives = 0
        false_positives = 0
        total_targets = sum(len(t['boxes']) for t in targets)
        
        for det in detections:
            for match in det['matches']:
                if match != -1:
                    true_positives += 1
                else:
                    false_positives += 1
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / total_targets if total_targets > 0 else 0
        
        return precision, recall
    
    def _calculate_average_precision(self, precision, recall):
        """Calculate average precision"""
        # Simplified AP calculation
        return precision * recall  # In practice, use proper integration
    
    def train(self, train_loader, val_loader, num_epochs: int):
        """Complete training loop for detection"""
        logger.info(f"Starting training for {self.model_name} for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            if self.early_stop:
                break
                
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss, loss_components = self.validate(val_loader)
            
            # Update learning rate scheduler if available
            if hasattr(self.model, 'scheduler'):
                self.model.scheduler.step(val_loss)
            
            # Log metrics
            self._log_metrics(epoch, train_loss, val_loss)
            
            # Log loss components
            loss_msg = "Loss components: "
            for loss_name, loss_value in loss_components.items():
                loss_msg += f"{loss_name}: {loss_value:.4f}, "
            logger.info(loss_msg[:-2])  # Remove trailing comma and space
            
            # Checkpoint based on validation loss
            if self._is_best_metric(val_loss):
                self._save_checkpoint(epoch, val_loss, {
                    'loss_components': loss_components
                })
            
            # Check early stopping
            self.early_stop = self._check_early_stopping(val_loss)
        
        # Load best model
        self._load_best_checkpoint()
        logger.info(f"Training completed for {self.model_name}")

class ModelTrainer:
    """Generic model trainer that automatically selects the right trainer type"""
    
    @staticmethod
    def create_trainer(model, model_name: str, checkpoint_dir: Path, model_type: str = None):
        """Create appropriate trainer based on model type"""
        if model_type is None:
            # Auto-detect model type
            if hasattr(model, 'criterion') and hasattr(model, 'optimizer'):
                if hasattr(model, 'calculate_iou') or 'segmentation' in model_name.lower():
                    model_type = 'segmentation'
                elif hasattr(model, 'roi_heads') or 'detection' in model_name.lower():
                    model_type = 'detection'
                else:
                    model_type = 'classification'
            else:
                raise ValueError("Cannot auto-detect model type. Please specify model_type.")
        
        if model_type == 'classification':
            return ClassificationTrainer(model, model_name, checkpoint_dir)
        elif model_type == 'segmentation':
            return SegmentationTrainer(model, model_name, checkpoint_dir)
        elif model_type == 'detection':
            return DetectionTrainer(model, model_name, checkpoint_dir)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    @staticmethod
    def get_available_trainers():
        """Get list of available trainer types"""
        return ['classification', 'segmentation', 'detection']

class CrossValidator:
    """Cross-validation for model evaluation"""
    
    def __init__(self, model_class, n_splits: int = 5):
        self.model_class = model_class
        self.n_splits = n_splits
        self.fold_metrics = []
        self.fold_models = []
    
    def cross_validate(self, dataset, model_params: dict, model_type: str = 'classification'):
        """Perform k-fold cross-validation"""
        from sklearn.model_selection import KFold
        
        kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            logger.info(f"Starting fold {fold + 1}/{self.n_splits}")
            
            # Create data loaders for this fold
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
            
            train_loader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=model_params.get('batch_size', 8), 
                sampler=train_subsampler
            )
            val_loader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=model_params.get('batch_size', 8), 
                sampler=val_subsampler
            )
            
            # Initialize model
            model = self.model_class(**model_params)
            
            # Create trainer
            trainer = ModelTrainer.create_trainer(
                model, 
                f"fold_{fold}", 
                Path("checkpoints") / f"fold_{fold}",
                model_type
            )
            
            # Train
            num_epochs = model_params.get('num_epochs', 50)
            trainer.train(train_loader, val_loader, num_epochs)
            
            # Store metrics
            fold_metrics = {
                'fold': fold,
                'best_train_loss': min(trainer.train_losses) if trainer.train_losses else float('inf'),
                'best_val_loss': min(trainer.val_losses) if trainer.val_losses else float('inf'),
            }
            
            # Add accuracy metrics for classification
            if hasattr(trainer, 'train_accuracies') and trainer.train_accuracies:
                fold_metrics['best_train_accuracy'] = max(trainer.train_accuracies)
                fold_metrics['best_val_accuracy'] = max(trainer.val_accuracies)
            
            # Add IoU for segmentation
            if model_type == 'segmentation' and hasattr(trainer, 'best_iou'):
                fold_metrics['best_iou'] = trainer.best_iou
            
            self.fold_metrics.append(fold_metrics)
            self.fold_models.append(model)
        
        return self._summarize_cv()
    
    def _summarize_cv(self) -> Dict:
        """Summarize cross-validation results"""
        if not self.fold_metrics:
            return {}
            
        df = pd.DataFrame(self.fold_metrics)
        summary = {
            'mean_val_loss': df['best_val_loss'].mean(),
            'std_val_loss': df['best_val_loss'].std(),
            'fold_results': df.to_dict('records')
        }
        
        # Add accuracy summary if available
        if 'best_val_accuracy' in df.columns:
            summary['mean_val_accuracy'] = df['best_val_accuracy'].mean()
            summary['std_val_accuracy'] = df['best_val_accuracy'].std()
        
        # Add IoU summary if available
        if 'best_iou' in df.columns:
            summary['mean_iou'] = df['best_iou'].mean()
            summary['std_iou'] = df['best_iou'].std()
        
        logger.info(f"CV Results - Val Loss: {summary['mean_val_loss']:.4f} ± {summary['std_val_loss']:.4f}")
        
        if 'mean_val_accuracy' in summary:
            logger.info(f"CV Results - Val Accuracy: {summary['mean_val_accuracy']:.2f}% ± {summary['std_val_accuracy']:.2f}%")
        
        if 'mean_iou' in summary:
            logger.info(f"CV Results - Mean IoU: {summary['mean_iou']:.4f} ± {summary['std_iou']:.4f}")
        
        return summary
    
    def get_best_fold_model(self, metric: str = 'val_loss', maximize: bool = False):
        """Get the best model from cross-validation"""
        if not self.fold_metrics:
            return None
        
        if maximize:
            best_fold_idx = np.argmax([fold[metric] for fold in self.fold_metrics])
        else:
            best_fold_idx = np.argmin([fold[metric] for fold in self.fold_metrics])
        
        return self.fold_models[best_fold_idx], self.fold_metrics[best_fold_idx]

class TrainingProgressTracker:
    """Track and visualize training progress"""
    
    def __init__(self):
        self.history = {}
    
    def add_training_run(self, run_name: str, trainer: BaseTrainer):
        """Add training run to tracker"""
        self.history[run_name] = {
            'train_losses': trainer.train_losses,
            'val_losses': trainer.val_losses,
            'train_accuracies': trainer.train_accuracies if hasattr(trainer, 'train_accuracies') else None,
            'val_accuracies': trainer.val_accuracies if hasattr(trainer, 'val_accuracies') else None,
            'learning_rates': trainer.learning_rates
        }
    
    def plot_training_history(self, run_names: List[str] = None, output_path: Path = None):
        """Plot training history for multiple runs"""
        import matplotlib.pyplot as plt
        
        if run_names is None:
            run_names = list(self.history.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for run_name in run_names:
            if run_name not in self.history:
                continue
                
            history = self.history[run_name]
            epochs = range(1, len(history['train_losses']) + 1)
            
            # Loss plot
            axes[0, 0].plot(epochs, history['train_losses'], label=f'{run_name} Train')
            axes[0, 0].plot(epochs, history['val_losses'], label=f'{run_name} Val', linestyle='--')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].set_xlabel('Epochs')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Accuracy plot (if available)
            if history['train_accuracies'] and history['val_accuracies']:
                axes[0, 1].plot(epochs, history['train_accuracies'], label=f'{run_name} Train')
                axes[0, 1].plot(epochs, history['val_accuracies'], label=f'{run_name} Val', linestyle='--')
                axes[0, 1].set_title('Training and Validation Accuracy')
                axes[0, 1].set_xlabel('Epochs')
                axes[0, 1].set_ylabel('Accuracy (%)')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # Learning rate plot
            if history['learning_rates']:
                axes[1, 0].plot(epochs, history['learning_rates'], label=run_name)
                axes[1, 0].set_title('Learning Rate Schedule')
                axes[1, 0].set_xlabel('Epochs')
                axes[1, 0].set_ylabel('Learning Rate')
                axes[1, 0].set_yscale('log')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
        
        # Hide empty subplot
        axes[1, 1].set_visible(False)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {output_path}")
        
        return fig
    
    def save_training_history(self, output_path: Path):
        """Save training history to file"""
        import json
        
        # Convert to serializable format
        serializable_history = {}
        for run_name, history in self.history.items():
            serializable_history[run_name] = {
                'train_losses': history['train_losses'],
                'val_losses': history['val_losses'],
                'train_accuracies': history['train_accuracies'],
                'val_accuracies': history['val_accuracies'],
                'learning_rates': history['learning_rates']
            }
        
        with open(output_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        logger.info(f"Training history saved to {output_path}")
    
    def load_training_history(self, input_path: Path):
        """Load training history from file"""
        import json
        
        with open(input_path, 'r') as f:
            self.history = json.load(f)
        
        logger.info(f"Training history loaded from {input_path}")