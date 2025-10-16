"""
Classification models for slide-level predictions
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path
from ..config.parameters import model_params
from .training import ModelTrainer  # Add this import
import logging

logger = logging.getLogger(__name__)

class SlideClassifier(nn.Module):
    """Whole slide image classifier for NAFLD grading"""
    
    def __init__(self, num_classes: int = 3, model_name: str = "resnet50"):
        super().__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        self.device = torch.device(model_params.device)
        
        # Initialize model
        self.model = self._initialize_model(model_name, num_classes)
        self.model.to(self.device)
        
        # Training setup
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=model_params.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5)
        
    def _initialize_model(self, model_name: str, num_classes: int) -> nn.Module:
        """Initialize pre-trained model"""
        if model_name == "resnet50":
            model = models.resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == "efficientnet":
            model = models.efficientnet_b0(pretrained=True)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif model_name == "densenet":
            model = models.densenet121(pretrained=True)
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        return model
    
    def forward(self, x):
        return self.model(x)
    
    def train_epoch(self, train_loader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        
        return epoch_loss, epoch_accuracy
    
    def validate(self, val_loader) -> Tuple[float, float]:
        """Validate classifier"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        
        return val_loss, val_accuracy
    
    def predict(self, image: np.ndarray) -> Dict:
        """Predict class for single image"""
        self.model.eval()
        
        # Preprocess image
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        return {
            'predicted_class': predicted.item(),
            'confidence': confidence.item(),
            'probabilities': probabilities.cpu().numpy()[0].tolist()
        }
    
    def predict_batch(self, images: List[np.ndarray]) -> List[Dict]:
        """Predict classes for batch of images"""
        return [self.predict(img) for img in images]

class NASClassifier:
    """Specialized classifier for NAFLD Activity Score (NAS) prediction"""
    
    def __init__(self):
        # Multi-output classifier for NAS components
        self.steatosis_classifier = SlideClassifier(num_classes=4)  # S0-S3
        self.ballooning_classifier = SlideClassifier(num_classes=3)  # 0-2
        self.inflammation_classifier = SlideClassifier(num_classes=3)  # 0-2
        
        # Store trainers for each component
        self.trainers = {}
    
    def train_component_classifiers(self, train_loaders: Dict, val_loaders: Dict, num_epochs: int):
        """Train individual NAS component classifiers"""
        components = ['steatosis', 'ballooning', 'inflammation']
        
        for component in components:
            logger.info(f"Training {component} classifier")
            
            # Create appropriate trainer - ModelTrainer is now imported
            trainer = ModelTrainer.create_trainer(
                getattr(self, f'{component}_classifier'), 
                f"{component}_classifier",
                Path("checkpoints"),
                model_type='classification'
            )
            
            # Store trainer for later use
            self.trainers[component] = trainer
            
            # Train the model
            trainer.train(
                train_loaders[component], 
                val_loaders[component], 
                num_epochs
            )
    
    def predict_nas(self, image: np.ndarray) -> Dict:
        """Predict complete NAS score"""
        steatosis_pred = self.steatosis_classifier.predict(image)
        ballooning_pred = self.ballooning_classifier.predict(image)
        inflammation_pred = self.inflammation_classifier.predict(image)
        
        # Calculate total NAS
        steatosis_score = steatosis_pred['predicted_class']
        ballooning_score = ballooning_pred['predicted_class']
        inflammation_score = inflammation_pred['predicted_class']
        total_nas = steatosis_score + ballooning_score + inflammation_score
        
        return {
            'steatosis_grade': steatosis_score,
            'ballooning_grade': ballooning_score,
            'inflammation_grade': inflammation_score,
            'total_nas': total_nas,
            'steatosis_confidence': steatosis_pred['confidence'],
            'ballooning_confidence': ballooning_pred['confidence'],
            'inflammation_confidence': inflammation_pred['confidence'],
            'component_probabilities': {
                'steatosis': steatosis_pred['probabilities'],
                'ballooning': ballooning_pred['probabilities'],
                'inflammation': inflammation_pred['probabilities']
            }
        }
    
    def predict_fibrosis_stage(self, image: np.ndarray) -> Dict:
        """Predict fibrosis stage (placeholder - would need separate model)"""
        # This would require a separate fibrosis-specific model
        # Using the main classifier as placeholder
        pred = self.steatosis_classifier.predict(image)
        
        return {
            'fibrosis_stage': min(pred['predicted_class'], 4),  # Cap at F4
            'confidence': pred['confidence']
        }
    
    def get_training_history(self, component: str) -> Dict:
        """Get training history for a specific component"""
        if component in self.trainers:
            trainer = self.trainers[component]
            return {
                'train_losses': trainer.train_losses,
                'val_losses': trainer.val_losses,
                'train_accuracies': trainer.train_accuracies,
                'val_accuracies': trainer.val_accuracies
            }
        return {}