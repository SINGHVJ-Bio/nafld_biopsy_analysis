"""
Object detection models for specific NAFLD features
"""

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from typing import Dict, List, Tuple, Optional
import numpy as np
from ..config.parameters import model_params
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DetectionModel(nn.Module):
    """Faster R-CNN based detection model for NAFLD features"""
    
    def __init__(self, num_classes: int = 5):  # background + 4 NAFLD classes
        super().__init__()
        self.num_classes = num_classes
        self.device = torch.device(model_params.device)
        
        # Load pre-trained model
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        
        # Replace the classifier with custom head
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        self.model.to(self.device)
        
        # Training setup
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=model_params.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        
    def forward(self, images, targets=None):
        """Forward pass for detection model"""
        if targets is None:
            return self.model(images)
        else:
            return self.model(images, targets)
    
    def train_epoch(self, train_loader) -> float:
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            self.optimizer.zero_grad()
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            self.optimizer.step()
            
            epoch_loss += losses.item()
            
            if batch_idx % 10 == 0:
                logger.info(f'Batch {batch_idx}, Loss: {losses.item():.4f}')
        
        self.scheduler.step()
        return epoch_loss / len(train_loader)
    
    def validate(self, val_loader) -> Tuple[float, Dict]:
        """Validate detection model"""
        self.model.eval()
        val_loss = 0
        
        # For detection, we might want to compute mAP, but for simplicity we'll just use loss
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
        
        avg_loss = val_loss / len(val_loader)
        
        # Return a tuple with loss and empty dict for compatibility
        return avg_loss, {}
    
    def predict(self, image: np.ndarray, confidence_threshold: float = 0.5) -> Dict:
        """Run detection on single image"""
        self.model.eval()
        
        # Preprocess image
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            predictions = self.model([image_tensor])
        
        # Filter by confidence
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        
        keep = scores >= confidence_threshold
        filtered_boxes = boxes[keep]
        filtered_scores = scores[keep]
        filtered_labels = labels[keep]
        
        return {
            'boxes': filtered_boxes,
            'scores': filtered_scores,
            'labels': filtered_labels
        }

class BallooningDetector:
    """Specialized detector for ballooned hepatocytes"""
    
    def __init__(self):
        self.detection_model = DetectionModel(num_classes=2)  # background + ballooning
        self.feature_extractor = None  # Could add feature extraction for ballooning
    
    def train(self, train_loader, val_loader, num_epochs: int):
        """Train specialized ballooning detector"""
        from .training import ModelTrainer
        trainer = ModelTrainer.create_trainer(
            self.detection_model, 
            "ballooning_detector",
            Path("checkpoints"),
            model_type='classification'  # We'll use classification trainer for simplicity
        )
        trainer.train(train_loader, val_loader, num_epochs)
    
    def detect_ballooned_cells(self, image: np.ndarray) -> List[Dict]:
        """Detect ballooned hepatocytes in image"""
        predictions = self.detection_model.predict(image, confidence_threshold=0.7)
        
        ballooning_detections = []
        for box, score, label in zip(predictions['boxes'], predictions['scores'], predictions['labels']):
            if label == 1:  # Assuming 1 is ballooning class
                detection = {
                    'bbox': box.tolist(),
                    'confidence': float(score),
                    'type': 'ballooning'
                }
                ballooning_detections.append(detection)
        
        return ballooning_detections
    
    def calculate_ballooning_score(self, detections: List[Dict], image_area: float) -> Dict:
        """Calculate ballooning score from detections"""
        if not detections:
            return {
                'ballooning_count': 0,
                'ballooning_density': 0,
                'mean_confidence': 0,
                'ballooning_score': 0
            }
        
        confidences = [det['confidence'] for det in detections]
        
        return {
            'ballooning_count': len(detections),
            'ballooning_density': len(detections) / image_area,
            'mean_confidence': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'ballooning_score': len(detections) * np.mean(confidences)  # Simple scoring
        }