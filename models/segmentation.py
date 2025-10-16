import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from ..config.parameters import model_params
import numpy as np

class DoubleConv(nn.Module):
    """(Conv2D -> BN -> ReLU) * 2"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    """U-Net architecture for segmentation"""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 5, features: List[int] = None):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512, 1024]
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        # Decoder
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(DoubleConv(feature * 2, feature))
        
        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # Decoder
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx//2]
            
            # Handle size mismatches
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx+1](concat_skip)
        
        return self.final_conv(x)

class SegmentationModel:
    """Wrapper for segmentation model with training logic"""
    
    def __init__(self, model_config: dict = None):
        self.config = model_config or {}
        self.model = UNet(
            in_channels=model_params.seg_input_channels,
            out_channels=model_params.seg_output_channels,
            features=model_params.seg_filters
        )
        self.device = torch.device(model_params.device)
        self.model.to(self.device)
        
        # Loss function with class weights for imbalance
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=model_params.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5)
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
        
        return epoch_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        val_loss = 0
        ious = []
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
                
                # Calculate IoU
                iou = self.calculate_iou(outputs, targets)
                ious.append(iou)
        
        return val_loss / len(val_loader), np.mean(ious)
    
    def calculate_iou(self, outputs, targets):
        """Calculate Intersection over Union"""
        # Convert outputs to predictions
        preds = torch.argmax(outputs, dim=1)
        
        # Calculate IoU for each class
        ious = []
        for class_idx in range(model_params.seg_output_channels):
            pred_mask = (preds == class_idx)
            target_mask = (targets == class_idx)
            
            intersection = (pred_mask & target_mask).float().sum()
            union = (pred_mask | target_mask).float().sum()
            
            if union > 0:
                ious.append(intersection / union)
            else:
                ious.append(float('nan'))
        
        return np.nanmean(ious)
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict segmentation for single image"""
        self.model.eval()
        
        # Preprocess image
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            output = self.model(image_tensor)
            prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        
        return prediction