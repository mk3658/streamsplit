"""
Encoder Models for StreamSplit Framework
Implements MobileNetV3-based encoders for edge and server
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math

def make_divisible(v: int, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block"""
    
    def __init__(self, in_channels: int, squeeze_factor: int = 4):
        super().__init__()
        squeeze_channels = make_divisible(in_channels // squeeze_factor, 8)
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels, squeeze_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeeze_channels, in_channels, 1, bias=False),
            nn.Hardsigmoid(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.squeeze(x)
        scale = self.excitation(scale)
        return x * scale

class InvertedResidualBlock(nn.Module):
    """Inverted Residual Block with optional SE and flexible activation"""
    
    def __init__(
        self,
        in_channels: int,
        expanded_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        use_se: bool,
        activation: str = "relu"
    ):
        super().__init__()
        
        self.use_residual = stride == 1 and in_channels == out_channels
        
        # Activation functions
        if activation == "relu":
            activation_fn = nn.ReLU
        elif activation == "hardswish":
            activation_fn = nn.Hardswish
        else:
            activation_fn = nn.ReLU
        
        layers = []
        
        # Expand
        if expanded_channels != in_channels:
            layers.extend([
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                activation_fn(inplace=True)
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv2d(
                expanded_channels,
                expanded_channels,
                kernel_size,
                stride,
                padding=kernel_size // 2,
                groups=expanded_channels,
                bias=False
            ),
            nn.BatchNorm2d(expanded_channels),
            activation_fn(inplace=True)
        ])
        
        # Squeeze-and-Excitation
        if use_se:
            layers.append(SqueezeExcitation(expanded_channels))
        
        # Project
        layers.extend([
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.block(x)
        if self.use_residual:
            result = result + x
        return result

class MobileNetV3EdgeEncoder(nn.Module):
    """
    MobileNetV3-Small based encoder for edge devices
    Supports partial forward pass for dynamic splitting
    """
    
    def __init__(
        self,
        input_dim: int = 128,  # Mel spectrogram features
        embedding_dim: int = 128,
        width_multiplier: float = 0.75,
        num_classes: Optional[int] = None
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # MobileNetV3-Small configuration
        # [input_channels, expanded_channels, output_channels, kernel_size, stride, use_se, activation]
        inverted_residual_setting = [
            [16, 16, 16, 3, 2, True, "relu"],      # Block 1
            [16, 72, 24, 3, 2, False, "relu"],     # Block 2
            [24, 88, 24, 3, 1, False, "relu"],     # Block 3
            [24, 96, 40, 5, 2, True, "hardswish"], # Block 4
            [40, 240, 40, 5, 1, True, "hardswish"], # Block 5
            [40, 240, 40, 5, 1, True, "hardswish"], # Block 6
            [40, 120, 48, 5, 1, True, "hardswish"], # Block 7
            [48, 144, 48, 5, 1, True, "hardswish"], # Block 8
            [48, 288, 96, 5, 2, True, "hardswish"], # Block 9
            [96, 576, 96, 5, 1, True, "hardswish"], # Block 10
            [96, 576, 96, 5, 1, True, "hardswish"], # Block 11
        ]
        
        # Apply width multiplier
        inverted_residual_setting = [
            [
                make_divisible(ic * width_multiplier, 8),
                make_divisible(ec * width_multiplier, 8),
                make_divisible(oc * width_multiplier, 8),
                k, s, se, act
            ]
            for ic, ec, oc, k, s, se, act in inverted_residual_setting
        ]
        
        # First conv layer
        first_conv_output_channels = make_divisible(16 * width_multiplier, 8)
        
        # Build the model
        self.features = nn.ModuleList()
        
        # Initial conv
        self.features.append(nn.Sequential(
            nn.Conv2d(1, first_conv_output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(first_conv_output_channels),
            nn.Hardswish(inplace=True)
        ))
        
        # Inverted residual blocks
        input_channels = first_conv_output_channels
        for ic, ec, oc, k, s, se, act in inverted_residual_setting:
            self.features.append(
                InvertedResidualBlock(input_channels, ec, oc, k, s, se, act)
            )
            input_channels = oc
        
        # Final conv layers
        last_conv_input_channels = input_channels
        last_conv_output_channels = make_divisible(576 * width_multiplier, 8)
        
        self.features.append(nn.Sequential(
            nn.Conv2d(last_conv_input_channels, last_conv_output_channels, 1, bias=False),
            nn.BatchNorm2d(last_conv_output_channels),
            nn.Hardswish(inplace=True)
        ))
        
        # Global average pooling and embedding
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = nn.Sequential(
            nn.Linear(last_conv_output_channels, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True)
        )
        
        # Optional classifier head
        if num_classes is not None:
            self.classifier = nn.Linear(embedding_dim, num_classes)
        else:
            self.classifier = None
        
        # Initialize weights
        self._initialize_weights()
        
        # Store split points for dynamic splitting
        self.split_points = list(range(len(self.features)))
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass"""
        return self.forward_partial(x, len(self.features))
    
    def forward_partial(self, x: torch.Tensor, split_point: int) -> torch.Tensor:
        """Forward pass up to split_point"""
        # Ensure input has correct shape (batch_size, 1, freq, time)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Apply layers up to split point
        for i in range(min(split_point, len(self.features))):
            x = self.features[i](x)
        
        return x
    
    def forward_from_partial(self, x: torch.Tensor, split_point: int) -> torch.Tensor:
        """Continue forward pass from split_point to get final embedding"""
        # Continue from split point
        for i in range(split_point, len(self.features)):
            x = self.features[i](x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Get embedding
        x = self.embedding(x)
        
        # Apply classifier if available
        if self.classifier is not None:
            x = self.classifier(x)
        
        return x
    
    def get_intermediate_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get all intermediate feature maps"""
        features = []
        
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        for layer in self.features:
            x = layer(x)
            features.append(x)
        
        return features

class MobileNetV3ServerEncoder(nn.Module):
    """
    Server-side encoder that continues from edge partial features
    Implements the final layers and refinement
    """
    
    def __init__(
        self,
        intermediate_dim: int = 256,
        embedding_dim: int = 128,
        num_layers: int = 4
    ):
        super().__init__()
        
        self.intermediate_dim = intermediate_dim
        self.embedding_dim = embedding_dim
        
        # Server-side refinement layers
        self.refinement_layers = nn.ModuleList()
        
        # Add several refinement blocks
        for i in range(num_layers):
            if i == 0:
                # First layer adapts from edge feature dimension
                layer = nn.Sequential(
                    nn.Conv2d(intermediate_dim, intermediate_dim, 3, 1, 1),
                    nn.BatchNorm2d(intermediate_dim),
                    nn.ReLU(inplace=True),
                    SqueezeExcitation(intermediate_dim)
                )
            else:
                # Subsequent layers maintain dimension
                layer = nn.Sequential(
                    nn.Conv2d(intermediate_dim, intermediate_dim, 3, 1, 1),
                    nn.BatchNorm2d(intermediate_dim),
                    nn.ReLU(inplace=True),
                    SqueezeExcitation(intermediate_dim)
                )
            self.refinement_layers.append(layer)
        
        # Final processing
        self.final_conv = nn.Sequential(
            nn.Conv2d(intermediate_dim, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Embedding layers
        self.embedding = nn.Sequential(
            nn.Linear(512, embedding_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass (for server-only mode)"""
        # This would typically take raw spectrogram and process it entirely
        # For now, we'll assume it's already partially processed
        return self.forward_from_split(x, 0)
    
    def forward_from_split(self, x: torch.Tensor, split_point: int) -> torch.Tensor:
        """Continue processing from edge split point"""
        # Apply refinement layers
        for layer in self.refinement_layers:
            x = layer(x)
        
        # Final convolution
        x = self.final_conv(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Generate embedding
        x = self.embedding(x)
        
        return x
    
    def full_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Complete forward pass for server-only processing"""
        # For server-only mode, we'd need a complete encoder
        # Here we simulate it by processing the input directly
        
        # Ensure correct input shape
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Simple feature extraction for server-only mode
        x = F.adaptive_avg_pool2d(x, (8, 8))  # Downsample
        x = x.expand(-1, self.intermediate_dim, -1, -1)  # Expand channels
        
        # Apply refinement
        return self.forward_from_split(x, 0)

class HybridMobileNetV3(nn.Module):
    """
    Complete MobileNetV3 model that can be split between edge and server
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        embedding_dim: int = 128,
        width_multiplier: float = 0.75,
        num_classes: Optional[int] = None
    ):
        super().__init__()
        
        self.edge_encoder = MobileNetV3EdgeEncoder(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            width_multiplier=width_multiplier,
            num_classes=None  # No classifier on edge
        )
        
        # Determine intermediate dimension from edge encoder
        # This should match the output of the edge encoder at various split points
        intermediate_dim = 96  # Typical output channels from MobileNetV3-Small
        
        self.server_encoder = MobileNetV3ServerEncoder(
            intermediate_dim=intermediate_dim,
            embedding_dim=embedding_dim
        )
        
        # Optional classifier
        if num_classes is not None:
            self.classifier = nn.Linear(embedding_dim, num_classes)
        else:
            self.classifier = None
    
    def forward(self, x: torch.Tensor, split_point: Optional[int] = None) -> torch.Tensor:
        """Forward pass with optional splitting"""
        if split_point is None:
            # Full edge processing
            return self.edge_encoder(x)
        
        # Split processing
        edge_features = self.edge_encoder.forward_partial(x, split_point)
        
        # Continue on server
        if split_point < len(self.edge_encoder.features):
            # Adjust channels if needed
            if edge_features.size(1) != self.server_encoder.intermediate_dim:
                # Add channel adjustment layer if dimensions don't match
                edge_features = F.adaptive_avg_pool2d(edge_features, (4, 4))
                edge_features = F.interpolate(
                    edge_features, 
                    size=(edge_features.size(2), edge_features.size(3)),
                    mode='bilinear',
                    align_corners=False
                )
                # Pad or project channels
                if edge_features.size(1) < self.server_encoder.intermediate_dim:
                    padding = self.server_encoder.intermediate_dim - edge_features.size(1)
                    edge_features = F.pad(edge_features, (0, 0, 0, 0, 0, padding))
                elif edge_features.size(1) > self.server_encoder.intermediate_dim:
                    edge_features = edge_features[:, :self.server_encoder.intermediate_dim]
            
            embedding = self.server_encoder.forward_from_split(edge_features, 0)
        else:
            # Edge completed everything, just get embedding
            embedding = self.edge_encoder.forward_from_partial(edge_features, split_point)
        
        # Apply classifier if available
        if self.classifier is not None:
            embedding = self.classifier(embedding)
        
        return embedding
    
    def get_split_points(self) -> List[int]:
        """Get available split points"""
        return self.edge_encoder.split_points

# Factory functions for easy model creation
def create_edge_encoder(config) -> MobileNetV3EdgeEncoder:
    """Create edge encoder from config"""
    return MobileNetV3EdgeEncoder(
        input_dim=getattr(config, 'n_mels', 128),
        embedding_dim=getattr(config, 'embedding_dim', 128),
        width_multiplier=getattr(config, 'width_multiplier', 0.75)
    )

def create_server_encoder(config) -> MobileNetV3ServerEncoder:
    """Create server encoder from config"""
    return MobileNetV3ServerEncoder(
        intermediate_dim=getattr(config, 'intermediate_dim', 256),
        embedding_dim=getattr(config, 'embedding_dim', 128),
        num_layers=getattr(config, 'server_layers', 4)
    )

def create_hybrid_model(config) -> HybridMobileNetV3:
    """Create complete hybrid model from config"""
    return HybridMobileNetV3(
        input_dim=getattr(config, 'n_mels', 128),
        embedding_dim=getattr(config, 'embedding_dim', 128),
        width_multiplier=getattr(config, 'width_multiplier', 0.75),
        num_classes=getattr(config, 'num_classes', None)
    )