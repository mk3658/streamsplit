"""
Dynamic Model Transformations for StreamSplit Framework
Implements adaptive neural architecture transformations at splitting points
Based on Section 3.3.2 of the paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from abc import ABC, abstractmethod

class ModelTransformation(ABC):
    """Abstract base class for model transformations"""
    
    @abstractmethod
    def apply(self, model: nn.Module, split_point: int, features: torch.Tensor) -> torch.Tensor:
        """Apply transformation at split point"""
        pass
    
    @abstractmethod
    def get_compression_ratio(self) -> float:
        """Get compression ratio achieved by transformation"""
        pass

class BottleneckInsertion(ModelTransformation):
    """
    Bottleneck insertion for feature compression
    T_bottleneck(f_θ, k) = f_θ,1:k ○ h_φ ○ f_θ,k+1:L (Equation 18)
    """
    
    def __init__(self, compression_ratio: float = 0.5, device: str = 'cpu'):
        self.compression_ratio = compression_ratio
        self.device = device
        self.bottleneck_layers = {}
        self.optimized = False
        
    def _create_bottleneck(self, input_dim: int, output_dim: int) -> nn.Module:
        """Create bottleneck layer h_φ"""
        bottleneck_dim = max(16, int(input_dim * self.compression_ratio))
        
        bottleneck = nn.Sequential(
            # Compression
            nn.Linear(input_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(bottleneck_dim),
            nn.Dropout(0.1),
            
            # Expansion
            nn.Linear(bottleneck_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(output_dim)
        ).to(self.device)
        
        return bottleneck
    
    def _optimize_bottleneck(self, model: nn.Module, split_point: int, 
                           data_loader: torch.utils.data.DataLoader):
        """
        Optimize bottleneck parameters φ according to Equation 18:
        min_φ E_x~D[||f_θ,k+1:L(f_θ,1:k(x)) - f_θ,k+1:L(h_φ(f_θ,1:k(x)))||²]
        """
        if split_point not in self.bottleneck_layers:
            return
        
        bottleneck = self.bottleneck_layers[split_point]
        optimizer = torch.optim.Adam(bottleneck.parameters(), lr=1e-3)
        
        model.eval()
        bottleneck.train()
        
        for batch in data_loader:
            optimizer.zero_grad()
            
            with torch.no_grad():
                # Get features at split point
                features = model.forward_partial(batch, split_point)
                # Get target output (without bottleneck)
                target = model.forward_from_partial(features, split_point)
            
            # Forward through bottleneck
            compressed_features = bottleneck(features.flatten(1))
            # Reshape to match original feature shape
            compressed_features = compressed_features.view_as(features)
            
            # Forward from compressed features
            output = model.forward_from_partial(compressed_features, split_point)
            
            # Compute reconstruction loss
            loss = F.mse_loss(output, target)
            loss.backward()
            optimizer.step()
    
    def apply(self, model: nn.Module, split_point: int, features: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck transformation at split point"""
        # Get feature dimensions
        if features.dim() > 2:
            # Flatten spatial dimensions for bottleneck
            original_shape = features.shape
            flat_features = features.flatten(1)
            input_dim = flat_features.size(1)
            output_dim = input_dim  # Maintain same output dimension
        else:
            flat_features = features
            input_dim = features.size(1)
            output_dim = input_dim
        
        # Create or retrieve bottleneck layer
        if split_point not in self.bottleneck_layers:
            self.bottleneck_layers[split_point] = self._create_bottleneck(input_dim, output_dim)
        
        bottleneck = self.bottleneck_layers[split_point]
        
        # Apply bottleneck transformation
        compressed = bottleneck(flat_features)
        
        # Reshape back to original shape if needed
        if features.dim() > 2:
            compressed = compressed.view(original_shape)
        
        return compressed
    
    def get_compression_ratio(self) -> float:
        """Get compression ratio achieved by bottleneck"""
        return self.compression_ratio
    
    def optimize(self, model: nn.Module, split_point: int, data_loader: torch.utils.data.DataLoader):
        """Optimize bottleneck for given split point"""
        self._optimize_bottleneck(model, split_point, data_loader)

class LayerQuantization(ModelTransformation):
    """
    Dynamic layer quantization for computation efficiency
    T_quant(f_θ, k, b) = Q_b(f_θ,1:k) ○ f_θ,k+1:L
    """
    
    def __init__(self, bits: int = 8, device: str = 'cpu'):
        self.bits = bits
        self.device = device
        self.quantization_params = {}
        
    def _compute_quantization_params(self, tensor: torch.Tensor) -> Tuple[float, float]:
        """Compute quantization scale and zero point"""
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        
        # Symmetric quantization
        max_range = max(abs(min_val), abs(max_val))
        scale = max_range / (2 ** (self.bits - 1) - 1)
        zero_point = 0
        
        return scale, zero_point
    
    def _quantize_tensor(self, tensor: torch.Tensor, scale: float, zero_point: float) -> torch.Tensor:
        """Quantize tensor using scale and zero point"""
        # Quantize
        quantized = torch.round(tensor / scale + zero_point)
        
        # Clamp to valid range
        min_val = -(2 ** (self.bits - 1))
        max_val = 2 ** (self.bits - 1) - 1
        quantized = torch.clamp(quantized, min_val, max_val)
        
        # Dequantize
        dequantized = (quantized - zero_point) * scale
        
        return dequantized
    
    def apply(self, model: nn.Module, split_point: int, features: torch.Tensor) -> torch.Tensor:
        """Apply quantization transformation"""
        # Compute or retrieve quantization parameters
        layer_key = f"{split_point}_{features.shape}"
        
        if layer_key not in self.quantization_params:
            scale, zero_point = self._compute_quantization_params(features)
            self.quantization_params[layer_key] = (scale, zero_point)
        else:
            scale, zero_point = self.quantization_params[layer_key]
        
        # Apply quantization
        quantized_features = self._quantize_tensor(features, scale, zero_point)
        
        return quantized_features
    
    def get_compression_ratio(self) -> float:
        """Get compression ratio based on bit reduction"""
        # Assuming original features are float32 (32 bits)
        return self.bits / 32.0

class ConditionalComputation(ModelTransformation):
    """
    Conditional computation based on input complexity
    T_cond(f_θ, k)(x) = { f_θ,1:k(x) ○ f_θ,k+1:L(x), if c(x) > c_threshold
                          { f_θ,1:k(x) ○ f_lite,k+1:L(x), otherwise (Equation 19)
    """
    
    def __init__(self, threshold: float = 0.5, device: str = 'cpu'):
        self.threshold = threshold
        self.device = device
        self.complexity_estimators = {}
        self.lite_models = {}
        
    def _create_complexity_estimator(self, input_shape: Tuple[int, ...]) -> nn.Module:
        """Create lightweight complexity estimator c(x)"""
        # Calculate input dimension
        if len(input_shape) > 2:
            # Convolutional features
            input_dim = int(np.prod(input_shape[1:]))
            estimator = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        else:
            # Linear features
            input_dim = input_shape[1]
            estimator = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 16),
                nn.ReLU(inplace=True),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
        
        return estimator.to(self.device)
    
    def _create_lite_model(self, model: nn.Module, split_point: int) -> nn.Module:
        """Create lightweight version of model layers f_lite,k+1:L"""
        # Extract layers after split point
        original_layers = list(model.children())[split_point:]
        
        # Create simplified version
        lite_layers = []
        for layer in original_layers:
            if isinstance(layer, nn.Conv2d):
                # Reduce channels by half
                lite_layer = nn.Conv2d(
                    layer.in_channels,
                    max(1, layer.out_channels // 2),
                    layer.kernel_size,
                    layer.stride,
                    layer.padding,
                    bias=layer.bias is not None
                )
                # Add channel adjustment if needed
                if lite_layer.out_channels != layer.out_channels:
                    channel_adj = nn.Conv2d(
                        lite_layer.out_channels,
                        layer.out_channels,
                        1, 1, 0,
                        bias=False
                    )
                    lite_layers.extend([lite_layer, channel_adj])
                else:
                    lite_layers.append(lite_layer)
                    
            elif isinstance(layer, nn.Linear):
                # Reduce hidden dimensions
                lite_layer = nn.Linear(
                    layer.in_features,
                    max(1, layer.out_features // 2)
                )
                # Add dimension adjustment
                dim_adj = nn.Linear(
                    lite_layer.out_features,
                    layer.out_features
                )
                lite_layers.extend([lite_layer, dim_adj])
            else:
                # Keep other layers as is
                lite_layers.append(layer)
        
        return nn.Sequential(*lite_layers).to(self.device)
    
    def _estimate_complexity(self, features: torch.Tensor, split_point: int) -> torch.Tensor:
        """Estimate input complexity c(x)"""
        # Create or retrieve complexity estimator
        shape_key = f"{split_point}_{features.shape[1:]}"
        
        if shape_key not in self.complexity_estimators:
            self.complexity_estimators[shape_key] = self._create_complexity_estimator(features.shape)
        
        estimator = self.complexity_estimators[shape_key]
        
        # Estimate complexity (multiple methods)
        with torch.no_grad():
            # Method 1: Neural estimator
            neural_complexity = estimator(features)
            
            # Method 2: Feature magnitude
            magnitude_complexity = torch.norm(features.flatten(1), dim=1, keepdim=True)
            magnitude_complexity = (magnitude_complexity - magnitude_complexity.min()) / \
                                   (magnitude_complexity.max() - magnitude_complexity.min() + 1e-8)
            
            # Method 3: Feature entropy (approximation)
            eps = 1e-8
            normalized_features = F.softmax(features.flatten(1), dim=1)
            entropy = -torch.sum(normalized_features * torch.log(normalized_features + eps), dim=1, keepdim=True)
            entropy_complexity = entropy / torch.log(torch.tensor(features.shape[1], dtype=torch.float))
            
            # Combine estimates
            complexity = (neural_complexity + magnitude_complexity.unsqueeze(1) + 
                         entropy_complexity.unsqueeze(1)) / 3.0
        
        return complexity.squeeze()
    
    def apply(self, model: nn.Module, split_point: int, features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply conditional computation transformation"""
        # Estimate complexity
        complexity = self._estimate_complexity(features, split_point)
        
        # Create lite model if not exists
        if split_point not in self.lite_models:
            self.lite_models[split_point] = self._create_lite_model(model, split_point)
        
        # Decide which path to use for each sample
        use_full_model = complexity > self.threshold
        
        # Split features based on complexity
        full_mask = use_full_model
        lite_mask = ~use_full_model
        
        output = torch.zeros_like(features)
        
        # Process with full model
        if full_mask.any():
            full_features = features[full_mask]
            # This would be done by the main model
            output[full_mask] = full_features
        
        # Process with lite model  
        if lite_mask.any():
            lite_features = features[lite_mask]
            # Apply lite model
            lite_output = self.lite_models[split_point](lite_features)
            # Adjust output shape if needed
            if lite_output.shape != lite_features.shape:
                lite_output = F.adaptive_avg_pool2d(lite_output, lite_features.shape[-2:])
            output[lite_mask] = lite_output
        
        # Metadata about the decision
        metadata = {
            'complexity_scores': complexity,
            'full_model_ratio': full_mask.float().mean().item(),
            'adaptive_threshold': self.threshold
        }
        
        return output, metadata
    
    def get_compression_ratio(self) -> float:
        """Get average compression ratio based on usage statistics"""
        # Simplified: assume lite model uses ~50% computation
        return 0.75  # Average of full (1.0) and lite (0.5) usage
    
    def update_threshold(self, accuracy_delta: float, target_accuracy: float):
        """Adaptively update threshold based on performance"""
        if accuracy_delta < -0.02:  # Accuracy dropped too much
            self.threshold -= 0.05  # Use full model more often
        elif accuracy_delta > 0.01:  # Good accuracy maintained
            self.threshold += 0.05  # Use lite model more often
        
        # Clamp threshold
        self.threshold = np.clip(self.threshold, 0.1, 0.9)

class DynamicTransformationManager:
    """
    Manager for dynamic model transformations
    Decides which transformations to apply based on current conditions
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.transformations = {
            'bottleneck': BottleneckInsertion(device=device),
            'quantization': LayerQuantization(device=device),
            'conditional': ConditionalComputation(device=device)
        }
        
        # Transformation policies
        self.policies = {
            'memory_constrained': ['quantization', 'bottleneck'],
            'bandwidth_constrained': ['bottleneck', 'quantization'],
            'cpu_constrained': ['conditional', 'quantization'],
            'balanced': ['conditional', 'bottleneck'],
            'quality_first': ['conditional'],
            'efficiency_first': ['quantization', 'bottleneck', 'conditional']
        }
        
        # Adaptive weights for combining transformations
        self.transformation_weights = {
            'bottleneck': 1.0,
            'quantization': 1.0,
            'conditional': 1.0
        }
        
        # Performance tracking
        self.performance_history = {
            'accuracy': [],
            'compression_ratio': [],
            'computation_time': []
        }
    
    def select_transformations(self, resource_state: Dict[str, float],
                             network_state: Dict[str, float],
                             privacy_requirements: bool = False) -> List[str]:
        """Select appropriate transformations based on current conditions"""
        selected = []
        
        # Memory pressure
        if resource_state.get('memory_usage', 0) > 0.8:
            if 'quantization' not in selected:
                selected.append('quantization')
            if 'bottleneck' not in selected:
                selected.append('bottleneck')
        
        # CPU pressure
        if resource_state.get('cpu_utilization', 0) > 0.7:
            if 'conditional' not in selected:
                selected.append('conditional')
        
        # Bandwidth constraints
        if network_state.get('bandwidth', float('inf')) < 2.0:  # Less than 2 Mbps
            if 'bottleneck' not in selected:
                selected.append('bottleneck')
        
        # Battery constraints
        if resource_state.get('battery_level', 1.0) < 0.3:
            if 'conditional' not in selected:
                selected.append('conditional')
            if 'quantization' not in selected:
                selected.append('quantization')
        
        # Privacy requirements (prefer more edge computation)
        if privacy_requirements:
            # Favor transformations that enable more edge processing
            if 'quantization' not in selected:
                selected.append('quantization')
        
        # Default to balanced approach if nothing selected
        if not selected:
            selected = self.policies['balanced']
        
        return selected
    
    def apply_transformations(self, model: nn.Module, split_point: int,
                            features: torch.Tensor, 
                            resource_state: Dict[str, float],
                            network_state: Dict[str, float]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply selected transformations to features"""
        # Select transformations
        selected_transforms = self.select_transformations(resource_state, network_state)
        
        # Apply transformations in sequence
        transformed_features = features
        transformation_metadata = {}
        total_compression = 1.0
        
        for transform_name in selected_transforms:
            if transform_name in self.transformations:
                transform = self.transformations[transform_name]
                
                if transform_name == 'conditional':
                    # Conditional computation returns additional metadata
                    transformed_features, metadata = transform.apply(model, split_point, transformed_features)
                    transformation_metadata[transform_name] = metadata
                else:
                    transformed_features = transform.apply(model, split_point, transformed_features)
                
                total_compression *= transform.get_compression_ratio()
                transformation_metadata[f'{transform_name}_compression'] = transform.get_compression_ratio()
        
        # Overall metadata
        metadata = {
            'applied_transformations': selected_transforms,
            'total_compression_ratio': total_compression,
            'transformation_details': transformation_metadata,
            'original_size': features.numel() * 4,  # Assuming float32
            'compressed_size': transformed_features.numel() * 4 * total_compression
        }
        
        return transformed_features, metadata
    
    def update_transformation_weights(self, performance_metrics: Dict[str, float]):
        """Update transformation weights based on performance feedback"""
        accuracy = performance_metrics.get('accuracy', 0.0)
        compression = performance_metrics.get('compression_ratio', 1.0)
        computation_time = performance_metrics.get('computation_time', 0.0)
        
        # Update performance history
        self.performance_history['accuracy'].append(accuracy)
        self.performance_history['compression_ratio'].append(compression)
        self.performance_history['computation_time'].append(computation_time)
        
        # Keep only recent history
        for key in self.performance_history:
            if len(self.performance_history[key]) > 100:
                self.performance_history[key] = self.performance_history[key][-100:]
        
        # Adaptive weight adjustment
        if len(self.performance_history['accuracy']) > 10:
            recent_accuracy = np.mean(self.performance_history['accuracy'][-10:])
            
            # If accuracy is dropping, reduce compression
            if recent_accuracy < 0.85:
                self.transformation_weights['quantization'] *= 0.95
                self.transformation_weights['bottleneck'] *= 0.95
            # If accuracy is good, can increase compression
            elif recent_accuracy > 0.92:
                self.transformation_weights['quantization'] *= 1.05
                self.transformation_weights['bottleneck'] *= 1.05
        
        # Clamp weights
        for key in self.transformation_weights:
            self.transformation_weights[key] = np.clip(self.transformation_weights[key], 0.1, 2.0)
    
    def get_transformation_stats(self) -> Dict[str, Any]:
        """Get statistics about transformation usage and performance"""
        stats = {
            'transformation_weights': self.transformation_weights.copy(),
            'average_compression_ratio': np.mean(self.performance_history['compression_ratio']) 
                                       if self.performance_history['compression_ratio'] else 1.0,
            'average_accuracy': np.mean(self.performance_history['accuracy'])
                               if self.performance_history['accuracy'] else 0.0,
            'performance_history_length': len(self.performance_history['accuracy'])
        }
        
        return stats
    
    def optimize_transformations(self, model: nn.Module, data_loader: torch.utils.data.DataLoader,
                                split_points: List[int]):
        """Optimize transformation parameters using data"""
        # Optimize bottleneck layers
        bottleneck_transform = self.transformations['bottleneck']
        for split_point in split_points:
            bottleneck_transform.optimize(model, split_point, data_loader)
        
        # Train complexity estimators for conditional computation
        conditional_transform = self.transformations['conditional']
        model.eval()
        
        with torch.no_grad():
            for batch in data_loader:
                for split_point in split_points:
                    features = model.forward_partial(batch, split_point)
                    # Run complexity estimation to update internal state
                    conditional_transform._estimate_complexity(features, split_point)

# Factory functions

def create_transformation_manager(device: str = 'cpu',
                                compression_ratio: float = 0.5,
                                quantization_bits: int = 8,
                                conditional_threshold: float = 0.5) -> DynamicTransformationManager:
    """Create transformation manager with custom parameters"""
    manager = DynamicTransformationManager(device=device)
    
    # Customize transformations
    manager.transformations['bottleneck'] = BottleneckInsertion(
        compression_ratio=compression_ratio, device=device
    )
    manager.transformations['quantization'] = LayerQuantization(
        bits=quantization_bits, device=device
    )
    manager.transformations['conditional'] = ConditionalComputation(
        threshold=conditional_threshold, device=device
    )
    
    return manager