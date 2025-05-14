"""
Evaluation Metrics for StreamSplit Framework
Implements comprehensive metrics for representation quality, resource efficiency, and adaptation
Based on Section 4.4 and Appendix W of the StreamSplit paper
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict, deque
import time
import logging
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import threading

# Export main classes for import
__all__ = ['MetricsCollector', 'RepresentationQualityMetrics', 'ResourceEfficiencyMetrics',
           'AdaptationMetrics', 'PerformanceTracker', 'evaluate_embeddings', 'create_metrics_collector']

@dataclass
class MetricResult:
    """Container for metric results with metadata"""
    value: float
    std: Optional[float] = None
    metadata: Dict[str, Any] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}

class RepresentationQualityMetrics:
    """
    Metrics for evaluating representation quality
    Implements downstream classification, nearest neighbor retrieval, and visualization metrics
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.logger = logging.getLogger(__name__)
        
    def downstream_classification_accuracy(self, embeddings: torch.Tensor, 
                                         labels: np.ndarray,
                                         test_size: float = 0.2,
                                         random_state: int = 42) -> MetricResult:
        """
        Evaluate downstream classification accuracy using linear SVM probe
        As described in Appendix W.1
        """
        # Convert embeddings to numpy
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train linear SVM
        svm = SVC(kernel='linear', random_state=random_state)
        svm.fit(X_train_scaled, y_train)
        
        # Predict and evaluate
        y_pred = svm.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
        
        metadata = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'num_classes': len(np.unique(labels)),
            'test_samples': len(y_test)
        }
        
        return MetricResult(value=accuracy, metadata=metadata)
    
    def nearest_neighbor_retrieval(self, embeddings: torch.Tensor, 
                                 labels: np.ndarray,
                                 k: int = 10,
                                 n_query_samples: int = 1000,
                                 metric: str = 'cosine',
                                 random_state: int = 42) -> MetricResult:
        """
        Evaluate nearest neighbor retrieval Precision@K
        As described in Appendix W.2
        """
        # Convert embeddings to numpy
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        
        # Normalize embeddings for cosine similarity
        if metric == 'cosine':
            embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Create nearest neighbor index
        nn_model = NearestNeighbors(n_neighbors=k+1, metric=metric)
        nn_model.fit(embeddings)
        
        # Sample query indices
        np.random.seed(random_state)
        query_indices = np.random.choice(len(embeddings), min(n_query_samples, len(embeddings)), replace=False)
        
        precisions = []
        recalls = []
        
        for query_idx in query_indices:
            query_embedding = embeddings[query_idx:query_idx+1]
            query_label = labels[query_idx]
            
            # Find nearest neighbors
            distances, indices = nn_model.kneighbors(query_embedding)
            neighbor_indices = indices[0][1:]  # Exclude self (first neighbor)
            neighbor_labels = labels[neighbor_indices]
            
            # Calculate precision@k
            relevant_retrieved = np.sum(neighbor_labels == query_label)
            precision_at_k = relevant_retrieved / k
            precisions.append(precision_at_k)
            
            # Calculate recall@k (if we know total relevant items)
            total_relevant = np.sum(labels == query_label) - 1  # Exclude query itself
            recall_at_k = relevant_retrieved / max(1, min(total_relevant, k))
            recalls.append(recall_at_k)
        
        mean_precision = np.mean(precisions)
        std_precision = np.std(precisions)
        mean_recall = np.mean(recalls)
        
        metadata = {
            'k': k,
            'n_query_samples': len(query_indices),
            'mean_recall_at_k': mean_recall,
            'precision_distribution': {
                'min': np.min(precisions),
                'max': np.max(precisions),
                'median': np.median(precisions),
                'q25': np.percentile(precisions, 25),
                'q75': np.percentile(precisions, 75)
            },
            'metric': metric
        }
        
        return MetricResult(value=mean_precision, std=std_precision, metadata=metadata)
    
    def embedding_separability(self, embeddings: torch.Tensor, 
                              labels: np.ndarray) -> MetricResult:
        """
        Measure embedding separability using between-class vs within-class distances
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        
        # Calculate class centroids
        centroids = []
        for label in unique_labels:
            mask = labels == label
            centroid = np.mean(embeddings[mask], axis=0)
            centroids.append(centroid)
        centroids = np.array(centroids)
        
        # Between-class distances
        between_class_dists = []
        for i in range(n_classes):
            for j in range(i+1, n_classes):
                dist = np.linalg.norm(centroids[i] - centroids[j])
                between_class_dists.append(dist)
        
        # Within-class distances
        within_class_dists = []
        for label in unique_labels:
            mask = labels == label
            class_embeddings = embeddings[mask]
            centroid = centroids[label == unique_labels][0]
            
            for embedding in class_embeddings:
                dist = np.linalg.norm(embedding - centroid)
                within_class_dists.append(dist)
        
        # Separability score (higher is better)
        mean_between = np.mean(between_class_dists)
        mean_within = np.mean(within_class_dists)
        separability = mean_between / (mean_within + 1e-8)
        
        metadata = {
            'mean_between_class_distance': mean_between,
            'mean_within_class_distance': mean_within,
            'std_between_class_distance': np.std(between_class_dists),
            'std_within_class_distance': np.std(within_class_dists),
            'n_classes': n_classes
        }
        
        return MetricResult(value=separability, metadata=metadata)
    
    def create_tsne_visualization(self, embeddings: torch.Tensor,
                                 labels: np.ndarray,
                                 perplexity: int = 30,
                                 learning_rate: float = 200,
                                 n_iter: int = 5000,
                                 save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create t-SNE visualization of embeddings
        As described in Appendix W.3
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        
        # Perform t-SNE
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter,
            random_state=42
        )
        
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       c=[colors[i]], label=f'Class {label}', alpha=0.7)
        
        plt.legend()
        plt.title('t-SNE Visualization of Audio Embeddings')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Calculate clustering quality metrics
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        
        silhouette = silhouette_score(embeddings_2d, labels)
        calinski_harabasz = calinski_harabasz_score(embeddings_2d, labels)
        
        return {
            'embeddings_2d': embeddings_2d,
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski_harabasz,
            'figure': plt.gcf()
        }

class ResourceEfficiencyMetrics:
    """
    Metrics for evaluating resource efficiency
    Implements CPU, memory, energy, and bandwidth measurement
    """
    
    def __init__(self, measurement_interval: float = 1.0):
        self.measurement_interval = measurement_interval
        self.logger = logging.getLogger(__name__)
        
        # Resource tracking
        self.cpu_measurements = deque(maxlen=3600)  # 1 hour at 1s intervals
        self.memory_measurements = deque(maxlen=3600)
        self.power_measurements = deque(maxlen=3600)
        self.bandwidth_measurements = deque(maxlen=3600)
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread = None
        self.start_time = None
        
    def start_monitoring(self):
        """Start continuous resource monitoring"""
        self.monitoring_active = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous resource monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        self.logger.info("Resource monitoring stopped")
    
    def _monitor_resources(self):
        """Internal method for continuous resource monitoring"""
        while self.monitoring_active:
            timestamp = time.time()
            
            # CPU utilization
            cpu_percent = psutil.cpu_percent(interval=None)
            self.cpu_measurements.append((timestamp, cpu_percent))
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)
            memory_percent = memory.percent
            self.memory_measurements.append((timestamp, memory_mb, memory_percent))
            
            # Network I/O (for bandwidth estimation)
            net_io = psutil.net_io_counters()
            self.bandwidth_measurements.append((timestamp, net_io.bytes_sent, net_io.bytes_recv))
            
            time.sleep(self.measurement_interval)
    
    def get_cpu_metrics(self) -> MetricResult:
        """Get CPU utilization metrics"""
        if not self.cpu_measurements:
            return MetricResult(value=0.0, metadata={'error': 'No CPU measurements available'})
        
        cpu_values = [measurement[1] for measurement in self.cpu_measurements]
        
        mean_cpu = np.mean(cpu_values)
        std_cpu = np.std(cpu_values)
        max_cpu = np.max(cpu_values)
        min_cpu = np.min(cpu_values)
        
        metadata = {
            'mean': mean_cpu,
            'std': std_cpu,
            'max': max_cpu,
            'min': min_cpu,
            'median': np.median(cpu_values),
            'percentile_95': np.percentile(cpu_values, 95),
            'measurements_count': len(cpu_values)
        }
        
        return MetricResult(value=mean_cpu, std=std_cpu, metadata=metadata)
    
    def get_memory_metrics(self) -> MetricResult:
        """Get memory usage metrics"""
        if not self.memory_measurements:
            return MetricResult(value=0.0, metadata={'error': 'No memory measurements available'})
        
        memory_mb_values = [measurement[1] for measurement in self.memory_measurements]
        memory_percent_values = [measurement[2] for measurement in self.memory_measurements]
        
        mean_memory_mb = np.mean(memory_mb_values)
        std_memory_mb = np.std(memory_mb_values)
        peak_memory_mb = np.max(memory_mb_values)
        
        metadata = {
            'mean_mb': mean_memory_mb,
            'std_mb': std_memory_mb,
            'peak_mb': peak_memory_mb,
            'mean_percent': np.mean(memory_percent_values),
            'peak_percent': np.max(memory_percent_values),
            'measurements_count': len(memory_mb_values)
        }
        
        return MetricResult(value=mean_memory_mb, std=std_memory_mb, metadata=metadata)
    
    def calculate_energy_efficiency(self, baseline_power: float = 1.8) -> MetricResult:
        """
        Calculate energy efficiency compared to baseline
        Based on Table 1 in the paper
        """
        if not self.power_measurements:
            return MetricResult(value=1.0, metadata={'error': 'No power measurements available'})
        
        power_values = [measurement[1] for measurement in self.power_measurements]
        mean_power = np.mean(power_values)
        
        # Energy efficiency as reduction from baseline
        energy_reduction = max(0, (baseline_power - mean_power) / baseline_power)
        energy_efficiency = 1 - energy_reduction
        
        # Total energy consumption
        if self.start_time:
            duration_hours = (time.time() - self.start_time) / 3600
            total_energy_wh = mean_power * duration_hours
        else:
            total_energy_wh = 0
        
        metadata = {
            'mean_power_w': mean_power,
            'baseline_power_w': baseline_power,
            'energy_reduction_percent': energy_reduction * 100,
            'total_energy_wh': total_energy_wh,
            'power_std': np.std(power_values),
            'measurements_count': len(power_values)
        }
        
        return MetricResult(value=energy_efficiency, metadata=metadata)
    
    def calculate_bandwidth_efficiency(self, baseline_bandwidth_mbps: float = 10.0) -> MetricResult:
        """
        Calculate bandwidth efficiency and reduction
        Based on communication cost analysis in Table 2b
        """
        if len(self.bandwidth_measurements) < 2:
            return MetricResult(value=1.0, metadata={'error': 'Insufficient bandwidth measurements'})
        
        # Calculate bandwidth usage over time
        bandwidth_usage_mbps = []
        
        for i in range(1, len(self.bandwidth_measurements)):
            prev_time, prev_sent, prev_recv = self.bandwidth_measurements[i-1]
            curr_time, curr_sent, curr_recv = self.bandwidth_measurements[i]
            
            time_diff = curr_time - prev_time
            bytes_diff = (curr_sent - prev_sent) + (curr_recv - prev_recv)
            
            if time_diff > 0:
                mbps = (bytes_diff * 8) / (time_diff * 1024 * 1024)
                bandwidth_usage_mbps.append(mbps)
        
        if not bandwidth_usage_mbps:
            return MetricResult(value=1.0, metadata={'error': 'No valid bandwidth measurements'})
        
        mean_bandwidth = np.mean(bandwidth_usage_mbps)
        bandwidth_reduction = max(0, (baseline_bandwidth_mbps - mean_bandwidth) / baseline_bandwidth_mbps)
        
        # Total data transmitted
        if self.bandwidth_measurements:
            first_measurement = self.bandwidth_measurements[0]
            last_measurement = self.bandwidth_measurements[-1]
            
            total_sent_mb = (last_measurement[1] - first_measurement[1]) / (1024 * 1024)
            total_recv_mb = (last_measurement[2] - first_measurement[2]) / (1024 * 1024)
            total_data_mb = total_sent_mb + total_recv_mb
        else:
            total_data_mb = 0
        
        metadata = {
            'mean_bandwidth_mbps': mean_bandwidth,
            'baseline_bandwidth_mbps': baseline_bandwidth_mbps,
            'bandwidth_reduction_percent': bandwidth_reduction * 100,
            'total_data_mb': total_data_mb,
            'peak_bandwidth_mbps': np.max(bandwidth_usage_mbps) if bandwidth_usage_mbps else 0,
            'measurements_count': len(bandwidth_usage_mbps)
        }
        
        return MetricResult(value=1 - bandwidth_reduction, metadata=metadata)
    
    def get_resource_summary(self) -> Dict[str, MetricResult]:
        """Get comprehensive resource efficiency summary"""
        return {
            'cpu_efficiency': self.get_cpu_metrics(),
            'memory_efficiency': self.get_memory_metrics(),
            'energy_efficiency': self.calculate_energy_efficiency(),
            'bandwidth_efficiency': self.calculate_bandwidth_efficiency()
        }

class AdaptationMetrics:
    """
    Metrics for evaluating adaptive behavior
    Measures adaptation latency and quality-resource tradeoffs
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.adaptation_events = []
        self.quality_resource_points = []
        
    def record_adaptation_event(self, adaptation_type: str, 
                               trigger_time: float,
                               completion_time: float,
                               old_state: Dict[str, Any],
                               new_state: Dict[str, Any],
                               performance_impact: float = None):
        """Record an adaptation event for latency analysis"""
        adaptation_latency = completion_time - trigger_time
        
        event = {
            'type': adaptation_type,
            'trigger_time': trigger_time,
            'completion_time': completion_time,
            'latency': adaptation_latency,
            'old_state': old_state,
            'new_state': new_state,
            'performance_impact': performance_impact
        }
        
        self.adaptation_events.append(event)
        
    def get_adaptation_latency_metrics(self) -> Dict[str, MetricResult]:
        """Analyze adaptation latency by type"""
        if not self.adaptation_events:
            return {'overall': MetricResult(value=0.0, metadata={'error': 'No adaptation events recorded'})}
        
        # Group by adaptation type
        events_by_type = defaultdict(list)
        for event in self.adaptation_events:
            events_by_type[event['type']].append(event['latency'])
        
        results = {}
        
        # Overall adaptation latency
        all_latencies = [event['latency'] for event in self.adaptation_events]
        results['overall'] = MetricResult(
            value=np.mean(all_latencies),
            std=np.std(all_latencies),
            metadata={
                'count': len(all_latencies),
                'median': np.median(all_latencies),
                'min': np.min(all_latencies),
                'max': np.max(all_latencies),
                'percentile_95': np.percentile(all_latencies, 95)
            }
        )
        
        # By adaptation type
        for adaptation_type, latencies in events_by_type.items():
            results[adaptation_type] = MetricResult(
                value=np.mean(latencies),
                std=np.std(latencies),
                metadata={
                    'count': len(latencies),
                    'median': np.median(latencies),
                    'min': np.min(latencies),
                    'max': np.max(latencies)
                }
            )
        
        return results
    
    def record_quality_resource_point(self, quality_score: float,
                                    resource_utilization: float,
                                    timestamp: float = None):
        """Record a quality-resource tradeoff point"""
        if timestamp is None:
            timestamp = time.time()
        
        point = {
            'quality': quality_score,
            'resource_utilization': resource_utilization,
            'timestamp': timestamp
        }
        
        self.quality_resource_points.append(point)
    
    def analyze_quality_resource_tradeoff(self) -> MetricResult:
        """
        Analyze quality-resource tradeoff curve
        As described in Appendix W.9
        """
        if len(self.quality_resource_points) < 2:
            return MetricResult(value=0.0, metadata={'error': 'Insufficient data points'})
        
        qualities = [point['quality'] for point in self.quality_resource_points]
        resources = [point['resource_utilization'] for point in self.quality_resource_points]
        
        # Calculate Pareto efficiency
        pareto_points = []
        for i, (q, r) in enumerate(zip(qualities, resources)):
            is_pareto = True
            for j, (q2, r2) in enumerate(zip(qualities, resources)):
                if i != j and q2 >= q and r2 <= r and (q2 > q or r2 < r):
                    is_pareto = False
                    break
            if is_pareto:
                pareto_points.append((q, r))
        
        # Calculate trade-off efficiency (area under normalized curve)
        if pareto_points:
            pareto_points_sorted = sorted(pareto_points, key=lambda x: x[1])
            # Normalize to [0, 1]
            min_quality, max_quality = min(qualities), max(qualities)
            min_resource, max_resource = min(resources), max(resources)
            
            if max_quality > min_quality and max_resource > min_resource:
                normalized_pareto = [
                    ((q - min_quality) / (max_quality - min_quality),
                     (r - min_resource) / (max_resource - min_resource))
                    for q, r in pareto_points_sorted
                ]
                
                # Calculate area under curve (higher is better for quality/resource tradeoff)
                auc = 0.0
                for i in range(1, len(normalized_pareto)):
                    q1, r1 = normalized_pareto[i-1]
                    q2, r2 = normalized_pareto[i]
                    auc += (r2 - r1) * (q1 + q2) / 2
                
                tradeoff_score = auc
            else:
                tradeoff_score = 0.0
        else:
            tradeoff_score = 0.0
        
        metadata = {
            'pareto_points': pareto_points,
            'total_points': len(self.quality_resource_points),
            'pareto_ratio': len(pareto_points) / len(self.quality_resource_points),
            'quality_range': (min(qualities), max(qualities)),
            'resource_range': (min(resources), max(resources)),
            'correlation': np.corrcoef(qualities, resources)[0, 1] if len(qualities) > 1 else 0.0
        }
        
        return MetricResult(value=tradeoff_score, metadata=metadata)

class PerformanceTracker:
    """
    Track performance metrics over time
    Provides windowed statistics and trend analysis
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics_history = defaultdict(lambda: deque(maxlen=window_size))
        self.logger = logging.getLogger(__name__)
        
    def update(self, metric_name: str, value: float, timestamp: float = None):
        """Update a metric with a new value"""
        if timestamp is None:
            timestamp = time.time()
        
        self.metrics_history[metric_name].append((timestamp, value))
    
    def get_windowed_stats(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric over the current window"""
        if metric_name not in self.metrics_history:
            return {'error': f'Metric {metric_name} not found'}
        
        values = [item[1] for item in self.metrics_history[metric_name]]
        
        if not values:
            return {'error': f'No values for metric {metric_name}'}
        
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'count': len(values),
            'trend': self._calculate_trend(metric_name)
        }
    
    def _calculate_trend(self, metric_name: str) -> float:
        """Calculate trend (slope) of metric over time"""
        if metric_name not in self.metrics_history:
            return 0.0
        
        data = list(self.metrics_history[metric_name])
        if len(data) < 2:
            return 0.0
        
        timestamps = np.array([item[0] for item in data])
        values = np.array([item[1] for item in data])
        
        # Normalize timestamps to start from 0
        timestamps = timestamps - timestamps[0]
        
        # Calculate linear regression slope
        if len(timestamps) > 1:
            slope = np.cov(timestamps, values)[0, 1] / np.var(timestamps)
        else:
            slope = 0.0
        
        return slope
    
    def get_all_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all tracked metrics"""
        summary = {}
        for metric_name in self.metrics_history:
            summary[metric_name] = self.get_windowed_stats(metric_name)
        return summary

class MetricsCollector:
    """
    Main metrics collection and coordination class
    Aggregates all metric types and provides unified interface
    """
    
    def __init__(self, device: str = 'cpu', measurement_interval: float = 1.0):
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Initialize metric components
        self.representation_metrics = RepresentationQualityMetrics(device=device)
        self.resource_metrics = ResourceEfficiencyMetrics(measurement_interval=measurement_interval)
        self.adaptation_metrics = AdaptationMetrics()
        self.performance_tracker = PerformanceTracker()
        
        # Overall state
        self.monitoring_active = False
        self.start_time = None
        
    def start_monitoring(self):
        """Start comprehensive monitoring"""
        self.monitoring_active = True
        self.start_time = time.time()
        self.resource_metrics.start_monitoring()
        self.logger.info("MetricsCollector monitoring started")
    
    def stop_monitoring(self):
        """Stop comprehensive monitoring"""
        self.monitoring_active = False
        self.resource_metrics.stop_monitoring()
        self.logger.info("MetricsCollector monitoring stopped")
    
    def update_resources(self, resources: Dict[str, float]):
        """Update resource metrics manually"""
        timestamp = time.time()
        for key, value in resources.items():
            self.performance_tracker.update(f"resource_{key}", value, timestamp)
    
    def update_network(self, network: Dict[str, float]):
        """Update network metrics manually"""
        timestamp = time.time()
        for key, value in network.items():
            self.performance_tracker.update(f"network_{key}", value, timestamp)
    
    def update_performance(self, performance: Dict[str, float]):
        """Update performance metrics manually"""
        timestamp = time.time()
        for key, value in performance.items():
            self.performance_tracker.update(f"performance_{key}", value, timestamp)
    
    def evaluate_embeddings(self, embeddings: torch.Tensor, 
                          labels: np.ndarray,
                          include_visualization: bool = False) -> Dict[str, MetricResult]:
        """
        Comprehensive evaluation of embeddings
        Returns all representation quality metrics
        """
        results = {}
        
        # Classification accuracy
        results['classification_accuracy'] = self.representation_metrics.downstream_classification_accuracy(
            embeddings, labels
        )
        
        # Nearest neighbor retrieval
        results['precision_at_10'] = self.representation_metrics.nearest_neighbor_retrieval(
            embeddings, labels, k=10
        )
        
        # Embedding separability
        results['embedding_separability'] = self.representation_metrics.embedding_separability(
            embeddings, labels
        )
        
        # t-SNE visualization if requested
        if include_visualization:
            visualization = self.representation_metrics.create_tsne_visualization(
                embeddings, labels
            )
            results['tsne_visualization'] = MetricResult(
                value=visualization['silhouette_score'],
                metadata=visualization
            )
        
        return results
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report
        Combines all metric types into unified report
        """
        report = {
            'timestamp': time.time(),
            'monitoring_duration': None,
            'resource_efficiency': {},
            'adaptation_performance': {},
            'performance_trends': {},
            'summary': {}
        }
        
        # Calculate monitoring duration
        if self.start_time:
            report['monitoring_duration'] = time.time() - self.start_time
        
        # Resource efficiency metrics
        resource_summary = self.resource_metrics.get_resource_summary()
        report['resource_efficiency'] = {
            name: {
                'value': metric.value,
                'std': metric.std,
                'metadata': metric.metadata
            }
            for name, metric in resource_summary.items()
        }
        
        # Adaptation metrics
        adaptation_latencies = self.adaptation_metrics.get_adaptation_latency_metrics()
        report['adaptation_performance'] = {
            name: {
                'value': metric.value,
                'std': metric.std,
                'metadata': metric.metadata
            }
            for name, metric in adaptation_latencies.items()
        }
        
        # Quality-resource tradeoff
        tradeoff_analysis = self.adaptation_metrics.analyze_quality_resource_tradeoff()
        report['adaptation_performance']['quality_resource_tradeoff'] = {
            'value': tradeoff_analysis.value,
            'metadata': tradeoff_analysis.metadata
        }
        
        # Performance trends
        report['performance_trends'] = self.performance_tracker.get_all_metrics_summary()
        
        # Summary metrics
        report['summary'] = self._calculate_summary_metrics(resource_summary, adaptation_latencies)
        
        return report
    
    def _calculate_summary_metrics(self, resource_summary: Dict[str, MetricResult],
                                  adaptation_latencies: Dict[str, MetricResult]) -> Dict[str, float]:
        """Calculate high-level summary metrics"""
        summary = {}
        
        # Overall efficiency score (weighted average of resource metrics)
        efficiency_weights = {
            'cpu_efficiency': 0.3,
            'memory_efficiency': 0.2,
            'energy_efficiency': 0.3,
            'bandwidth_efficiency': 0.2
        }
        
        efficiency_score = 0.0
        total_weight = 0.0
        
        for metric_name, weight in efficiency_weights.items():
            if metric_name in resource_summary:
                efficiency_score += resource_summary[metric_name].value * weight
                total_weight += weight
        
        if total_weight > 0:
            summary['overall_efficiency_score'] = efficiency_score / total_weight
        else:
            summary['overall_efficiency_score'] = 0.0
        
        # Adaptation responsiveness (inverse of average adaptation latency)
        if 'overall' in adaptation_latencies:
            avg_latency = adaptation_latencies['overall'].value
            summary['adaptation_responsiveness'] = 1.0 / (avg_latency + 1.0) if avg_latency > 0 else 1.0
        else:
            summary['adaptation_responsiveness'] = 1.0
        
        # Resource stability (inverse of coefficient of variation)
        cpu_metrics = resource_summary.get('cpu_efficiency')
        if cpu_metrics and cpu_metrics.std is not None and cpu_metrics.value > 0:
            cpu_cv = cpu_metrics.std / cpu_metrics.value
            summary['resource_stability'] = 1.0 / (1.0 + cpu_cv)
        else:
            summary['resource_stability'] = 1.0
        
        return summary
    
    def export_metrics(self, filepath: str, format: str = 'json'):
        """Export metrics to file"""
        report = self.get_comprehensive_report()
        
        if format.lower() == 'json':
            import json
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        elif format.lower() == 'csv':
            import pandas as pd
            # Flatten the report for CSV export
            flattened_data = []
            
            # Resource efficiency metrics
            for metric_name, metric_data in report['resource_efficiency'].items():
                flattened_data.append({
                    'category': 'resource_efficiency',
                    'metric': metric_name,
                    'value': metric_data['value'],
                    'std': metric_data.get('std'),
                    'timestamp': report['timestamp']
                })
            
            # Adaptation metrics
            for metric_name, metric_data in report['adaptation_performance'].items():
                flattened_data.append({
                    'category': 'adaptation_performance',
                    'metric': metric_name,
                    'value': metric_data['value'],
                    'std': metric_data.get('std'),
                    'timestamp': report['timestamp']
                })
            
            # Performance trends
            for metric_name, metric_stats in report['performance_trends'].items():
                if 'mean' in metric_stats:
                    flattened_data.append({
                        'category': 'performance_trends',
                        'metric': metric_name,
                        'value': metric_stats['mean'],
                        'std': metric_stats.get('std'),
                        'timestamp': report['timestamp']
                    })
            
            df = pd.DataFrame(flattened_data)
            df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'")
        
        self.logger.info(f"Metrics exported to {filepath}")


# Standalone utility functions

def evaluate_embeddings(embeddings: torch.Tensor, 
                       labels: np.ndarray,
                       device: str = 'cpu',
                       include_visualization: bool = False) -> Dict[str, MetricResult]:
    """
    Standalone function to evaluate embeddings
    
    Args:
        embeddings: Tensor of shape (N, D) containing embeddings
        labels: Array of shape (N,) containing class labels
        device: Device to use for computations
        include_visualization: Whether to include t-SNE visualization
    
    Returns:
        Dictionary containing evaluation results
    """
    metrics = RepresentationQualityMetrics(device=device)
    
    results = {}
    
    # Classification accuracy using linear probe
    results['classification_accuracy'] = metrics.downstream_classification_accuracy(
        embeddings, labels
    )
    
    # Nearest neighbor retrieval metrics
    for k in [1, 5, 10]:
        results[f'precision_at_{k}'] = metrics.nearest_neighbor_retrieval(
            embeddings, labels, k=k
        )
    
    # Embedding quality metrics
    results['embedding_separability'] = metrics.embedding_separability(
        embeddings, labels
    )
    
    # Optional visualization
    if include_visualization:
        visualization = metrics.create_tsne_visualization(embeddings, labels)
        results['tsne_visualization'] = MetricResult(
            value=visualization['silhouette_score'],
            metadata=visualization
        )
    
    return results


def calculate_bandwidth_reduction(baseline_bandwidth: float,
                                current_bandwidth: float) -> float:
    """
    Calculate bandwidth reduction percentage
    
    Args:
        baseline_bandwidth: Baseline bandwidth usage (e.g., server-only)
        current_bandwidth: Current bandwidth usage
    
    Returns:
        Bandwidth reduction as a percentage (0-100)
    """
    if baseline_bandwidth <= 0:
        return 0.0
    
    reduction = (baseline_bandwidth - current_bandwidth) / baseline_bandwidth
    return max(0.0, reduction * 100.0)


def calculate_latency_reduction(baseline_latency: float,
                              current_latency: float) -> float:
    """
    Calculate latency reduction percentage
    
    Args:
        baseline_latency: Baseline latency (e.g., server-only)
        current_latency: Current latency
    
    Returns:
        Latency reduction as a percentage (0-100)
    """
    if baseline_latency <= 0:
        return 0.0
    
    reduction = (baseline_latency - current_latency) / baseline_latency
    return max(0.0, reduction * 100.0)


def calculate_energy_reduction(baseline_power: float,
                             current_power: float) -> float:
    """
    Calculate energy reduction percentage
    
    Args:
        baseline_power: Baseline power consumption
        current_power: Current power consumption
    
    Returns:
        Energy reduction as a percentage (0-100)
    """
    if baseline_power <= 0:
        return 0.0
    
    reduction = (baseline_power - current_power) / baseline_power
    return max(0.0, reduction * 100.0)


def create_metrics_collector(device: str = 'cpu',
                           measurement_interval: float = 1.0) -> MetricsCollector:
    """
    Factory function to create a MetricsCollector with default configuration
    
    Args:
        device: Device to use for computations
        measurement_interval: Interval between resource measurements in seconds
    
    Returns:
        Configured MetricsCollector instance
    """
    return MetricsCollector(device=device, measurement_interval=measurement_interval)


# Testing and validation functions

def test_metrics_functionality():
    """Test basic metrics functionality"""
    print("Testing StreamSplit Metrics...")
    
    # Create synthetic embeddings and labels
    n_samples = 500
    n_features = 128
    n_classes = 5
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate class-separated embeddings
    embeddings = []
    labels = []
    
    for class_id in range(n_classes):
        class_center = np.random.randn(n_features) * 2
        class_embeddings = np.random.randn(n_samples // n_classes, n_features) * 0.5 + class_center
        embeddings.append(class_embeddings)
        labels.extend([class_id] * (n_samples // n_classes))
    
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    embeddings_tensor = torch.FloatTensor(embeddings)
    
    # Test representation quality metrics
    print("✓ Testing representation quality metrics...")
    rep_metrics = RepresentationQualityMetrics()
    
    # Classification accuracy
    acc_result = rep_metrics.downstream_classification_accuracy(embeddings_tensor, labels)
    print(f"  Classification accuracy: {acc_result.value:.3f}")
    
    # Nearest neighbor retrieval
    precision_result = rep_metrics.nearest_neighbor_retrieval(embeddings_tensor, labels, k=5)
    print(f"  Precision@5: {precision_result.value:.3f}")
    
    # Embedding separability
    sep_result = rep_metrics.embedding_separability(embeddings_tensor, labels)
    print(f"  Embedding separability: {sep_result.value:.3f}")
    
    # Test resource metrics
    print("✓ Testing resource efficiency metrics...")
    resource_metrics = ResourceEfficiencyMetrics(measurement_interval=0.1)
    resource_metrics.start_monitoring()
    time.sleep(1.0)  # Let it collect some measurements
    resource_metrics.stop_monitoring()
    
    cpu_metrics = resource_metrics.get_cpu_metrics()
    memory_metrics = resource_metrics.get_memory_metrics()
    print(f"  CPU utilization: {cpu_metrics.value:.1f}%")
    print(f"  Memory usage: {memory_metrics.value:.1f} MB")
    
    # Test adaptation metrics
    print("✓ Testing adaptation metrics...")
    adaptation_metrics = AdaptationMetrics()
    
    # Simulate some adaptation events
    for i in range(5):
        start_time = time.time()
        time.sleep(0.01)  # Simulate adaptation time
        end_time = time.time()
        
        adaptation_metrics.record_adaptation_event(
            adaptation_type='split_decision',
            trigger_time=start_time,
            completion_time=end_time,
            old_state={'split_point': i},
            new_state={'split_point': i + 1}
        )
        
        # Record quality-resource points
        adaptation_metrics.record_quality_resource_point(
            quality_score=0.8 + 0.1 * np.random.randn(),
            resource_utilization=0.5 + 0.2 * np.random.randn()
        )
    
    latency_metrics = adaptation_metrics.get_adaptation_latency_metrics()
    print(f"  Average adaptation latency: {latency_metrics['overall'].value * 1000:.2f} ms")
    
    tradeoff_result = adaptation_metrics.analyze_quality_resource_tradeoff()
    print(f"  Quality-resource tradeoff score: {tradeoff_result.value:.3f}")
    
    # Test comprehensive collector
    print("✓ Testing MetricsCollector...")
    collector = create_metrics_collector()
    collector.start_monitoring()
    
    # Simulate some updates
    collector.update_resources({'cpu': 65.2, 'memory': 820})
    collector.update_network({'bandwidth': 5.5, 'latency': 85})
    collector.update_performance({'accuracy': 0.876, 'loss': 0.234})
    
    time.sleep(0.5)
    collector.stop_monitoring()
    
    # Generate comprehensive report
    report = collector.get_comprehensive_report()
    print(f"  Summary efficiency score: {report['summary']['overall_efficiency_score']:.3f}")
    
    # Test evaluation function
    print("✓ Testing standalone evaluation function...")
    eval_results = evaluate_embeddings(embeddings_tensor, labels, include_visualization=False)
    print(f"  Standalone evaluation accuracy: {eval_results['classification_accuracy'].value:.3f}")
    
    print("\nAll metrics tests completed successfully!")
    
    return True


if __name__ == "__main__":
    # Run tests
    test_metrics_functionality()