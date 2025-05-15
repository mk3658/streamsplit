#!/usr/bin/env python3
"""
Network Simulation for StreamSplit Framework
Simulates realistic network conditions as described in Appendix S
Supports dynamic bandwidth and latency variations, packet loss, and mobile network patterns
"""

import argparse
import asyncio
import logging
import time
import random
import sys
import signal
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum
import threading
import subprocess
import psutil

# Try imports for traffic control (Linux)
try:
    import pyroute2
    HAS_PYROUTE2 = True
except ImportError:
    HAS_PYROUTE2 = False

@dataclass
class NetworkCondition:
    """Represents network conditions at a point in time"""
    bandwidth_mbps: float
    latency_ms: float
    packet_loss_percent: float
    jitter_ms: float
    timestamp: float
    location_type: str = "unknown"
    connection_type: str = "wifi"

class NetworkEnvironment(Enum):
    """Types of network environments from the paper"""
    HOME_WIFI = "home_wifi"
    OFFICE_WIFI = "office_wifi"
    MOBILE_4G = "mobile_4g"
    MOBILE_5G = "mobile_5g"
    PUBLIC_WIFI = "public_wifi"
    POOR_CONNECTION = "poor_connection"
    EDGE_COMPUTING = "edge_computing"

class MobilityPattern(Enum):
    """Mobility patterns affecting network conditions"""
    STATIONARY = "stationary"
    WALKING = "walking"
    DRIVING = "driving"
    TRAIN = "train"
    INDOOR_MOVEMENT = "indoor_movement"

@dataclass
class NetworkProfile:
    """Network profile for different environments"""
    name: str
    bandwidth_range: Tuple[float, float]  # Mbps
    latency_range: Tuple[float, float]    # ms
    packet_loss_range: Tuple[float, float]  # %
    jitter_range: Tuple[float, float]     # ms
    stability_factor: float  # 0-1, higher = more stable
    transition_probability: float  # Probability of changing conditions

# Network profiles based on paper's measurement methodology (Appendix S)
NETWORK_PROFILES = {
    NetworkEnvironment.HOME_WIFI: NetworkProfile(
        name="Home WiFi",
        bandwidth_range=(10.0, 50.0),
        latency_range=(10, 30),
        packet_loss_range=(0.0, 0.5),
        jitter_range=(1, 5),
        stability_factor=0.9,
        transition_probability=0.05
    ),
    NetworkEnvironment.OFFICE_WIFI: NetworkProfile(
        name="Office WiFi",
        bandwidth_range=(20.0, 100.0),
        latency_range=(5, 20),
        packet_loss_range=(0.0, 0.2),
        jitter_range=(1, 3),
        stability_factor=0.95,
        transition_probability=0.03
    ),
    NetworkEnvironment.MOBILE_4G: NetworkProfile(
        name="Mobile 4G",
        bandwidth_range=(5.0, 30.0),
        latency_range=(30, 100),
        packet_loss_range=(0.5, 2.0),
        jitter_range=(5, 20),
        stability_factor=0.6,
        transition_probability=0.15
    ),
    NetworkEnvironment.MOBILE_5G: NetworkProfile(
        name="Mobile 5G",
        bandwidth_range=(50.0, 200.0),
        latency_range=(5, 25),
        packet_loss_range=(0.1, 1.0),
        jitter_range=(2, 10),
        stability_factor=0.8,
        transition_probability=0.1
    ),
    NetworkEnvironment.PUBLIC_WIFI: NetworkProfile(
        name="Public WiFi",
        bandwidth_range=(1.0, 15.0),
        latency_range=(50, 150),
        packet_loss_range=(1.0, 5.0),
        jitter_range=(10, 30),
        stability_factor=0.4,
        transition_probability=0.25
    ),
    NetworkEnvironment.POOR_CONNECTION: NetworkProfile(
        name="Poor Connection",
        bandwidth_range=(0.5, 2.0),
        latency_range=(200, 500),
        packet_loss_range=(5.0, 15.0),
        jitter_range=(50, 100),
        stability_factor=0.2,
        transition_probability=0.4
    ),
    NetworkEnvironment.EDGE_COMPUTING: NetworkProfile(
        name="Edge Computing",
        bandwidth_range=(100.0, 1000.0),
        latency_range=(1, 5),
        packet_loss_range=(0.0, 0.1),
        jitter_range=(0.1, 1.0),
        stability_factor=0.98,
        transition_probability=0.01
    )
}

class NetworkSimulator:
    """
    Simulates realistic network conditions with temporal patterns
    Based on measurement methodology from Appendix S
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Current state
        self.current_environment = NetworkEnvironment.HOME_WIFI
        self.current_conditions = None
        self.mobility_pattern = MobilityPattern.STATIONARY
        
        # History tracking
        self.conditions_history = deque(maxlen=10000)
        self.transition_history = deque(maxlen=1000)
        
        # Temporal patterns
        self.time_of_day_factor = 1.0
        self.congestion_patterns = {}
        self.weather_effects = False
        
        # Simulation state
        self.is_running = False
        self.start_time = None
        self.simulation_speed = config.get('simulation_speed', 1.0)
        
        # Observers (for StreamSplit integration)
        self.observers = []
        
        # Initialize with stable conditions
        self._generate_initial_conditions()
        
        self.logger.info(f"NetworkSimulator initialized with {self.current_environment.value}")
    
    def _generate_initial_conditions(self):
        """Generate initial network conditions"""
        profile = NETWORK_PROFILES[self.current_environment]
        
        # Generate conditions within profile ranges
        bandwidth = random.uniform(*profile.bandwidth_range)
        latency = random.uniform(*profile.latency_range)
        packet_loss = random.uniform(*profile.packet_loss_range)
        jitter = random.uniform(*profile.jitter_range)
        
        self.current_conditions = NetworkCondition(
            bandwidth_mbps=bandwidth,
            latency_ms=latency,
            packet_loss_percent=packet_loss,
            jitter_ms=jitter,
            timestamp=time.time(),
            location_type=self.current_environment.value,
            connection_type="wifi" if "wifi" in self.current_environment.value else "cellular"
        )
    
    async def start_simulation(self, duration: Optional[float] = None):
        """Start network simulation"""
        self.is_running = True
        self.start_time = time.time()
        
        self.logger.info(f"Starting network simulation for {duration or 'unlimited'} seconds")
        
        # Main simulation loop
        update_interval = 1.0 / self.simulation_speed  # 1 second real-time by default
        
        try:
            while self.is_running:
                if duration and (time.time() - self.start_time) >= duration:
                    break
                
                # Update network conditions
                await self._update_conditions()
                
                # Notify observers
                await self._notify_observers()
                
                # Sleep for update interval
                await asyncio.sleep(update_interval)
                
        except asyncio.CancelledError:
            self.logger.info("Network simulation cancelled")
        except Exception as e:
            self.logger.error(f"Error in network simulation: {e}")
        finally:
            self.is_running = False
    
    async def stop_simulation(self):
        """Stop network simulation"""
        self.is_running = False
        self.logger.info("Network simulation stopped")
    
    async def _update_conditions(self):
        """Update network conditions based on various factors"""
        current_time = time.time()
        
        # Check for environment transitions
        await self._check_environment_transition()
        
        # Apply time-of-day effects
        self._apply_time_of_day_effects()
        
        # Apply mobility effects
        self._apply_mobility_effects()
        
        # Apply congestion patterns
        self._apply_congestion_patterns()
        
        # Generate new conditions
        new_conditions = self._generate_conditions()
        
        # Store in history
        self.conditions_history.append(new_conditions)
        
        # Log significant changes
        if self._is_significant_change(new_conditions):
            self.logger.info(
                f"Network change: {new_conditions.bandwidth_mbps:.1f}Mbps, "
                f"{new_conditions.latency_ms:.1f}ms, "
                f"{new_conditions.packet_loss_percent:.2f}% loss"
            )
        
        self.current_conditions = new_conditions
    
    async def _check_environment_transition(self):
        """Check if network environment should transition"""
        profile = NETWORK_PROFILES[self.current_environment]
        
        if random.random() < profile.transition_probability:
            # Transition to new environment
            new_environment = self._choose_next_environment()
            
            if new_environment != self.current_environment:
                self.logger.info(f"Environment transition: {self.current_environment.value} -> {new_environment.value}")
                
                # Record transition
                self.transition_history.append({
                    'from': self.current_environment.value,
                    'to': new_environment.value,
                    'timestamp': time.time()
                })
                
                self.current_environment = new_environment
    
    def _choose_next_environment(self) -> NetworkEnvironment:
        """Choose next environment based on mobility pattern and probabilities"""
        # Transition probabilities based on current environment and mobility
        if self.mobility_pattern == MobilityPattern.STATIONARY:
            # Less likely to change environment when stationary
            transition_weights = {
                self.current_environment: 0.7,
                NetworkEnvironment.HOME_WIFI: 0.1,
                NetworkEnvironment.OFFICE_WIFI: 0.1,
                NetworkEnvironment.POOR_CONNECTION: 0.1
            }
        elif self.mobility_pattern == MobilityPattern.DRIVING:
            # More likely to have mobile connections
            transition_weights = {
                NetworkEnvironment.MOBILE_4G: 0.4,
                NetworkEnvironment.MOBILE_5G: 0.3,
                NetworkEnvironment.POOR_CONNECTION: 0.2,
                NetworkEnvironment.PUBLIC_WIFI: 0.1
            }
        else:
            # Balanced transitions
            transition_weights = {env: 1.0 for env in NetworkEnvironment}
        
        # Normalize weights
        total_weight = sum(transition_weights.values())
        choices = list(transition_weights.keys())
        weights = [transition_weights[choice] / total_weight for choice in choices]
        
        return np.random.choice(choices, p=weights)
    
    def _apply_time_of_day_effects(self):
        """Apply time-of-day effects on network conditions"""
        current_hour = time.localtime().tm_hour
        
        # Peak hours: 9-11 AM, 7-9 PM
        if 9 <= current_hour <= 11 or 19 <= current_hour <= 21:
            self.time_of_day_factor = 0.7  # Reduced performance during peak
        # Off-peak hours: late night/early morning
        elif 1 <= current_hour <= 6:
            self.time_of_day_factor = 1.2  # Better performance at night
        else:
            self.time_of_day_factor = 1.0  # Normal performance
    
    def _apply_mobility_effects(self):
        """Apply effects based on mobility pattern"""
        if not hasattr(self, '_mobility_state'):
            self._mobility_state = {
                'velocity': 0.0,
                'signal_strength_variation': 0.0,
                'handover_probability': 0.0
            }
        
        if self.mobility_pattern == MobilityPattern.DRIVING:
            self._mobility_state['velocity'] = random.uniform(30, 80)  # km/h
            self._mobility_state['signal_strength_variation'] = 0.3
            self._mobility_state['handover_probability'] = 0.1
        elif self.mobility_pattern == MobilityPattern.WALKING:
            self._mobility_state['velocity'] = random.uniform(3, 6)  # km/h
            self._mobility_state['signal_strength_variation'] = 0.1
            self._mobility_state['handover_probability'] = 0.02
        elif self.mobility_pattern == MobilityPattern.TRAIN:
            self._mobility_state['velocity'] = random.uniform(60, 150)  # km/h
            self._mobility_state['signal_strength_variation'] = 0.4
            self._mobility_state['handover_probability'] = 0.15
        else:  # STATIONARY
            self._mobility_state['velocity'] = 0.0
            self._mobility_state['signal_strength_variation'] = 0.05
            self._mobility_state['handover_probability'] = 0.001
    
    def _apply_congestion_patterns(self):
        """Apply network congestion based on realistic patterns"""
        # Simulate congestion for public/shared networks
        if self.current_environment in [NetworkEnvironment.PUBLIC_WIFI, NetworkEnvironment.OFFICE_WIFI]:
            # Congestion varies throughout the day
            current_time = time.time()
            congestion_cycle = 3600  # 1 hour cycle
            congestion_phase = (current_time % congestion_cycle) / congestion_cycle
            
            # Sinusoidal congestion pattern
            congestion_factor = 0.8 + 0.2 * np.sin(2 * np.pi * congestion_phase)
            
            # Store congestion state
            if not hasattr(self, '_congestion_factor'):
                self._congestion_factor = congestion_factor
            else:
                # Smooth transitions
                self._congestion_factor = 0.9 * self._congestion_factor + 0.1 * congestion_factor
        else:
            self._congestion_factor = 1.0
    
    def _generate_conditions(self) -> NetworkCondition:
        """Generate new network conditions based on current state"""
        profile = NETWORK_PROFILES[self.current_environment]
        
        # Base values from profile
        base_bandwidth = random.uniform(*profile.bandwidth_range)
        base_latency = random.uniform(*profile.latency_range)
        base_packet_loss = random.uniform(*profile.packet_loss_range)
        base_jitter = random.uniform(*profile.jitter_range)
        
        # Apply stability factor (how much conditions change from previous)
        if self.current_conditions:
            stability = profile.stability_factor
            bandwidth = stability * self.current_conditions.bandwidth_mbps + (1 - stability) * base_bandwidth
            latency = stability * self.current_conditions.latency_ms + (1 - stability) * base_latency
            packet_loss = stability * self.current_conditions.packet_loss_percent + (1 - stability) * base_packet_loss
            jitter = stability * self.current_conditions.jitter_ms + (1 - stability) * base_jitter
        else:
            bandwidth, latency, packet_loss, jitter = base_bandwidth, base_latency, base_packet_loss, base_jitter
        
        # Apply time-of-day effects
        bandwidth *= self.time_of_day_factor
        latency /= self.time_of_day_factor
        
        # Apply congestion effects
        bandwidth *= getattr(self, '_congestion_factor', 1.0)
        latency /= getattr(self, '_congestion_factor', 1.0)
        
        # Apply mobility effects
        mobility_state = getattr(self, '_mobility_state', {})
        signal_variation = mobility_state.get('signal_strength_variation', 0.0)
        
        # Signal strength affects bandwidth and latency
        signal_factor = 1.0 + random.uniform(-signal_variation, signal_variation)
        bandwidth *= signal_factor
        latency *= (2.0 - signal_factor)  # Inverse relationship
        
        # Handover events cause temporary spikes
        if random.random() < mobility_state.get('handover_probability', 0.0):
            latency += random.uniform(100, 500)  # Handover latency spike
            packet_loss += random.uniform(1.0, 5.0)  # Temporary packet loss
        
        # Add random variations
        bandwidth += random.gauss(0, bandwidth * 0.1)  # 10% gaussian noise
        latency += random.gauss(0, latency * 0.1)
        packet_loss += random.gauss(0, packet_loss * 0.1)
        jitter += random.gauss(0, jitter * 0.1)
        
        # Ensure values are within reasonable bounds
        bandwidth = max(0.1, bandwidth)
        latency = max(1.0, latency)
        packet_loss = max(0.0, min(50.0, packet_loss))
        jitter = max(0.0, jitter)
        
        return NetworkCondition(
            bandwidth_mbps=bandwidth,
            latency_ms=latency,
            packet_loss_percent=packet_loss,
            jitter_ms=jitter,
            timestamp=time.time(),
            location_type=self.current_environment.value,
            connection_type=self.current_conditions.connection_type if self.current_conditions else "wifi"
        )
    
    def _is_significant_change(self, new_conditions: NetworkCondition) -> bool:
        """Check if network change is significant enough to log"""
        if not self.current_conditions:
            return True
        
        # Define thresholds for significant changes
        bandwidth_threshold = 0.2  # 20% change
        latency_threshold = 0.3    # 30% change
        packet_loss_threshold = 1.0  # 1% absolute change
        
        bandwidth_change = abs(new_conditions.bandwidth_mbps - self.current_conditions.bandwidth_mbps) / self.current_conditions.bandwidth_mbps
        latency_change = abs(new_conditions.latency_ms - self.current_conditions.latency_ms) / self.current_conditions.latency_ms
        packet_loss_change = abs(new_conditions.packet_loss_percent - self.current_conditions.packet_loss_percent)
        
        return (bandwidth_change > bandwidth_threshold or 
                latency_change > latency_threshold or 
                packet_loss_change > packet_loss_threshold)
    
    def add_observer(self, observer: Callable[[NetworkCondition], None]):
        """Add observer for network condition changes"""
        self.observers.append(observer)
    
    async def _notify_observers(self):
        """Notify all observers of current network conditions"""
        for observer in self.observers:
            try:
                if asyncio.iscoroutinefunction(observer):
                    await observer(self.current_conditions)
                else:
                    observer(self.current_conditions)
            except Exception as e:
                self.logger.error(f"Error notifying observer: {e}")
    
    def get_current_conditions(self) -> NetworkCondition:
        """Get current network conditions"""
        return self.current_conditions
    
    def get_conditions_history(self, duration: Optional[float] = None) -> List[NetworkCondition]:
        """Get history of network conditions"""
        if duration is None:
            return list(self.conditions_history)
        
        cutoff_time = time.time() - duration
        return [cond for cond in self.conditions_history if cond.timestamp >= cutoff_time]
    
    def set_environment(self, environment: NetworkEnvironment):
        """Manually set network environment"""
        old_env = self.current_environment
        self.current_environment = environment
        self.logger.info(f"Environment manually changed: {old_env.value} -> {environment.value}")
    
    def set_mobility_pattern(self, pattern: MobilityPattern):
        """Set mobility pattern"""
        self.mobility_pattern = pattern
        self.logger.info(f"Mobility pattern set to: {pattern.value}")

class NetworkTrafficShaper:
    """
    Applies network conditions using traffic control (Linux only)
    Integrates with StreamSplit for realistic testing
    """
    
    def __init__(self, interface: str = "lo", use_tc: bool = True):
        self.interface = interface
        self.use_tc = use_tc and sys.platform.startswith('linux')
        self.logger = logging.getLogger(__name__)
        
        # Validate interface exists
        if self.use_tc and not self._interface_exists():
            self.logger.warning(f"Interface {interface} not found, disabling traffic shaping")
            self.use_tc = False
        
        # Current applied conditions
        self.applied_conditions = None
        
    def _interface_exists(self) -> bool:
        """Check if network interface exists"""
        try:
            interfaces = psutil.net_if_stats()
            return self.interface in interfaces
        except:
            return False
    
    async def apply_conditions(self, conditions: NetworkCondition):
        """Apply network conditions using traffic control"""
        if not self.use_tc:
            self.applied_conditions = conditions
            return
        
        try:
            # Clear existing rules
            await self._clear_tc_rules()
            
            # Apply new rules
            await self._apply_tc_rules(conditions)
            
            self.applied_conditions = conditions
            self.logger.debug(f"Applied network conditions: {conditions.bandwidth_mbps}Mbps, {conditions.latency_ms}ms")
            
        except Exception as e:
            self.logger.error(f"Failed to apply network conditions: {e}")
    
    async def _clear_tc_rules(self):
        """Clear existing traffic control rules"""
        try:
            # Clear qdisc
            process = await asyncio.create_subprocess_exec(
                'tc', 'qdisc', 'del', 'dev', self.interface, 'root',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
        except:
            pass  # Ignore errors if no rules exist
    
    async def _apply_tc_rules(self, conditions: NetworkCondition):
        """Apply traffic control rules"""
        # Convert conditions to tc parameters
        bandwidth_kbps = int(conditions.bandwidth_mbps * 1024)
        latency_ms = int(conditions.latency_ms)
        jitter_ms = int(conditions.jitter_ms)
        packet_loss_percent = conditions.packet_loss_percent
        
        # Create qdisc with netem for latency/jitter/loss
        cmd_netem = [
            'tc', 'qdisc', 'add', 'dev', self.interface, 'root', 'handle', '1:',
            'netem', 'delay', f'{latency_ms}ms'
        ]
        
        # Add jitter if significant
        if jitter_ms > 0:
            cmd_netem.extend([f'{jitter_ms}ms'])
        
        # Add packet loss if significant
        if packet_loss_percent > 0:
            cmd_netem.extend(['loss', f'{packet_loss_percent}%'])
        
        # Execute netem command
        process = await asyncio.create_subprocess_exec(
            *cmd_netem,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            self.logger.error(f"Failed to apply netem: {stderr.decode()}")
            return
        
        # Add bandwidth limitation with tbf
        cmd_tbf = [
            'tc', 'qdisc', 'add', 'dev', self.interface, 'parent', '1:1',
            'handle', '10:', 'tbf', 'rate', f'{bandwidth_kbps}kbit',
            'burst', '32k', 'latency', '50ms'
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd_tbf,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            self.logger.error(f"Failed to apply tbf: {stderr.decode()}")
    
    def get_applied_conditions(self) -> Optional[NetworkCondition]:
        """Get currently applied network conditions"""
        return self.applied_conditions
    
    async def remove_shaping(self):
        """Remove all traffic shaping"""
        if self.use_tc:
            await self._clear_tc_rules()
        self.applied_conditions = None
        self.logger.info("Network shaping removed")

class NetworkAnalyzer:
    """
    Analyzes network simulation results
    Provides visualization and statistics
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_conditions_history(self, history: List[NetworkCondition]) -> Dict[str, Any]:
        """Analyze network conditions history"""
        if not history:
            return {'error': 'No history available'}
        
        # Extract metrics
        bandwidths = [c.bandwidth_mbps for c in history]
        latencies = [c.latency_ms for c in history]
        packet_losses = [c.packet_loss_percent for c in history]
        jitters = [c.jitter_ms for c in history]
        
        # Calculate statistics
        stats = {
            'bandwidth': {
                'mean': np.mean(bandwidths),
                'std': np.std(bandwidths),
                'min': np.min(bandwidths),
                'max': np.max(bandwidths),
                'median': np.median(bandwidths)
            },
            'latency': {
                'mean': np.mean(latencies),
                'std': np.std(latencies),
                'min': np.min(latencies),
                'max': np.max(latencies),
                'median': np.median(latencies)
            },
            'packet_loss': {
                'mean': np.mean(packet_losses),
                'std': np.std(packet_losses),
                'min': np.min(packet_losses),
                'max': np.max(packet_losses),
                'median': np.median(packet_losses)
            },
            'jitter': {
                'mean': np.mean(jitters),
                'std': np.std(jitters),
                'min': np.min(jitters),
                'max': np.max(jitters),
                'median': np.median(jitters)
            },
            'duration': history[-1].timestamp - history[0].timestamp,
            'sample_count': len(history)
        }
        
        # Environment distribution
        env_counts = defaultdict(int)
        for condition in history:
            env_counts[condition.location_type] += 1
        
        stats['environment_distribution'] = dict(env_counts)
        
        # Connection stability
        changes = 0
        for i in range(1, len(history)):
            prev_bw = history[i-1].bandwidth_mbps
            curr_bw = history[i].bandwidth_mbps
            if abs(curr_bw - prev_bw) / prev_bw > 0.2:  # >20% change
                changes += 1
        
        stats['stability'] = {
            'bandwidth_changes': changes,
            'change_rate': changes / len(history) if history else 0
        }
        
        return stats
    
    def plot_network_conditions(self, history: List[NetworkCondition], 
                               save_path: Optional[str] = None):
        """Plot network conditions over time"""
        if not history:
            self.logger.warning("No history to plot")
            return None
        
        # Prepare data
        timestamps = [(c.timestamp - history[0].timestamp) / 60 for c in history]  # Minutes
        bandwidths = [c.bandwidth_mbps for c in history]
        latencies = [c.latency_ms for c in history]
        packet_losses = [c.packet_loss_percent for c in history]
        
        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Bandwidth plot
        axes[0].plot(timestamps, bandwidths, 'b-', alpha=0.7)
        axes[0].set_ylabel('Bandwidth (Mbps)')
        axes[0].set_title('Network Conditions Over Time')
        axes[0].grid(True, alpha=0.3)
        
        # Latency plot
        axes[1].plot(timestamps, latencies, 'r-', alpha=0.7)
        axes[1].set_ylabel('Latency (ms)')
        axes[1].grid(True, alpha=0.3)
        
        # Packet loss plot
        axes[2].plot(timestamps, packet_losses, 'g-', alpha=0.7)
        axes[2].set_ylabel('Packet Loss (%)')
        axes[2].set_xlabel('Time (minutes)')
        axes[2].grid(True, alpha=0.3)
        
        # Color background by environment
        environments = [c.location_type for c in history]
        env_colors = {
            'home_wifi': 'lightblue',
            'office_wifi': 'lightgreen',
            'mobile_4g': 'orange',
            'mobile_5g': 'purple',
            'public_wifi': 'yellow',
            'poor_connection': 'red',
            'edge_computing': 'cyan'
        }
        
        # Add environment backgrounds
        current_env = environments[0]
        start_time = timestamps[0]
        
        for i, env in enumerate(environments[1:], 1):
            if env != current_env:
                # Add background for previous environment
                color = env_colors.get(current_env, 'gray')
                for ax in axes:
                    ax.axvspan(start_time, timestamps[i], alpha=0.2, color=color, label=current_env if i == 1 else "")
                
                current_env = env
                start_time = timestamps[i]
        
        # Add final environment background
        color = env_colors.get(current_env, 'gray')
        for ax in axes:
            ax.axvspan(start_time, timestamps[-1], alpha=0.2, color=color)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Network conditions plot saved to {save_path}")
        
        return fig
    
    def generate_report(self, history: List[NetworkCondition], 
                       output_path: str = "network_simulation_report.json"):
        """Generate comprehensive network simulation report"""
        stats = self.analyze_conditions_history(history)
        
        # Add additional analysis
        report = {
            'simulation_summary': stats,
            'recommendations': self._generate_recommendations(stats),
            'streamsplit_implications': self._analyze_streamsplit_implications(history),
            'timestamp': time.time()
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Network simulation report saved to {output_path}")
        return report
    
    def _generate_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on network analysis"""
        recommendations = []
        
        # Bandwidth recommendations
        if stats['bandwidth']['mean'] < 5.0:
            recommendations.append("Low bandwidth detected. Consider increasing edge processing ratio.")
        
        # Latency recommendations
        if stats['latency']['mean'] > 100:
            recommendations.append("High latency detected. Dynamic splitting should favor edge processing.")
        
        # Stability recommendations
        if stats['stability']['change_rate'] > 0.2:
            recommendations.append("High network variability. Increase adaptation frequency.")
        
        # Packet loss recommendations
        if stats['packet_loss']['mean'] > 2.0:
            recommendations.append("Significant packet loss. Enable compression and error correction.")
        
        return recommendations
    
    def _analyze_streamsplit_implications(self, history: List[NetworkCondition]) -> Dict[str, Any]:
        """Analyze implications for StreamSplit performance"""
        implications = {
            'optimal_split_ratio': self._estimate_optimal_split_ratio(history),
            'transmission_policy': self._recommend_transmission_policy(history),
            'adaptation_frequency': self._recommend_adaptation_frequency(history),
            'quality_predictions': self._predict_quality_impacts(history)
        }
        
        return implications
    
    def _estimate_optimal_split_ratio(self, history: List[NetworkCondition]) -> float:
        """Estimate optimal edge/server split ratio"""
        # Simple heuristic based on network conditions
        total_score = 0
        count = 0
        
        for condition in history:
            # Score based on bandwidth and latency
            # Higher score = more edge processing
            bandwidth_score = max(0, 1 - condition.bandwidth_mbps / 50.0)
            latency_score = min(1, condition.latency_ms / 200.0)
            loss_score = min(1, condition.packet_loss_percent / 5.0)
            
            score = (bandwidth_score + latency_score + loss_score) / 3.0
            total_score += score
            count += 1
        
        return total_score / count if count > 0 else 0.5
    
    def _recommend_transmission_policy(self, history: List[NetworkCondition]) -> str:
        """Recommend transmission policy based on network characteristics"""
        avg_bandwidth = np.mean([c.bandwidth_mbps for c in history])
        avg_latency = np.mean([c.latency_ms for c in history])
        avg_loss = np.mean([c.packet_loss_percent for c in history])
        
        if avg_bandwidth > 20 and avg_latency < 50 and avg_loss < 1.0:
            return "aggressive"  # Transmit more frequently
        elif avg_bandwidth < 5 or avg_latency > 150 or avg_loss > 3.0:
            return "conservative"  # Transmit only high-uncertainty embeddings
        else:
            return "adaptive"  # Standard adaptive policy
    
    def _recommend_adaptation_frequency(self, history: List[NetworkCondition]) -> float:
        """Recommend adaptation frequency in seconds"""
        # Calculate network stability
        bandwidths = [c.bandwidth_mbps for c in history]
        cv = np.std(bandwidths) / np.mean(bandwidths) if bandwidths else 0
        
        # More unstable networks need more frequent adaptation
        if cv > 0.5:
            return 5.0  # Adapt every 5 seconds
        elif cv > 0.2:
            return 10.0  # Adapt every 10 seconds
        else:
            return 30.0  # Adapt every 30 seconds
    
    def _predict_quality_impacts(self, history: List[NetworkCondition]) -> Dict[str, float]:
        """Predict quality impacts under different scenarios"""
        predictions = {}
        
        # Estimate bandwidth reduction potential
        avg_bandwidth = np.mean([c.bandwidth_mbps for c in history])
        predictions['bandwidth_reduction_potential'] = min(0.8, avg_bandwidth / 10.0)
        
        # Estimate latency reduction potential
        avg_latency = np.mean([c.latency_ms for c in history])
        predictions['latency_reduction_potential'] = min(0.8, avg_latency / 200.0)
        
        # Estimate accuracy preservation
        stability = 1.0 - (np.std([c.bandwidth_mbps for c in history]) / avg_bandwidth)
        predictions['accuracy_preservation'] = max(0.85, stability)
        
        return predictions

class StreamSplitNetworkIntegration:
    """
    Integration between network simulation and StreamSplit framework
    Provides real-time network conditions to StreamSplit components
    """
    
    def __init__(self, network_simulator: NetworkSimulator):
        self.network_simulator = network_simulator
        self.logger = logging.getLogger(__name__)
        
        # StreamSplit components (would be injected in real usage)
        self.splitting_agent = None
        self.edge_module = None
        self.server_module = None
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        
        # Register as observer
        self.network_simulator.add_observer(self._on_network_change)
    
    async def _on_network_change(self, conditions: NetworkCondition):
        """Handle network condition changes"""
        # Convert to format expected by StreamSplit
        network_metrics = {
            'bandwidth': conditions.bandwidth_mbps,
            'latency': conditions.latency_ms,
            'packet_loss': conditions.packet_loss_percent,
            'connection_quality': self._calculate_connection_quality(conditions),
            'timestamp': conditions.timestamp
        }
        
        # Update splitting agent if available
        if self.splitting_agent:
            await self._update_splitting_agent(network_metrics)
        
        # Update transmission policies
        await self._update_transmission_policies(network_metrics)
        
        # Log significant changes
        self.logger.debug(f"Network update: {network_metrics}")
    
    def _calculate_connection_quality(self, conditions: NetworkCondition) -> float:
        """Calculate overall connection quality score (0-1)"""
        # Normalize individual metrics
        bandwidth_score = min(1.0, conditions.bandwidth_mbps / 50.0)
        latency_score = max(0.0, 1.0 - conditions.latency_ms / 500.0)
        loss_score = max(0.0, 1.0 - conditions.packet_loss_percent / 10.0)
        
        # Weighted average
        return 0.4 * bandwidth_score + 0.4 * latency_score + 0.2 * loss_score
    
    async def _update_splitting_agent(self, network_metrics: Dict[str, Any]):
        """Update splitting agent with new network metrics"""
        if hasattr(self.splitting_agent, 'network_simulator'):
            # Inject network state into splitting agent's network simulator
            self.splitting_agent.network_simulator.current_bandwidth = network_metrics['bandwidth']
            self.splitting_agent.network_simulator.current_latency = network_metrics['latency']
            self.splitting_agent.network_simulator.current_packet_loss = network_metrics['packet_loss']
    
    async def _update_transmission_policies(self, network_metrics: Dict[str, Any]):
        """Update transmission policies based on network conditions"""
        # Update edge module transmission threshold
        if self.edge_module and hasattr(self.edge_module, 'update_transmission_threshold'):
            quality = network_metrics['connection_quality']
            # Lower quality = higher threshold (transmit less)
            threshold = 0.3 + 0.4 * (1.0 - quality)
            await self.edge_module.update_transmission_threshold(threshold)
        
        # Update server module aggregation frequency
        if self.server_module and hasattr(self.server_module, 'update_aggregation_frequency'):
            # Poor networks need more frequent aggregation
            freq = max(5.0, 30.0 * network_metrics['connection_quality'])
            await self.server_module.update_aggregation_frequency(freq)
    
    def set_streamsplit_components(self, splitting_agent=None, edge_module=None, server_module=None):
        """Set StreamSplit components for integration"""
        self.splitting_agent = splitting_agent
        self.edge_module = edge_module
        self.server_module = server_module
        self.logger.info("StreamSplit components configured for network integration")

class NetworkScenarioGenerator:
    """
    Generates realistic network scenarios for testing
    Based on real-world patterns and edge computing use cases
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_daily_scenario(self, duration_hours: float = 24) -> List[Tuple[float, NetworkEnvironment, MobilityPattern]]:
        """Generate a realistic daily network scenario"""
        scenario = []
        current_time = 0
        
        # Morning routine (home -> commute -> office)
        scenario.extend([
            (0, NetworkEnvironment.HOME_WIFI, MobilityPattern.STATIONARY),
            (1, NetworkEnvironment.MOBILE_4G, MobilityPattern.DRIVING),
            (2, NetworkEnvironment.OFFICE_WIFI, MobilityPattern.STATIONARY)
        ])
        
        # Work hours with occasional poor connection
        for hour in range(3, 11):
            env = NetworkEnvironment.OFFICE_WIFI
            if random.random() < 0.1:  # 10% chance of poor connection
                env = NetworkEnvironment.POOR_CONNECTION
            scenario.append((hour, env, MobilityPattern.INDOOR_MOVEMENT))
        
        # Lunch break (might go out)
        if random.random() < 0.5:
            scenario.extend([
                (11, NetworkEnvironment.MOBILE_4G, MobilityPattern.WALKING),
                (12, NetworkEnvironment.PUBLIC_WIFI, MobilityPattern.STATIONARY),
                (13, NetworkEnvironment.OFFICE_WIFI, MobilityPattern.WALKING)
            ])
        else:
            scenario.append((11, NetworkEnvironment.OFFICE_WIFI, MobilityPattern.STATIONARY))
        
        # Afternoon work
        for hour in range(14, 18):
            scenario.append((hour, NetworkEnvironment.OFFICE_WIFI, MobilityPattern.STATIONARY))
        
        # Commute home
        scenario.extend([
            (18, NetworkEnvironment.MOBILE_4G, MobilityPattern.DRIVING),
            (19, NetworkEnvironment.HOME_WIFI, MobilityPattern.STATIONARY)
        ])
        
        # Evening at home
        for hour in range(20, 24):
            scenario.append((hour, NetworkEnvironment.HOME_WIFI, MobilityPattern.STATIONARY))
        
        # Convert hours to seconds and filter by duration
        scenario_seconds = []
        for hour, env, mobility in scenario:
            if hour < duration_hours:
                scenario_seconds.append((hour * 3600, env, mobility))
        
        return scenario_seconds
    
    def generate_mobile_scenario(self, duration_hours: float = 4) -> List[Tuple[float, NetworkEnvironment, MobilityPattern]]:
        """Generate mobile/traveling scenario"""
        scenario = []
        
        # Start at home
        scenario.append((0, NetworkEnvironment.HOME_WIFI, MobilityPattern.STATIONARY))
        
        # Travel sequence
        travel_time = 0.5  # 30 minutes
        for i, (env, mobility) in enumerate([
            (NetworkEnvironment.MOBILE_5G, MobilityPattern.DRIVING),
            (NetworkEnvironment.MOBILE_4G, MobilityPattern.DRIVING),
            (NetworkEnvironment.PUBLIC_WIFI, MobilityPattern.STATIONARY),
            (NetworkEnvironment.MOBILE_5G, MobilityPattern.TRAIN),
            (NetworkEnvironment.POOR_CONNECTION, MobilityPattern.TRAIN),
            (NetworkEnvironment.MOBILE_4G, MobilityPattern.WALKING),
            (NetworkEnvironment.OFFICE_WIFI, MobilityPattern.STATIONARY)
        ]):
            time_offset = (i + 1) * travel_time * 3600
            if time_offset < duration_hours * 3600:
                scenario.append((time_offset, env, mobility))
        
        return scenario
    
    def generate_edge_computing_scenario(self, duration_hours: float = 2) -> List[Tuple[float, NetworkEnvironment, MobilityPattern]]:
        """Generate edge computing deployment scenario"""
        scenario = []
        
        # Start with edge computing (ideal conditions)
        scenario.append((0, NetworkEnvironment.EDGE_COMPUTING, MobilityPattern.STATIONARY))
        
        # Simulate various degradations
        degradation_time = duration_hours * 3600 / 6
        
        scenarios = [
            (NetworkEnvironment.OFFICE_WIFI, MobilityPattern.STATIONARY),
            (NetworkEnvironment.MOBILE_5G, MobilityPattern.STATIONARY),
            (NetworkEnvironment.PUBLIC_WIFI, MobilityPattern.STATIONARY),
            (NetworkEnvironment.MOBILE_4G, MobilityPattern.STATIONARY),
            (NetworkEnvironment.POOR_CONNECTION, MobilityPattern.STATIONARY),
            (NetworkEnvironment.EDGE_COMPUTING, MobilityPattern.STATIONARY)  # Return to ideal
        ]
        
        for i, (env, mobility) in enumerate(scenarios):
            time_offset = (i + 1) * degradation_time
            if time_offset < duration_hours * 3600:
                scenario.append((time_offset, env, mobility))
        
        return scenario

async def run_scenario(simulator: NetworkSimulator, scenario: List[Tuple[float, NetworkEnvironment, MobilityPattern]], duration: float):
    """Run a predefined scenario"""
    simulator.logger.info(f"Running scenario with {len(scenario)} transitions over {duration/3600:.1f} hours")
    
    # Start simulation
    simulation_task = asyncio.create_task(simulator.start_simulation(duration))
    
    # Apply scenario transitions
    async def apply_scenario():
        start_time = time.time()
        
        for transition_time, environment, mobility in scenario:
            # Wait for transition time
            elapsed = time.time() - start_time
            wait_time = transition_time - elapsed
            
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            
            # Apply transition
            simulator.set_environment(environment)
            simulator.set_mobility_pattern(mobility)
            simulator.logger.info(f"Scenario transition: {environment.value} ({mobility.value})")
    
    # Run both tasks
    await asyncio.gather(simulation_task, apply_scenario())

def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('network_simulation.log')
        ]
    )

async def main():
    """Main entry point for network simulation"""
    parser = argparse.ArgumentParser(description='StreamSplit Network Simulation')
    parser.add_argument('--duration', type=float, default=300, help='Simulation duration in seconds')
    parser.add_argument('--scenario', choices=['daily', 'mobile', 'edge', 'random'], default='random', 
                       help='Predefined scenario to run')
    parser.add_argument('--environment', choices=[e.value for e in NetworkEnvironment], 
                       help='Initial network environment')
    parser.add_argument('--mobility', choices=[m.value for m in MobilityPattern], default='stationary',
                       help='Initial mobility pattern')
    parser.add_argument('--interface', default='lo', help='Network interface for traffic shaping')
    parser.add_argument('--enable-shaping', action='store_true', help='Enable traffic shaping (Linux only)')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--output-dir', default='.', help='Output directory for results')
    parser.add_argument('--plot', action='store_true', help='Generate plots of network conditions')
    parser.add_argument('--streamsplit-integration', action='store_true', 
                       help='Enable StreamSplit integration')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Create simulation configuration
    config = {
        'simulation_speed': 1.0,
        'enable_weather_effects': False,
        'enable_congestion_simulation': True
    }
    
    # Initialize simulator
    simulator = NetworkSimulator(config)
    
    # Set initial conditions
    if args.environment:
        env = NetworkEnvironment(args.environment)
        simulator.set_environment(env)
    
    mobility = MobilityPattern(args.mobility)
    simulator.set_mobility_pattern(mobility)
    
    # Initialize traffic shaper if requested
    traffic_shaper = None
    if args.enable_shaping:
        traffic_shaper = NetworkTrafficShaper(interface=args.interface)
        simulator.add_observer(traffic_shaper.apply_conditions)
        logger.info(f"Traffic shaping enabled on interface {args.interface}")
    
    # Initialize StreamSplit integration if requested
    integration = None
    if args.streamsplit_integration:
        integration = StreamSplitNetworkIntegration(simulator)
        logger.info("StreamSplit integration enabled")
    
    # Generate scenario if specified
    scenario_generator = NetworkScenarioGenerator()
    scenario = None
    
    if args.scenario == 'daily':
        scenario = scenario_generator.generate_daily_scenario(args.duration / 3600)
        logger.info("Generated daily scenario")
    elif args.scenario == 'mobile':
        scenario = scenario_generator.generate_mobile_scenario(args.duration / 3600)
        logger.info("Generated mobile scenario")
    elif args.scenario == 'edge':
        scenario = scenario_generator.generate_edge_computing_scenario(args.duration / 3600)
        logger.info("Generated edge computing scenario")
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Simulation interrupted, shutting down...")
        asyncio.create_task(simulator.stop_simulation())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Run simulation
        logger.info(f"Starting network simulation for {args.duration} seconds...")
        
        if scenario:
            await run_scenario(simulator, scenario, args.duration)
        else:
            await simulator.start_simulation(args.duration)
        
        logger.info("Simulation completed successfully")
        
        # Analyze results
        analyzer = NetworkAnalyzer()
        history = simulator.get_conditions_history()
        
        # Generate statistics
        stats = analyzer.analyze_conditions_history(history)
        logger.info(f"Simulation stats: {json.dumps(stats, indent=2)}")
        
        # Generate report
        report_path = f"{args.output_dir}/network_simulation_report.json"
        report = analyzer.generate_report(history, report_path)
        
        # Generate plots if requested
        if args.plot:
            plot_path = f"{args.output_dir}/network_conditions.png"
            analyzer.plot_network_conditions(history, plot_path)
            logger.info(f"Network conditions plot saved to {plot_path}")
        
        # Save raw data
        raw_data_path = f"{args.output_dir}/network_conditions_raw.json"
        with open(raw_data_path, 'w') as f:
            json.dump([{
                'timestamp': c.timestamp,
                'bandwidth_mbps': c.bandwidth_mbps,
                'latency_ms': c.latency_ms,
                'packet_loss_percent': c.packet_loss_percent,
                'jitter_ms': c.jitter_ms,
                'location_type': c.location_type,
                'connection_type': c.connection_type
            } for c in history], f, indent=2)
        
        logger.info(f"Raw network data saved to {raw_data_path}")
        
        # Print summary
        print("\n=== Network Simulation Summary ===")
        print(f"Duration: {stats['duration']:.1f} seconds")
        print(f"Average bandwidth: {stats['bandwidth']['mean']:.2f} Mbps")
        print(f"Average latency: {stats['latency']['mean']:.2f} ms")
        print(f"Average packet loss: {stats['packet_loss']['mean']:.2f}%")
        print(f"Network changes: {stats['stability']['bandwidth_changes']}")
        print(f"Environment distribution: {stats['environment_distribution']}")
        
        # Print StreamSplit recommendations
        if 'streamsplit_implications' in report:
            implications = report['streamsplit_implications']
            print(f"\n=== StreamSplit Recommendations ===")
            print(f"Optimal edge/server split ratio: {implications['optimal_split_ratio']:.2f}")
            print(f"Recommended transmission policy: {implications['transmission_policy']}")
            print(f"Recommended adaptation frequency: {implications['adaptation_frequency']:.1f}s")
        
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        await simulator.stop_simulation()
        
        if traffic_shaper:
            await traffic_shaper.remove_shaping()
            logger.info("Traffic shaping removed")

if __name__ == "__main__":
    asyncio.run(main())