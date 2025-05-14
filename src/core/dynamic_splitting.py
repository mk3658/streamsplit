"""
Dynamic Computation Splitting Agent for StreamSplit Framework
Implements RL-based adaptive splitting decisions using PPO
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from dataclasses import dataclass
from enum import Enum

import psutil
import random

@dataclass
class NetworkState:
    """Network condition state"""
    bandwidth: float  # Mbps
    latency: float    # ms
    packet_loss: float  # percentage
    connection_quality: float  # 0-1 score

@dataclass
class ResourceState:
    """Device resource state"""
    cpu_utilization: float
    memory_usage: float
    available_memory: float
    battery_level: float
    temperature: float

@dataclass
class SplitDecision:
    """Split decision output"""
    split_point: int
    confidence: float
    reasoning: Dict[str, float]
    expected_performance: Dict[str, float]

class SplitAction(Enum):
    """Available splitting actions (layers to split at)"""
    EDGE_ONLY = 0      # Process everything on edge
    SPLIT_EARLY = 1    # Split at 25% of layers
    SPLIT_MID = 2      # Split at 50% of layers
    SPLIT_LATE = 3     # Split at 75% of layers
    SERVER_ONLY = 4    # Process everything on server

class SplitPolicyNetwork(nn.Module):
    """Neural network for split policy using PPO"""
    
    def __init__(self, state_dim: int = 20, hidden_dim: int = 128, 
                 num_actions: int = 5):
        super().__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
            nn.Softmax(dim=-1)
        )
        
        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action probabilities and state value"""
        features = self.feature_extractor(state)
        action_probs = self.policy_head(features)
        state_value = self.value_head(features)
        return action_probs, state_value
    
    def act(self, state: torch.Tensor) -> Tuple[int, float, float]:
        """Sample action from policy"""
        action_probs, state_value = self.forward(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), state_value.item()

class ExperienceBuffer:
    """Buffer for storing RL experiences"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.states = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.log_probs = deque(maxlen=capacity)
        self.values = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)
        
    def add(self, state: np.ndarray, action: int, reward: float,
            log_prob: float, value: float, done: bool = False):
        """Add experience to buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def get_batch(self) -> Dict[str, torch.Tensor]:
        """Get all experiences as batch"""
        return {
            'states': torch.FloatTensor(list(self.states)),
            'actions': torch.LongTensor(list(self.actions)),
            'rewards': torch.FloatTensor(list(self.rewards)),
            'log_probs': torch.FloatTensor(list(self.log_probs)),
            'values': torch.FloatTensor(list(self.values)),
            'dones': torch.BoolTensor(list(self.dones))
        }
    
    def clear(self):
        """Clear all experiences"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
    
    def __len__(self):
        return len(self.states)

class NetworkSimulator:
    """Simulates network conditions based on configuration"""
    
    def __init__(self, config):
        self.config = config
        self.bandwidth_range = config.bandwidth_range
        self.latency_range = config.latency_range
        
        # Current conditions
        self.current_bandwidth = random.uniform(*self.bandwidth_range)
        self.current_latency = random.uniform(*self.latency_range)
        self.current_packet_loss = random.uniform(0, 0.05)  # 0-5%
        
        # Change patterns
        self.bandwidth_trend = 0.0
        self.latency_trend = 0.0
        self.last_update = time.time()
        
    def update(self):
        """Update network conditions with realistic variations"""
        current_time = time.time()
        dt = current_time - self.last_update
        
        # Add some randomness to trends
        if random.random() < 0.1:  # 10% chance to change trend
            self.bandwidth_trend = random.uniform(-0.5, 0.5)
            self.latency_trend = random.uniform(-10, 10)
        
        # Update with trends and noise
        bandwidth_noise = random.gauss(0, 0.1)
        latency_noise = random.gauss(0, 5)
        
        self.current_bandwidth += self.bandwidth_trend * dt + bandwidth_noise
        self.current_latency += self.latency_trend * dt + latency_noise
        
        # Clamp to valid ranges
        self.current_bandwidth = np.clip(
            self.current_bandwidth, *self.bandwidth_range
        )
        self.current_latency = np.clip(
            self.current_latency, *self.latency_range
        )
        
        # Update packet loss based on conditions
        if self.current_bandwidth < 1.0 or self.current_latency > 150:
            self.current_packet_loss = min(0.1, self.current_packet_loss + 0.01)
        else:
            self.current_packet_loss = max(0.0, self.current_packet_loss - 0.005)
        
        self.last_update = current_time
    
    def get_state(self) -> NetworkState:
        """Get current network state"""
        self.update()
        
        # Calculate connection quality score
        bandwidth_score = min(1.0, self.current_bandwidth / 5.0)
        latency_score = max(0.0, 1.0 - self.current_latency / 300.0)
        loss_score = max(0.0, 1.0 - self.current_packet_loss / 0.1)
        
        quality = (bandwidth_score + latency_score + loss_score) / 3.0
        
        return NetworkState(
            bandwidth=self.current_bandwidth,
            latency=self.current_latency,
            packet_loss=self.current_packet_loss,
            connection_quality=quality
        )

class DynamicSplittingAgent:
    """
    RL-based agent for dynamic computation splitting decisions
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model architecture parameters
        self.state_dim = 20  # Dimension of state vector
        self.num_actions = len(SplitAction)
        self.total_layers = 16  # Total layers in the model
        
        # Initialize policy network
        self.policy_net = SplitPolicyNetwork(
            state_dim=self.state_dim,
            hidden_dim=128,
            num_actions=self.num_actions
        )
        
        # PPO parameters
        self.learning_rate = config.split_agent_lr
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), 
            lr=self.learning_rate
        )
        
        self.clip_epsilon = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.gamma = 0.99  # Discount factor
        self.gae_lambda = 0.95  # GAE parameter
        
        # Experience buffer
        self.experience_buffer = ExperienceBuffer(capacity=10000)
        
        # Reward weights
        self.reward_weights = config.split_reward_weights
        
        # Network simulator
        self.network_simulator = NetworkSimulator(config)
        
        # State tracking
        self.current_state = None
        self.last_action = None
        self.last_log_prob = None
        self.last_value = None
        self.episode_rewards = deque(maxlen=100)
        
        # Performance tracking
        self.performance_history = {
            'accuracy': deque(maxlen=1000),
            'latency': deque(maxlen=1000),
            'bandwidth_usage': deque(maxlen=1000),
            'resource_usage': deque(maxlen=1000),
            'energy_consumption': deque(maxlen=1000)
        }
        
        # Adaptation parameters
        self.update_frequency = 32  # Update policy every N steps
        self.steps_since_update = 0
        
        self.is_running = False
        
        self.logger.info("DynamicSplittingAgent initialized")
    
    async def start(self):
        """Start the splitting agent"""
        self.is_running = True
        self.logger.info("DynamicSplittingAgent started")
    
    async def stop(self):
        """Stop the splitting agent"""
        self.is_running = False
        self.logger.info("DynamicSplittingAgent stopped")
    
    def _get_resource_state(self) -> ResourceState:
        """Get current device resource state"""
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # Simulate battery and temperature (would be real on actual device)
        battery = random.uniform(20, 100)  # 20-100%
        temperature = random.uniform(40, 80)  # 40-80°C
        
        return ResourceState(
            cpu_utilization=cpu_percent / 100.0,
            memory_usage=memory.percent / 100.0,
            available_memory=memory.available / (1024**3),  # GB
            battery_level=battery / 100.0,
            temperature=temperature
        )
    
    def _get_state_vector(self, resource_state: ResourceState, 
                         network_state: NetworkState) -> np.ndarray:
        """Convert states to feature vector for RL agent"""
        # Resource features (5 dimensions)
        resource_features = [
            resource_state.cpu_utilization,
            resource_state.memory_usage,
            resource_state.available_memory / 8.0,  # Normalize by max expected
            resource_state.battery_level,
            resource_state.temperature / 100.0  # Normalize
        ]
        
        # Network features (4 dimensions)
        network_features = [
            network_state.bandwidth / 10.0,  # Normalize by max expected
            network_state.latency / 500.0,   # Normalize by max expected
            network_state.packet_loss / 0.1, # Normalize by max expected
            network_state.connection_quality
        ]
        
        # Performance history features (5 dimensions)
        perf_features = []
        for metric in ['accuracy', 'latency', 'bandwidth_usage', 
                      'resource_usage', 'energy_consumption']:
            if self.performance_history[metric]:
                avg_value = np.mean(list(self.performance_history[metric])[-10:])
                perf_features.append(avg_value)
            else:
                perf_features.append(0.0)
        
        # Recent decisions features (6 dimensions)
        recent_actions = [0] * self.num_actions
        if len(self.experience_buffer.actions) > 0:
            # One-hot encode last action
            last_action = list(self.experience_buffer.actions)[-1]
            recent_actions[last_action] = 1.0
        
        # Average recent performance
        recent_perf = np.mean(list(self.episode_rewards)[-5:]) if self.episode_rewards else 0.0
        
        # Combine all features
        state_vector = np.array(resource_features + network_features + 
                               perf_features + recent_actions + [recent_perf])
        
        # Ensure we have exactly state_dim features
        assert len(state_vector) == self.state_dim
        
        return state_vector.astype(np.float32)
    
    def _action_to_split_point(self, action: int) -> int:
        """Convert action index to actual split layer"""
        split_percentages = {
            SplitAction.EDGE_ONLY.value: 0.0,
            SplitAction.SPLIT_EARLY.value: 0.25,
            SplitAction.SPLIT_MID.value: 0.5,
            SplitAction.SPLIT_LATE.value: 0.75,
            SplitAction.SERVER_ONLY.value: 1.0
        }
        
        percentage = split_percentages[action]
        split_layer = int(percentage * self.total_layers)
        return split_layer
    
    def _calculate_reward(self, performance_metrics: Dict[str, float],
                         resource_state: ResourceState, 
                         network_state: NetworkState,
                         action: int) -> float:
        """Calculate reward based on current performance and action"""
        reward = 0.0
        
        # Accuracy reward (higher is better)
        accuracy = performance_metrics.get('accuracy', 0.0)
        reward += self.reward_weights['accuracy'] * accuracy
        
        # Resource usage penalty (lower is better)
        resource_usage = (resource_state.cpu_utilization + 
                         resource_state.memory_usage) / 2.0
        reward += self.reward_weights['resource_usage'] * resource_usage
        
        # Latency penalty (lower is better)
        latency = performance_metrics.get('latency', 0.0) / 1000.0  # Normalize ms to seconds
        reward += self.reward_weights['latency'] * latency
        
        # Privacy risk penalty (edge processing is more private)
        split_point = self._action_to_split_point(action)
        privacy_risk = split_point / self.total_layers  # More server compute = higher risk
        reward += self.reward_weights['privacy_risk'] * privacy_risk
        
        # Bonus for good network utilization
        if network_state.connection_quality > 0.8 and action in [2, 3, 4]:
            reward += 0.1  # Bonus for using server when network is good
        elif network_state.connection_quality < 0.3 and action in [0, 1]:
            reward += 0.1  # Bonus for using edge when network is poor
        
        # Battery conservation bonus
        if resource_state.battery_level < 0.2 and action in [0, 1]:
            reward += 0.15  # Bonus for edge processing when battery low
        
        return reward
    
    async def get_split_decision(self) -> Dict[str, Any]:
        """Get split decision from RL agent"""
        # Get current states
        resource_state = self._get_resource_state()
        network_state = self.network_simulator.get_state()
        
        # Create state vector
        state_vector = self._get_state_vector(resource_state, network_state)
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
        
        # Get action from policy
        with torch.no_grad():
            action, log_prob, value = self.policy_net.act(state_tensor)
        
        # Convert action to split point
        split_point = self._action_to_split_point(action)
        
        # Store for reward calculation
        self.current_state = state_vector
        self.last_action = action
        self.last_log_prob = log_prob
        self.last_value = value
        
        # Calculate confidence and reasoning
        action_probs, _ = self.policy_net(state_tensor)
        confidence = action_probs[0, action].item()
        
        reasoning = {
            'resource_pressure': resource_state.cpu_utilization,
            'network_quality': network_state.connection_quality,
            'battery_level': resource_state.battery_level,
            'recent_performance': np.mean(list(self.episode_rewards)[-5:]) if self.episode_rewards else 0.0
        }
        
        expected_performance = {
            'expected_latency': self._estimate_latency(split_point, network_state),
            'expected_accuracy': self._estimate_accuracy(split_point, resource_state),
            'expected_energy': self._estimate_energy(split_point, resource_state)
        }
        
        decision = SplitDecision(
            split_point=split_point,
            confidence=confidence,
            reasoning=reasoning,
            expected_performance=expected_performance
        )
        
        return {
            'split_point': decision.split_point,
            'confidence': decision.confidence,
            'reasoning': decision.reasoning,
            'expected_performance': decision.expected_performance,
            'action': action
        }
    
    def _estimate_latency(self, split_point: int, network_state: NetworkState) -> float:
        """Estimate latency for given split point"""
        # Edge processing time (proportional to layers processed)
        edge_ratio = split_point / self.total_layers
        edge_latency = edge_ratio * 50.0  # Base edge processing time
        
        # Network latency (if data needs transmission)
        network_latency = 0.0
        if split_point < self.total_layers:
            network_latency = network_state.latency
        
        # Server processing time
        server_ratio = (self.total_layers - split_point) / self.total_layers
        server_latency = server_ratio * 20.0  # Server is faster
        
        return edge_latency + network_latency + server_latency
    
    def _estimate_accuracy(self, split_point: int, resource_state: ResourceState) -> float:
        """Estimate accuracy for given split point"""
        # Base accuracy depends on resource constraints
        base_accuracy = 0.8 if resource_state.cpu_utilization < 0.7 else 0.75
        
        # Full edge processing might have lower accuracy due to resource constraints
        if split_point == 0:
            accuracy_penalty = 0.05 * resource_state.cpu_utilization
            return max(0.7, base_accuracy - accuracy_penalty)
        
        # Server involvement generally improves accuracy
        server_ratio = (self.total_layers - split_point) / self.total_layers
        accuracy_boost = server_ratio * 0.1
        
        return min(1.0, base_accuracy + accuracy_boost)
    
    def _estimate_energy(self, split_point: int, resource_state: ResourceState) -> float:
        """Estimate energy consumption for given split point"""
        # Edge computation energy
        edge_ratio = split_point / self.total_layers
        edge_energy = edge_ratio * 1.0  # Normalized energy units
        
        # Network transmission energy
        transmission_energy = 0.0
        if split_point < self.total_layers:
            transmission_energy = 0.2  # Fixed cost for transmission
        
        return edge_energy + transmission_energy
    
    async def update_performance(self, performance_history: Dict[str, List[float]]):
        """Update agent with recent performance metrics"""
        if not self.current_state or self.last_action is None:
            return
        
        # Extract latest metrics
        latest_metrics = {}
        for key, values in performance_history.items():
            if values:
                latest_metrics[key] = values[-1]
                # Update performance history
                self.performance_history[key].append(values[-1])
        
        # Calculate reward
        resource_state = self._get_resource_state()
        network_state = self.network_simulator.get_state()
        reward = self._calculate_reward(
            latest_metrics, resource_state, network_state, self.last_action
        )
        
        # Store experience
        self.experience_buffer.add(
            state=self.current_state,
            action=self.last_action,
            reward=reward,
            log_prob=self.last_log_prob,
            value=self.last_value
        )
        
        self.episode_rewards.append(reward)
        self.steps_since_update += 1
        
        # Update policy if we have enough experience
        if (self.steps_since_update >= self.update_frequency and 
            len(self.experience_buffer) >= self.update_frequency):
            await self._update_policy()
            self.steps_since_update = 0
    
    async def _update_policy(self):
        """Update policy using PPO"""
        if len(self.experience_buffer) < self.update_frequency:
            return
        
        # Get batch of experiences
        batch = self.experience_buffer.get_batch()
        
        # Calculate advantages using GAE
        advantages = self._calculate_advantages(
            batch['rewards'], batch['values'], batch['dones']
        )
        returns = advantages + batch['values']
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(4):  # Multiple epochs per update
            # Forward pass
            new_action_probs, new_values = self.policy_net(batch['states'])
            
            # Get probabilities for taken actions
            new_log_probs = torch.log(
                new_action_probs.gather(1, batch['actions'].unsqueeze(1)).squeeze()
            )
            
            # Calculate policy ratio
            ratio = torch.exp(new_log_probs - batch['log_probs'])
            
            # Policy loss with clipping
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(new_values.squeeze(), returns)
            
            # Entropy loss for exploration
            entropy_loss = -(new_action_probs * torch.log(new_action_probs + 1e-8)).sum(1).mean()
            
            # Total loss
            total_loss = (policy_loss + 
                         self.value_loss_coef * value_loss - 
                         self.entropy_coef * entropy_loss)
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.optimizer.step()
        
        # Clear experience buffer
        self.experience_buffer.clear()
        
        # Log training info
        avg_reward = np.mean(list(self.episode_rewards)[-50:]) if self.episode_rewards else 0.0
        self.logger.info(f"Policy updated. Average reward: {avg_reward:.3f}")
    
    def _calculate_advantages(self, rewards: torch.Tensor, values: torch.Tensor, 
                            dones: torch.Tensor) -> torch.Tensor:
        """Calculate advantages using Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        advantage = 0.0
        
        # Work backwards through the episode
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            advantage = delta + self.gamma * self.gae_lambda * advantage
            advantages[t] = advantage
        
        return advantages
    
    def get_network_quality(self) -> float:
        """Get current network quality score"""
        network_state = self.network_simulator.get_state()
        return network_state.connection_quality
    
    def get_network_metrics(self) -> Dict[str, float]:
        """Get detailed network metrics"""
        network_state = self.network_simulator.get_state()
        return {
            'bandwidth': network_state.bandwidth,
            'latency': network_state.latency,
            'packet_loss': network_state.packet_loss,
            'connection_quality': network_state.connection_quality
        }
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get agent performance metrics"""
        return {
            'average_reward': np.mean(list(self.episode_rewards)) if self.episode_rewards else 0.0,
            'reward_std': np.std(list(self.episode_rewards)) if self.episode_rewards else 0.0,
            'episodes_completed': len(self.episode_rewards),
            'experiences_collected': len(self.experience_buffer)
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get current agent state for checkpointing"""
        return {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': list(self.episode_rewards),
            'performance_history': {
                k: list(v) for k, v in self.performance_history.items()
            },
            'steps_since_update': self.steps_since_update
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load agent state from checkpoint"""
        self.policy_net.load_state_dict(state['policy_net_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.episode_rewards = deque(state['episode_rewards'], maxlen=100)
        
        for k, v in state['performance_history'].items():
            self.performance_history[k] = deque(v, maxlen=1000)
        
        self.steps_since_update = state['steps_since_update']
        
        self.logger.info("DynamicSplittingAgent state loaded from checkpoint")