import numpy as np
import torch
import copy
from collections import defaultdict
import logging

from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCTSNode:
    """
    Node in the Monte Carlo Tree Search for counterfactual generation.
    """
    def __init__(self, state, parent=None, action=None, prior=0.0):
        self.state = state  # Environment state (encoded features or tokens)
        self.parent = parent  # Parent node
        self.action = action  # Action that led to this node
        self.children = {}  # Maps actions to child nodes
        self.visit_count = 0  # Number of times node was visited
        self.value_sum = 0.0  # Sum of values from all visits
        self.prior = prior  # Prior probability from policy network
    
    @property
    def value(self):
        """Mean value of the node."""
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0
    
    def is_expanded(self):
        """Check if node has been expanded."""
        return len(self.children) > 0
    
    def add_child(self, action, child_node):
        """Add a child node."""
        self.children[action] = child_node


class PPOMCTS:
    """
    Monte Carlo Tree Search implementation for PPO counterfactual generation.
    """
    def __init__(self, env, ppo_model, c_puct=8.0, distance_weight=0.1, discount_factor=0.99, num_simulations=50, kl_beta=0.15):
        """
        Initialize MCTS for counterfactual generation.
        
        Parameters:
        - env: PPOEnv instance
        - ppo_model: Trained PPO model (policy and value networks)
        - c_puct: Exploration constant for PUCT formula
        - distance_weight: Weight for distance penalty in PUCT formula
        - discount_factor: Discount factor for future rewards (gamma)
        - num_simulations: Number of simulations per move
        - kl_beta: KL coefficient for step-level reward
        """
        self.env = env
        self.ppo_model = ppo_model
        self.c_puct = c_puct
        self.distance_weight = distance_weight
        self.discount_factor = discount_factor
        self.num_simulations = num_simulations
        self.kl_beta = kl_beta
        self.feature_dim = len(self.env.feature_order)  # Derive feature dimension from environment
        self.q_values = defaultdict(float)  # Maps (state_hash, action) to Q-value
        
    def hash_state(self, state):
        """Create a hashable representation of the state."""
        if isinstance(state, np.ndarray):
            return tuple(state.flatten())
        elif isinstance(state, torch.Tensor):
            return tuple(state.cpu().numpy().flatten())
        return tuple(state)
    
    def _set_env_state(self, env, state):
        """
        Set the environment state based on the node’s state representation.
        Uses env.reset() and updates modified_features without assuming a set_state method.
        """
        state_shape = state.shape[0] if isinstance(state, (np.ndarray, torch.Tensor)) else len(state)
        expected_obs_dim = self.env.observation_space.shape[0]
        if state_shape != expected_obs_dim:
            raise ValueError(f"State has {state_shape} features, but {expected_obs_dim} expected")
        
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        
        # Extract modified_encoded from state (assumes observation structure from PPOEnv._get_observation)
        feature_dim = len(self.env.feature_order)
        modified_encoded = state[feature_dim:2 * feature_dim].copy()  # modified_encoded is second block
        env.reset()
        env.modified_encoded = modified_encoded
        env.modified_features = env.encode_features(env.modified_features)  # Ensure consistency
        env.current_prediction = env.generate_prediction(env.model, env.modified_features)
    
    def initialize_tree(self, root_state):
        """Initialize the search tree with the root state."""
        expected_obs_dim = self.env.observation_space.shape[0]
        if root_state.shape[0] != expected_obs_dim:
            raise ValueError(f"Root state has {root_state.shape[0]} features, but {expected_obs_dim} expected")
        
        with torch.no_grad():
            obs_tensor = torch.tensor(root_state, dtype=torch.float32).unsqueeze(0)
            value = self.ppo_model.policy.predict_values(obs_tensor).item()
        
        self.root = MCTSNode(state=root_state, prior=1.0)
        self.root.value_sum = value
        self.root.visit_count = 1
        return self.root
    
    def select(self, node):
        """
        Select a path through the tree until a leaf node using PUCT with distance penalty.
        """
        while node.is_expanded() and not self._is_terminal_state(node.state):
            best_action = None
            best_score = float('-inf')
            
            virtual_env = copy.deepcopy(self.env)
            self._set_env_state(virtual_env, node.state)
            
            # Estimate max distance for normalization
            max_distance = sum(
                virtual_env.tab_dataset.iloc[:, i].max() - virtual_env.tab_dataset.iloc[:, i].min()
                if i not in virtual_env.categorical_indices
                else 1.0
                for i in range(len(virtual_env.feature_order))
            ) or 1.0  # Avoid division by zero
            
            for action, child in node.children.items():
                action_policy = self.get_policy_for_action(node.state, action)
                exploration_score = self.c_puct * action_policy * np.sqrt(node.visit_count) / (1 + child.visit_count)
                q_value = self.q_values[(self.hash_state(node.state), action)]
                
                # Calculate distance for the child state
                child_env = copy.deepcopy(self.env)
                self._set_env_state(child_env, child.state)
                distance = child_env.calculate_distance(child_env.original_features, child_env.modified_features)
                normalized_distance = distance / max_distance  # Normalize to [0, 1]
                
                # Modified UCB score with distance penalty
                ucb_score = q_value + exploration_score - self.distance_weight * normalized_distance
                
                if ucb_score > best_score:
                    best_score = ucb_score
                    best_action = action
            
            if best_action is None:
                break
            node = node.children[best_action]
        
        return node
    
    def get_policy_for_action(self, state, action):
        """Get the policy probability for a specific action."""
        with torch.no_grad():
            obs_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_dist = self.ppo_model.policy.get_distribution(obs_tensor)
            num_actions = self.env.action_space.n
            action_probs = torch.exp(action_dist.log_prob(torch.arange(num_actions))).cpu().numpy()
        
        if action < len(action_probs):
            return action_probs[action]
        return 0.01  # Small default probability
    
    def expand(self, leaf_node, temperature=2.0):
        """
        Expand the leaf node by adding top-k child nodes.
        """
        if leaf_node.is_expanded() or self._is_terminal_state(leaf_node.state):
            return leaf_node
        
        virtual_env = copy.deepcopy(self.env)
        self._set_env_state(virtual_env, leaf_node.state)
        
        with torch.no_grad():
            obs_tensor = torch.tensor(leaf_node.state, dtype=torch.float32).unsqueeze(0)
            action_dist = self.ppo_model.policy.get_distribution(obs_tensor)
            num_actions = self.env.action_space.n
            action_probs = torch.exp(action_dist.log_prob(torch.arange(num_actions))).cpu().numpy()
            action_probs = action_probs ** (1 / temperature)
            action_probs /= action_probs.sum() + 1e-8
        
        top_k = min(len(action_probs), 50)  # Match branching factor
        top_action_indices = np.argsort(action_probs)[-top_k:]
        
        for action_idx in top_action_indices:
            if action_probs[action_idx] < 0.005:
                continue
            next_state, reward, done, info = virtual_env.step(action_idx)
            reward += self.MCTS_reward(virtual_env.calculate_distance(
                virtual_env.original_features, virtual_env.modified_features))
            if next_state.shape[0] != self.env.observation_space.shape[0]:
                raise ValueError(f"Next state has {next_state.shape[0]} features, but {self.env.observation_space.shape[0]} expected")
            child = MCTSNode(
                state=next_state,
                parent=leaf_node,
                action=action_idx,
                prior=action_probs[action_idx]
            )
            leaf_node.add_child(action_idx, child)
            self.q_values[(self.hash_state(leaf_node.state), action_idx)] = reward
        
        return leaf_node
    
    def evaluate(self, node):
        """
        Evaluate the value of a node using the PPO value model.
        """
        virtual_env = copy.deepcopy(self.env)
        self._set_env_state(virtual_env, node.state)
        
        if self._is_terminal_state(node.state):
            counterfactual_found = virtual_env.current_prediction != virtual_env.original_prediction
            reward = virtual_env.calculate_reward(counterfactual_found=counterfactual_found)
            reward += self.MCTS_reward(virtual_env.calculate_distance(
                virtual_env.original_features, virtual_env.modified_features))
            value = reward
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor(node.state, dtype=torch.float32).unsqueeze(0)
                value = self.ppo_model.policy.predict_values(obs_tensor).item()
        
        node.visit_count = 1
        node.value_sum = value
        
        if node.parent is not None:
            state_hash = self.hash_state(node.parent.state)
            self.q_values[(state_hash, node.action)] = value  # Initialize Q with V
        
        return value
    
    def backup(self, node, value):
        """
        Update statistics for all nodes in the path from leaf to root.
        """
        current = node
        while current.parent is not None:
            parent = current.parent
            action = current.action
            parent_state_hash = self.hash_state(parent.state)
            
            virtual_env = copy.deepcopy(self.env)
            self._set_env_state(virtual_env, parent.state)
            _, reward, _, _ = virtual_env.step(action)
            reward += self.MCTS_reward(virtual_env.calculate_distance(
                virtual_env.original_features, virtual_env.modified_features))
            
            # Approximate KL term if reference policy is unavailable
            try:
                kl_term = -self.kl_beta * np.log(self.get_policy_for_action(parent.state, action) / 
                                               self.ppo_model.reference_policy.get_probability(parent.state, action))
            except AttributeError:
                kl_term = 0.0
            
            self.q_values[(parent_state_hash, action)] = reward + kl_term + self.discount_factor * value
            
            parent.visit_count += 1
            total_visits = sum(child.visit_count for child in parent.children.values())
            value_sum = sum(child.visit_count * self.q_values[(parent_state_hash, a)] 
                           for a, child in parent.children.items())
            
            parent.value_sum = value_sum
            value = value_sum / total_visits if total_visits > 0 else parent.value
            current = parent
    
    def _is_terminal_state(self, state):
        """
        Check if a state is terminal (counterfactual found or max steps reached).
        """
        virtual_env = copy.deepcopy(self.env)
        self._set_env_state(virtual_env, state)
        return (virtual_env.current_prediction != virtual_env.original_prediction or
                virtual_env.steps_taken >= virtual_env.max_steps)
    

    def MCTS_reward(self, distance):
        """
        Calculate the MCTS reward based on distance.
        """
        if distance < 5:
            return 50
        elif distance < 20:
            return 30
        elif distance < 100:
            return 20
        elif distance < 500:
            return 10
        elif distance < 1000:
            return 0
        elif distance < 5000:
            return -10
        elif distance < 10000:
            return -25
        else:
            return -40
        


    def run_mcts(self, root_state, temperature=2.0):
        """
        Run the full MCTS process to find the best action.
        """
        expected_obs_dim = self.env.observation_space.shape[0]
        if isinstance(root_state, np.ndarray) and root_state.shape[0] != expected_obs_dim:
            raise ValueError(f"Root state has {root_state.shape[0]} features, but {expected_obs_dim} expected")
        
        self.initialize_tree(root_state)
        
        for _ in range(self.num_simulations):
            leaf = self.select(self.root)
            if not self._is_terminal_state(leaf.state):
                leaf = self.expand(leaf, temperature)
            value = self.evaluate(leaf)
            self.backup(leaf, value)
        
        visit_counts = np.array([child.visit_count for child in self.root.children.values()])
        actions = list(self.root.children.keys())
        
        if len(visit_counts) > 0:
            if temperature == 0:
                return actions[np.argmax(visit_counts)]
            visit_counts = visit_counts ** (1 / temperature)
            visit_counts = visit_counts / (np.sum(visit_counts) + 1e-8)
            return actions[np.random.choice(len(actions), p=visit_counts)]
        
        action, _ = self.ppo_model.predict(root_state, deterministic=(temperature == 0))
        return action