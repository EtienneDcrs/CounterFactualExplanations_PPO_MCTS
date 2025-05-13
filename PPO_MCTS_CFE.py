import numpy as np
import torch
import gymnasium as gym
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict
from tqdm import tqdm
import time

class CounterfactualMCTS:
    """
    Monte Carlo Tree Search for counterfactual explanations, adapted from PPO-MCTS
    """
    
    def __init__(
        self,
        policy,  # SB3 Policy object
        value_fn=None,  # Optional custom value function, will use policy's value function if None
        classifier=None,  # Classifier model we're explaining
        original_instance=None,  # Original instance we're finding counterfactuals for
        target_class=None,  # Target class for counterfactual
        feature_ranges=None,  # Min/max values for each feature
        action_mapper=None,  # Function to map abstract actions to feature modifications
        
        # MCTS parameters
        max_depth=20,  # Maximum tree depth
        sim_count=10,  # Number of simulations per action
        k=10,  # Number of actions to consider from policy distribution
        c_puct=1.0,  # Exploration constant
        gamma=0.99,  # Discount factor
        
        # Additional parameters
        use_dirichlet_noise=False,  # Add dirichlet noise to root prior
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        
        # Objective function parameters
        distance_weight=1.0,  # Weight for distance penalty
        sparsity_weight=0.1,  # Weight for sparsity reward
    ):
        
        self.policy = policy
        self.value_fn = value_fn if value_fn is not None else policy
        self.classifier = classifier
        self.original_instance = original_instance
        self.target_class = target_class
        self.feature_ranges = feature_ranges
        self.action_mapper = action_mapper
        
        # MCTS parameters
        self.max_depth = max_depth
        self.sim_count = sim_count
        self.k = k
        self.c_puct = c_puct
        self.gamma = gamma
        
        # Additional parameters
        self.use_dirichlet_noise = use_dirichlet_noise
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        
        # Objective function parameters
        self.distance_weight = distance_weight
        self.sparsity_weight = sparsity_weight
        
        # Initialize tree structures
        self._tree = None
        self.reset_tree()
        
    def reset_tree(self):
        """Initialize or reset the search tree"""
        self._tree = {
            "visit_counts": defaultdict(int),  # N(s,a): visit count for each state-action pair
            "total_values": defaultdict(float),  # W(s,a): total value for each state-action pair
            "mean_values": defaultdict(float),  # Q(s,a): mean value for each state-action pair
            "priors": defaultdict(lambda: np.zeros(self.k)),  # P(s,a): prior probability for each action
            "children": defaultdict(list),  # Maps state to list of child states
            "actions": defaultdict(list),  # Maps state to action that led to it
            "states": {},  # Maps state_id to actual state representation
            "is_terminal": defaultdict(bool),  # Whether a state is terminal
            "depth": defaultdict(int),  # Depth of each state in the tree
            "action_mapping": defaultdict(list),  # Maps abstract actions to actual feature modifications
        }
        
    def search(self, state):
        """
        Perform MCTS search from the given state and return action probabilities
        """
        # Convert state to hashable representation
        state_id = self._get_state_id(state)
        
        # Initialize root state if not already in tree
        if state_id not in self._tree["states"]:
            self._tree["states"][state_id] = state.copy()
            self._tree["depth"][state_id] = 0
            self._evaluate_state(state_id)
        
        # Perform simulations
        for _ in range(self.sim_count):
            self._simulate(state_id)
            
        # Return action probabilities based on visit counts
        visit_counts = np.zeros(self.k)
        for i in range(self.k):
            if (state_id, i) in self._tree["visit_counts"]:
                visit_counts[i] = self._tree["visit_counts"][(state_id, i)]
                
        # Apply temperature to visit counts
        visit_counts = visit_counts ** (1.0 / 1.0)  # Temperature parameter could be added
        
        # Normalize to get probabilities
        if visit_counts.sum() > 0:
            probs = visit_counts / visit_counts.sum()
        else:
            probs = np.ones(self.k) / self.k
            
        return probs
        
    def _simulate(self, state_id):
        """
        Perform a single MCTS simulation starting from state_id
        """
        path = []  # To track visited states for backpropagation
        current_id = state_id
        
        # Selection phase - select actions according to UCT until we reach a leaf node
        while (self._has_children(current_id) and 
               not self._tree["is_terminal"][current_id] and
               self._tree["depth"][current_id] < self.max_depth):
               
            # Select action according to UCT formula
            action = self._select_action(current_id)
            path.append((current_id, action))
            
            # Get next state
            next_id = self._get_child(current_id, action)
            current_id = next_id
            
        # Expansion phase - if node is not terminal and not at max depth
        if not self._tree["is_terminal"][current_id] and self._tree["depth"][current_id] < self.max_depth:
            # Evaluate and expand the node
            self._evaluate_state(current_id)
            
            # If not terminal after evaluation, select a child and continue
            if not self._tree["is_terminal"][current_id]:
                action = self._select_action(current_id)
                path.append((current_id, action))
                
                # Create child state by applying the action
                child_state = self._apply_action(self._tree["states"][current_id], action)
                child_id = self._get_state_id(child_state)
                
                # Add child to tree if new
                if child_id not in self._tree["states"]:
                    self._tree["states"][child_id] = child_state
                    self._tree["depth"][child_id] = self._tree["depth"][current_id] + 1
                    self._tree["actions"][(current_id, action)] = child_id
                    self._tree["children"][current_id].append(child_id)
                    
                    # Check if child is terminal (valid counterfactual)
                    prediction = self.classifier.predict([child_state])[0]
                    self._tree["is_terminal"][child_id] = (prediction == self.target_class)
                    
                    # Evaluate the state
                    self._evaluate_state(child_id)
                
                current_id = child_id
        
        # Simulation - for our counterfactual problem, we don't need random rollouts 
        # as the value function estimates quality directly
        
        # Backpropagation - update values and visit counts along the path
        value = self._get_value(current_id)
        for state_id, action in reversed(path):
            key = (state_id, action)
            self._tree["visit_counts"][key] += 1
            self._tree["total_values"][key] += value
            self._tree["mean_values"][key] = self._tree["total_values"][key] / self._tree["visit_counts"][key]
            
            # Discount the value for the next state in the path
            value *= self.gamma
    
    def _select_action(self, state_id):
        """
        Select an action from state using UCT formula
        """
        best_score = -float('inf')
        best_action = 0
        
        # Get parent visit count for UCT normalization
        parent_visit_count = sum(self._tree["visit_counts"][(state_id, a)] 
                                for a in range(self.k)
                                if (state_id, a) in self._tree["visit_counts"]) + 1  # Add 1 to avoid division by zero
        
        for action in range(self.k):
            if (state_id, action) in self._tree["visit_counts"]:
                # UCT formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
                exploit = self._tree["mean_values"][(state_id, action)]
                explore = (self.c_puct * 
                           self._tree["priors"][state_id][action] * 
                           np.sqrt(parent_visit_count) / 
                           (1 + self._tree["visit_counts"][(state_id, action)]))
                score = exploit + explore
            else:
                # For unvisited actions, prioritize by prior probability
                score = self.c_puct * self._tree["priors"][state_id][action]
                
            if score > best_score:
                best_score = score
                best_action = action
                
        return best_action
    
    def _evaluate_state(self, state_id):
        """
        Evaluate a state to get prior probabilities and value
        """
        state = self._tree["states"][state_id]
        
        # Get policy priors and value estimate from policy network
        with torch.no_grad():
            # Convert state to torch tensor for policy
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Get action distribution from policy
            action_dist = self.policy.get_distribution(state_tensor) 
            logits = action_dist.distribution.logits.squeeze().cpu().numpy()
            
            # Get top-k actions
            top_k_actions = np.argsort(logits)[-self.k:][::-1]
            
            # Compute softmax over top-k logits
            top_k_logits = logits[top_k_actions]
            top_k_probs = np.exp(top_k_logits - np.max(top_k_logits))
            top_k_probs = top_k_probs / np.sum(top_k_probs)
            
            # Store priors for top-k actions
            self._tree["priors"][state_id] = np.zeros(self.k)
            for i, action_idx in enumerate(top_k_actions):
                # Map policy actions to MCTS action space
                mcts_action = i  # In our simplified case, just use index
                self._tree["priors"][state_id][mcts_action] = top_k_probs[i]
                self._tree["action_mapping"][state_id].append(action_idx)
            
            # Add Dirichlet noise to root prior
            if self.use_dirichlet_noise and self._tree["depth"][state_id] == 0:
                noise = np.random.dirichlet([self.dirichlet_alpha] * self.k)
                self._tree["priors"][state_id] = (
                    (1 - self.dirichlet_epsilon) * self._tree["priors"][state_id] + 
                    self.dirichlet_epsilon * noise
                )
                
            # Get value estimate
            value = self.value_fn.predict_values(state_tensor).item()
        
        return value
    
    def _has_children(self, state_id):
        """Check if a state has any children"""
        return len(self._tree["children"][state_id]) > 0
    
    def _get_child(self, state_id, action):
        """Get child state resulting from taking action in state_id"""
        if (state_id, action) in self._tree["actions"]:
            return self._tree["actions"][(state_id, action)]
        else:
            # Apply action to state to get new state
            child_state = self._apply_action(self._tree["states"][state_id], action)
            child_id = self._get_state_id(child_state)
            
            # Add to tree
            self._tree["states"][child_id] = child_state
            self._tree["depth"][child_id] = self._tree["depth"][state_id] + 1
            self._tree["actions"][(state_id, action)] = child_id
            self._tree["children"][state_id].append(child_id)
            
            return child_id
    
    def _apply_action(self, state, action):
        """Apply a MCTS action to a state to get a new state"""
        new_state = state.copy()
        
        # Get actual action index from mapping
        if len(self._tree["action_mapping"][self._get_state_id(state)]) > 0:
            policy_action = self._tree["action_mapping"][self._get_state_id(state)][action]
        else:
            # Fallback if mapping not available
            policy_action = action
            
        # Use custom action mapper if provided
        if self.action_mapper is not None:
            new_state = self.action_mapper(new_state, policy_action)
        else:
            # Default implementation - modify one feature at a time
            feature_idx = policy_action // 2
            direction = 1 if policy_action % 2 == 0 else -1
            step_size = (self.feature_ranges[feature_idx][1] - self.feature_ranges[feature_idx][0]) / 20.0
            
            new_state[feature_idx] += direction * step_size
            
            # Apply bounds
            new_state[feature_idx] = max(self.feature_ranges[feature_idx][0], 
                                        min(self.feature_ranges[feature_idx][1], 
                                            new_state[feature_idx]))
        
        return new_state
    
    def _get_value(self, state_id):
        """Get value estimate for a state"""
        if self._tree["is_terminal"][state_id]:
            # If terminal (valid counterfactual), compute reward based on:
            # - Distance from original instance (minimize)
            # - Number of changed features (minimize for sparsity)
            state = self._tree["states"][state_id]
            
            # Compute distance from original instance
            distance = np.linalg.norm(state - self.original_instance)
            
            # Compute sparsity (count of changed features)
            changes = np.count_nonzero(state != self.original_instance)
            
            # Combine into reward (higher is better)
            reward = 10.0 - self.distance_weight * distance - self.sparsity_weight * changes
            return reward
        else:
            # For non-terminal states, use value function estimate
            state = self._tree["states"][state_id]
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                return self.value_fn.predict_values(state_tensor).item()
    
    def _get_state_id(self, state):
        """Convert state to hashable representation"""
        return tuple(np.round(state, 6))  # Round to avoid floating point issues
        
    def get_counterfactual(self, state, deterministic=False):
        """
        Find a counterfactual explanation for the given state
        Returns: action, action_probs
        """
        # Run MCTS search
        action_probs = self.search(state)
        
        # Select action
        if deterministic:
            action = np.argmax(action_probs)
        else:
            # Sample from distribution
            action = np.random.choice(len(action_probs), p=action_probs)
            
        return action, action_probs
        
    def get_best_counterfactual(self, max_steps=100):
        """
        Run the search for multiple steps to find the best counterfactual
        """
        current_state = self.original_instance.copy()
        best_counterfactual = None
        best_reward = -float('inf')
        
        for step in range(max_steps):
            # Run MCTS search from current state
            action, _ = self.get_counterfactual(current_state, deterministic=(step > max_steps // 2))
            
            # Apply action
            current_state = self._apply_action(current_state, action)
            
            # Check if valid counterfactual
            prediction = self.classifier.predict([current_state])[0]
            if prediction == self.target_class:
                # Calculate reward for this counterfactual
                distance = np.linalg.norm(current_state - self.original_instance)
                changes = np.count_nonzero(current_state != self.original_instance)
                reward = 10.0 - self.distance_weight * distance - self.sparsity_weight * changes
                
                # Track best counterfactual
                if best_counterfactual is None or reward > best_reward:
                    best_counterfactual = current_state.copy()
                    best_reward = reward
        
        return best_counterfactual, best_reward


class MCTSPolicyWrapper:
    """
    Wrapper class to integrate MCTS with Stable Baselines PPO policy
    """
    
    def __init__(
        self,
        policy,
        classifier,
        feature_ranges,
        mcts_sims=10,
        c_puct=1.0,
        k=10,
        use_mcts=True  # Flag to enable/disable MCTS (for comparison)
    ):
        self.policy = policy
        self.mcts = CounterfactualMCTS(
            policy=policy,
            classifier=classifier,
            feature_ranges=feature_ranges,
            sim_count=mcts_sims,
            c_puct=c_puct,
            k=k
        )
        self.use_mcts = use_mcts
        
    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        """Compatible with SB3 policy interface"""
        if not self.use_mcts:
            # Use regular policy
            return self.policy.predict(observation, state, episode_start, deterministic)
        
        # Set original instance for counterfactual search
        if self.mcts.original_instance is None:
            self.mcts.original_instance = observation.copy()
            
        # Run MCTS
        action, action_probs = self.mcts.get_counterfactual(observation, deterministic)
        
        return action, None  # Return format compatible with SB3
        
    def set_target_class(self, target_class):
        """Set target class for counterfactual search"""
        self.mcts.target_class = target_class
        
    def reset_tree(self):
        """Reset the MCTS search tree"""
        self.mcts.reset_tree()
        self.mcts.original_instance = None


# Example usage with SB3 PPO
def integrate_mcts_with_ppo(env, ppo_model, classifier, feature_ranges, target_class):
    """
    Integrate MCTS with a trained PPO model for counterfactual generation
    """
    # Create MCTS wrapper around PPO policy
    mcts_policy = MCTSPolicyWrapper(
        policy=ppo_model.policy,
        classifier=classifier,
        feature_ranges=feature_ranges,
        mcts_sims=20,
        c_puct=2.0,
        k=10
    )
    
    # Set target class
    mcts_policy.set_target_class(target_class)
    
    # Replace PPO's policy with MCTS-enhanced policy
    # Note: This is just for inference, not for training
    original_policy = ppo_model.policy
    ppo_model.policy = mcts_policy
    
    # The PPO model can now be used as normal, but with MCTS-enhanced action selection
    return ppo_model, original_policy  # Return original policy so we can restore it later