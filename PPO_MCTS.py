import numpy as np
import torch
import copy
from collections import defaultdict
import graphviz

class MCTSNode:
    """
    Node in the Monte Carlo Tree Search for counterfactual generation.
    """
    def __init__(self, state, parent=None, action=None, prior=0.0):
        # The state represents the modified features at this node
        self.state = state  # Environment state (encoded features)
        self.parent = parent  # Parent node
        self.action = action  # Action that led to this node
        self.children = {}  # Maps actions to child nodes
        
        # MCTS statistics
        self.visit_count = 0  # Number of times node was visited 
        self.value_sum = 0.0  # Sum of values from all visits
        self.prior = prior  # Prior probability from policy network
    
    @property
    def value(self):
        """Mean value of the node"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def is_expanded(self):
        """Check if node has been expanded"""
        return len(self.children) > 0
    
    def add_child(self, action, child_node):
        """Add a child node"""
        self.children[action] = child_node


class PPOMCTS:
    """
    Monte Carlo Tree Search implementation for CERTIFAI counterfactual generation.
    """
    def __init__(self, env, ppo_model, c_puct=1.0, discount_factor=0.99, num_simulations=10):
        """
        Initialize MCTS for counterfactual generation.
        
        Parameters:
        - env: PPOEnv instance
        - ppo_model: Trained PPO model (for policy and value estimates)
        - c_puct: Exploration constant for PUCT formula
        - discount_factor: Discount factor for future rewards (gamma)
        - num_simulations: Number of simulations per move
        """
        self.env = env
        self.ppo_model = ppo_model
        self.c_puct = c_puct
        self.discount_factor = discount_factor
        self.num_simulations = num_simulations
        
        # Track state values and Q-values
        self.q_values = defaultdict(float)  # Maps (state_hash, action) to Q-value
        
    def hash_state(self, state):
        """Create a hashable representation of the state"""
        if isinstance(state, np.ndarray):
            return tuple(state.flatten())
        return state
    
    def initialize_tree(self, root_state):
        """Initialize the search tree with the root state"""
        # Get policy prediction for the root
        root_policy, _ = self.ppo_model.predict(root_state, deterministic=False)
        
        # Create root node with priors from policy
        self.root = MCTSNode(state=root_state)
        
        # Evaluate the root node
        with torch.no_grad():
            value = self.ppo_model.policy.predict_values(
                torch.tensor(root_state, dtype=torch.float32).unsqueeze(0)
            ).item()
        
        self.root.value_sum = value
        self.root.visit_count = 1
        
        return self.root
    
    def _set_env_state(self, env, state):
        """
        Set the environment state based on the state representation from a node.
        
        Parameters:
        - env: Copy of the environment
        - state: State representation from MCTS node
        """
        # Extract features from state representation
        env.modified_features = state[:len(env.original_features)]
 
        # Ensure the prediction is updated
        env.current_prediction = env.generate_prediction(env.model, env.modified_features)


    def select(self, node):
        """
        Select a path through the tree until we reach a leaf node.
        Uses Upper Confidence Bound for Trees (UCT) to balance exploration/exploitation.
        
        Returns:
        - The selected leaf node
        """
        while node.is_expanded():
            # Find the action that maximizes UCB score
            best_action = None
            best_score = float('-inf')
            
            # Calculate UCB score for each action (child node)
            for action, child in node.children.items():
                # Get policy probability for this action 
                action_policy = self.get_policy_for_action(node.state, action)
                
                # Calculate UCB score using PUCT formula
                exploration_score = self.c_puct * action_policy * np.sqrt(node.visit_count) / (1 + child.visit_count)
                
                # Get Q-value for this state-action pair
                q_value = self.q_values[(self.hash_state(node.state), action)]
                
                # Total score is Q-value plus exploration bonus
                ucb_score = q_value + exploration_score
                
                if ucb_score > best_score:
                    best_score = ucb_score
                    best_action = action
            
            # Move to the selected child
            node = node.children[best_action]
            
        return node
    
    def get_policy_for_action(self, state, action):
        """Get the policy probability for a specific action"""
        # Get probabilities from the policy network
        action_probs, _ = self.ppo_model.predict(state, deterministic=False)

        with torch.no_grad():

            obs_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            # Get the action distribution
            action_distribution = self.ppo_model.policy.get_distribution(obs_tensor)

            # Get the probabilities
            action_probs = torch.exp(action_distribution.log_prob(torch.arange(action_distribution.action_dim))).numpy()

        # Convert action_probs to numpy array
        if isinstance(action_probs, torch.Tensor):
            action_probs = action_probs.numpy()
        elif isinstance(action_probs, list):
            action_probs = np.array(action_probs)
        
        # If action_probs is a scalar (single action), convert to list
        if np.isscalar(action_probs):
            return 1.0 if action_probs == action else 0.0
        
        # Return probability for the specific action
        if isinstance(action_probs, np.ndarray) and action < len(action_probs):
            return action_probs[action]
        return 0.01  # Small default probability for unknown actions

    def expand(self, leaf_node):
        """
        Expand the leaf node by adding all possible child nodes.
        
        In the counterfactual generation context, this means:
        1. Creating a virtual copy of the environment
        2. Getting possible actions from the PPO policy
        3. Adding child nodes for promising actions
        
        Returns:
        - The newly expanded leaf node
        """
        # Skip expansion if this node was already expanded
        if leaf_node.is_expanded():
            return leaf_node
            
        # Create a virtual copy of the environment state to simulate actions
        virtual_env = copy.deepcopy(self.env)
        
        # Set the environment state to match the leaf node's state
        self._set_env_state(virtual_env, leaf_node.state)

        with torch.no_grad():

            obs_tensor = torch.tensor(leaf_node.state, dtype=torch.float32).unsqueeze(0)

            # Get the action distribution
            action_distribution = self.ppo_model.policy.get_distribution(obs_tensor)

            # Get the probabilities
            action_probs = torch.exp(action_distribution.log_prob(torch.arange(action_distribution.action_dim))).numpy()

            # Sample an action
            action = action_distribution.sample().squeeze().numpy()


            #print("action_probabilities", action_probs)
            #print("action", action)

        # Convert action_probs to numpy array
        if isinstance(action_probs, torch.Tensor):
            action_probs = action_probs.numpy()
        elif isinstance(action_probs, list):
            action_probs = np.array(action_probs)

        # If action_probs is a scalar, convert to an array
        if np.isscalar(action_probs):
            action_probs = np.array([action_probs])
        
        # Choose top-k actions to expand (for efficiency)
        top_k = min(len(action_probs), 10)  # Expand at most 10 nodes
        top_action_indices = np.argsort(action_probs)[-top_k:]
        
        # Expand the node with each chosen action
        for action_idx in top_action_indices:
            # Skip actions with very low probability
            if action_probs[action_idx] < 0.005:
                continue
                
            # Execute the action in the virtual environment
            next_state, reward, done, info = virtual_env.step(action_idx)
            
            # Create a new child node
            child = MCTSNode(
                state=next_state,
                parent=leaf_node,
                action=action_idx,
                prior=action_probs[action_idx]
            )
            
            # Initialize Q-value for this state-action pair
            state_hash = self.hash_state(leaf_node.state)
            self.q_values[(state_hash, action_idx)] = reward
            
            # Add the child node to the tree
            leaf_node.add_child(action_idx, child)
            
        return leaf_node
    
    def evaluate(self, node):
        """
        Evaluate the value of a node using the PPO value model.
        
        As described in the paper, we don't do Monte-Carlo rollout for efficiency,
        but use the value model to evaluate the node directly.
        
        Parameters:
        - node: The node to evaluate
        
        Returns:
        - value: The estimated value of the node
        """
        # If this is a terminal node (counterfactual found or max steps reached)
        # we can use the direct reward
        virtual_env = copy.deepcopy(self.env)
        self._set_env_state(virtual_env, node.state)
        
        # Check if counterfactual was found
        original_prediction = virtual_env.original_prediction
        modified_prediction = virtual_env.generate_prediction(virtual_env.model, virtual_env.modified_features)
        counterfactual_found = (modified_prediction != original_prediction)
        
        # Check if max steps reached
        max_steps_reached = virtual_env.steps_taken >= virtual_env.max_steps
        
        # If terminal state, use the terminal reward
        if counterfactual_found or max_steps_reached:
            # Calculate the terminal reward
            distance = virtual_env.calculate_distance()
            reward = virtual_env.calculate_reward(
                distance=distance,
                counterfactual_found=counterfactual_found,
                modified_prediction=modified_prediction
            )
            value = reward
        else:
            # Use the value model to estimate the value of this state
            with torch.no_grad():
                # Use the proper SB3 PPO value prediction method
                value = self.ppo_model.policy.predict_values(
                    torch.tensor(node.state, dtype=torch.float32).unsqueeze(0)
                ).item()
        
        # Initialize the node's statistics
        node.visit_count = 1
        node.value_sum = value
        
        # Initialize Q-values for child edges with the value
        if node.parent is not None:
            state_hash = self.hash_state(node.parent.state)
            action = node.action
            # As per the paper, "initialize Q with V" to prevent exploration suppression 
            self.q_values[(state_hash, action)] = value
        
        return value
    
    def backup(self, node, value):
        """
        Update statistics for all nodes in the path from leaf to root.
        
        Following the paper:
        1. Update Q-values using rewards and discounted values
        2. Update value estimates for each node
        3. Increment visit counts
        
        Parameters:
        - node: The leaf node where evaluation occurred
        - value: The value to back up
        """
        # Track current node as we move up the tree
        current = node
        
        while current.parent is not None:
            # Get the parent node and the action that led to this node
            parent = current.parent
            action = current.action
            
            # Get the parent state hash for Q-value lookup
            parent_state_hash = self.hash_state(parent.state)
            
            # Calculate step-level reward (simplified KL term)
            # In the paper: r(s,a) = -β log(pθ(a|s)/pθ0(a|s))
            # We approximate this using the prior from the policy
            virtual_env = copy.deepcopy(self.env)
            self._set_env_state(virtual_env, parent.state)
            _, reward, _, _ = virtual_env.step(action)
            
            # Update Q-value using the formula from the paper: Q(s,a) = r(s,a) + γ·V(s')
            self.q_values[(parent_state_hash, action)] = reward + self.discount_factor * value
            
            # Update parent node statistics
            parent.visit_count += 1
            
            # Update parent value: V(s) = ∑a N(s')Q(s,a) / ∑a N(s')
            # Calculate the new mean value for the parent based on all children Q-values
            total_visits = 0
            value_sum = 0.0
            
            for child_action, child_node in parent.children.items():
                child_visits = child_node.visit_count
                total_visits += child_visits
                value_sum += child_visits * self.q_values[(parent_state_hash, child_action)]
            
            if total_visits > 0:
                parent.value_sum = value_sum
                mean_value = value_sum / total_visits
            else:
                mean_value = parent.value
            
            # Update value for next iteration (moving up the tree)
            value = mean_value
            
            # Move to parent for next iteration
            current = parent
    
    def _is_terminal_state(self, state):
        """
        Check if a state is terminal (counterfactual found).
        
        Returns:
        - Boolean indicating if this is a terminal state
        """
        virtual_env = copy.deepcopy(self.env)
        self._set_env_state(virtual_env, state)
        
        # Check if we found a counterfactual
        original_prediction = virtual_env.original_prediction
        modified_prediction = virtual_env.generate_prediction(
            virtual_env.model, 
            virtual_env.modified_features
        )
        
        # Terminal if counterfactual found or max steps reached
        return (modified_prediction != original_prediction) or (virtual_env.steps_taken >= virtual_env.max_steps)
    
    def run_mcts(self, root_state, temperature=1.0):
        """
        Run the full MCTS process to find the best action.
        
        Parameters:
        - root_state: The initial state
        - temperature: Controls the exploration in the final policy
        
        Returns:
        - The best action based on visit counts
        """
        # Initialize the tree
        self.initialize_tree(root_state)
        
        # Run simulations
        for _ in range(self.num_simulations):
            # 1. Selection phase: navigate tree until we reach a leaf node
            leaf = self.select(self.root)
            
            # 2. Expansion phase: add child nodes to the leaf
            if not self._is_terminal_state(leaf.state):
                leaf = self.expand(leaf)
                
            # 3. Evaluation phase: estimate value of leaf node
            value = self.evaluate(leaf)
            
            # 4. Backup phase: update statistics up the tree
            self.backup(leaf, value)
        
        # Return action based on visit counts according to the temperature
        # (as described in the paper)
        visit_counts = np.array([
            child.visit_count for action, child in self.root.children.items()
        ])
        actions = list(self.root.children.keys())
        
        # Apply temperature and normalize
        if len(visit_counts) > 0:
            # Apply temperature: p(a|s) ∝ N(s,a)^(1/τ)
            if temperature == 0:  # Deterministic selection
                best_action = actions[np.argmax(visit_counts)]
                return best_action
            else:
                # Apply temperature and normalize
                visit_counts = visit_counts ** (1 / temperature)
                visit_counts = visit_counts / np.sum(visit_counts)
                
                # Sample from the distribution
                selected_idx = np.random.choice(len(actions), p=visit_counts)
                return actions[selected_idx]
        
        # Fallback to PPO policy if no visits
        action, _ = self.ppo_model.predict(root_state, deterministic=(temperature == 0))
        #self.draw_tree()
        return action