import sys
import random
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

class ACA:
    def __init__(self, rule_num, n_cells, max_steps=1000):
        self.rule_num = rule_num
        self.n_cells = n_cells
        self.max_steps = max_steps
        self.rule_table = self._get_rule_table(rule_num)

    def _get_rule_table(self, rule):
        """Returns rule table: dict from 3-bit tuples to 0/1 output."""
        binary = f"{rule:08b}"[::-1]  # reverse to match Wolfram convention
        table = {}
        for i in range(8):
            key = tuple(int(b) for b in f"{7 - i:03b}")
            table[key] = int(binary[i])
        return table

    def _get_neighborhood(self, state, idx):
        """Returns (left, center, right) neighborhood with wraparound."""
        n = self.n_cells
        return (state[(idx - 1) % n], state[idx], state[(idx + 1) % n])

class NeuralACA(ACA):
    def __init__(self, rule_num, n_cells, max_steps=1000, learning_rate=0.01):
        super().__init__(rule_num, n_cells, max_steps)
        # Initialize weights and biases for the neural network
        self.weights = np.random.randn(n_cells) * 0.01  # Small initial weights
        self.biases = np.random.randn(n_cells) * 0.01   # Small initial biases
        self.learning_rate = learning_rate
        self.threshold = 0.5  # Threshold for binary activation
        
        # Pre-compute all possible neighborhoods for faster lookup
        self.neighborhood_lookup = {}
        for i in range(n_cells):
            self.neighborhood_lookup[i] = [(i-1) % n_cells, i, (i+1) % n_cells]

    def sigmoid(self, x):
        """Sigmoid activation function - vectorized"""
        return 1 / (1 + np.exp(-np.clip(x, -15, 15)))  # Clip to avoid overflow
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function - vectorized"""
        sig = self.sigmoid(x)
        return sig * (1 - sig)
    
    def compute_mask(self, state):
        """Compute mask based on neural network - vectorized"""
        # Forward pass - vectorized operations
        z = np.dot(state, self.weights) + self.biases
        activation = self.sigmoid(z)
        # Convert to binary mask based on threshold
        mask = (activation > self.threshold).astype(int)
        return mask, z, activation
    
    def apply_rule_vectorized(self, state, mask):
        """Apply CA rules to all cells where mask is 1 in a vectorized manner"""
        new_state = state.copy()
        
        # Get indices where mask is 1
        idx_to_update = np.where(mask == 1)[0]
        
        # Create arrays for left, center, right values for all indices at once
        left_indices = (idx_to_update - 1) % self.n_cells
        right_indices = (idx_to_update + 1) % self.n_cells
        
        left_values = state[left_indices]
        center_values = state[idx_to_update]
        right_values = state[right_indices]
        
        # Apply rules using vectorized operations
        for i, idx in enumerate(idx_to_update):
            neighborhood = (left_values[i], center_values[i], right_values[i])
            new_state[idx] = self.rule_table[neighborhood]
            
        return new_state
    
    def evolve(self, initial_state=None, training=False, target_phase=0):
        """Optimized version that evolves the ACA using neural network mask"""
        if initial_state is None:
            state = np.random.randint(0, 2, size=self.n_cells)
        else:
            state = np.array(initial_state, dtype=np.int8)
            
            # Pad or truncate if necessary
            if len(state) < self.n_cells:
                state = np.pad(state, (0, self.n_cells - len(state)))
            elif len(state) > self.n_cells:
                state = state[:self.n_cells]
        
        # Pre-allocate history array for speed
        if training:
            # During training we only need final state
            history = None
        else:
            history = [state.copy()]
        
        for _ in range(self.max_steps):
            # Compute mask using neural network
            mask, z, activation = self.compute_mask(state)
            
            # Apply cellular automaton rules to masked cells
            state = self.apply_rule_vectorized(state, mask)
            
            if not training and history is not None:
                history.append(state.copy())
            
            # Perform backpropagation during training
            if training:
                self._backpropagate(state, z, target_phase)
        
        return state, history
    
    def _backpropagate(self, state, z, target_phase):
        """Performs vectorized backpropagation to update weights and biases"""
        # Create target pattern based on phase transition type
        half_point = self.n_cells // 2
        target_pattern = np.zeros(self.n_cells, dtype=np.int8)
        
        if target_phase == 0:
            target_pattern[:half_point] = 1
        else:
            target_pattern[half_point:] = 1
        
        # Compute vectorized gradients
        activation = self.sigmoid(z)
        error = target_pattern - activation
        d_activation = error * self.sigmoid_derivative(z)
        
        # Update weights and biases using gradient descent
        self.weights += self.learning_rate * np.dot(state, d_activation)
        self.biases += self.learning_rate * d_activation
    
    def load_dataset(self, filepath):
        """Load dataset from CSV file without headers"""
        # Load CSV without headers and assign column names
        df = pd.read_csv(filepath, header=None, names=['input', 'pt', 'num_pt'])
        
        # Convert all at once using numpy for better performance
        inputs = []
        pts = df['pt'].astype(int).values
        
        for input_str in df['input'].values:
            inputs.append(np.array([int(bit) for bit in input_str], dtype=np.int8))
        
        return list(zip(inputs, pts))
    
    def train_on_dataset(self, dataset, epochs=100, batch_size=32):
        """Train the neural network using the provided dataset with mini-batches"""
        dataset_size = len(dataset)
        losses = np.zeros(epochs)
        
        # Convert inputs to numpy arrays for better performance
        inputs = []
        targets = []
        for input_state, target in dataset:
            # Ensure correct length
            if len(input_state) < self.n_cells:
                input_state = np.pad(input_state, (0, self.n_cells - len(input_state)))
            elif len(input_state) > self.n_cells:
                input_state = input_state[:self.n_cells]
            inputs.append(input_state)
            targets.append(target)
        
        inputs = np.array(inputs, dtype=np.int8)
        targets = np.array(targets, dtype=np.int8)
        
        for epoch in range(epochs):
            # Shuffle dataset indices
            indices = np.random.permutation(dataset_size)
            epoch_loss = 0
            
            # Process in mini-batches for better performance
            for start_idx in range(0, dataset_size, batch_size):
                end_idx = min(start_idx + batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                batch_loss = 0
                for idx in batch_indices:
                    final_state, _ = self.evolve(inputs[idx], training=True, target_phase=targets[idx])
                    
                    # Calculate loss
                    half_point = self.n_cells // 2
                    if targets[idx] == 0:
                        expected = np.concatenate([np.ones(half_point), np.zeros(self.n_cells - half_point)])
                    else:
                        expected = np.concatenate([np.zeros(half_point), np.ones(self.n_cells - half_point)])
                    
                    batch_loss += np.mean((final_state - expected) ** 2)
                
                epoch_loss += batch_loss / len(batch_indices)
            
            losses[epoch] = epoch_loss / ((dataset_size + batch_size - 1) // batch_size)
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}, Loss: {losses[epoch]:.4f}")
        
        return losses
    
    def evaluate_dataset(self, dataset):
        """Efficiently evaluate the model on dataset"""
        dataset_size = len(dataset)
        confidence_scores = np.zeros(dataset_size)
        half_point = self.n_cells // 2
        
        for i, (initial_state, target_phase) in enumerate(dataset):
            # Ensure correct length
            if len(initial_state) < self.n_cells:
                initial_state = np.pad(initial_state, (0, self.n_cells - len(initial_state)))
            elif len(initial_state) > self.n_cells:
                initial_state = initial_state[:self.n_cells]
                
            final_state, _ = self.evolve(initial_state)
            
            # Calculate confidence score with vectorized operations
            if target_phase == 0:
                first_half_score = np.sum(final_state[:half_point]) / half_point
                second_half_score = (half_point - np.sum(final_state[half_point:])) / (self.n_cells - half_point)
            else:
                first_half_score = (half_point - np.sum(final_state[:half_point])) / half_point
                second_half_score = np.sum(final_state[half_point:]) / (self.n_cells - half_point)
            
            confidence_scores[i] = (first_half_score + second_half_score) / 2
        
        return np.mean(confidence_scores)

# Example usage with optimized performance:
if __name__ == "__main__":
    import time

    args = sys.argv[1:]
    if len(args) != 1:
        print("Usage: python neural_aca.py <dataset_path>")
        sys.exit(1)
    dataset_path = args[0]
    
    # Create a Neural ACA with appropriate cell count
    n_cells = 16  # For this example
    neural_aca = NeuralACA(rule_num=30, n_cells=n_cells, max_steps=50, learning_rate=0.1)
    
    # Load dataset
    start_time = time.time()
    dataset = neural_aca.load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} samples in {time.time() - start_time:.4f} seconds")
    
    # Train on dataset
    print("\nTraining on dataset...")
    start_time = time.time()
    losses = neural_aca.train_on_dataset(dataset, epochs=200, batch_size=32)
    print(f"Training completed in {time.time() - start_time:.4f} seconds")
    
    # Evaluate
    start_time = time.time()
    avg_confidence = neural_aca.evaluate_dataset(dataset)
    print(f"\nEvaluation completed in {time.time() - start_time:.4f} seconds")
    print(f"Average confidence score on dataset: {avg_confidence * 100:.2f}%")