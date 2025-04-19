import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

class ACA:
    def __init__(self, rule_num, n_cells):
        self.rule_num = rule_num
        self.n_cells = n_cells
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


class PhaseDataset(Dataset):
    """Dataset for phase transition data"""
    def __init__(self, data, n_cells):
        self.data = data
        self.n_cells = n_cells
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        initial_state, target_phase = self.data[idx]
        
        # Pad or truncate to n_cells
        if len(initial_state) < self.n_cells:
            initial_state = np.pad(initial_state, (0, self.n_cells - len(initial_state)))
        elif len(initial_state) > self.n_cells:
            initial_state = initial_state[:self.n_cells]
        
        # Convert to tensors
        initial_state_tensor = torch.FloatTensor(initial_state)
        target_phase_tensor = torch.tensor(target_phase, dtype=torch.long)
        
        return initial_state_tensor, target_phase_tensor


class NeuralACAModel(nn.Module):
    """PyTorch model for Neural ACA"""
    def __init__(self, n_cells):
        super(NeuralACAModel, self).__init__()
        self.linear = nn.Linear(n_cells, n_cells)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Simple feedforward network that outputs mask probabilities
        return self.sigmoid(self.linear(x))


class NeuralACA(ACA):
    def __init__(self, rule_num, n_cells, max_steps=100, learning_rate=0.01, device=None):
        super().__init__(rule_num, n_cells)
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        
        # Set up device (GPU if available)
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Set up PyTorch model
        self.model = NeuralACAModel(n_cells).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.threshold = 0.5
        
        # Create tensors for rule lookup
        self.setup_differentiable_rule_table()
        
    def setup_differentiable_rule_table(self):
        """Set up tensors for differentiable rule application"""
        # Create rule tensor that maps neighborhood patterns to outputs
        rule_tensor = torch.zeros(2, 2, 2, device=self.device)
        
        # Fill the tensor based on the rule table
        for left in range(2):
            for center in range(2):
                for right in range(2):
                    neighborhood = (left, center, right)
                    rule_tensor[left, center, right] = self.rule_table[neighborhood]
                    
        self.rule_tensor = rule_tensor
    
    def load_dataset(self, filepath):
        """Load dataset from CSV file without headers"""
        # Load CSV without headers and assign column names
        df = pd.read_csv(filepath, header=None, names=['input', 'pt', 'num_pt'])
        data = []
        
        for _, row in df.iterrows():
            # Check if input is already a number and convert to binary representation
            if isinstance(row['input'], (int, np.int64, float, np.float64)):
                # Convert number to binary string and then to list of integers
                input_str = format(int(row['input']), f'0{self.n_cells}b')
                input_state = [int(bit) for bit in input_str]
            else:
                # Convert input string to list of integers
                input_state = [int(bit) for bit in row['input']]
            
            # Get phase transition label
            pt = int(row['pt'])
            
            data.append((input_state, pt))
            
        return data
    
    def apply_differentiable_rule(self, state, mask_probs):
        """Apply CA rules in a differentiable manner using soft masking"""
        batch_size = 1 if len(state.shape) == 1 else state.shape[0]
        
        # Ensure state is properly shaped
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            mask_probs = mask_probs.unsqueeze(0)
        
        result = torch.zeros_like(state)
        
        for b in range(batch_size):
            current_state = state[b]
            current_mask = mask_probs[b]
            
            for i in range(self.n_cells):
                # Get neighborhood
                left = current_state[(i - 1) % self.n_cells]
                center = current_state[i]
                right = current_state[(i + 1) % self.n_cells]
                
                # Compute neighborhood probabilities - each value is between 0 and 1
                # Interpolate between 0 and 1 for each position in the neighborhood
                p_left_0 = 1.0 - left
                p_left_1 = left
                p_center_0 = 1.0 - center
                p_center_1 = center
                p_right_0 = 1.0 - right
                p_right_1 = right
                
                # Calculate rule outputs for all 8 possible neighborhoods
                # Weighted by the probability of each neighborhood
                rule_output = 0.0
                rule_output += p_left_0 * p_center_0 * p_right_0 * self.rule_table[(0, 0, 0)]
                rule_output += p_left_0 * p_center_0 * p_right_1 * self.rule_table[(0, 0, 1)]
                rule_output += p_left_0 * p_center_1 * p_right_0 * self.rule_table[(0, 1, 0)]
                rule_output += p_left_0 * p_center_1 * p_right_1 * self.rule_table[(0, 1, 1)]
                rule_output += p_left_1 * p_center_0 * p_right_0 * self.rule_table[(1, 0, 0)]
                rule_output += p_left_1 * p_center_0 * p_right_1 * self.rule_table[(1, 0, 1)]
                rule_output += p_left_1 * p_center_1 * p_right_0 * self.rule_table[(1, 1, 0)]
                rule_output += p_left_1 * p_center_1 * p_right_1 * self.rule_table[(1, 1, 1)]
                
                # Soft update based on mask probability
                result[b, i] = current_mask[i] * rule_output + (1 - current_mask[i]) * current_state[i]
        
        # If input was 1D, return 1D
        if len(state.shape) == 1:
            result = result.squeeze(0)
            
        return result
    
    def compute_expected_state(self, target_phase):
        """Compute expected final state based on target phase"""
        half_point = self.n_cells // 2
        expected = torch.zeros(self.n_cells, device=self.device)
        
        if target_phase == 0:
            expected[:half_point] = 1.0
        else:
            expected[half_point:] = 1.0
            
        return expected
    
    def calculate_confidence(self, state, target_phase):
        """Calculate confidence score for a state"""
        state_np = state.cpu().numpy() if torch.is_tensor(state) else state
        half_point = self.n_cells // 2
        
        if target_phase == 0:
            first_half_score = np.sum(state_np[:half_point]) / half_point
            second_half_score = (half_point - np.sum(state_np[half_point:])) / (self.n_cells - half_point)
        else:
            first_half_score = (half_point - np.sum(state_np[:half_point])) / half_point
            second_half_score = np.sum(state_np[half_point:]) / (self.n_cells - half_point)
        
        return (first_half_score + second_half_score) / 2
    
    def evolve_step(self, state):
        """One step of evolution using the neural network mask"""
        # Get mask from neural network
        self.model.eval()
        with torch.no_grad():
            mask_probs = self.model(state)
            # For evaluation, use hard thresholding
            mask = (mask_probs > self.threshold).float()
            # Apply CA rules using traditional method for evaluation
            new_state = self.apply_rule(state, mask)
        return new_state
    
    def apply_rule(self, state, mask):
        """Apply CA rules to cells marked by mask (non-differentiable version)"""
        state_np = state.cpu().numpy() if torch.is_tensor(state) else state.copy()
        mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask.copy()
        
        # Find indices where mask is 1
        indices = np.where(mask_np > 0.5)[0]
        
        # Create new state array
        new_state = state_np.copy()
        
        # Apply rule to each marked cell
        for idx in indices:
            neighborhood = self._get_neighborhood(state_np, idx)
            new_state[idx] = self.rule_table[neighborhood]
            
        return torch.tensor(new_state, dtype=torch.float32, device=self.device) if torch.is_tensor(state) else new_state
    
    def evolve(self, initial_state=None, keep_history=False):
        """Run CA evolution for max_steps"""
        if initial_state is None:
            initial_state = torch.randint(0, 2, (self.n_cells,), device=self.device).float()
        elif not torch.is_tensor(initial_state):
            # Convert to tensor and ensure right shape
            if len(initial_state) < self.n_cells:
                initial_state = np.pad(initial_state, (0, self.n_cells - len(initial_state)))
            elif len(initial_state) > self.n_cells:
                initial_state = initial_state[:self.n_cells]
            initial_state = torch.tensor(initial_state, dtype=torch.float32, device=self.device)
        
        state = initial_state
        history = [state.clone().cpu().numpy()] if keep_history else None
        
        for _ in range(self.max_steps):
            state = self.evolve_step(state)
            
            if keep_history:
                history.append(state.clone().cpu().numpy())
        
        return state, history
    
    def train_on_dataset(self, dataset, epochs=100, batch_size=32):
        """Train using PyTorch DataLoader with differentiable operations"""
        # Create dataset and dataloader for efficient batching
        phase_dataset = PhaseDataset(dataset, self.n_cells)
        dataloader = DataLoader(phase_dataset, batch_size=batch_size, shuffle=True)
        
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            print(f"Batch:", end=" ", flush=True)
            for bn, (batch_states, batch_targets) in enumerate(dataloader):
                
                print(f"{bn + 1}/{len(dataloader)}", end=" ", flush=True)
                batch_states = batch_states.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                # Process entire batch at once
                self.model.train()
                self.optimizer.zero_grad()
                
                # Keep track of states for all examples in batch
                current_batch_states = batch_states
                
                # Run evolution for specified steps
                for _ in range(self.max_steps):
                    # Get mask probabilities from neural network
                    mask_probs = self.model(current_batch_states)
                    
                    # Apply differentiable rule using mask probabilities
                    current_batch_states = self.apply_differentiable_rule(current_batch_states, mask_probs)
                
                # Compute loss for each example against expected state
                batch_loss = 0.0
                for i in range(len(batch_states)):
                    expected = self.compute_expected_state(batch_targets[i].item())
                    loss = self.criterion(current_batch_states[i], expected)
                    batch_loss += loss
                
                # Average loss over batch
                batch_loss = batch_loss / len(batch_states)
                
                # Backward pass and optimization
                batch_loss.backward()
                self.optimizer.step()
                
                epoch_loss += batch_loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            
            # if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
                
        return losses
    
    # def evaluate_dataset(self, dataset):
    #     """Evaluate model on dataset"""
    #     self.model.eval()
    #     confidences = []
        
    #     with torch.no_grad():
    #         for initial_state, target_phase in dataset:
    #             # Convert to tensor and ensure right shape
    #             if len(initial_state) < self.n_cells:
    #                 initial_state = np.pad(initial_state, (0, self.n_cells - len(initial_state)))
    #             elif len(initial_state) > self.n_cells:
    #                 initial_state = initial_state[:self.n_cells]
                    
    #             initial_state_tensor = torch.tensor(initial_state, dtype=torch.float32, device=self.device)
                
    #             # Run evolution
    #             final_state, _ = self.evolve(initial_state_tensor)
                
    #             # Calculate confidence score
    #             confidence = self.calculate_confidence(final_state, target_phase)
    #             confidences.append(confidence)
                
    #     return np.mean(confidences)
    
    def save_model(self, path):
        """Save trained model"""
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path):
        """Load trained model"""
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def load_and_split_dataset(self, filepath, test_size=0.2, random_seed=42):
        """Load dataset from CSV file and split into training and test sets"""
        # Load CSV without headers and assign column names
        df = pd.read_csv(filepath, header=None, names=['input', 'pt', 'num_pt'])
        data = []
        
        for _, row in df.iterrows():
            # Check if input is already a number and convert to binary representation
            if isinstance(row['input'], (int, np.int64, float, np.float64)):
                # Convert number to binary string and then to list of integers
                input_str = format(int(row['input']), f'0{self.n_cells}b')
                input_state = [int(bit) for bit in input_str]
            else:
                # Original code path for string inputs
                input_state = [int(bit) for bit in row['input']]
            
            # Get phase transition label
            pt = int(row['pt'])
            
            data.append((input_state, pt))
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        # Shuffle and split the data
        random.shuffle(data)
        split_idx = int(len(data) * (1 - test_size))
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        
        print(f"Dataset split: {len(train_data)} training samples, {len(test_data)} test samples")
        
        return train_data, test_data

    def train_and_evaluate(self, train_data, test_data, epochs=100, batch_size=32):
        """Train the model and evaluate after each epoch"""
        # Create datasets and dataloaders
        train_dataset = PhaseDataset(train_data, self.n_cells)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Track metrics
        train_losses = []
        train_accuracies = []
        test_accuracies = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            epoch_loss = 0.0
            
            for batch_states, batch_targets in train_dataloader:
                batch_states = batch_states.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Keep track of states for all examples in batch
                current_batch_states = batch_states
                
                # Run evolution for specified steps
                for _ in range(self.max_steps):
                    # Get mask probabilities from neural network
                    mask_probs = self.model(current_batch_states)
                    
                    # Apply differentiable rule using mask probabilities
                    current_batch_states = self.apply_differentiable_rule(current_batch_states, mask_probs)
                
                # Compute loss for each example against expected state
                batch_loss = 0.0
                for i in range(len(batch_states)):
                    expected = self.compute_expected_state(batch_targets[i].item())
                    loss = self.criterion(current_batch_states[i], expected)
                    batch_loss += loss
                
                # Average loss over batch
                batch_loss = batch_loss / len(batch_states)
                
                # Backward pass and optimization
                batch_loss.backward()
                self.optimizer.step()
                
                epoch_loss += batch_loss.item()
            
            avg_loss = epoch_loss / len(train_dataloader)
            train_losses.append(avg_loss)
            
            # Evaluate on train and test sets after each epoch
            train_accuracy = self.evaluate_dataset(train_data)
            test_accuracy = self.evaluate_dataset(test_data)
            
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
            
            # Print progress
            # if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}")
        
        # Return collected metrics
        return {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies
        }

    def evaluate_dataset(self, dataset):
        """Evaluate model on dataset and return accuracy"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for initial_state, target_phase in dataset:
                # Convert to tensor and ensure right shape
                if len(initial_state) < self.n_cells:
                    initial_state = np.pad(initial_state, (0, self.n_cells - len(initial_state)))
                elif len(initial_state) > self.n_cells:
                    initial_state = initial_state[:self.n_cells]
                    
                initial_state_tensor = torch.tensor(initial_state, dtype=torch.float32, device=self.device)
                
                # Run evolution
                final_state, _ = self.evolve(initial_state_tensor)
                
                # Calculate confidence score
                confidence = self.calculate_confidence(final_state, target_phase)
                
                # Determine if prediction is correct (confidence > 0.5)
                if confidence > 0.5:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        return accuracy

# Modified main function
if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    args = sys.argv[1:]
    if len(args) != 1:
        print(f"Usage: python {sys.argv[0]} <dataset_path>")
        sys.exit(1)
    dataset_path = args[0]
    
    # Create a Neural ACA with appropriate cell count
    n_cells = 16  # For this example
    neural_aca = NeuralACA(rule_num=30, n_cells=n_cells, max_steps=10, learning_rate=0.01)
    
    # Load and split dataset
    start_time = time.time()
    train_data, test_data = neural_aca.load_and_split_dataset(dataset_path)
    print(f"Loaded and split dataset in {time.time() - start_time:.4f} seconds")
    
    # Train on dataset with evaluation
    print("\nTraining on dataset with evaluation after each epoch...")
    start_time = time.time()
    metrics = neural_aca.train_and_evaluate(train_data, test_data, epochs=15, batch_size=32)
    print(f"Training and evaluation completed in {time.time() - start_time:.4f} seconds")
    
    # Plot training metrics
    plt.figure(figsize=(12, 8))
    
    # Plot loss
    plt.subplot(2, 1, 1)
    plt.plot(metrics['train_losses'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Plot accuracies
    plt.subplot(2, 1, 2)
    plt.plot(metrics['train_accuracies'], label='Train Accuracy')
    plt.plot(metrics['test_accuracies'], label='Test Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()
    print("\nTraining metrics plot saved to 'training_metrics.png'")
    
    # Test with a specific example from test set
    print("\nTesting with first example from test set:")
    initial_state, target_phase = test_data[0]
    print(f"Initial state: {''.join(map(str, initial_state))}")
    print(f"Target phase: {target_phase}")
    
    # Run evolution
    start_time = time.time()
    final_state, history = neural_aca.evolve(initial_state, keep_history=True)
    print(f"Evolution completed in {time.time() - start_time:.4f} seconds")
    
    # Convert final state to string representation
    final_state_np = final_state.cpu().numpy()
    final_state_str = ''.join(['1' if cell > 0.5 else '0' for cell in final_state_np])
    print(f"Final state: {final_state_str}")
    
    # Calculate confidence score
    confidence = neural_aca.calculate_confidence(final_state, target_phase)
    print(f"Confidence score: {confidence:.4f}")
    
    # Save model
    neural_aca.save_model("neural_aca_model.pth")
    print("\nModel saved to neural_aca_model.pth")