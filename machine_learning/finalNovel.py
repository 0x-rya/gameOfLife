import sys
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from classify_ml import load_encoded_data

def apply_eca_rule(state, update_mask, rule):
    """Apply ECA rule to selected cells using the update_mask."""
    rule_bin = torch.tensor([int(b) for b in f"{rule:08b}"], device=state.device).float()
    padded = torch.cat([state[:, -1:], state, state[:, :1]], dim=1)  # circular padding

    new_state = state.clone()
    for i in range(state.shape[1]):
        left = padded[:, i]
        center = padded[:, i+1]
        right = padded[:, i+2]

        neighborhood = (left * 4 + center * 2 + right).long()
        updated_bits = rule_bin[7 - neighborhood]  # use tensor indexing

        # use update_mask to blend original vs updated bits
        new_state[:, i] = update_mask[:, i] * updated_bits + (1 - update_mask[:, i]) * state[:, i]

    return new_state

class ACALayer(nn.Module):
    def __init__(self, size):
        super(ACALayer, self).__init__()
        self.linear = nn.Linear(size, size)

    def forward(self, x):
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x

class ACAClassifier(nn.Module):
    def __init__(self, size, depth, rule):
        super(ACAClassifier, self).__init__()
        self.layers = nn.ModuleList([ACALayer(size) for _ in range(depth)])
        self.rule = rule

    def forward(self, x):
        state = x
        for layer in self.layers:
            update_mask = layer(state)
            state = apply_eca_rule(state, update_mask, self.rule)
        return state

def make_target_pattern(y, size):
    # For pt == 0 → [1]*half + [0]*half, else the reverse
    half = size // 2
    targets = torch.zeros((len(y), size))
    for i, label in enumerate(y):
        if label == 0:
            targets[i, :half] = 1
        else:
            targets[i, half:] = 1
    return targets

def train_aca_model(X_train, y_train, size, depth=25, epochs=20, lr=0.01, X_test=None, y_test=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ACAClassifier(size, depth, rule=110).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    targets_train = make_target_pattern(y_train_tensor, size).to(device)

    if X_test is not None and y_test is not None:
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
        targets_test = make_target_pattern(y_test_tensor, size).to(device)
    else:
        X_test_tensor = targets_test = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output_train = model(X_train_tensor)
        loss_train = loss_fn(output_train, targets_train)
        loss_train.backward()
        optimizer.step()

        # Evaluate test loss
        loss_test = None
        if X_test_tensor is not None:
            model.eval()
            with torch.no_grad():
                output_test = model(X_test_tensor)
                loss_test = loss_fn(output_test, targets_test)

        log = f"Epoch {epoch + 1}/{epochs}, Train Loss: {loss_train.item():.4f}"
        if loss_test is not None:
            log += f", Test Loss: {loss_test.item():.4f}"
        print(log)

    return model

def predict_pt_from_output(outputs):
    preds = []
    for out in outputs:
        out = (out > 0.5).float()
        half = len(out) // 2
        ones_left = out[:half].sum()
        ones_right = out[half:].sum()
        pred = 0 if ones_left > ones_right else 1
        preds.append(pred)
    return preds

def evaluate_aca_model(model, X, y, num_samples_to_print=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)
    
    # Create target patterns in the same way as during training
    size = X.shape[1]
    targets = make_target_pattern(y_tensor, size).to(device)
    
    with torch.no_grad():
        # Print input for a few samples
        print("\n===== DETAILED EVALUATION DEBUGGING =====")
        for sample_idx in range(min(num_samples_to_print, len(X))):
            print(f"\nSAMPLE {sample_idx} (True class: {y[sample_idx]}):")
            print(f"Input: {X_tensor[sample_idx].cpu().numpy()}")
            
            # Track intermediate states through the model's layers
            current_state = X_tensor[sample_idx:sample_idx+1]  # Add batch dimension
            
            print("\nIntermediate states through layers:")
            for i, layer in enumerate(model.layers):
                update_mask = layer(current_state)
                binary_mask = (update_mask > 0.5).float()
                
                print(f"Layer {i} update mask (raw): {update_mask.squeeze().cpu().numpy()}")
                print(f"Layer {i} binary mask: {binary_mask.squeeze().cpu().numpy()}")
                
                # Apply ECA rule
                new_state = apply_eca_rule(current_state, update_mask, model.rule)
                print(f"Layer {i} output state: {new_state.squeeze().cpu().numpy()}")
                
                current_state = new_state
            
            print(f"Final output: {current_state.squeeze().cpu().numpy()}")
            print(f"Target pattern: {targets[sample_idx].cpu().numpy()}")
            
            # Calculate match percentage for this sample
            binary_output = (current_state.squeeze() > 0.5).float()
            binary_target = (targets[sample_idx] > 0.5).float()
            match_percent = (binary_output == binary_target).float().mean().item()
            print(f"Binary match percentage: {match_percent:.4f}")
            
            # Determine predicted class
            half = len(binary_output) // 2
            ones_left = binary_output[:half].sum().item()
            ones_right = binary_output[half:].sum().item()
            pred_class = 0 if ones_left > ones_right else 1
            print(f"Predicted class: {pred_class} (ones_left: {ones_left}, ones_right: {ones_right})")
            print("-" * 50)
        
        # Run full evaluation on all samples
        output = model(X_tensor)
        
        # Calculate binary accuracy
        binary_output = (output > 0.5).float()
        binary_targets = (targets > 0.5).float()
        match_percentage = (binary_output == binary_targets).float().mean().item()
        
        # Calculate loss
        loss_fn = nn.BCELoss()
        test_loss = loss_fn(output, targets).item()
        
        print("\n===== OVERALL EVALUATION RESULTS =====")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Binary Match Percentage: {match_percentage:.4f}")
        
        # Class accuracy
        thresholded = binary_output.cpu().numpy()
        y_np = y_tensor.cpu().numpy()
        
        correct = 0
        total = len(y_np)
        
        for i, (pred_pattern, true_label) in enumerate(zip(thresholded, y_np)):
            half = len(pred_pattern) // 2
            ones_left = pred_pattern[:half].sum()
            ones_right = pred_pattern[half:].sum()
            pred_class = 0 if ones_left > ones_right else 1
            
            if pred_class == true_label:
                correct += 1
        
        acc = correct / total
        print(f"Classification Accuracy: {acc:.4f}")
        
    return acc, test_loss, match_percentage

if __name__ == "__main__":

    args = sys.argv[1:]

    if len(args) != 1:
        print(f"Invalid Usage. Correct Usage `{sys.argv[0]} encoded_input_file.csv`")
        exit()

    X, y_pt, y_num, input_size = load_encoded_data(args[0])

    X_train, X_test, y_train, y_test = train_test_split(X, y_pt, test_size=0.2, random_state=42)
    input_size = X.shape[1]

    model = train_aca_model(X_train, y_train, size=input_size, depth=10, epochs=50, lr=0.001, X_test=X_test, y_test=y_test)
    evaluate_aca_model(model, X_test, y_test)
