import sys
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from acaClassifier import ACAClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

import torch
import torch.nn as nn
import torch.optim as optim

# Load and preprocess the data
def load_encoded_data(path):
    features = []
    pt_labels = []
    num_pt_labels = []

    with open(path, 'r') as file:
        for line in file:
            binary_str, pt, num_pt = line.strip().split(',')
            binary_vec = [int(bit) for bit in binary_str.strip()]
            features.append(binary_vec)
            pt_labels.append(int(pt))
            num_pt_labels.append(int(num_pt))

    X = np.array(features)
    y_pt = np.array(pt_labels)
    y_num = np.array(num_pt_labels)
    input_size = X.shape[1]

    return X, y_pt, y_num, input_size

# FNN Model
class FNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(FNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.model(x)

CANDIDATE_ACA_RULES = [
    4, 5, 12, 13, 36, 44, 68, 69, 72, 77, 78, 79, 92, 93, 94, 95, 100, 104, 128, 130,
    132, 133, 136, 138, 141, 144, 146, 152, 160, 162, 164, 168, 170, 172, 174, 176,
    178, 182, 184, 186, 188, 190, 192, 194, 197, 202, 203, 207, 208, 216, 217, 218,
    219, 221, 222, 223, 224, 226, 228, 230, 232, 233, 234, 237, 238, 240, 242, 244,
    246, 248, 250, 252, 254
]

# Train FNN
def train_fnn(X_train, y_train, X_test, y_test, input_size, num_classes=2, epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FNN(input_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_train = model(X_train_tensor).argmax(dim=1).cpu().numpy()
        pred_test = model(X_test_tensor).argmax(dim=1).cpu().numpy()

    return accuracy_score(y_train, pred_train), accuracy_score(y_test, pred_test)

def evaluate_aca_classifier(X, y, rule_num, label_name="pt"):
    n_cells = X.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifier = ACAClassifier(rule_num, n_cells)
    classifier.fit(X_train, y_train)

    train_preds = classifier.predict(X_train)
    test_preds = classifier.predict(X_test)

    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)

    print(f"[Rule {rule_num}] ACA Classifier ({label_name}): Train Acc = {train_acc:.3f}, Test Acc = {test_acc:.3f}")
    return train_acc, test_acc

# Run experiments
def evaluate_models(X, y, task_name="pt", input_size=32):
    print(f"\n=== Evaluating models for {task_name.upper()} ===")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "SVM (RBF Kernel)": SVC(kernel="rbf"),
        "Naive Bayes (Bernoulli)": BernoulliNB(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Extra Trees": ExtraTreesClassifier()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        acc_train = accuracy_score(y_train, model.predict(X_train))
        acc_test = accuracy_score(y_test, model.predict(X_test))
        print(f"{name}: Train Acc = {acc_train:.3f}, Test Acc = {acc_test:.3f}")

    # Neural model
    num_classes = len(set(y))
    fnn_train_acc, fnn_test_acc = train_fnn(X_train, y_train, X_test, y_test, input_size, num_classes)
    print(f"Feedforward NN: Train Acc = {fnn_train_acc:.3f}, Test Acc = {fnn_test_acc:.3f}")

    # ACA models
    print(f"\n--- ACA Classifier Accuracy ({task_name.upper()}) ---")
    for rule in CANDIDATE_ACA_RULES:
        evaluate_aca_classifier(X, y, rule_num=rule, label_name=task_name)

# Main
if __name__ == "__main__":

    args = sys.argv[1:]

    if len(args) != 1:
        print(f"Invalid Usage. Correct Usage `{sys.argv[0]} encoded_input_file.csv`")
        exit()

    X, y_pt, y_num, input_size = load_encoded_data(args[0])

    evaluate_models(X, y_pt, "pt", input_size)
    evaluate_models(X, y_num, "num_pt", input_size)

