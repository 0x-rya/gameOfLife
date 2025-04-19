import random
import numpy as np
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

    def evolve(self, initial_state=None):
        """Runs the ACA asynchronously until convergence or timeout."""
        if initial_state is None:
            state = [random.randint(0, 1) for _ in range(self.n_cells)]
        else:
            state = list(initial_state)

        state = np.array(state)
        history = [state.copy()]
        no_change_counter = 0

        for _ in range(self.max_steps):
            idx = random.randint(0, self.n_cells - 1)
            neighborhood = self._get_neighborhood(state, idx)
            new_val = self.rule_table[neighborhood]

            if state[idx] == new_val:
                no_change_counter += 1
            else:
                no_change_counter = 0
                state[idx] = new_val

            history.append(state.copy())

            if no_change_counter > self.n_cells:
                break  # likely converged

        return state, history

class ACAClassifier:
    def __init__(self, rule_num, n_cells, max_steps=None):
        self.aca = ACA(rule_num, n_cells, max_steps or n_cells * 10)
        self.rule_num = rule_num
        self.n_cells = n_cells
        self.label_map = {}

    def _state_to_key(self, state):
        return ''.join(map(str, state))

    def fit(self, X, y):
        """
        Train ACA classifier: map final states to majority label.
        """
        state_to_labels = defaultdict(list)

        for x, label in zip(X, y):
            final_state, _ = self.aca.evolve(initial_state=x)
            key = self._state_to_key(final_state)
            state_to_labels[key].append(label)

        self.label_map = {
            key: Counter(labels).most_common(1)[0][0]
            for key, labels in state_to_labels.items()
        }

    def predict(self, X):
        """
        Predict label by evolving each pattern and finding its sink.
        """
        preds = []
        for x in X:
            final_state, _ = self.aca.evolve(initial_state=x)
            key = self._state_to_key(final_state)
            pred = self.label_map.get(key, 0)  # fallback to class 0
            preds.append(pred)
        return preds

