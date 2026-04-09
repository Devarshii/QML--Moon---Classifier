import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import pennylane as qml
from pennylane import numpy as pnp


# --------------------------------------------------
# Create results folder
# --------------------------------------------------
os.makedirs("results", exist_ok=True)


# --------------------------------------------------
# Generate dataset
# --------------------------------------------------
X, y = make_moons(n_samples=300, noise=0.1, random_state=42)

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)


# --------------------------------------------------
# Plot and save dataset
# --------------------------------------------------
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("make_moons Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("results/dataset_plot.png", bbox_inches="tight")
plt.show()


# --------------------------------------------------
# Train-test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# --------------------------------------------------
# Feature scaling
# --------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# Classical baseline model
# --------------------------------------------------
clf = LogisticRegression()
clf.fit(X_train_scaled, y_train)

y_pred_classical = clf.predict(X_test_scaled)
classical_acc = accuracy_score(y_test, y_pred_classical)

print("\n===== Classical Model =====")
print("Accuracy:", classical_acc)
print(classification_report(y_test, y_pred_classical))


# --------------------------------------------------
# Quantum device setup
# --------------------------------------------------
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)


# --------------------------------------------------
# Quantum circuit
# --------------------------------------------------
@qml.qnode(dev)
def quantum_circuit(x, weights):
    # Encode input features
    qml.RX(x[0], wires=0)
    qml.RY(x[1], wires=1)

    # Variational layer
    qml.RY(weights[0], wires=0)
    qml.RY(weights[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(weights[2], wires=0)
    qml.RZ(weights[3], wires=1)

    return qml.expval(qml.PauliZ(0))


# --------------------------------------------------
# Prediction functions
# --------------------------------------------------
def predict_quantum(x, weights):
    return 1 if quantum_circuit(x, weights) >= 0 else 0


def predict_batch(X, weights):
    return np.array([predict_quantum(x, weights) for x in X])


# --------------------------------------------------
# Loss function
# --------------------------------------------------
def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2
    return loss / len(labels)


# Convert labels {0,1} to {-1,1}
y_train_q = pnp.array(np.where(y_train == 0, -1, 1), requires_grad=False)
y_test_q = pnp.array(np.where(y_test == 0, -1, 1), requires_grad=False)


def cost(weights, X, Y):
    predictions = [quantum_circuit(x, weights) for x in X]
    return square_loss(Y, predictions)


# --------------------------------------------------
# Initialize trainable weights
# --------------------------------------------------
np.random.seed(42)
weights = pnp.array(np.random.randn(4) * 0.1, requires_grad=True)


# --------------------------------------------------
# Train quantum model
# --------------------------------------------------
opt = qml.GradientDescentOptimizer(stepsize=0.1)
epochs = 50
loss_history = []

for epoch in range(epochs):
    weights = opt.step(lambda w: cost(w, X_train_scaled, y_train_q), weights)
    current_loss = cost(weights, X_train_scaled, y_train_q)
    loss_history.append(current_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}: Loss = {current_loss:.4f}")


# --------------------------------------------------
# Evaluate quantum model
# --------------------------------------------------
y_pred_quantum = predict_batch(X_test_scaled, weights)
quantum_acc = accuracy_score(y_test, y_pred_quantum)

print("\n===== Quantum Model =====")
print("Accuracy:", quantum_acc)
print(classification_report(y_test, y_pred_quantum))


# --------------------------------------------------
# Plot and save training loss
# --------------------------------------------------
plt.figure(figsize=(6, 4))
plt.plot(loss_history)
plt.title("Quantum Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("results/training_loss.png", bbox_inches="tight")
plt.show()


# --------------------------------------------------
# Plot and save model comparison
# --------------------------------------------------
models = ["Logistic Regression", "Quantum Classifier"]
scores = [classical_acc, quantum_acc]

plt.figure(figsize=(6, 4))
plt.bar(models, scores)
plt.ylim(0, 1)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.savefig("results/model_comparison.png", bbox_inches="tight")
plt.show()

import os
print(os.listdir("results"))