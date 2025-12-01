import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Loading dataset
data = load_breast_cancer()
X = data.data
y = data.target

print("Dataset shape:", X.shape)
print("Number of classes:", len(np.unique(y)))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train with different activation functions
accuracies = {}

for act in activations:
    model = MLPClassifier(hidden_layer_sizes=(32, 16),
                          activation=act,
                          max_iter=1000,
                          random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    accuracies[act] = acc

accuracies


# Visualizing activation functions
def plot_act_fun(name: str, x_vals, func, idx):
    plt.subplot(1, 3, idx)
    plt.plot(x_vals, func)
    plt.title(name)
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.grid(True)

# Input range
x_vals = np.linspace(-5, 5, 200)

# Activation outputs
sigmoid = 1 / (1 + np.exp(-x_vals))
tanh = np.tanh(x_vals)
relu = np.maximum(0, x_vals)

# Create subplots
plt.figure(figsize=(15, 4))

plot_act_fun("Sigmoid", x_vals, sigmoid, 1)
plot_act_fun("TanH", x_vals, tanh, 2)
plot_act_fun("ReLu", x_vals, relu, 3)

plt.tight_layout()
plt.show()


# Finding best activation function
best_act = max(accuracies, key=accuracies.get)

print("Best Activation Function:", best_act.upper())

best_model = MLPClassifier(hidden_layer_sizes=(32, 16),
                           activation=best_act,
                           max_iter=1000,
                           random_state=42)

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

