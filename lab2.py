import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


#### Credit dataset
data = fetch_openml("credit-g", version=1, as_frame=True)
X = data.data
y = (data.target == "good").astype(int)

numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object", "category"]).columns

# Preprocessing pipeline
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("preprocess", preprocess),
    ("mlp", MLPClassifier(max_iter=1500, random_state=42))
])


# ----------------------- Grid Search --------------------
param_grid = {
    "mlp__hidden_layer_sizes": [(50,), (100,), (64, 32)],
    "mlp__activation": ["relu", "tanh"],
    "mlp__alpha": [0.0001, 0.001, 0.01],
    "mlp__learning_rate": ["constant", "adaptive"],
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best GridSearch Parameters:")
print(grid_search.best_params_)


y_pred = grid_search.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("GridSearch Test Accuracy:", acc)

#------------------------Random Search---------------------------------
param_dist = {
    "mlp__hidden_layer_sizes": [(50,), (100,), (128, 64), (64, 32, 16)],
    "mlp__activation": ["relu", "tanh", "logistic"],
    "mlp__alpha": np.logspace(-4, -1, 10),
    "mlp__learning_rate_init": np.logspace(-4, -2, 10),
    "mlp__solver": ["adam", "sgd"]
}

random_search = RandomizedSearchCV(
    pipeline,
    param_dist,
    n_iter=15,
    cv=3,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)

print("Best RandomSearch Parameters:")
print(random_search.best_params_)

y_pred = random_search.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("RandomSearch Test Accuracy:", acc)


# ---------------------------- MNIST Dataset ---------------------
mnist = fetch_openml("mnist_784", version=1)
X = mnist.data.astype("float32")
y = mnist.target.astype("int")

X /= 255.0  # scale pixels 0–1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

mnist_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(max_iter=200, random_state=42))
])


#-------------------------- Grid Search -----------------
param_grid_mnist = {
    "mlp__hidden_layer_sizes": [(64,), (128,), (256,)],
    "mlp__activation": ["relu", "tanh"],
    "mlp__alpha": [0.0001, 0.001]
}

grid_mnist = GridSearchCV(
    mnist_pipe,
    param_grid_mnist,
    cv=2,
    n_jobs=-1
)

grid_mnist.fit(X_train[:2000], y_train[:2000])

print("Best MNIST GridSearch Parameters:")
print(grid_mnist.best_params_)

y_pred = grid_mnist.predict(X_test[:2000])
acc = accuracy_score(y_test[:2000], y_pred)

print("MNIST GridSearch Accuracy:", acc)


#--------------------------- Random Search -----------------
param_dist_mnist = {
    "mlp__hidden_layer_sizes": [(128,), (256,), (128, 64)],
    "mlp__activation": ["relu", "tanh"],
    "mlp__alpha": np.logspace(-4, -2, 5),
    "mlp__learning_rate_init": np.logspace(-4, -2, 5),
}

rand_mnist = RandomizedSearchCV(
    mnist_pipe,
    param_dist_mnist,
    n_iter=8,
    cv=2,
    n_jobs=-1,
    random_state=42
)

rand_mnist.fit(X_train[:2000], y_train[:2000])

print("Best MNIST RandomSearch Parameters:")
print(rand_mnist.best_params_)


y_pred = rand_mnist.predict(X_test[:2000])
acc = accuracy_score(y_test[:2000], y_pred)

print("MNIST RandomSearch Accuracy:", acc)
