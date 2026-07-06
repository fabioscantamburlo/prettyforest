"""MNIST classification with RandomForest + PrettyForest visualization.

Trains a 50-tree RandomForest on MNIST digits, does 5-fold cross-validation,
then visualizes the model with test predictions and true labels.
"""

from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np
import polars as pl

from prettyforest import prettygrow


def main():
    print("Loading MNIST...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X, y = mnist.data, mnist.target.astype(int)

    # Use a subset for speed (full MNIST is 70k samples)
    X, _, y, _ = train_test_split(X, y, train_size=10000, random_state=42, stratify=y)
    print(
        f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes"
    )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # Cross-validation
    print("\n5-fold Cross-Validation...")
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=15,
        max_features=10,
        random_state=42,
        n_jobs=-1,
    )
    scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"  CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"  Per-fold: {[f'{s:.4f}' for s in scores]}")

    # Final model on full training set
    print("\nTraining final model (50 trees, depth=15, max_features=10)...")
    rf.fit(X_train, y_train)
    train_acc = rf.score(X_train, y_train)
    test_acc = rf.score(X_test, y_test)
    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Test accuracy:  {test_acc:.4f}")

    # Use consistent feature names — must match what the model sees
    feat_names = [f"feature_{i}" for i in range(784)]
    X_test_pl = pl.DataFrame({name: X_test[:, i] for i, name in enumerate(feat_names)})

    # Visualize with true labels
    print("\nGenerating forest visualization...")
    prettygrow(
        rf,
        data=X_test_pl,
        target=y_test,
        feature_names=feat_names,
        output_path="tmp/mnist_forest.html",
    )
    print("✓ Saved: tmp/mnist_forest.html")
    print("\nOpen it in your browser. Try:")
    print("  - Trace sample #0 to see how the forest classifies a digit")
    print("  - Compare 'Ensemble prediction' vs 'True' label")
    print("  - Double-click a tree to see its decision path")
    print("  - Sort by Leaf Purity to find the most/least confident trees")


if __name__ == "__main__":
    main()
