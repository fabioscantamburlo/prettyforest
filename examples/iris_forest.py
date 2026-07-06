"""Iris dataset with all supported model types — easy to inspect."""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
import polars as pl

from prettyforest import prettygrow


def main():
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    X_test_pl = pl.DataFrame(
        {name: X_test[:, i] for i, name in enumerate(feature_names)}
    )

    models = [
        ("RF", RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42)),
        (
            "GBM",
            GradientBoostingClassifier(n_estimators=10, max_depth=3, random_state=42),
        ),
        (
            "LGBM",
            lgb.LGBMClassifier(
                n_estimators=10, max_depth=3, verbose=-1, random_state=42
            ),
        ),
        (
            "CatBoost",
            CatBoostClassifier(iterations=10, depth=3, verbose=0, random_seed=42),
        ),
    ]

    for name, model in models:
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        path = f"tmp/iris_{name.lower()}.html"
        prettygrow(
            model,
            data=X_test_pl,
            target=y_test,
            feature_names=feature_names,
            output_path=path,
        )
        print(f"  ✓ {name}: accuracy={acc:.3f} → {path}")

    print("\nDone! Open any HTML to test prediction tracing on Iris.")


if __name__ == "__main__":
    main()
