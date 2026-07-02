"""Run PrettyForest experiments with configurable parameters.

Usage:
    uv run run_experiments.py
    uv run run_experiments.py --n-samples 5000 --n-trees 100 --max-depth 12
    uv run run_experiments.py --models rf lgbm --task cls
    uv run run_experiments.py --n-predict 200 --seed 99
"""

import argparse
from pathlib import Path

import pandas as pd
import polars as pl
from sklearn.datasets import make_classification, make_regression


def main():
    parser = argparse.ArgumentParser(description="PrettyForest experiment runner")
    parser.add_argument("--n-samples", type=int, default=500, help="Training samples (default: 500)")
    parser.add_argument("--n-features", type=int, default=8, help="Number of features (default: 8)")
    parser.add_argument("--n-trees", type=int, default=20, help="Number of estimators (default: 20)")
    parser.add_argument("--max-depth", type=int, default=6, help="Max tree depth (default: 6)")
    parser.add_argument("--n-predict", type=int, default=100, help="Samples to embed for prediction (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--task", choices=["cls", "reg", "both"], default="both", help="Task type (default: both)")
    parser.add_argument("--models", nargs="+", default=["rf", "gbm", "lgbm", "catboost"],
                        choices=["rf", "gbm", "lgbm", "catboost", "dt"],
                        help="Models to run (default: rf gbm lgbm catboost)")
    parser.add_argument("--outdir", type=str, default="tmp", help="Output directory (default: tmp)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    feat_names = [f"f{i}" for i in range(args.n_features)]

    # Generate datasets
    X_cls, y_cls, X_reg, y_reg = None, None, None, None
    if args.task in ("cls", "both"):
        X_cls, y_cls = make_classification(
            n_samples=args.n_samples, n_features=args.n_features,
            n_informative=max(2, args.n_features // 2), n_redundant=2,
            n_classes=3, n_clusters_per_class=2, random_state=args.seed,
        )
    if args.task in ("reg", "both"):
        X_reg, y_reg = make_regression(
            n_samples=args.n_samples, n_features=args.n_features,
            n_informative=max(2, args.n_features // 2), noise=15.0,
            random_state=args.seed,
        )

    from prettyforest import visualize

    def run(name, model, X_train, y_train, X_pl):
        model.fit(X_train, y_train)
        # Limit embedded samples to n_predict
        data_subset = X_pl.head(args.n_predict)
        path = outdir / f"{name}.html"
        visualize(model, data=data_subset, output_path=str(path))
        n_trees = getattr(model, "n_estimators", None) or getattr(model, "tree_count_", None) or getattr(model, "iterations", None) or 1
        print(f"  ✓ {name}: {n_trees} trees, {args.n_samples} samples → {path}")

    # Classification
    if X_cls is not None:
        X_cls_df = pd.DataFrame(X_cls, columns=feat_names)
        X_cls_pl = pl.DataFrame({n: X_cls[:, i] for i, n in enumerate(feat_names)})

        print(f"\n=== Classification ({args.n_samples} samples, {args.n_features} features) ===\n")

        if "dt" in args.models:
            from sklearn.tree import DecisionTreeClassifier
            run("dt_cls", DecisionTreeClassifier(max_depth=args.max_depth, random_state=args.seed),
                X_cls_df, y_cls, X_cls_pl)

        if "rf" in args.models:
            from sklearn.ensemble import RandomForestClassifier
            run("rf_cls", RandomForestClassifier(n_estimators=args.n_trees, max_depth=args.max_depth, random_state=args.seed),
                X_cls_df, y_cls, X_cls_pl)

        if "gbm" in args.models:
            from sklearn.ensemble import GradientBoostingClassifier
            run("gbm_cls", GradientBoostingClassifier(n_estimators=args.n_trees, max_depth=args.max_depth, random_state=args.seed),
                X_cls_df, y_cls, X_cls_pl)

        if "lgbm" in args.models:
            from lightgbm import LGBMClassifier
            run("lgbm_cls", LGBMClassifier(n_estimators=args.n_trees, max_depth=args.max_depth, verbose=-1, random_state=args.seed),
                X_cls_df, y_cls, X_cls_pl)

        if "catboost" in args.models:
            from catboost import CatBoostClassifier
            run("catboost_cls", CatBoostClassifier(iterations=args.n_trees, depth=min(args.max_depth, 16), verbose=0, random_seed=args.seed),
                X_cls_df, y_cls, X_cls_pl)

    # Regression
    if X_reg is not None:
        X_reg_df = pd.DataFrame(X_reg, columns=feat_names)
        X_reg_pl = pl.DataFrame({n: X_reg[:, i] for i, n in enumerate(feat_names)})

        print(f"\n=== Regression ({args.n_samples} samples, {args.n_features} features) ===\n")

        if "dt" in args.models:
            from sklearn.tree import DecisionTreeRegressor
            run("dt_reg", DecisionTreeRegressor(max_depth=args.max_depth, random_state=args.seed),
                X_reg_df, y_reg, X_reg_pl)

        if "rf" in args.models:
            from sklearn.ensemble import RandomForestRegressor
            run("rf_reg", RandomForestRegressor(n_estimators=args.n_trees, max_depth=args.max_depth, random_state=args.seed),
                X_reg_df, y_reg, X_reg_pl)

        if "gbm" in args.models:
            from sklearn.ensemble import GradientBoostingRegressor
            run("gbm_reg", GradientBoostingRegressor(n_estimators=args.n_trees, max_depth=args.max_depth, random_state=args.seed),
                X_reg_df, y_reg, X_reg_pl)

        if "lgbm" in args.models:
            from lightgbm import LGBMRegressor
            run("lgbm_reg", LGBMRegressor(n_estimators=args.n_trees, max_depth=args.max_depth, verbose=-1, random_state=args.seed),
                X_reg_df, y_reg, X_reg_pl)

        if "catboost" in args.models:
            from catboost import CatBoostRegressor
            run("catboost_reg", CatBoostRegressor(iterations=args.n_trees, depth=min(args.max_depth, 16), verbose=0, random_seed=args.seed),
                X_reg_df, y_reg, X_reg_pl)

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
