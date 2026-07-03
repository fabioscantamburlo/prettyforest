import marimo

__generated_with = "0.23.13"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    import polars as pl
    from prettyforest import visualize

    return (
        GradientBoostingClassifier,
        RandomForestClassifier,
        load_iris,
        mo,
        pl,
        train_test_split,
        visualize,
    )


@app.cell
def _(RandomForestClassifier, load_iris, pl, train_test_split):
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    X_test_pl = pl.DataFrame(
        {name: X_test[:, i] for i, name in enumerate(iris.feature_names)}
    )
    return X_test_pl, accuracy, iris, model, y_test


@app.cell
def _(accuracy, mo):
    mo.md(f"""
    # 🌲 PrettyForest Demo

    **Model**: Random Forest (20 trees, depth=5) on Iris dataset
    **Test accuracy**: {accuracy:.1%}

    👇 Trace a sample, double-click a tree, sort by purity.
    """)
    return


@app.cell
def _(X_test_pl, mo, model, visualize, y_test):
    html_rf = visualize(model, data=X_test_pl, target=y_test)
    mo.iframe(html_rf, height="700px")
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Gradient Boosting (boosted trees)

    Same data, different model. Notice the ⚠️ warning on double-click
    and gradient correction badges instead of class votes.
    """)
    return


@app.cell
def _(GradientBoostingClassifier, X_test_pl, iris, mo, visualize, y_test):
    gbm = GradientBoostingClassifier(n_estimators=10, max_depth=3, random_state=42)
    gbm.fit(iris.data[:105], iris.target[:105])

    html_gbm = visualize(gbm, data=X_test_pl, target=y_test)
    mo.iframe(html_gbm, height="700px")
    return


if __name__ == "__main__":
    app.run()
