import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression


def find_predictors(max_features=10, filepath="../data/weather_processed.pkl"):
    print("finding best predictors...")

    # ---------------------------------------
    # Load Data
    # ---------------------------------------
    weather = pd.read_pickle(filepath)

    # ---------------------------------------
    # Prepare Data
    # ---------------------------------------
    # Set the target values from the original data
    weather["target"] = weather.shift(-1)["max_temp"]
    weather = weather.iloc[:-1, :].copy()

    X = weather.drop("target", axis=1)
    y = weather["target"]

    # ---------------------------------------
    # Select the best predictors
    # ---------------------------------------
    k_best = SelectKBest(f_regression, k=max_features)
    X_selected = k_best.fit_transform(X, y)
    selected_indices = k_best.get_support(indices=True)
    best_features = X.columns[selected_indices]

    print(f"    best features: {best_features}")
    return best_features

