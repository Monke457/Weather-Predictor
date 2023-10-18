import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns


def train_and_test_models(predictors=None, filepath="../data/weather_processed.pkl"):
    # ---------------------------------------
    # Load data
    # ---------------------------------------
    weather = pd.read_pickle(filepath)

    # always load the original data set make sure the target values are correct
    weather_orig = pd.read_pickle("../data/london_weather.pkl")

    # ---------------------------------------
    # Set the predictors and target
    # ---------------------------------------
    if predictors is None:
        predictors = weather.columns

    weather["target"] = weather_orig.shift(-1)["max_temp"]
    weather = weather.iloc[:-1, :].copy()

    # Take subset of data to avoid overfitting
    weather = weather.iloc[::2]

    # ---------------------------------------
    # Create Models
    # ---------------------------------------
    reg_ridge = Ridge(alpha=.1)
    reg_tree = DecisionTreeRegressor(random_state=0)
    reg_forest = RandomForestRegressor(max_depth=90, random_state=0, n_estimators=100)

    col_count = len(predictors)
    params = {
        'hidden_layer_sizes': [col_count, col_count, col_count],
        'alpha': 0.1,
        'random_state': 0,
        'learning_rate_init': 0.005,
        'shuffle': False
    }
    reg_mlp = MLPRegressor(**params)

    # Select which models to test
    models = [reg_ridge, reg_tree, reg_mlp, reg_forest]

    # ---------------------------------------
    # Start training and testing
    # ---------------------------------------
    print("training and testing models...")
    for m in models:
        # Train model and generate predictions
        error, combined = create_predictions(predictors, weather, m)

        # Print the accuracy
        modelName = m.__class__.__name__
        print(f"{modelName} MAE: {error}")

        # Plot the predictions
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=combined, x=combined.index, y="actual")
        sns.lineplot(data=combined, x=combined.index, y="prediction")
        plt.xlabel("Date")
        plt.ylabel("Celsius")
        plt.title(f"{modelName}")
        plt.show()


def create_predictions(features, data, model):
    """
    A function for predicting the daily maximum temperature using a regression model.
    :param features: an array of strings describing the columns of the data to use as predictors.
    :param data: a DataFrame object containing all the weather data.
    :param model: the machine learning model for generating the predictions.
    :return err: the average difference between the target values and predictions.
            result: DataFrame object containing the target values and predictions.
    """

    # Split the data
    train = data.loc[:"2018-12-31"]
    test = data.loc["2019-01-01":]

    # Fit the data to the model
    model.fit(train[features], train["target"])

    # Make predictions
    predictions = model.predict(test[features])

    # Get the average prediction accuracy
    err = mean_absolute_error(test["target"], predictions)

    # Combine the target and prediction values
    result = pd.concat([test["target"], pd.Series(predictions, index=test.index)], axis=1)
    result.columns = ["actual", "prediction"]

    return err, result

