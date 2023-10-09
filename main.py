import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Configure pandas to display all columns.
pd.set_option("display.width", 400)
pd.set_option("display.max_columns", None)

# Load the data.
weather = pd.read_csv("london_weather.csv", index_col="date")

# Select relevant columns.
core_weather = weather[["max_temp", "min_temp", "precipitation", "snow_depth",
                        "mean_temp", "pressure", "cloud_cover", "sunshine"]].copy()

# Convert the date to datetime.
core_weather.index = pd.to_datetime(core_weather.index, format="%Y%m%d")

# Fill null values.
core_weather["snow_depth"] = core_weather["snow_depth"].fillna(0)
core_weather["precipitation"] = core_weather["precipitation"].fillna(0)
core_weather["max_temp"] = core_weather["max_temp"].ffill()
core_weather["min_temp"] = core_weather["min_temp"].ffill()
core_weather["cloud_cover"] = core_weather["cloud_cover"].fillna(7)
core_weather["pressure"] = core_weather["pressure"].fillna(101790.0)

# Create mean temps to fill null
mean_temps = (core_weather["max_temp"] + core_weather["min_temp"]) / 2
core_weather["mean_temp"] = core_weather["mean_temp"].fillna(mean_temps)

# Add rolling means
core_weather["rolling_max"] = core_weather["max_temp"].rolling(30).mean()
core_weather = core_weather.iloc[30:, :].copy()

# Add difference between monthly average and daily temperature
core_weather["rolling_max_diff"] = core_weather["rolling_max"] - core_weather["max_temp"]

# Add difference between max and min temperatures
core_weather["max_min_diff"] = core_weather["max_temp"] - core_weather["min_temp"]

# Add monthly average
core_weather["monthly_avg"] = core_weather["max_temp"].groupby(core_weather.index.month)\
    .transform(lambda x: x.expanding(1).mean())

# Add day of year average
core_weather["day_of_year_avg"] = core_weather["max_temp"].groupby(core_weather.index.day_of_year)\
    .transform(lambda x: x.expanding(1).mean())

# Set the target values.
core_weather["target"] = core_weather.shift(-1)["max_temp"]
core_weather = core_weather.iloc[:-1, :].copy()

# Set the predictors (features).
predictors = ["max_temp", "min_temp", "precipitation", "snow_depth", "rolling_max",
              "rolling_max_diff", "max_min_diff", "monthly_avg", "day_of_year_avg",
              "mean_temp", "pressure", "cloud_cover", "sunshine"]

# Initialize a ridge model.
reg = Ridge(alpha=.1)

# Initialize a decision tree model.
reg_tree = DecisionTreeRegressor(random_state=0)

# Initialize a random forest model
reg_forest = RandomForestRegressor(max_depth=90, random_state=0, n_estimators=100)

# Initialize a multi-layer perceptron regressor (neural network)
params = {
    'hidden_layer_sizes': [13, 13, 13],
    'alpha': 0.1,
    'random_state': 0,
    'learning_rate_init': 0.005,
    'shuffle': False
}
reg_mlp = MLPRegressor(**params)


def create_predictions(features, data, model):
    """
    A function for predicting the daily maximum temperature using a regression model.
    :param features: an array of strings describing the columns of the data to use as predictors.
    :param data: a DataFrame object containing all the weather data.
    :param model: the machine learning model for generating the predictions.
    :return err: the average difference between the target values and predictions.
            result: DataFrame object containing the target values and predictions.
    """

    # Split the data.
    train = data.loc[:"2018-12-31"]
    test = data.loc["2019-01-01":]

    # Fit the data to the model.
    model.fit(train[features], train["target"])

    # Make predictions.
    predictions = model.predict(test[predictors])

    # Get the average prediction accuracy.
    err = mean_absolute_error(test["target"], predictions)

    # Combine the target and prediction values.
    result = pd.concat([test["target"], pd.Series(predictions, index=test.index)], axis=1)
    result.columns = ["actual", "prediction"]

    return err, result


# Train model and generate predictions.
error, combined = create_predictions(predictors, core_weather, reg_mlp)

# Print the results.
print(error)
print(combined)

# Plot the predictions.
combined.plot()
plt.show()

"""
## Further analysis.

# Show the difference between the predictions and target.
combined["diff"] = (combined["actual"] - combined["prediction"]).abs()
print(combined.sort_values("diff", ascending=False))

# Show the weights for each predictor.
print(reg.coef_)

# Show predictor correlation
print(core_weather.corr()["target"])
"""
