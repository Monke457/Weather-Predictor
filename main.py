import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Configure pandas to display all columns.
pd.set_option("display.width", 400)
pd.set_option("display.max_columns", None)

# Load the data.
weather = pd.read_csv("london_weather.csv", index_col="date")

# Select relevant columns.
core_weather = weather[["max_temp", "min_temp", "precipitation", "snow_depth"]].copy()

# Convert the date to datetime.
core_weather.index = pd.to_datetime(core_weather.index, format="%Y%m%d")

# Fill null values.
core_weather["snow_depth"] = core_weather["snow_depth"].fillna(0)
core_weather["precipitation"] = core_weather["precipitation"].fillna(0)
core_weather["max_temp"] = core_weather["max_temp"].ffill()
core_weather["min_temp"] = core_weather["min_temp"].ffill()

# Set the target values.
core_weather["target"] = core_weather.shift(-1)["max_temp"]
core_weather = core_weather.iloc[:-1, :].copy()


def create_predictions(predictors, data, reg):
    """
    A function for predicting the daily maximum temperature using a regression model.
    :param predictors: an array of strings describing the columns of the data to use as predictors.
    :param data: a DataFrame object containing all the weather data.
    :param reg: the Ridge regression model to use to generate the predictions.
    :return error: the average difference between the target values and predictions.
            combined: a DataFrame object containing the target values and predictions.
    """

    # Set the predictors (features).
    predictors = ["max_temp", "min_temp", "precipitation", "snow_depth"]

    # Split the data.
    train = data.loc[:"2018-12-31"]
    test = data.loc["2019-01-01":]

    # Initialize a regression model.
    reg = Ridge(alpha=.1)

    # Fit the data to the model.
    reg.fit(train[predictors], train["target"])

    # Make predictions.
    predictions = reg.predict(test[predictors])

    # Get the average prediction accuracy.
    error = mean_absolute_error(test["target"], predictions)

    # Combine the target and prediction values.
    combined = pd.concat([test["target"], pd.Series(predictions, index=test.index)], axis=1)
    combined.columns = ["actual", "predictions"]

    return error, combined


core_weather["month_max"] = core_weather["max_temp"].rolling(30).mean()
core_weather["month_day_max"] = core_weather["month_max"] / core_weather["max_temp"]
print(core_weather)

""" 
## Prediction analysis.

# Plot the predictions.
combined.plot()
plt.show()

# Show the weights for each predictor.
print(reg.coef_)
"""

# print(core_weather.apply(pd.isnull).sum())
# print(core_weather.groupby(core_weather.index.year).sum())
