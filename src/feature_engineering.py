import pandas as pd
import numpy as np


def aggregate_value(function):
    if function == "std":
        return np.std
    if function == "mean":
        return np.mean
    if function == "median":
        return np.median
    return np.nan


def add_temporal_abstraction(df, columns, function):
    print(
        f"adding temporal abstraction {function}"
    )
    temporal_abstraction = (
        df[columns]
        .rolling(window=30, min_periods=1)
        .apply(aggregate_value(function))
    )

    df_temp = pd.concat(
        [df, temporal_abstraction.add_suffix(f"_{function}_30")], axis=1
    )

    return df_temp


def engineer_features(filepath="../data/london_weather.pkl"):
    print("engineering features...")

    # Load the data.
    weather = pd.read_pickle(filepath)

    # Set columns for temporal abstraction
    columns = ["cloud_cover", "sunshine", "max_temp", "mean_temp", "min_temp", "precipitation",
               "pressure", "snow_depth"]
    aggregate_functions = ["std", "mean", "median"]

    # Add temporal abstraction for each aggregate function
    for fun in aggregate_functions:
        weather = add_temporal_abstraction(weather, columns, fun)

    # Take subset of data to avoid overfitting
    weather = weather.iloc[::2]

    # Add difference between max and min temperatures
    weather["max_min_diff"] = weather["max_temp"] - weather["min_temp"]

    # Add monthly average
    weather["monthly_avg"] = weather["max_temp"].groupby(weather.index.month) \
        .transform(lambda x: x.expanding(1).mean())

    # Add day of year average
    weather["day_of_year_avg"] = weather["max_temp"].groupby(weather.index.day_of_year) \
        .transform(lambda x: x.expanding(1).mean())

    weather.to_pickle(filepath)

    print(f"features generated -> {filepath}")

    return weather
