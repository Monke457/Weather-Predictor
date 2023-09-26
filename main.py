import pandas as pd

# Configure pandas to display all columns
pd.set_option("display.width", 400)
pd.set_option("display.max_columns", None)

# Load the data.
weather = pd.read_csv("london_weather.csv", index_col="date")

# Select relevant columns
core_weather = weather[["max_temp", "min_temp", "precipitation", "snow_depth"]].copy()

# Convert the date to datetime
core_weather.index = pd.to_datetime(core_weather.index, format="%Y%m%d")

# fill null values
core_weather["precipitation"] = core_weather["precipitation"].fillna(0)
core_weather["snow_depth"] = core_weather["snow_depth"].fillna(0)
core_weather["max_temp"] = core_weather["max_temp"].ffill()
core_weather["min_temp"] = core_weather["min_temp"].ffill()

print(core_weather.apply(pd.isnull).sum()/core_weather.shape[0])

# 15:46