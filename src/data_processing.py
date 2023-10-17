import pandas as pd


def process_data(filepath="../data/london_weather.csv"):
    print("processing data...")
    # Load the data.
    weather = pd.read_csv(filepath, index_col="date")

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

    # Store as a pickle file
    core_weather.to_pickle("../data/london_weather.pkl")

    print("data processed -> data/london_weather.pkl")

    return core_weather

