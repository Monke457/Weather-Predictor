import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_transformation import LowPassFilter

LowPass = LowPassFilter()


def apply_filter(df, fs, cutoff):
    df_filtered = df.copy()

    for col in df_filtered.columns:
        df_filtered = LowPass.low_pass_filter(df_filtered, col, fs, cutoff)
        df_filtered[col] = df_filtered[col + "_lowpass"]
        del df_filtered[col + "_lowpass"]

    return df_filtered


def plot_comparison(df, df_lp, col, year=2000, month=6):
    subset_1 = df[df.index.year == year]
    subset_2 = df_lp[df_lp.index.year == year]

    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=subset_1,
        x=subset_1.index,
        y=col,
        label=f"{col} before filter",
    )
    sns.lineplot(
        data=subset_2,
        x=subset_2.index,
        y=col,
        label=f"{col} after filter",
    )
    plt.xlabel("Date")
    plt.ylabel("Celsius")
    plt.title(f"Max Temperature {year} Before and After Lowpass Filter")
    plt.show()

    subset_1 = df[(df.index.year == year) & (df.index.month == month)]
    subset_2 = df_lp[(df_lp.index.year == year) & (df_lp.index.month == month)]

    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=subset_1,
        x=subset_1.index,
        y=col,
        label=f"{col} before filter",
    )
    sns.lineplot(
        data=subset_2,
        x=subset_2.index,
        y=col,
        label=f"{col} after filter",
    )
    plt.xlabel("Date")
    plt.ylabel("Celsius")
    plt.title(f"Max Temperature {year} / {month} Before and After Lowpass Filter")
    plt.show()


def process_data(filepath="../data/london_weather.csv", plot=False):
    print("processing data...")

    # ---------------------------------------
    # Load data
    # ---------------------------------------
    weather = pd.read_csv(filepath, index_col="date")
    core_weather = weather[["max_temp", "min_temp", "precipitation", "snow_depth",
                            "mean_temp", "pressure", "cloud_cover", "sunshine"]].copy()
    core_weather.index = pd.to_datetime(core_weather.index, format="%Y%m%d")

    # ---------------------------------------
    # Fill null values
    # ---------------------------------------
    core_weather["snow_depth"] = core_weather["snow_depth"].fillna(0)
    core_weather["precipitation"] = core_weather["precipitation"].fillna(0)
    core_weather["max_temp"] = core_weather["max_temp"].ffill()
    core_weather["min_temp"] = core_weather["min_temp"].ffill()
    core_weather["cloud_cover"] = core_weather["cloud_cover"].fillna(7)
    core_weather["pressure"] = core_weather["pressure"].fillna(101790.0)
    mean_temps = (core_weather["max_temp"] + core_weather["min_temp"]) / 2
    core_weather["mean_temp"] = core_weather["mean_temp"].fillna(mean_temps)

    # ---------------------------------------
    # Butterworth lowpass filter
    # ---------------------------------------
    weather_lowpass = apply_filter(core_weather, 10, 1.5)
    if plot:
        plot_comparison(core_weather, weather_lowpass, "max_temp", 1996, 4)

    # ---------------------------------------
    # Export
    # ---------------------------------------
    core_weather.to_pickle("../data/london_weather.pkl")
    weather_lowpass.to_pickle("../data/weather_processed.pkl")
    print("    data pickled -> data/london_weather.pkl")
    print("    data processed -> data/weather_processed.pkl")
    return weather_lowpass

