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


def analyse_data(filepath="../data/london_weather.pkl"):
    print("running data analysis...")

    # Load the data.
    weather = pd.read_pickle(filepath)

    # ---------------------------------------
    # Butterworth lowpass filter
    # ---------------------------------------
    weather_lowpass = apply_filter(weather, 10, 1.5)
    plot_comparison(weather, weather_lowpass, "max_temp", 1996, 4)

    weather_lowpass.to_pickle("../data/lowpass_weather.pkl")

    print("lowpass created -> data/lowpass_weather.pkl")
    return weather_lowpass

