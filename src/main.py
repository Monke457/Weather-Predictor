import pandas as pd
import data_processing as process
import data_analysis as analyse
import feature_engineering as feat
import feature_select as select
import models

# Configure pandas to display all columns
pd.set_option("display.width", 400)
pd.set_option("display.max_columns", None)

# Selects relevant columns, fills null values, sets datetime index
# Stores as a pickle file
# process.process_data("../data/london_weather.csv")

# Creates a new dataframe and applies a lowpass filter
# Plots the results as a comparison
# Stores as a pickle file
# analyse.analyse_data("../data/london_weather.pkl")

# Adds predictor columns to the dataset
# Overwrites the pickle file
# feat.engineer_features("../data/lowpass_weather.pkl")

# Finds the 10 best predictors
# predictors = select.find_predictors(filepath="../data/lowpass_weather.pkl")

# Trains models: Ridge, Decision Tree, Random Forest, MLP
# Prints mean absolute errors
# Plots results
models.train_and_test_models(filepath="../data/lowpass_weather.pkl")
