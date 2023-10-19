import pandas as pd
import data_processing as process
import data_analysis as analyse
import feature_engineering as feat
import feature_select as select
import models

# -----------------------------------------------------------------
# Configure Display
# -----------------------------------------------------------------
pd.set_option("display.width", 400)
pd.set_option("display.max_columns", None)

# -----------------------------------------------------------------
# Selects relevant columns, fills null values, sets datetime index
# Applies a lowpass filter
# Stores original data as pickle file
# Stores processed data as a pickle file
# -----------------------------------------------------------------
process.process_data()

# -----------------------------------------------------------------
# Adds predictor columns to the dataset
# Overwrites the pickle file
# -----------------------------------------------------------------
feat.engineer_features()

# -----------------------------------------------------------------
# Runs clustering analysis and PCA
# Overwrites the pickle file
# -----------------------------------------------------------------
analyse.analyse_data()

# -----------------------------------------------------------------
# Finds the 15 best predictors
# -----------------------------------------------------------------
predictors = select.find_predictors()

# -----------------------------------------------------------------
# Trains models: Ridge, Decision Tree, Random Forest, MLP
# Prints mean absolute errors
# Plots results
# -----------------------------------------------------------------
models.create_and_test_models(predictors=predictors)
