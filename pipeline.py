from utils.data_management.data_analysis import *
from utils.data_management.metric_calculator import *

import pandas as pd

df = pd.read_csv("Titanic-Dataset.csv")

if __name__ == '__main__':
    # populating age with median
    populate_nan_columns(df, "Age", find_column_median(df, "Age"))

    # populating embarked location with mode
    populate_nan_columns(df, "Embarked", find_column_mode(df, "Embarked"))

    # removing redundant columns

    run_eda(df, "Survived")
