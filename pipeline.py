from utils.data_management.data_analysis import *
from utils.data_management.metric_calculator import *
from utils.data_management.Training import *

import pandas as pd

df = pd.read_csv("Titanic-Dataset.csv")

if __name__ == '__main__':
    # populating age with median
    populate_nan_columns(df, "Age", find_column_median(df, "Age"))

    # populating embarked location with mode
    populate_nan_columns(df, "Embarked", find_column_mode(df, "Embarked"))

    # removing redundant columns


    remove_columns(df, ["PassengerId", "Cabin", "Ticket", "Name"])

    df = pd.get_dummies(df, columns = ["Sex", "Embarked"], dtype = int)

    run_eda(df, "Survived")

    split_tuple = split_data(df)

    train_data = split_tuple[0]
    val_data = split_tuple[1]
    test_data = split_tuple[2]

    # data preperation using dummy variables


    build_decision_tree(train_data)

    

