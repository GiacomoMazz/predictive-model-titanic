from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

import pandas as pd

# splits into train/validation/test

def split_data(df: pd.DataFrame):

    x = df.drop("Survived", axis = 1)
    y = df["Survived"]

    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size = 0.4, train_size = 0.6, random_state = 67, stratify=y)

    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size = 0.5, train_size = 0.5, random_state = 67, stratify=y_temp)

    training_data = (x_train, y_train)
    validating_data = (x_val, y_val)
    testing_data = (x_test, y_test)

    return training_data, validating_data, testing_data

def build_decision_tree(train_data):
    x_train, y_train = train_data

    model = DecisionTreeClassifier(random_state = 67)
    model.fit(x_train, y_train)

    return model