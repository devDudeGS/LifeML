import pandas as pd
from lasso_regression import LassoRegression
import numpy as np


class CsvDataset:
    def __init__(self, file_path):
        """
        Initialize a dataframe from a CSV file.
        """
        self.df = pd.read_csv(file_path, index_col=0)
        self.X = None
        self.y = None

    def drop_columns(self, columns):
        """
        Drop columns from the dataframe.
        """
        self.df.drop(columns, axis=1, inplace=True)

    def drop_index_before_date(self, date):
        """
        Drop rows with index before a given date.
        """
        self.df = self.df[self.df.index >= date]

    def drop_index_after_date(self, date):
        """
        Drop rows with index after a given date.
        """
        self.df = self.df[self.df.index <= date]

    def encode_categorical_columns(self, columns):
        """
        Encode categorical columns in the dataframe.
        """
        self.df = pd.get_dummies(self.df, columns=columns)

    def discretize_feature(self, feature, bins):
        """
        Split continuous feature into discrete bins
        """
        self.df[feature] = pd.to_numeric(self.df[feature], errors='coerce')
        digitized = np.digitize(self.df[feature], bins=bins)
        self.df[feature] = digitized
        self.encode_categorical_columns([feature])

    def split_features_and_target(self, target_column):
        """
        Set remaining values as floats, and split into features and target.
        """
        self.df = self.df.astype('float')
        self.X = self.df.drop(target_column, axis=1)
        self.y = self.df[target_column]


def get_excess_cols(df, keep_cols):
    """
    Get columns to drop based on columns to keep
    """
    all_cols = df.columns.tolist()
    return set(all_cols) - set(keep_cols)


def run_model(X, y):
    """
    Run the ML model on the dataset, and return the coefficients and feature names.
    """
    model = LassoRegression(alpha=1.0)
    model.split_data(X, y)
    best_a = model.get_best_alpha(
        model.X_train, model.y_train, model.X_val, model.y_val)
    model.set_alpha(best_a)
    model.fit(model.X_train, model.y_train)
    model.score(model.X_test, model.y_test)
    return model.get_coefficients_and_features(X)


def print_results(coefficients, feature_names):
    """
    Print the results of the ML model.
    """
    coef_df = pd.DataFrame(
        data=coefficients, index=feature_names, columns=['Coefficient'])
    coef_df = coef_df.sort_values(by='Coefficient', ascending=False)
    print(coef_df)
    print()
    print("Legend")
    print("High positive coefficient = strong positive relationship")
    print("Low positive coefficient = weak positive relationship")
    print("High negative coefficient = strong negative relationship")
    print("Low negative coefficient = weak negative relationship")
    print("Coefficient near 0 = little linear relationship")
    print()
