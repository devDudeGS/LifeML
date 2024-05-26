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

    def insert_column_from_column_additions(self, columns, new_column_name):
        """
        Insert new column, by adding given columns.
        """
        self.df[new_column_name] = 0
        for col in columns:
            self.df[new_column_name] += self.df[col]

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

    def drop_rows_without_column_values(self, columns):
        """
        Drop rows without values in given columns.
        """
        self.df = self.df.dropna(subset=columns)

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

    def breakout_categorical_codes(self, feature, category_prefix):
        """
        Given a feature that contains multiple categorical codes,
        break out each code into its own column, then drop the original.
        """
        unique_categories = sorted(set(''.join(self.df[feature].unique())))
        for category in unique_categories:
            self.df[category_prefix + category] = self.df[feature].apply(lambda x: 1 if category in x else 0)
        self.df.drop(feature, axis=1, inplace=True)

    def split_features_and_target(self, target_column):
        """
        Set remaining values as floats, and split into features and target.
        """
        self.df = self.df.astype('float')
        self.X = self.df.drop(target_column, axis=1)
        self.y = self.df[target_column]

    def get_avg_in_column(self, column, days):
        """
        Get average for a column over a certain number of days.
        """
        subset = self.df[column].tail(days)
        mean = round(subset.mean(), 1)
        median = round(subset.median(), 1)
        print(f"Mean for {column} in last {days} days: {mean}")
        print(f"Median for {column} in last {days} days: {median}")
        print()


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
    best_a, best_error = model.get_best_alpha(
        model.X_train, model.y_train, model.X_val, model.y_val)
    model.set_alpha(best_a)
    model.fit(model.X_train, model.y_train)
    model.score(model.X_test, model.y_test)
    return model, best_error


def print_failure(best_rmse, rmse_threshold):
    """
    Print failure if RMSE doesn't meet threshold.
    """
    print(f"RMSE of {best_rmse} was not lower than the threshold of {rmse_threshold}")
    print()


def print_coef(coefficients, feature_names):
    """
    Print coefficients with feature names.
    """
    coef_df = pd.DataFrame(
        data=coefficients, index=feature_names, columns=['Coefficient'])
    coef_df = coef_df[coef_df['Coefficient'] != 0]
    coef_df = coef_df.sort_values(by='Coefficient', ascending=False)
    print(coef_df)
    print()


def print_results(model, best_rmse, rmse_threshold, X):
    """
    Print the results of the ML model.
    """
    if best_rmse < rmse_threshold:
        coefficients, feature_names = model.get_coefficients_and_features(X)

        print_coef(coefficients, feature_names)
    else:
        print_failure(best_rmse, rmse_threshold)


def label_discretized_features(feature_names, feature, bins):
    """
    Updates the discretized feature label with actual bin values
    """
    updated_names = []
    for name in feature_names:
        if name.startswith(feature):
            index = int(name.split('_')[-1])
            bin_value = bins[index]
            name = name.replace(str(index), str(bin_value))
        updated_names.append(name)
    return updated_names


def print_results_discretized(model, best_rmse, rmse_threshold, X, feature, bins):
    """
    Print the results of the discretized ML model.
    """
    if best_rmse < rmse_threshold:
        coefficients, feature_names = model.get_coefficients_and_features(X)
        feature_names = label_discretized_features(feature_names, feature, bins)

        print_coef(coefficients, feature_names)
    else:
        print_failure(best_rmse, rmse_threshold)


def print_legend():
    """
    Print the legend to understand the results.
    """
    print("Legend")
    print("High positive coefficient = strong positive relationship")
    print("Low positive coefficient = weak positive relationship")
    print("High negative coefficient = strong negative relationship")
    print("Low negative coefficient = weak negative relationship")
    print("Coefficient near 0 = little linear relationship")
    print()
