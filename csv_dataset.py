import pandas as pd
from lasso_regression import LassoRegression


class CsvDataset:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path, index_col=0)

    def drop_columns(self, columns):
        self.df.drop(columns, axis=1, inplace=True)

    def encode_categorical_columns(self, columns):
        self.df = pd.get_dummies(self.df, columns=columns)

    def split_features_and_target(self, target_column):
        self.df = self.df.astype('float')
        self.X = self.df.drop(target_column, axis=1)
        self.y = self.df[target_column]

    def run_model(self, X, y):
        model = LassoRegression(alpha=1.0)
        model.split_data(X, y)
        best_a = model.get_best_alpha(
            model.X_train, model.y_train, model.X_val, model.y_val)
        model.set_alpha(best_a)
        model.fit(model.X_train, model.y_train)
        model.score(model.X_test, model.y_test)
        return model.get_coefficients_and_features(X)

    def print_results(self, coefficients, feature_names):
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
