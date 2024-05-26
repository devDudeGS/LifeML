from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np


class LassoRegression:
    def __init__(self, alpha=1.0):
        """
        Initialize a LassoRegression object with the specified alpha value.
        """
        self.set_alpha(alpha)

    def set_alpha(self, alpha):
        """
        Set an alpha value on a LassoRegression object.
        """
        self.alpha = float(alpha)
        self.model = Lasso(alpha=self.alpha)

    def split_data(self, X, y):
        """
        Split the data into training, test, and validation sets.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.25, random_state=42)
        self.X_train = X_train  # X training set
        self.X_test = X_test  # X test set
        self.X_val = X_val  # X validation set
        self.y_train = y_train  # y training set
        self.y_test = y_test  # y test set
        self.y_val = y_val  # y validation set
        return self

    def get_best_alpha(self, X_train, y_train, X_val, y_val):
        """
        Find the alpha with the lowest RMSE for the Lasso regression model.
        """
        alphas = [1, 0.8, 0.5, 0.3, 0.1, 0.08, 0.05, 0.03, 0.01]
        best_rmse = float(np.inf)
        best_alpha = None
        for alpha in alphas:
            model = Lasso(alpha=alpha)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            if rmse < best_rmse:
                best_rmse = rmse
                best_alpha = alpha

        print('Best alpha:', best_alpha)
        print('Best RMSE (lower is better):', best_rmse)
        return best_alpha, best_rmse

    def fit(self, X_train, y_train):
        """
        Fit the Lasso regression model to the specified training data.
        """
        self.model.fit(X_train, y_train)
        return self

    def score(self, X_test, y_test):
        """
        Get the R-squared score for the Lasso regression model on the specified test data.
        """
        score = self.model.score(X_test, y_test)
        print('R-squared score (higher is better):', score)
        print()
        return score

    def get_coefficients_and_features(self, X):
        """
        Get the coefficients and feature names for the Lasso regression model.
        Returns a tuple containing two lists: the coefficients and the feature names.
        """
        coefficients = self.model.coef_
        coefficients = [round(coefficient, 6) for coefficient in coefficients]
        feature_names = X.columns
        return coefficients, feature_names
