from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.metrics import r2_score


class RidgeRegression:
    def __init__(self, alpha=1.0):
        """
        Initialize a RidgeRegression object with the specified alpha value.
        """
        self.set_alpha(alpha)

    def set_alpha(self, alpha):
        """
        Set an alpha value on a RidgeRegression object.
        """
        self.alpha = float(alpha)
        self.model = Ridge(alpha=self.alpha)

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
        Find the best alpha value for the Ridge regression model.
        """
        alphas = [1, 0.8, 0.5, 0.3, 0.1, 0.08, 0.05, 0.03, 0.01]
        best_score = float(-np.inf)
        best_alpha = None
        for alpha in alphas:
            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)

            score = model.score(X_val, y_val)
            if score > best_score:
                best_score = score
                best_alpha = alpha

        print('Best alpha:', best_alpha)
        print('Best R-squared score (higher is better)', best_score)
        return best_alpha

    def fit(self, X_train, y_train):
        """
        Fit the Ridge regression model to the specified training data.
        """
        self.model.fit(X_train, y_train)
        return self

    def score(self, X, y):
        """
        Calculate the R-squared score for the model.
        """
        y_pred = self.predict(X)
        score = r2_score(y, y_pred)
        return score

    def predict(self, X):
        """
        Predict the target variable for the specified input data.
        """
        return self.model.predict(X)

    def calculate_rmse(self, X, y):
        """
        Calculate the RMSE for the model.
        """
        y_pred = self.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        print('RMSE:', rmse)
        print()
        return rmse

    def get_coefficients_and_features(self, X):
        """
        Get the coefficients and feature names for the Ridge regression model.
        Returns a tuple containing two lists: the coefficients and the feature names.
        """
        coefficients = self.model.coef_
        coefficients = [round(coefficient, 6) for coefficient in coefficients]
        feature_names = X.columns
        return coefficients, feature_names
