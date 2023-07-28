from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class LassoRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.model = Lasso(alpha=self.alpha)

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        self.score = self.model.score(X_test, y_test)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_coefficients_and_features(self, X):
        coefficients = self.model.coef_

        # Round the coefficients to 6 decimal places
        coefficients = [round(coefficient, 6) for coefficient in coefficients]

        feature_names = X.columns
        return coefficients, feature_names
