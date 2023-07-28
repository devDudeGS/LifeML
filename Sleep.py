import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


"""
Used GitHub Copilot and Claude 2

LR important numerical features for me:
1. consecutive days met
2. adj time awake
3. sleep score / zone minutes yesterday

LR important categorical features for me:
1. NOT diet yesterday (cat 0 or 1 or 2)_0
2. goal met (cat 0 or 1)_0
3. diet yesterday (cat 0 or 1 or 2)_1

Lasso important features for me:
1. wake up diff (7:30)
2. zone minutes yesterday
3. LESS bedtime diff (22:30)
"""

# Read in data from csv file
# Set index to column 0
df = pd.read_csv('data/sleep.csv', index_col=0)

# Drop unnecessary columns
df.drop(['how awesome (0-4)', 'how unneeded caffeine (0-4)',
        'other notes', 'goal', 'time awake', 'stages %'], axis=1, inplace=True)

# Drop categorical columns
# df.drop(['diet yesterday (cat 0 or 1 or 2)', 'meditated yesterday (cat 0 or 1)', 'alcohol yesterday (cat 0 or 1)',
#          'goal met (cat 0 or 1)', 'estimated oxygen variation high (cat 0 or 1)'], axis=1, inplace=True)

# Encode categorical columns
df = pd.get_dummies(df, columns=['diet yesterday (cat 0 or 1 or 2)', 'meditated yesterday (cat 0 or 1)',
                                 'alcohol yesterday (cat 0 or 1)', 'goal met (cat 0 or 1)',
                                 'estimated oxygen variation high (cat 0 or 1)'])

# Convert data to numeric float values
df = df.astype('float')

# Linear Regression
# # Split data into X (features) and y (target)
# X = df.drop('target', axis=1)
# y = df['target']

# # Create linear regression model
# lr = LinearRegression()

# # Fit model to data
# lr.fit(X, y)

# # Print model coefficients
# print("Coefficients:", lr.coef_)

# # Print R-squared score
# print("R-squared score:", lr.score(X, y))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('target', axis=1), df['target'], test_size=0.2, random_state=42)

# # needed?
# X_train, X_val, y_train, y_val = train_test_split(
#     X_train, y_train, test_size=0.25, random_state=42)

# alphas = [1, 0.8, 0.5, 0.3, 0.1, 0.08, 0.05, 0.03, 0.01]

# Lasso Regression
# Create a Lasso regression object
lasso = Lasso(alpha=1.0)

# Fit the model to your data
lasso.fit(X_train, y_train)

# for a in alphas:
#     print('alpha:', a)
#     model = Lasso(alpha=a)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_val)
#     rmse = mean_squared_error(y_val, y_pred, squared=False)
#     print('RMSE:', rmse)

# Evaluate the model on your test data
score = lasso.score(X_test, y_test)
print('Lasso regression score:', score)

# Extract the feature names
# feature_names = X.columns
feature_names = X_train.columns

# Get the coefficients from the model
# coefficients = lr.coef_
coefficients = lasso.coef_

# Round the coefficients to 6 decimal places
coefficients = [round(coefficient, 6) for coefficient in coefficients]

# Combine to dataframe
coef_df = pd.DataFrame(
    data=coefficients, index=feature_names, columns=['Coefficient'])

# Sort by coefficient value
coef_df = coef_df.sort_values(by='Coefficient', ascending=False)

# Print results
print(coef_df)
print()
print("Legend")
print("High positive coefficient = strong positive relationship")
print("Low positive coefficient = weak positive relationship")
print("High negative coefficient = strong negative relationship")
print("Low negative coefficient = weak negative relationship")
print("Coefficient near 0 = little linear relationship")

# # Plot bar chart
# plt.bar(feature_names, coefficients)

# # Set the x-tick labels to be diagonal
# plt.xticks(rotation=75)

# plt.title("Linear Regression Coefficients")
# plt.xlabel("Feature")
# plt.ylabel("Coefficient")

# # Annotate bars with the value
# for i, v in enumerate(coefficients):
#     plt.text(i, v, str(round(v, 3)))

# # Prevent cut off labels
# plt.tight_layout()

# plt.show()
