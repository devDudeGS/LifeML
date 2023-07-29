import pandas as pd
from lasso_regression import LassoRegression
import matplotlib.pyplot as plt

"""
Used GitHub Copilot and Claude 2

Most important features for me:
1. consecutive_meditation_days
2. active_zone_mins_today
3. first_meditation_time
"""

# Read in data from csv file
# Set index to column 0
df = pd.read_csv('data/meditations.csv', index_col=0)

# Drop unnecessary columns
df.drop(['inability_to_cope_with_responsibilities (0-4)', 'feeling_score_fitbit_am', 'feeling_score_fitbit_noon',
         'feeling_score_fitbit_pm', 'satisfaction_with_life_as_whole (0-4)', 'fully_mentally_alert (0-4)',
         'outward_happiness', 'self_transcendence_glimpsed (0 or 1)', 'self_insight_obtained (0 or 1)',
         'meditation_type (cat)'], axis=1, inplace=True)

# Encode categorical columns
df = pd.get_dummies(df, columns=['diet_today (0 or 1 or 2)', 'diet_yesterday (0 or 1 or 2)',
                                 'alcohol_today (0 or 1)', 'alcohol_yesterday (0 or 1)', 'caffeine (0 or 1)'])

# Convert data to numeric float values
df = df.astype('float')

# Split data into X (features) and y (target)
X = df.drop('target_feelings', axis=1)
y = df['target_feelings']

# Create and fit a Lasso regression model
lasso = LassoRegression(alpha=1.0)
lasso.split_data(X, y)
best_a = lasso.get_best_alpha(
    lasso.X_train, lasso.y_train, lasso.X_val, lasso.y_val)
lasso.set_alpha(best_a)
lasso.fit(lasso.X_train, lasso.y_train)
lasso.score(lasso.X_test, lasso.y_test)

coefficients, feature_names = lasso.get_coefficients_and_features(X)

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
