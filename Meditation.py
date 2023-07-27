import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


"""
Used GitHub Copilot and Claude 2

Most important numerical features for me:
1. total_meditation_mins
2. consecutive_meditation_days
3. active_zone_mins_today

Most important categorical features for me:
1. diet_yesterday_1
2. diet_today_2
3. caffeine_0
"""

# Read in data from csv file
# Set index to column 0
df = pd.read_csv('data/meditations.csv', index_col=0)

# Drop unnecessary columns
df.drop(['inability_to_cope_with_responsibilities (0-4)', 'feeling_score_fitbit_am', 'feeling_score_fitbit_noon', 'feeling_score_fitbit_pm',
         'satisfaction_with_life_as_whole (0-4)', 'fully_mentally_alert (0-4)', 'outward_happiness', 'self_transcendence_glimpsed (0 or 1)',
         'self_insight_obtained (0 or 1)', 'meditation_type (cat)'], axis=1, inplace=True)

# Drop categorical columns
df.drop(['diet_today (0 or 1 or 2)', 'diet_yesterday (0 or 1 or 2)', 'alcohol_today (0 or 1)',
         'alcohol_yesterday (0 or 1)', 'caffeine (0 or 1)',], axis=1, inplace=True)

# Encode categorical columns
# df = pd.get_dummies(df, columns=['diet_today (0 or 1 or 2)', 'diet_yesterday (0 or 1 or 2)',
#                                  'alcohol_today (0 or 1)', 'alcohol_yesterday (0 or 1)', 'caffeine (0 or 1)'])

# Convert data to numeric float values
df = df.astype('float')

# Split data into X (features) and y (target)
X = df.drop('target_feelings', axis=1)
y = df['target_feelings']

# Create linear regression model
lr = LinearRegression()

# Fit model to data
lr.fit(X, y)

# Print model coefficients
print("Coefficients:", lr.coef_)

# Print R-squared score
print("R-squared score:", lr.score(X, y))

# Extract the feature names
feature_names = X.columns

# Get the coefficients from the model
coefficients = lr.coef_

# Round the coefficients to 5 decimal places
coefficients = [round(coefficient, 5) for coefficient in coefficients]

# Combine to dataframe
coef_df = pd.DataFrame(
    data=coefficients, index=feature_names, columns=['Coefficient'])

# Sort by coefficient value
coef_df = coef_df.sort_values(by='Coefficient', ascending=False)

# Print results
print(coef_df)

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
