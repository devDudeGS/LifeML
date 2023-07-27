import pandas as pd
from sklearn.linear_model import LinearRegression


"""
Used GitHub Copilot and Claude 2
"""

# Read in data from csv file
# Set index to column 0
df = pd.read_csv('data/meditations.csv', index_col=0)

# Drop unnecessary columns
df.drop(['inability_to_cope_with_responsibilities (0-4)', 'feeling_score_fitbit_am', 'feeling_score_fitbit_noon', 'feeling_score_fitbit_pm',
         'satisfaction_with_life_as_whole (0-4)', 'fully_mentally_alert (0-4)', 'outward_happiness', 'self_transcendence_glimpsed (0 or 1)',
         'self_insight_obtained (0 or 1)', 'meditation_type (cat)'], axis=1, inplace=True)
# temp drop
df.drop(['diet_today (0 or 1 or 2)', 'diet_yesterday (0 or 1 or 2)', 'alcohol_today (0 or 1)', 'alcohol_yesterday (0 or 1)', 'caffeine (0 or 1)',
         'sleep_length_hrs'], axis=1, inplace=True)

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
