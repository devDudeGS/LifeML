import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


"""
Used GitHub Copilot and Claude 2

Most important numerical features for me:
1. consecutive days met
2. adj time awake
3. sleep score / zone minutes yesterday

time asleep ?? (#2 when categorical features incl)

Most important categorical features for me:
1. NOT diet yesterday (cat 0 or 1 or 2)_0
2. goal met (cat 0 or 1)_0
3. diet yesterday (cat 0 or 1 or 2)_1
"""

# Read in data from csv file
# Set index to column 0
df = pd.read_csv('data/sleep.csv', index_col=0)

# Drop unnecessary columns
df.drop(['how awesome (0-4)', 'how unneeded caffeine (0-4)',
        'other notes', 'goal'], axis=1, inplace=True)

# Drop categorical columns
# df.drop(['diet yesterday (cat 0 or 1 or 2)', 'meditated yesterday (cat 0 or 1)', 'alcohol yesterday (cat 0 or 1)',
#          'goal met (cat 0 or 1)', 'estimated oxygen variation high (cat 0 or 1)'], axis=1, inplace=True)

# Encode categorical columns
df = pd.get_dummies(df, columns=['diet yesterday (cat 0 or 1 or 2)', 'meditated yesterday (cat 0 or 1)',
                                 'alcohol yesterday (cat 0 or 1)', 'goal met (cat 0 or 1)',
                                 'estimated oxygen variation high (cat 0 or 1)'])

# Convert data to numeric float values
df = df.astype('float')

# Split data into X (features) and y (target)
X = df.drop('target', axis=1)
y = df['target']

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
