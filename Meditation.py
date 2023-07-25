import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Load the CSV data into a pandas DataFrame
data = pd.read_csv('data/meditations.csv')

# Split the data into features (X) and target (y)
X = data.iloc[:, 1:]
y = data.iloc[:, 0]

# Create a Logistic Regression model
model = LogisticRegression()

# Create an RFE object and fit it to the data
rfe = RFE(model, n_features_to_select=1)
rfe.fit(X, y)

# Print the feature rankings
for i, rank in enumerate(rfe.ranking_):
    print(f'Feature {i+1}: Rank {rank}')
