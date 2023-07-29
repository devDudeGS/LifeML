from csv_dataset import CsvDataset

"""
Analyzes sleep data from 2020-09-08 to 2020-10-27,
to determine which features are correlated with higher sleep quality.

Most important features for me:
1. zone minutes yesterday
2. wake up diff (7:30)
3. adj time awake
4. bedtime diff (22:30)

Alpha: 1.0
RMSE: 0.511

Used GitHub Copilot and Claude 2.0 to help write this code.
"""


def analyze_sleep_data():
    print()
    print("Meditation Data Analysis")
    print()

    data = CsvDataset('data/sleep.csv')
    columns_to_drop = get_columns_to_drop()
    data.drop_columns(columns_to_drop)
    categorical_columns = get_categorical_columns()
    data.encode_categorical_columns(categorical_columns)
    data.split_features_and_target('target')
    coefficients, feature_names = data.run_model(data.X, data.y)
    data.print_results(coefficients, feature_names)


def get_columns_to_drop():
    """
    Return columns to drop from the dataset.
    Includes columns used to derive target or other features.
    """
    return ['how awesome (0-4)', 'how unneeded caffeine (0-4)', 'other notes', 'goal', 'time awake', 'stages %']


def get_categorical_columns():
    """
    Return categorical columns to encode.
    """
    return ['diet yesterday (cat 0 or 1 or 2)', 'meditated yesterday (cat 0 or 1)', 'alcohol yesterday (cat 0 or 1)',
            'goal met (cat 0 or 1)', 'estimated oxygen variation high (cat 0 or 1)']


if __name__ == "__main__":
    analyze_sleep_data()
