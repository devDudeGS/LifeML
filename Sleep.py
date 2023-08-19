from csv_dataset import CsvDataset, run_model, print_results

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
R-squared score: -0.039

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
    feature_to_discretize, bins_to_discretize = get_sleep_length_discretized()
    data.discretize_feature(feature_to_discretize, bins_to_discretize)
    data.split_features_and_target('target')
    coefficients, feature_names = run_model(data.X, data.y)
    print_results(coefficients, feature_names)


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


def get_sleep_length_columns():
    return ['time asleep', 'target']


def get_sleep_length_discretized():
    feature = 'time asleep'
    bins = [5.75, 6, 6.25, 6.5, 6.75, 7, 7.25, 7.5, 7.75, 8, 8.25, 8.5, 8.75, 9]
    return feature, bins


if __name__ == "__main__":
    analyze_sleep_data()
