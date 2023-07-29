from csv_dataset import CsvDataset

"""
Analyzes meditation data from 2021-09-27 to 2021-12-17,
to determine which features are correlated with happiness scores.

Most important features for me:
1. consecutive_meditation_days
2. active_zone_mins_today
3. first_meditation_time

Used GitHub Copilot and Claude 2.0 to help write this code.
"""


def analyze_meditation_data():
    data = CsvDataset('data/meditations.csv')
    columns_to_drop = get_columns_to_drop()
    data.drop_columns(columns_to_drop)
    categorical_columns = get_categorical_columns()
    data.encode_categorical_columns(categorical_columns)
    data.split_features_and_target('target_feelings')
    coefficients, feature_names = data.run_model(data.X, data.y)
    data.print_results(coefficients, feature_names)


def get_columns_to_drop():
    """
    Return columns to drop from the dataset.
    Includes columns used to derive target or other features.
    """
    return ['feeling_score_fitbit_am', 'feeling_score_fitbit_noon', 'feeling_score_fitbit_pm', 'satisfaction_with_life_as_whole (0-4)',
            'inability_to_cope_with_responsibilities (0-4)', 'fully_mentally_alert (0-4)', 'outward_happiness',
            'self_transcendence_glimpsed (0 or 1)', 'self_insight_obtained (0 or 1)', 'meditation_type (cat)']


def get_categorical_columns():
    """
    Return categorical columns to encode.
    """
    return ['diet_today (0 or 1 or 2)', 'diet_yesterday (0 or 1 or 2)', 'alcohol_today (0 or 1)', 'alcohol_yesterday (0 or 1)',
            'caffeine (0 or 1)']


if __name__ == "__main__":
    analyze_meditation_data()
