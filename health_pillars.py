from csv_dataset import CsvDataset

"""
Analyzes health data from 2020-09-08 to 2020-10-27,
and 2021-09-27 to 2021-12-17,
and 2023-08-01 to present,
to determine which features are correlated with happiness scores.

Most important features for me:
1. sleep_wakeup_diff (6:15-6:45)
2. active_zone_mins_today
3. sleep_bedtime_diff (22:30-23:00)
4. active_zone_mins_prev_week
5. active_zone_mins_yesterday

Alpha: 1.0
RMSE: 0.231
R-squared score: -0.104

Used GitHub Copilot and Claude 2.0 to help write the initial logic.
"""


def analyze_health_data():
    print()
    print("Health Pillars Data Analysis")
    print()

    data = CsvDataset('data/health_pillars.csv')
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
    return ['meditation_type (cat)', 'non_duality_glimpsed (0 or 1)', 'self_insight_obtained (0 or 1)',
            'feeling_score_fitbit_am', 'feeling_score_fitbit_pm', 'satisfaction_with_life_as_whole (0-4)',
            'inability_to_cope_with_responsibilities (0-4)']


def get_categorical_columns():
    """
    Return categorical columns to encode.
    """
    return ['caffeine (0 or 1)', 'diet_today (0 or 1 or 2)', 'diet_yesterday (0 or 1 or 2)', 'exercise_type (cat)']


if __name__ == "__main__":
    analyze_health_data()
