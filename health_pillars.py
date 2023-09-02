from csv_dataset import CsvDataset, run_model, print_results, get_excess_cols, print_legend

"""
Analyzes health data from:
2020-09-08 to 2020-10-27,
2021-09-27 to 2021-12-17,
2023-08-01 to present,
to determine which features are correlated with happiness scores.

Baseline important features for me:
1. sleep_wakeup_diff
2. active_zone_mins_today
3. sleep_bedtime_diff
4. active_zone_mins_prev_week
5. active_zone_mins_yesterday

Alpha: 1.0
RMSE: 0.231
R-squared score: -0.104

Used GitHub Copilot and Claude 2.0 to help write the initial logic.
"""

HEALTH_CSV = 'data/health_pillars.csv'
TARGET_MAIN = 'target_feelings'
TARGET_WAKEUP = 'feeling_score_fitbit_am'
SLEEP_DATASET_START = '2020-09-08'
SLEEP_DATASET_END = '2020-10-27'
MEDITATION_DATASET_START = '2021-09-27'
MEDITATION_DATASET_END = '2021-12-17'
HEALTH_DATASET_START = '2023-08-01'
LATEST_DATA_END = '2023-08-26'


def analyze_health_data():
    print()
    print("Health Pillars Data Analysis")
    print()

    analyze_all_features(TARGET_MAIN, SLEEP_DATASET_START, MEDITATION_DATASET_END)
    analyze_sleep_length(TARGET_WAKEUP, SLEEP_DATASET_START, MEDITATION_DATASET_END)
    print_legend()


def analyze_all_features(target, date_to_start, date_to_end):
    print("--ALL FEATURES--")
    print()

    data = CsvDataset(HEALTH_CSV)

    # set dates
    data.drop_index_before_date(date_to_start)
    data.drop_index_after_date(date_to_end)

    # set columns
    data.drop_columns(get_columns_to_drop())
    data.encode_categorical_columns(get_categorical_columns())

    # run model
    data.split_features_and_target(target)
    coefficients, feature_names = run_model(data.X, data.y)

    print_results(coefficients, feature_names)


def analyze_sleep_length(target, date_to_start, date_to_end):
    print("--SLEEP LENGTH--")
    print()

    data = CsvDataset(HEALTH_CSV)

    # set dates
    data.drop_index_before_date(date_to_start)
    data.drop_index_after_date(date_to_end)

    # set columns
    columns_to_keep = get_sleep_length_columns(target)
    columns_to_drop = get_excess_cols(data.df, columns_to_keep)
    data.drop_columns(columns_to_drop)
    feature, bins = get_sleep_length_discretized()
    data.discretize_feature(feature, bins)

    # run model
    data.split_features_and_target(target)
    coefficients, feature_names = run_model(data.X, data.y)

    print_results(coefficients, feature_names)


def get_columns_to_drop():
    """
    Return columns to drop from the dataset.
    Includes columns used to derive target or other features.
    """
    return ['meditation_type (cat)', 'non_duality_glimpsed (0 or 1)', 'self_insight_obtained (0 or 1)',
            'feeling_score_fitbit_am', 'feeling_score_fitbit_pm', 'satisfaction_with_life_as_whole (0-4)',
            'inability_to_cope_with_responsibilities (0-4)', 'sleep_awake_mins']


def get_categorical_columns():
    """
    Return categorical columns to encode.
    """
    return ['caffeine (0 or 1)', 'diet_today (0 or 1 or 2)', 'diet_yesterday (0 or 1 or 2)', 'exercise_type (cat)']


def get_sleep_length_columns(target):
    """
    Return columns related to sleep length.
    Target could be target_feelings or feeling_score_fitbit_am.
    """
    return ['sleep_length', target]


def get_sleep_length_discretized():
    """
    sleep_length_10  8.25   0.466667
    sleep_length_11  8.50   0.150000
    sleep_length_12  8.75   0.040000
    sleep_length_8   7.75  -0.090000
    sleep_length_3   6.50  -0.125000
    """
    feature = 'sleep_length'
    bins = [5.75, 6, 6.25, 6.5, 6.75, 7, 7.25, 7.5, 7.75, 8, 8.25, 8.5, 8.75, 9]
    return feature, bins


if __name__ == "__main__":
    analyze_health_data()
