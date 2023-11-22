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

Latest important features:
1. diet_yesterday (0 or 1 or 2)_1.0
2. exercise_type (cat)_0.0 INVERSE
3. diet_today (0 or 1 or 2)_0.0 INVERSE

Latest sleep conclusions:
1. sleep_length_8   7.75      = BEST
2. sleep_length_0-3 5.75-6.50 = WORST

Avg sleep awake mins: 52.7
"""

HEALTH_CSV = 'data/health_pillars.csv'
TARGET_MAIN = 'target_feelings'
TARGET_WAKEUP = 'feeling_score_fitbit_am'
TARGET_NON_DUALITY = 'non_duality_glimpsed (0 or 1)'
TARGET_SELF_INSIGHT = 'self_insight_obtained (0 or 1)'
SLEEP_DATASET_START = '2020-09-08'
SLEEP_DATASET_END = '2020-10-27'
MEDITATION_DATASET_START = '2021-09-27'
MEDITATION_DATASET_END = '2021-12-17'
HEALTH_DATASET_START = '2023-08-01'
LATEST_DATA_END = '2023-11-18'


def analyze_health_data():
    print()
    print("Health Pillars Data Analysis")
    print()

    print("--ALL FEATURES--")
    print()
    analyze_all_features(TARGET_MAIN, SLEEP_DATASET_START, LATEST_DATA_END)

    print("--SLEEP LENGTH--")
    print()
    analyze_sleep_length(TARGET_WAKEUP, HEALTH_DATASET_START, LATEST_DATA_END)

    print("--MEDITATION--")
    print()
    analyze_meditation(TARGET_NON_DUALITY, TARGET_SELF_INSIGHT)

    print_legend()


def analyze_all_features(target, date_to_start, date_to_end):
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
    data = CsvDataset(HEALTH_CSV)

    # set dates
    data.drop_index_before_date(date_to_start)
    data.drop_index_after_date(date_to_end)

    # get avg awake mins
    data.get_avg_in_column('sleep_awake_mins', 30)

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


def analyze_meditation(target_1, target_2):
    data = CsvDataset(HEALTH_CSV)

    # set rows
    targets = [target_1, target_2]
    data.drop_rows_without_column_values(targets)

    # set columns
    target = "target_meditation"
    data.insert_column_from_column_additions(targets, target)
    columns_to_keep = get_meditation_columns(target)
    columns_to_drop = get_excess_cols(data.df, columns_to_keep)
    data.drop_columns(columns_to_drop)
    feature, category_prefix = get_meditation_type_feature()
    data.breakout_categorical_codes(feature, category_prefix)

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
    Return sleep length in discretized bins.
    """
    feature = 'sleep_length'
    bins = [5.75, 6, 6.25, 6.5, 6.75, 7, 7.25, 7.5, 7.75, 8, 8.25, 8.5, 8.75, 9]
    return feature, bins


def get_meditation_columns(target):
    """
    Return columns related to meditation.
    """
    return ['meditation_first_time', 'meditation_total_mins', 'meditation_consecutive', 'meditation_mins_prev_week',
            'meditation_type (cat)', target]


def get_meditation_type_feature():
    """
    Return meditation type feature, and prefix for breakout columns.
    """
    feature = 'meditation_type (cat)'
    category_prefix = 'meditation_type_'
    return feature, category_prefix


if __name__ == "__main__":
    analyze_health_data()
