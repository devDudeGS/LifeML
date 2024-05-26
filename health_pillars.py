from csv_dataset import CsvDataset, run_model, print_results, print_results_discretized, get_excess_cols, print_legend

"""
Analyzes health data from:
2020-09-08 to 2020-10-27,
2021-09-27 to 2021-12-17,
2023-08-01 to present,
to determine which features are correlated with happiness scores.

Used GitHub Copilot and Claude 2.0 to help write the initial logic.
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
LATEST_DATA_END = '2024-04-30'
MEDIAN_ERROR_ALL = 0.337845
MEDIAN_ERROR_SLEEP_LENGTH = 0.569224
MEDIAN_ERROR_SLEEP_QUALITY = 0.533276
MEDIAN_ERROR_MEDITATION = 0.543210
CUSTOM_DAYS_BEFORE = 90


def analyze_health_data():
    print()
    print("Health Pillars Data Analysis")
    print()

    print("--ALL FEATURES--")
    print()
    analyze_all_features(TARGET_MAIN, HEALTH_DATASET_START, LATEST_DATA_END)

    print("--SLEEP LENGTH--")
    print()
    analyze_sleep_length(TARGET_WAKEUP, HEALTH_DATASET_START, LATEST_DATA_END)

    # TODO: determine how to return useful data
    # print("--SLEEP QUALITY PILLAR--")
    # print()
    # analyze_sleep(TARGET_WAKEUP, HEALTH_DATASET_START, LATEST_DATA_END)

    print("--MEDITATION QUALITY PILLAR--")
    print()
    analyze_meditation(TARGET_NON_DUALITY, TARGET_SELF_INSIGHT, LATEST_DATA_END)

    print_legend()


def analyze_all_features(target, date_to_start, date_to_end):
    """
    Analyzes the target using all features.
    """
    data = CsvDataset(HEALTH_CSV)

    # set dates
    data.drop_index_after_date(date_to_end)
    data.drop_index_before_date(date_to_start)

    # set columns
    data.drop_columns(get_columns_to_drop_with_meditation())
    data.encode_categorical_columns(get_categorical_columns())

    # run model
    data.split_features_and_target(target)
    model, best_rmse = run_model(data.X, data.y)

    print_results(model, best_rmse, MEDIAN_ERROR_ALL, data.X)


def analyze_sleep_length(target, date_to_start, date_to_end):
    """
    Analyzes the target using sleep length.
    """
    data = CsvDataset(HEALTH_CSV)

    # set dates
    data.drop_index_after_date(date_to_end)
    data.drop_index_before_date(date_to_start)

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
    model, best_rmse = run_model(data.X, data.y)

    print_results_discretized(model, best_rmse, MEDIAN_ERROR_SLEEP_LENGTH, data.X, feature, bins)


def analyze_sleep(target, date_to_start, date_to_end):
    """
    Analyzes sleep target using all features.
    """
    data = CsvDataset(HEALTH_CSV)

    # set dates
    data.drop_index_after_date(date_to_end)
    data.drop_index_before_date(date_to_start)

    # set columns
    columns_to_keep = get_all_sleep_columns(target)
    columns_to_drop = get_excess_cols(data.df, columns_to_keep)
    data.drop_columns(columns_to_drop)
    data.encode_categorical_columns(get_past_categorical_columns())

    # run model
    data.split_features_and_target(target)
    model, best_rmse = run_model(data.X, data.y)

    print_results(model, best_rmse, MEDIAN_ERROR_SLEEP_QUALITY, data.X)


def analyze_meditation(target_1, target_2, date_to_end):
    """
    Analyzes meditation targets using all features.
    """
    data = CsvDataset(HEALTH_CSV)

    # set rows
    data.drop_index_after_date(date_to_end)
    targets = [target_1, target_2]
    data.drop_rows_without_column_values(targets)

    # set columns
    target = "target_meditation"
    data.insert_column_from_column_additions(targets, target)
    data.drop_columns(get_columns_to_drop())
    data.encode_categorical_columns(get_categorical_columns())
    feature, category_prefix = get_meditation_type_feature()
    data.breakout_categorical_codes(feature, category_prefix)

    # run model
    data.split_features_and_target(target)
    model, best_rmse = run_model(data.X, data.y)

    print_results(model, best_rmse, MEDIAN_ERROR_MEDITATION, data.X)


def get_columns_to_drop():
    """
    Return columns to drop from the dataset.
    Includes columns used to derive target or other features.
    """
    return ['non_duality_glimpsed (0 or 1)', 'self_insight_obtained (0 or 1)', 'heart_rate_mins_weekly_goal',
            'steps_daily_goal', 'feeling_score_fitbit_am', 'feeling_score_fitbit_pm',
            'satisfaction_with_life_as_whole (0-4)', 'inability_to_cope_with_responsibilities (0-4)',
            'sleep_awake_mins', 'feeling_score_funk', 'stress_mgmt_score_fitbit', 'sleep_score']


def get_columns_to_drop_with_meditation():
    """
    Return columns to drop from the dataset, plus meditation.
    """
    columns_to_drop = get_columns_to_drop()
    columns_to_drop.append('meditation_type (cat)')
    return columns_to_drop


def get_categorical_columns():
    """
    Return categorical columns to encode.
    """
    return ['caffeine (0 or 1)', 'diet_today (0 or 1 or 2)', 'diet_yesterday (0 or 1 or 2)', 'exercise_type (cat)']


def get_past_categorical_columns():
    """
    Return categorical columns from the past to encode.
    """
    return ['diet_yesterday (0 or 1 or 2)']


def get_sleep_length_columns(target):
    """
    Return columns related to sleep length.
    Target could be target_feelings or feeling_score_fitbit_am.
    """
    return ['sleep_length', target]


def get_all_sleep_columns(target):
    """
    Return columns potentially related to sleep quality.
    Target is feeling_score_fitbit_am.
    """
    return ['diet_yesterday (0 or 1 or 2)', 'diet_prev_week', 'diet_rut', 'meditation_consecutive',
            'meditation_mins_prev_week', 'meditation_rut', 'exercise_count_prev_week', 'heart_rate_mins_yesterday',
            'heart_rate_mins_prev_week', 'exercise_rut', 'sleep_bedtime_diff (21:45-22:20-23:05)',
            'sleep_bedtime_prev_week', 'sleep_wakeup_diff (5:45-6:15-6:45)', 'sleep_length', 'sleep_score',
            'sleep_awake_mins', 'sleep_rut', 'steps_yesterday', 'steps_prev_week', target]


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
