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
TARGET_NON_DUALITY = 'non_duality_glimpsed (0 or 1)'
TARGET_SELF_INSIGHT = 'self_insight_obtained (0 or 1)'
TARGET_FUNK = 'feeling_score_funk'
SLEEP_DATASET_START = '2020-09-08'
SLEEP_DATASET_END = '2020-10-27'
MEDITATION_DATASET_START = '2021-09-27'
MEDITATION_DATASET_END = '2021-12-17'
HEALTH_DATASET_START = '2023-08-01'
LATEST_DATA_END = '2023-12-31'
CUSTOM_DAYS_BEFORE = 90  # TODO: implement


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

    print("--SLEEP QUALITY PILLAR--")
    print()
    analyze_sleep(TARGET_WAKEUP, HEALTH_DATASET_START, LATEST_DATA_END)

    print("--MEDITATION QUALITY PILLAR--")
    print()
    analyze_meditation(TARGET_NON_DUALITY, TARGET_SELF_INSIGHT, LATEST_DATA_END)

    print_legend()


def analyze_all_features(target, date_to_start, date_to_end):
    """
    Latest total analysis:
    1. sleep_wakeup_diff
    2. sleep_bedtime_diff INVERSE
    3. sleep_score

    Best alpha: 0.5
    Best RMSE (lower is better): 0.34262807680830715
    R-squared score (higher is better): -0.31079255502771974
    """
    data = CsvDataset(HEALTH_CSV)

    # set dates
    data.drop_index_before_date(date_to_start)
    data.drop_index_after_date(date_to_end)

    # set columns
    data.drop_columns(get_columns_to_drop_with_meditation())
    data.encode_categorical_columns(get_categorical_columns())

    # run model
    data.split_features_and_target(target)
    coefficients, feature_names = run_model(data.X, data.y)

    print_results(coefficients, feature_names)


def analyze_sleep_length(target, date_to_start, date_to_end):
    """
    Latest sleep length analysis:
    1. sleep_length_7.25      = BEST
    2. sleep_length_5.75-6.50 = WORST

    Best alpha: 0.01
    Best RMSE (lower is better): 0.5814541666705516
    R-squared score (higher is better): 0.07092727465969717

    Avg sleep awake mins: 42.5
    """
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

    feature_names = label_discretized_features(feature_names, feature, bins)
    print_results(coefficients, feature_names)


def analyze_sleep(target, date_to_start, date_to_end):
    """
    Latest sleep analysis:
    1. sleep_score
    2. diet_prev_week INVERSE ???
    3. exercise_rut ???

    Best alpha: 0.1
    Best RMSE (lower is better): 0.5466253000647345
    R-squared score (higher is better): -0.06447544036724828
    """
    data = CsvDataset(HEALTH_CSV)

    # set dates
    data.drop_index_before_date(date_to_start)
    data.drop_index_after_date(date_to_end)

    # set columns
    columns_to_keep = get_all_sleep_columns(target)
    columns_to_drop = get_excess_cols(data.df, columns_to_keep)
    data.drop_columns(columns_to_drop)
    data.encode_categorical_columns(get_past_categorical_columns())

    # run model
    data.split_features_and_target(target)
    coefficients, feature_names = run_model(data.X, data.y)

    print_results(coefficients, feature_names)


def analyze_meditation(target_1, target_2, date_to_end):
    """
    Latest meditation analysis:
    1. active_zone_mins_today
    2. sleep_bedtime_diff INVERSE
    3. meditation_mins_prev_week INVERSE ???

    Best alpha: 1
    Best RMSE (lower is better): 0.6135820247822891
    R-squared score (higher is better): -0.1865736268030005
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
    coefficients, feature_names = run_model(data.X, data.y)

    print_results(coefficients, feature_names)


def get_columns_to_drop():
    """
    Return columns to drop from the dataset.
    Includes columns used to derive target or other features.
    """
    return ['non_duality_glimpsed (0 or 1)', 'self_insight_obtained (0 or 1)', 'active_zone_mins_weekly_goal',
            'steps_daily_goal', 'feeling_score_fitbit_am', 'feeling_score_fitbit_pm',
            'satisfaction_with_life_as_whole (0-4)', 'inability_to_cope_with_responsibilities (0-4)',
            'sleep_awake_mins', 'feeling_score_funk']


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
            'meditation_mins_prev_week', 'meditation_rut', 'exercise_count_prev_week', 'active_zone_mins_yesterday',
            'active_zone_mins_prev_week', 'exercise_rut', 'sleep_bedtime_diff (21:45-22:20-23:05)',
            'sleep_bedtime_prev_week', 'sleep_wakeup_diff (5:45-6:15-6:45)', 'sleep_length', 'sleep_score',
            'sleep_awake_mins', 'sleep_rut', 'steps_yesterday', 'steps_prev_week', target]


def get_sleep_length_discretized():
    """
    Return sleep length in discretized bins.
    """
    feature = 'sleep_length'
    bins = [5.75, 6, 6.25, 6.5, 6.75, 7, 7.25, 7.5, 7.75, 8, 8.25, 8.5, 8.75, 9]
    return feature, bins


def label_discretized_features(feature_names, feature, bins):
    """
    Updates the discretized feature label with actual bin values
    """
    updated_names = []
    for name in feature_names:
        if name.startswith(feature):
            index = int(name.split('_')[-1])
            bin_value = bins[index]
            name = name.replace(str(index), str(bin_value))
        updated_names.append(name)
    return updated_names


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
