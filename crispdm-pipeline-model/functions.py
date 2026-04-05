"""
functions.py — Reusable data science utility functions
Built from Chapters 6–18 of Predictive AI ML

Chapter 6: Automating Feature-Level Exploration
  - histogram()
  - boxplot()
  - countplot()
  - univariate_viz()
  - unistats()

Chapter 7: Automating Data Preparation Pipelines
  - basic_wrangling()
  - parse_date()
  - manage_dates()
  - bin_categories()
  - skew_correct()
  - missing_drop()
  - missing_fill()
  - clean_outlier()
  - clean_outliers()

Chapter 8: Automating Relationship Discovery
  - bivariate_stats()
  - scatterplot()
  - bar_chart()
  - crosstab()
  - bivariate()

Chapter 9: Regression
  - fit_regression()
  - regression_summary()

Chapter 10: MLR Diagnostics for Causal Inference
  - regression_diagnostics()
  - assumption_checks()
  - diagnostic_model()

Chapter 11: MLR for Predictive Inference
  - holdout_split()
  - predict_and_evaluate()

Chapter 12: Decision Trees for Predictive Regression
  - fit_tree()
  - tree_summary()
  - tree_depth_sweep()
  - extract_tree_rules()

Chapter 13: Classification Modeling
  - fit_classifier()
  - classification_report_custom()
  - compare_classifiers()
  - threshold_analysis()

Chapter 14: Ensemble Methods
  - fit_ensemble()
  - compare_models()
  - ensemble_feature_importance()
  - ensemble_depth_sweep()

Chapter 15: Model Evaluation, Selection & Tuning
  - algorithm_selector()
  - _get_estimator_map()
  - model_comparison_report()
  - learning_curve_report()
  - tuning_pipeline()
  - _get_param_spaces()

Chapter 16: Feature Selection
  - feature_selector()
  - feature_importance_report()

Chapter 17: Deploying ML Pipelines
  - save_model()
  - load_and_predict()

Chapter 18: Monitoring and Managing ML Pipelines
  - monitor_drift()
  - performance_tracker()
"""


# =============================================================================
# Chapter 6: Automating Feature-Level Exploration
# =============================================================================

def histogram(df, col):
  """
  Draw a histogram with KDE overlay for a single numeric column.

  Parameters
  ----------
  df : pandas.DataFrame
      Input DataFrame.
  col : str
      Name of the numeric column to plot.
  """
  import matplotlib.pyplot as plt
  import seaborn as sns
  plt.figure(figsize=(10, 5))
  sns.histplot(data=df, x=col, kde=True, color="orange")
  plt.title(f'Distribution of {col}')
  plt.ylabel('Frequency')
  plt.xlabel(col)
  sns.despine()
  plt.tight_layout()
  plt.show()


def boxplot(df, col):
  """
  Draw a compact horizontal box plot for a single numeric column.

  Parameters
  ----------
  df : pandas.DataFrame
      Input DataFrame.
  col : str
      Name of the numeric column to plot.
  """
  import matplotlib.pyplot as plt
  import seaborn as sns
  plt.figure(figsize=(10, 2))
  sns.set_style('ticks')
  flierprops = dict(marker='o', markersize=4, markerfacecolor='none',
                    linestyle='none', markeredgecolor='gray')
  sns.boxplot(data=df, x=col, fliersize=4, saturation=0.50,
              width=0.50, linewidth=0.5, flierprops=flierprops)
  plt.title(f'Box Plot for {col}')
  plt.yticks([])
  sns.despine(left=True)
  plt.tight_layout()
  plt.show()


def countplot(df, col):
  """
  Draw a count plot with percentage labels for a single column.

  Parameters
  ----------
  df : pandas.DataFrame
      Input DataFrame.
  col : str
      Name of the column to plot (typically categorical or boolean 0/1).
  """
  import matplotlib.pyplot as plt
  import seaborn as sns
  plt.figure(figsize=(10, 6))
  ax = sns.countplot(data=df, x=col)
  plt.title(f'Count Plot for {col}')
  plt.xlabel(col)
  plt.ylabel('Count')
  plt.xticks(rotation=45, ha='right')
  total = len(df[col].dropna())
  for p in ax.patches:
    height = p.get_height()
    percentage = (height / total) * 100
    ax.text(p.get_x() + p.get_width() / 2., height,
            f'{percentage:.1f}%', ha='center', va='bottom')
  plt.tight_layout()
  plt.show()


def univariate_viz(df, col, stacked=True):
  """
  Dispatcher: choose the right chart for a single column based on dtype.

  Numeric columns (non-boolean) get a stacked boxplot + histogram (default)
  or separate boxplot and histogram when stacked=False. Boolean 0/1 columns
  and categorical columns get a count plot with percentage labels.

  Parameters
  ----------
  df : pandas.DataFrame
      Input DataFrame.
  col : str
      Name of the column to visualize.
  stacked : bool
      If True (default), numeric columns display a combined boxplot-over-histogram.
      If False, boxplot and histogram are drawn as separate figures.
  """
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns

  if pd.api.types.is_numeric_dtype(df[col]):
    unique_vals = set(df[col].dropna().unique())
    is_boolean = unique_vals.issubset({0, 1})

    if not is_boolean:
      if stacked:
        f, (ax_box, ax_hist) = plt.subplots(
          2, sharex=True, figsize=(10, 6),
          gridspec_kw={"height_ratios": (.15, .85)}
        )
        sns.set_style('ticks')
        flierprops = dict(marker='o', markersize=4, markerfacecolor='none',
                          linestyle='none', markeredgecolor='gray')
        sns.boxplot(data=df, x=col, ax=ax_box, fliersize=4, saturation=0.50,
                    width=0.50, linewidth=0.5, flierprops=flierprops)
        sns.histplot(data=df, x=col, ax=ax_hist, kde=True, color="orange")
        ax_box.set(yticks=[], xticks=[])
        ax_box.set_xlabel('')
        ax_box.set_ylabel('')
        ax_hist.set_ylabel('Frequency')
        ax_hist.set_xlabel(col)
        plt.suptitle(f'Box Plot and Distribution for {col}', y=1.02)
        sns.despine(ax=ax_hist)
        sns.despine(ax=ax_box, left=True, bottom=True)
        plt.tight_layout()
        plt.show()
      else:
        boxplot(df, col)
        histogram(df, col)
    else:
      countplot(df, col)
  else:
    countplot(df, col)


def unistats(df, viz=True):
  """
  Generate univariate statistics (and optional visualizations) for every
  column in a DataFrame.

  For each column, reports: count, unique values, data type. Numeric columns
  additionally get: min, max, quartiles, mean, median, mode, std, skew, kurtosis.
  When viz=True, calls univariate_viz() per column to produce the appropriate chart.

  Parameters
  ----------
  df : pandas.DataFrame
      Input DataFrame to analyze.
  viz : bool
      If True (default), produce a chart for each column via univariate_viz().

  Returns
  -------
  pandas.DataFrame
      Summary table with one row per column and statistics as columns.
  """
  import pandas as pd

  output_df = pd.DataFrame(columns=[
    'Count', 'Unique', 'Type',
    'Min', 'Max', '25%', '50%', '75%',
    'Mean', 'Median', 'Mode', 'Std', 'Skew', 'Kurt'
  ])

  for col in df.columns:
    count = df[col].count()
    unique = df[col].nunique()
    dtype = str(df[col].dtype)
    # Initialize branch-specific values to placeholders each iteration
    min_val = '-'
    max_val = '-'
    q1 = '-'
    q2 = '-'
    q3 = '-'
    mean_val = '-'
    median_val = '-'
    mode_val = '-'
    std_val = '-'
    skew_val = '-'
    kurt_val = '-'

    if pd.api.types.is_bool_dtype(df[col]):
      series_num = df[col].astype(int)
      min_val = int(series_num.min())
      max_val = int(series_num.max())
      q1 = round(series_num.quantile(0.25), 2)
      q2 = round(series_num.quantile(0.50), 2)
      q3 = round(series_num.quantile(0.75), 2)
      mean_val = round(series_num.mean(), 2)
      median_val = round(series_num.median(), 2)
      mode_series = series_num.mode()
      mode_val = int(mode_series.values[0]) if len(mode_series) > 0 else '-'
      std_val = round(series_num.std(), 2)
      skew_val = round(series_num.skew(), 2)
      kurt_val = round(series_num.kurt(), 2)
    elif pd.api.types.is_numeric_dtype(df[col]):
      min_val = round(df[col].min(), 2)
      max_val = round(df[col].max(), 2)
      q1 = round(df[col].quantile(0.25), 2)
      q2 = round(df[col].quantile(0.50), 2)
      q3 = round(df[col].quantile(0.75), 2)
      mean_val = round(df[col].mean(), 2)
      median_val = round(df[col].median(), 2)
      mode_series = df[col].mode()
      mode_val = round(mode_series.values[0], 2) if len(mode_series) > 0 else '-'
      std_val = round(df[col].std(), 2)
      skew_val = round(df[col].skew(), 2)
      kurt_val = round(df[col].kurt(), 2)

    if viz:
      univariate_viz(df, col)

    output_df.loc[col] = (
      count, unique, dtype,
      min_val, max_val, q1, q2, q3,
      mean_val, median_val, mode_val, std_val, skew_val, kurt_val
    )

  return output_df


# =============================================================================
# Chapter 7: Automating Data Preparation Pipelines
# =============================================================================

def basic_wrangling(df, features=None, missing_threshold=0.95, unique_threshold=0.95, messages=True):
  """
  Perform basic structural data wrangling by removing low-value columns.

  Removes columns that are: (1) mostly missing (above missing_threshold),
  (2) mostly unique values likely to be IDs (above unique_threshold for int/object),
  or (3) constant (only one unique value). Standardizes column names to lowercase
  with underscores replacing spaces.

  Parameters
  ----------
  df : pandas.DataFrame
      Input DataFrame.
  features : list of str or None
      Columns to evaluate. If None, all columns are checked.
  missing_threshold : float
      Proportion of missing values above which a column is dropped.
  unique_threshold : float
      Proportion of unique values above which an int/object column is dropped.
  messages : bool
      If True, print messages about dropped columns.

  Returns
  -------
  pandas.DataFrame
      Cleaned DataFrame (copy of input; original is not modified).
  """
  import pandas as pd

  out = df.copy()

  # Standardize column names
  out.columns = out.columns.str.strip().str.lower().str.replace(' ', '_')

  if features is None:
    features = list(out.columns)
  else:
    # Also standardize the feature names provided
    features = [f.strip().lower().replace(' ', '_') for f in features]

  dropped = []
  for feat in features:
    if feat not in out.columns:
      if messages:
        print(f'Skipping "{feat}": column not found in DataFrame')
      continue

    rows = out.shape[0]
    missing = out[feat].isna().sum()
    unique = out[feat].nunique()

    if missing / rows >= missing_threshold:
      if messages:
        print(f'Dropping {feat}: {missing} missing values out of {rows} ({missing/rows:.2%})')
      out = out.drop(columns=[feat])
      dropped.append(feat)
    elif unique / rows >= unique_threshold and out[feat].dtype in ['int64', 'object']:
      if messages:
        print(f'Dropping {feat}: {unique} unique values out of {rows} ({unique/rows:.2%})')
      out = out.drop(columns=[feat])
      dropped.append(feat)
    elif unique <= 1:
      val = out[feat].unique()[0] if unique == 1 else 'N/A'
      if messages:
        print(f'Dropping {feat}: contains only one unique value ({val})')
      out = out.drop(columns=[feat])
      dropped.append(feat)

  if messages and dropped:
    print(f'\nTotal columns dropped: {len(dropped)}')

  return out


def parse_date(df, features=None, days_since_today=False, drop_date=True, messages=True):
  """
  Parse date columns and extract temporal features (year, month, day, weekday).

  Converts specified columns to datetime, extracts structured features, and
  optionally computes elapsed days from today. Handles invalid dates gracefully.

  Parameters
  ----------
  df : pandas.DataFrame
      Input DataFrame.
  features : list of str or None
      Column names containing date values. If None, attempts auto-detection.
  days_since_today : bool
      If True, add a column with days elapsed from today.
  drop_date : bool
      If True, drop the original date column after extracting features.
  messages : bool
      If True, print status messages.

  Returns
  -------
  pandas.DataFrame
      DataFrame with date features added (copy; original not modified).
  """
  import pandas as pd
  from datetime import datetime as dt

  out = df.copy()

  if features is None:
    # Auto-detect: try to parse object columns as dates
    features = []
    for col in out.columns:
      if out[col].dtype == 'object':
        parsed = pd.to_datetime(out[col], errors='coerce')
        non_null = out[col].notna().sum()
        if non_null > 0 and parsed.notna().sum() / non_null > 0.5:
          features.append(col)
    if messages and features:
      print(f'Auto-detected date columns: {features}')

  for feat in features:
    if feat not in out.columns:
      if messages:
        print(f'{feat} does not exist in the DataFrame. Skipping.')
      continue

    parsed = pd.to_datetime(out[feat], errors='coerce')
    valid_count = parsed.notna().sum()

    if valid_count == 0:
      if messages:
        print(f'{feat}: no valid dates found. Skipping.')
      continue

    out[feat] = parsed
    out[f'{feat}_year'] = parsed.dt.year
    out[f'{feat}_month'] = parsed.dt.month
    out[f'{feat}_day'] = parsed.dt.day
    out[f'{feat}_weekday'] = parsed.dt.day_name()

    # Add hour only if the column has time components
    has_time = (parsed.dt.hour != 0) | (parsed.dt.minute != 0) | (parsed.dt.second != 0)
    if has_time.any():
      out[f'{feat}_hour'] = parsed.dt.hour

    if days_since_today:
      out[f'{feat}_days_until_today'] = (dt.today() - parsed).dt.days

    if drop_date:
      out = out.drop(columns=[feat])

    if messages:
      print(f'{feat}: extracted year, month, day, weekday' +
            (', hour' if has_time.any() else '') +
            (', days_until_today' if days_since_today else '') +
            ('. Original column dropped.' if drop_date else '.'))

  return out


def manage_dates(df, startdate=None, enddate=None, retain_original=False,
                 show_details=True):
  """
  Identify date-like columns, extract temporal features, and optionally compute
  days relative to reference dates or between two date columns.

  For each detected date column, creates: year, month, weekday, day, and hour
  (if time is present). Optionally adds days_from_startdate, days_to_enddate,
  or days_between when startdate/enddate are provided.

  Parameters
  ----------
  df : pandas.DataFrame
      Input DataFrame.
  startdate : str, datetime, or None
      If a column name in df: used with enddate to compute days between columns.
      If a date value (str or datetime): days from this date to each date column.
  enddate : str, datetime, or None
      If a column name in df: used with startdate to compute days between columns.
      If a date value (str or datetime): days from each date column to this date.
  retain_original : bool
      If True, keep the original date columns. If False (default), drop them.
  show_details : bool
      If True (default), print cleaning details. If False, run silently.

  Returns
  -------
  pandas.DataFrame
      DataFrame with date features added (copy; original not modified).

  Examples
  --------
  >>> df = manage_dates(df)  # Extract year, month, day, weekday, hour
  >>> df = manage_dates(df, enddate='2024-12-31')  # Days to enddate
  >>> df = manage_dates(df, startdate='order_date', enddate='ship_date')  # Days between columns
  """
  import pandas as pd

  out = df.copy()

  # Auto-detect date-like columns
  date_cols = []
  for col in out.columns:
    if out[col].dtype == 'object':
      parsed = pd.to_datetime(out[col], errors='coerce')
      non_null = out[col].notna().sum()
      if non_null > 0 and parsed.notna().sum() / non_null > 0.5:
        date_cols.append(col)
    elif pd.api.types.is_datetime64_any_dtype(out[col]):
      date_cols.append(col)

  if not date_cols:
    if show_details:
      print('No date-like columns detected.')
    return out

  if show_details:
    print(f'Date columns detected: {date_cols}')

  # Check if startdate and enddate are both column names in the dataset
  start_is_col = isinstance(startdate, str) and startdate in out.columns
  end_is_col = isinstance(enddate, str) and enddate in out.columns
  both_are_cols = start_is_col and end_is_col

  if both_are_cols:
    # Compute days between the two date columns
    start_parsed = pd.to_datetime(out[startdate], errors='coerce')
    end_parsed = pd.to_datetime(out[enddate], errors='coerce')
    out['days_between'] = (end_parsed - start_parsed).dt.days
    if show_details:
      print(f'Added days_between ({startdate} → {enddate}).')

  # Parse reference dates for value-based calculations
  start_val = None
  end_val = None
  if startdate is not None and not start_is_col:
    start_val = pd.to_datetime(startdate, errors='coerce')
    if pd.isna(start_val):
      if show_details:
        print(f'startdate "{startdate}" could not be parsed as a date. Skipping.')
  if enddate is not None and not end_is_col:
    end_val = pd.to_datetime(enddate, errors='coerce')
    if pd.isna(end_val):
      if show_details:
        print(f'enddate "{enddate}" could not be parsed as a date. Skipping.')

  for col in date_cols:
    parsed = pd.to_datetime(out[col], errors='coerce')
    valid_count = parsed.notna().sum()

    if valid_count == 0:
      if show_details:
        print(f'  {col}: no valid dates found. Skipping.')
      continue

    # Extract temporal features
    out[f'{col}_year'] = parsed.dt.year
    out[f'{col}_month'] = parsed.dt.month
    out[f'{col}_weekday'] = parsed.dt.day_name()
    out[f'{col}_day'] = parsed.dt.day

    has_time = (parsed.dt.hour != 0) | (parsed.dt.minute != 0) | (parsed.dt.second != 0)
    if has_time.any():
      out[f'{col}_hour'] = parsed.dt.hour

    # Days from startdate (value) to each date
    if start_val is not None:
      out[f'{col}_days_from_startdate'] = (parsed - start_val).dt.days

    # Days from each date to enddate (value)
    if end_val is not None:
      out[f'{col}_days_to_enddate'] = (end_val - parsed).dt.days

    if not retain_original:
      out = out.drop(columns=[col])

    extras = []
    if has_time.any():
      extras.append('hour')
    if start_val is not None:
      extras.append('days_from_startdate')
    if end_val is not None:
      extras.append('days_to_enddate')

    if show_details:
      msg = f'  {col}: year, month, weekday, day'
      if extras:
        msg += ', ' + ', '.join(extras)
      msg += '.' + (' Original dropped.' if not retain_original else ' Original retained.')
      print(msg)

  return out


def bin_categories(df, features=None, cutoff=0.05, min_count=15,
                   replace_with='Other', drop_below_threshold_other=False, messages=True):
  """
  Bin low-frequency categorical values into a single group.

  For each specified categorical column, values that fall below BOTH the
  percentage cutoff AND the minimum count threshold are replaced with
  replace_with (default 'Other').

  Parameters
  ----------
  df : pandas.DataFrame
      Input DataFrame.
  features : list of str or None
      Columns to process. If None, all object-type columns are processed.
  cutoff : float
      Minimum proportion of rows a category must represent to avoid binning.
  min_count : int
      Minimum absolute count a category must have to avoid binning.
  replace_with : str
      Label to assign to binned categories.
  drop_below_threshold_other : bool
      If True, drop rows where 'Other' still doesn't meet thresholds.
  messages : bool
      If True, print binning actions.

  Returns
  -------
  pandas.DataFrame
      DataFrame with rare categories binned (copy; original not modified).
  """
  import pandas as pd

  out = df.copy()

  if features is None:
    features = [c for c in out.columns if out[c].dtype == 'object']

  for feat in features:
    if feat not in out.columns:
      if messages:
        print(f'{feat} not found in DataFrame. Skipping.')
      continue
    if pd.api.types.is_numeric_dtype(out[feat]):
      continue

    n_total = len(out)
    value_counts = out[feat].value_counts()
    to_bin = []

    for val, count in value_counts.items():
      pct = count / n_total
      # Bin only if BOTH thresholds are violated
      if count < min_count and pct < cutoff:
        to_bin.append(val)

    if to_bin:
      out[feat] = out[feat].replace(to_bin, replace_with)
      if messages:
        print(f'{feat}: binned {len(to_bin)} categories into "{replace_with}"')

    # Optionally drop rows where Other is still too small
    if drop_below_threshold_other and replace_with in out[feat].values:
      other_count = (out[feat] == replace_with).sum()
      other_pct = other_count / len(out)
      if other_count < min_count and other_pct < cutoff:
        out = out[out[feat] != replace_with]
        if messages:
          print(f'{feat}: dropped {other_count} "{replace_with}" rows (below thresholds)')

  return out


def skew_correct(df, feature, methods=None, messages=True, visualize=True):
  """
  Reduce skewness of a numeric feature by trying a menu of transformations
  and keeping the one that brings skewness closest to zero.

  Candidates: none, cbrt, sqrt, log1p, yeojohnson.

  Parameters
  ----------
  df : pandas.DataFrame
      Input DataFrame.
  feature : str
      Name of the numeric column to transform.
  methods : list of str or None
      Transformation candidates. Default: ["none", "cbrt", "sqrt", "log1p", "yeojohnson"].
  messages : bool
      If True, print a transformation report.
  visualize : bool
      If True, show before/after histograms.

  Returns
  -------
  pandas.DataFrame
      DataFrame with a new column named '{feature}_skewfix' (copy; original not modified).
  """
  import pandas as pd
  import numpy as np

  if methods is None:
    methods = ["none", "cbrt", "sqrt", "log1p", "yeojohnson"]

  if feature not in df.columns:
    if messages:
      print(f"{feature} is not found in the DataFrame. No transformation performed.")
    return df.copy()

  x = pd.to_numeric(df[feature], errors="coerce")
  if x.notna().sum() == 0:
    if messages:
      print(f"{feature} could not be converted to numeric. No transformation performed.")
    return df.copy()

  out = df.copy()

  # Shift negatives for transforms that need non-negative values
  min_val = x.min(skipna=True)
  shift = -float(min_val) if (not pd.isna(min_val) and min_val < 0) else 0.0
  x_shifted = x + shift

  candidates = {}
  candidates["none"] = x.astype("float64")

  if "cbrt" in methods:
    candidates["cbrt"] = np.cbrt(x_shifted.clip(lower=0)).astype("float64")

  if "sqrt" in methods:
    candidates["sqrt"] = np.sqrt(x_shifted.clip(lower=0)).astype("float64")

  if "log1p" in methods:
    candidates["log1p"] = np.log1p(x_shifted.clip(lower=0)).astype("float64")

  if "yeojohnson" in methods:
    try:
      from scipy.stats import yeojohnson
      x_nonmissing = x.dropna().to_numpy(dtype="float64")
      yj_vals, yj_lambda = yeojohnson(x_nonmissing)
      yj_series = x.astype("float64").copy()
      yj_series.loc[x.dropna().index] = yj_vals
      candidates["yeojohnson"] = yj_series
    except Exception:
      if messages:
        print("Yeo-Johnson failed or scipy not available. Skipping.")

  # Select the candidate closest to zero skewness
  best_name = "none"
  best_series = candidates["none"]
  best_score = abs(candidates["none"].skew(skipna=True))

  for name in methods:
    if name not in candidates:
      continue
    sk = candidates[name].skew(skipna=True)
    score = abs(sk) if not pd.isna(sk) else np.inf
    if score < best_score:
      best_score = score
      best_name = name
      best_series = candidates[name]

  new_col = f"{feature}_skewfix"
  out[new_col] = best_series.astype("float64")

  if messages:
    before = x.skew(skipna=True)
    after = out[new_col].skew(skipna=True)
    print(f"Feature: {feature}")
    print(f"Skew before: {round(before, 5)}")
    print(f"Chosen method: {best_name}")
    if best_name in ["cbrt", "sqrt", "log1p"] and shift > 0:
      print(f"Shift used (to handle negatives): {round(shift, 5)}")
    print(f"Skew after: {round(after, 5)}")
    print(f"New column: {new_col}")

  if visualize:
    import seaborn as sns
    import matplotlib.pyplot as plt

    df_temp = pd.DataFrame({feature: x.astype("float64"), "transformed": out[new_col].astype("float64")})
    f, axes = plt.subplots(1, 2, figsize=[7, 3.5])
    sns.despine(left=True)
    sns.histplot(df_temp[feature].dropna(), ax=axes[0], kde=True)
    axes[0].set_title("Before")
    sns.histplot(df_temp["transformed"].dropna(), ax=axes[1], kde=True)
    axes[1].set_title("After")
    plt.setp(axes, yticks=[])
    plt.tight_layout()
    plt.show()

  return out


def missing_drop(df, label="", features=None, messages=True, row_threshold=0.9, col_threshold=0.5):
  """
  Drop rows and columns with excessive missing data while retaining as much
  non-null data as possible.

  Strategy: (1) Drop columns above col_threshold. (2) Drop rows above row_threshold.
  (3) Protect the label column. (4) Iteratively drop the column or set of rows
  that preserves the most non-null cells until no missing values remain.

  Parameters
  ----------
  df : pandas.DataFrame
      Input DataFrame.
  label : str
      Target column name. If provided, rows missing this column are dropped.
  features : list of str or None
      Columns to consider. If None, all columns are used.
  messages : bool
      If True, print progress.
  row_threshold : float
      Minimum proportion of non-missing values to keep a row.
  col_threshold : float
      Minimum proportion of non-missing values to keep a column.

  Returns
  -------
  pandas.DataFrame
      Cleaned DataFrame (copy; original not modified).
  """
  import pandas as pd

  out = df.copy()
  start_count = out.count().sum()

  # Drop columns with too much missing
  out = out.dropna(axis=1, thresh=round(col_threshold * out.shape[0]))

  # Drop rows with too much missing
  out = out.dropna(axis=0, thresh=round(row_threshold * out.shape[1]))

  # Protect label column
  if label and label in out.columns:
    out = out.dropna(axis=0, subset=[label])

  def generate_missing_table():
    results = pd.DataFrame(columns=['Missing', 'column', 'rows'])
    for feat in out.columns:
      missing = out[feat].isna().sum()
      if missing > 0:
        memory_col = out.drop(columns=[feat]).count().sum()
        memory_rows = out.dropna(subset=[feat]).count().sum()
        results.loc[feat] = [missing, memory_col, memory_rows]
    return results

  results = generate_missing_table()

  while results.shape[0] > 0:
    best = results[['column', 'rows']].max(axis=1).iloc[0]
    max_axis = results.columns[results.isin([best]).any()][0]

    if messages:
      print(f'{int(best)} {max_axis}')

    results = results.sort_values(by=[max_axis], ascending=False)

    if messages:
      print('\n', results)

    if max_axis == 'rows':
      out = out.dropna(axis=0, subset=[results.index[0]])
    else:
      out = out.drop(columns=[results.index[0]])

    results = generate_missing_table()

  if messages:
    end_count = out.count().sum()
    pct = round(end_count / start_count * 100, 2) if start_count > 0 else 0
    print(f'{pct}% ({end_count}) / ({start_count}) of non-null cells were kept.')

  return out


def missing_fill(df, label, features=None, row_threshold=0.9, col_threshold=0.5,
                 acceptable=0.1, mar='drop', force_impute=False, large_dataset=200000, messages=True):
  """
  Handle missing data by first dropping extreme cases, then testing for
  missing-data bias, and finally imputing remaining values.

  For numeric labels: uses t-tests to detect MAR (Missing at Random).
  For categorical labels: uses proportion z-tests.
  If bias is detected (MAR), either drops rows or fills with median/mode.
  If no bias (MCAR), uses KNNImputer (large datasets) or IterativeImputer.

  Parameters
  ----------
  df : pandas.DataFrame
      Input DataFrame.
  label : str
      Target column name (required for bias testing).
  features : list of str or None
      Columns to check. If None, all columns are used.
  row_threshold : float
      Minimum proportion of non-missing values to keep a row.
  col_threshold : float
      Minimum proportion of non-missing values to keep a column.
  acceptable : float
      Max proportion of significant tests before treating as MAR.
  mar : str
      How to handle MAR data: 'drop' or 'impute'.
  force_impute : bool
      If True, force ML-based imputation even when MAR is detected.
  large_dataset : int
      Row count threshold above which KNNImputer is used instead of IterativeImputer.
  messages : bool
      If True, print progress and display diagnostic tables.

  Returns
  -------
  pandas.DataFrame
      Cleaned DataFrame (copy; original not modified).
  """
  import pandas as pd
  import numpy as np
  from scipy import stats

  out = df.copy()

  if label not in out.columns:
    print(f'The label provided ({label}) does not exist in the DataFrame.')
    return out

  start_count = out.count().sum()

  # Drop columns and rows with too much missing
  out = out.dropna(axis=1, thresh=round(col_threshold * out.shape[0]))
  out = out.dropna(axis=0, thresh=round(row_threshold * out.shape[1]))

  if label in out.columns:
    out = out.dropna(axis=0, subset=[label])

  if features is None:
    features = list(out.columns)

  # No missing data remaining? Return early.
  if not out.isna().any().any():
    if messages:
      print('No missing values remaining after threshold drops.')
    return out

  # Test for missing-data bias
  if pd.api.types.is_numeric_dtype(out[label]):
    df_results = pd.DataFrame(columns=['total missing', 'null mean', 'non-null mean',
                                       'null std', 'non-null std', 't', 'p'])
    for feat in features:
      if feat == label or feat not in out.columns:
        continue
      missing = out[feat].isna().sum()
      if missing > 0:
        null_group = out[out[feat].isna()]
        nonnull_group = out[~out[feat].isna()]
        if len(null_group) >= 2 and len(nonnull_group) >= 2:
          t, p = stats.ttest_ind(null_group[label], nonnull_group[label], nan_policy='omit')
          df_results.loc[feat] = [
            int(missing), round(null_group[label].mean(), 6), round(nonnull_group[label].mean(), 6),
            round(null_group[label].std(), 6), round(nonnull_group[label].std(), 6), t, p
          ]
  else:
    from statsmodels.stats.proportion import proportions_ztest

    df_results = pd.DataFrame(columns=['total missing', 'null prop', 'non-null prop', 'Z', 'p'])
    for feat in features:
      if feat == label or feat not in out.columns:
        continue
      missing = out[feat].isna().sum()
      if missing > 0:
        null_group = out[out[feat].isna()]
        nonnull_group = out[~out[feat].isna()]
        for group in null_group[label].unique():
          p1_num = null_group[null_group[label] == group].shape[0]
          p1_den = null_group[null_group[label] != group].shape[0]
          p2_num = nonnull_group[nonnull_group[label] == group].shape[0]
          p2_den = nonnull_group[nonnull_group[label] != group].shape[0]
          if p1_den > 0 and p2_den > 0 and p1_num < p1_den:
            numerators = np.array([p1_num, p2_num])
            denominators = np.array([p1_den, p2_den])
            try:
              z, p = proportions_ztest(numerators, denominators)
              df_results.loc[f'{feat}_{group}'] = [
                int(missing), round(p1_num / p1_den, 6), round(p2_num / p2_den, 6), z, p
              ]
            except Exception:
              pass

  if messages and len(df_results) > 0:
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    from IPython.display import display
    display(df_results)

  # Decide: MAR or MCAR?
  if len(df_results) > 0:
    sig_proportion = df_results[df_results['p'] < 0.05].shape[0] / df_results.shape[0]
  else:
    sig_proportion = 0

  if sig_proportion > acceptable and not force_impute:
    # Treat as MAR
    if mar == 'drop':
      out = out.dropna()
      if messages:
        print('Missing data appears biased (MAR). Null rows dropped.')
    else:
      for feat in df_results.index:
        col_name = feat.split('_')[0] if '_' in feat else feat
        if col_name in out.columns:
          if pd.api.types.is_numeric_dtype(out[col_name]):
            fill_val = out[col_name].median()
            out[col_name] = out[col_name].fillna(fill_val)
            if messages:
              print(f'{col_name} filled with median ({fill_val})')
          else:
            out[col_name] = out[col_name].fillna('missing')
            if messages:
              print(f'{col_name} filled with "missing"')
  else:
    # Treat as MCAR — use ML imputation
    if not out.isna().any().any():
      if messages:
        print('No missing values remaining.')
      return out

    from sklearn.preprocessing import OrdinalEncoder

    # Separate numeric and categorical handling
    numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    cat_cols = [c for c in out.columns if c not in numeric_cols]

    # Impute numeric columns
    if numeric_cols and out[numeric_cols].isna().any().any():
      if out.shape[0] > large_dataset:
        from sklearn.impute import KNNImputer
        imp = KNNImputer()
      else:
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        imp = IterativeImputer(max_iter=10, random_state=0)

      imputed = imp.fit_transform(out[numeric_cols])
      out[numeric_cols] = pd.DataFrame(imputed, index=out.index, columns=numeric_cols)

    # Impute categorical columns with mode
    for col in cat_cols:
      if out[col].isna().any():
        mode_val = out[col].mode()
        fill_val = mode_val[0] if len(mode_val) > 0 else "Unknown"
        out[col] = out[col].fillna(fill_val)

    if messages:
      print('Null values imputed (MCAR).')

  return out


def clean_outlier(df, features=None, method="remove", messages=True, skew_threshold=1):
  """
  Identify and handle outliers one feature at a time using either the
  Empirical Rule (Z-score, for normally distributed features) or Tukey's
  IQR Rule (for skewed features).

  The method is chosen automatically based on the skewness of each feature.

  Parameters
  ----------
  df : pandas.DataFrame
      Input DataFrame.
  features : list of str or None
      Numeric columns to evaluate. If None, all columns are used.
  method : str
      How to handle outliers: 'remove', 'replace', 'impute', or 'null'.
  messages : bool
      If True, print outlier counts per feature.
  skew_threshold : float
      Absolute skewness above which Tukey's IQR rule is used instead of Z-score.

  Returns
  -------
  pandas.DataFrame
      Cleaned DataFrame (copy; original not modified).
  """
  import pandas as pd
  import numpy as np

  out = df.copy()

  if features is None:
    features = list(out.columns)

  for feat in features:
    if feat not in out.columns:
      if messages:
        print(f'{feat} is not found in the DataFrame.')
      continue

    if not pd.api.types.is_numeric_dtype(out[feat]):
      if messages:
        print(f'{feat} is categorical and was ignored.')
      continue

    if out[feat].nunique() <= 1:
      if messages:
        print(f'{feat} has only one unique value and was ignored.')
      continue

    # Skip boolean 0/1 columns
    if set(out[feat].dropna().unique()).issubset({0, 1}):
      if messages:
        print(f'{feat} is a dummy code (0/1) and was ignored.')
      continue

    skew = out[feat].skew()

    if abs(skew) > skew_threshold:
      # Tukey's IQR rule
      q1 = out[feat].quantile(0.25)
      q3 = out[feat].quantile(0.75)
      iqr = q3 - q1
      lower = q1 - (1.5 * iqr)
      upper = q3 + (1.5 * iqr)
    else:
      # Empirical rule (Z-score)
      mean = out[feat].mean()
      std = out[feat].std()
      lower = mean - (3 * std)
      upper = mean + (3 * std)

    below_count = (out[feat] < lower).sum()
    above_count = (out[feat] > upper).sum()

    if messages:
      print(f'{feat} has {above_count} values above max={upper:.4f} '
            f'and {below_count} below min={lower:.4f}')

    if below_count == 0 and above_count == 0:
      continue

    if method == "remove":
      out = out[(out[feat] >= lower) & (out[feat] <= upper)]
    elif method == "replace":
      out.loc[out[feat] < lower, feat] = lower
      out.loc[out[feat] > upper, feat] = upper
    elif method == "impute":
      out.loc[out[feat] < lower, feat] = np.nan
      out.loc[out[feat] > upper, feat] = np.nan
      from sklearn.experimental import enable_iterative_imputer
      from sklearn.impute import IterativeImputer

      # Impute only numeric columns
      numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
      imp = IterativeImputer(max_iter=10, random_state=0)
      imputed = imp.fit_transform(out[numeric_cols])
      out[numeric_cols] = pd.DataFrame(imputed, index=out.index, columns=numeric_cols)
    elif method == "null":
      out.loc[out[feat] < lower, feat] = np.nan
      out.loc[out[feat] > upper, feat] = np.nan

  return out


def clean_outliers(df, messages=True, drop_percent=0.02, distance='manhattan', min_samples=5):
  """
  Detect and remove multivariate outliers using DBSCAN clustering.

  Automatically finds an epsilon value that removes approximately drop_percent
  of the dataset as outliers (noise points).

  Parameters
  ----------
  df : pandas.DataFrame
      Input DataFrame.
  messages : bool
      If True, print progress and show a diagnostic plot.
  drop_percent : float
      Target proportion of rows to remove as outliers.
  distance : str
      Distance metric for DBSCAN. Options: 'cityblock', 'cosine', 'euclidean',
      'l1', 'l2', 'manhattan'.
  min_samples : int
      Minimum points for a DBSCAN core point.

  Returns
  -------
  pandas.DataFrame
      DataFrame with outlier rows removed (copy; original not modified).
  """
  import pandas as pd
  import numpy as np
  from sklearn.cluster import DBSCAN
  from sklearn import preprocessing

  out = df.copy()

  # Drop columns and rows with missing data (required for DBSCAN)
  cols_before = out.shape[1]
  out = out.dropna(axis='columns')
  if messages and out.shape[1] < cols_before:
    print(f"{cols_before - out.shape[1]} columns were dropped due to missing data")

  rows_before = out.shape[0]
  out = out.dropna()
  if messages and out.shape[0] < rows_before:
    print(f"{rows_before - out.shape[0]} rows were dropped due to missing data")

  # Prepare for clustering: bin categories, wrangle, encode
  df_temp = out.copy()
  df_temp = bin_categories(df_temp, messages=False)
  df_temp = basic_wrangling(df_temp, messages=False)
  df_temp = pd.get_dummies(df_temp, drop_first=True)

  # Normalize
  df_temp = pd.DataFrame(
    preprocessing.MinMaxScaler().fit_transform(df_temp),
    columns=df_temp.columns, index=df_temp.index
  )

  # Find optimal eps by iterating
  outliers_per_eps = []
  outliers = df_temp.shape[0]
  eps = 0

  if df_temp.shape[0] < 500:
    iterator = 0.01
  elif df_temp.shape[0] < 2000:
    iterator = 0.05
  elif df_temp.shape[0] < 10000:
    iterator = 0.1
  else:
    iterator = 0.2

  while outliers > 0:
    eps += iterator
    db = DBSCAN(metric=distance, min_samples=min_samples, eps=eps).fit(df_temp)
    outliers = np.count_nonzero(db.labels_ == -1)
    outliers_per_eps.append(outliers)
    if messages:
      print(f'eps: {round(eps, 2)}, outliers: {outliers}, '
            f'percent: {round((outliers / df_temp.shape[0]) * 100, 3)}%')

  # Find eps closest to desired drop_percent
  target_drops = round(df_temp.shape[0] * drop_percent)
  drops = min(outliers_per_eps, key=lambda x: abs(x - target_drops))
  best_eps = (outliers_per_eps.index(drops) + 1) * iterator

  # Re-run DBSCAN at optimal eps
  db = DBSCAN(metric=distance, min_samples=min_samples, eps=best_eps).fit(df_temp)
  out['outlier'] = db.labels_

  if messages:
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_outliers = (out['outlier'] == -1).sum()
    print(f"\n{n_outliers} outlier rows removed from the DataFrame")

    sns.lineplot(x=range(1, len(outliers_per_eps) + 1), y=outliers_per_eps)
    sns.scatterplot(x=[best_eps / iterator], y=[drops])
    plt.xlabel(f'eps (multiply by {iterator})')
    plt.ylabel('Number of Outliers')
    plt.show()

  # Remove outlier rows
  out = out[out['outlier'] != -1]

  return out


# =============================================================================
# Chapter 8: Automating Relationship Discovery
# =============================================================================

def bivariate_stats(df, label, roundto=4):
  """
  Compute bivariate statistics for all features relative to a label.

  Automatically selects the appropriate test based on data types:
  - Numeric vs Numeric: Pearson r, Kendall tau, Spearman rho, linear regression
  - Categorical vs Numeric: one-way ANOVA F-statistic
  - Categorical vs Categorical: chi-square test of independence

  Parameters
  ----------
  df : pandas.DataFrame
      Input DataFrame.
  label : str
      Target column name.
  roundto : int
      Decimal places for rounding.

  Returns
  -------
  pandas.DataFrame
      Summary table sorted by p-value (ascending).
  """
  import pandas as pd
  from scipy import stats

  output_df = pd.DataFrame(
    columns=['missing', 'p', 'r', 'tau', 'rho', 'y = m(x) + b', 'F', 'X2', 'skew', 'unique', 'values']
  )

  for feature in df.columns:
    if feature == label:
      continue

    df_temp = df[[feature, label]].dropna()
    missing = (df.shape[0] - df_temp.shape[0]) / df.shape[0]
    unique = df_temp[feature].nunique()

    is_num_feat = pd.api.types.is_numeric_dtype(df_temp[feature])
    is_num_label = pd.api.types.is_numeric_dtype(df_temp[label])

    if is_num_feat and is_num_label:
      # N2N
      if len(df_temp) < 3:
        continue
      m, b, r, p, err = stats.linregress(df_temp[feature], df_temp[label])
      tau, _ = stats.kendalltau(df_temp[feature], df_temp[label])
      rho, _ = stats.spearmanr(df_temp[feature], df_temp[label])
      output_df.loc[feature] = [
        f'{missing:.2%}', round(p, roundto), round(r, roundto), round(tau, roundto),
        round(rho, roundto), f'y = {round(m, roundto)}(x) + {round(b, roundto)}',
        '-', '-', round(df_temp[feature].skew(), roundto), unique, '-'
      ]

    elif not is_num_feat and not is_num_label:
      # C2C
      contingency_table = pd.crosstab(df_temp[feature], df_temp[label])
      if contingency_table.size < 2:
        continue
      X2, p, dof, expected = stats.chi2_contingency(contingency_table)
      output_df.loc[feature] = [
        f'{missing:.2%}', round(p, roundto), '-', '-', '-', '-', '-',
        round(X2, roundto), '-', unique, list(df_temp[feature].unique())
      ]

    else:
      # C2N or N2C
      if is_num_feat:
        num, cat = feature, label
        skew = round(df_temp[feature].skew(), roundto)
      else:
        num, cat = label, feature
        skew = '-'

      groups = df_temp[cat].unique()
      if len(groups) < 2:
        continue
      group_lists = [df_temp[df_temp[cat] == g][num] for g in groups]
      F_val, p = stats.f_oneway(*group_lists)
      output_df.loc[feature] = [
        f'{missing:.2%}', round(p, roundto), '-', '-', '-', '-',
        round(F_val, roundto), '-', skew, unique, list(df_temp[cat].unique())
      ]

  return output_df.sort_values(by=['p'])


def scatterplot(df, feature, label, roundto=3, linecolor='darkorange',
                title=None, savepath=None, show=True):
  """
  Create a scatterplot with regression line and embedded statistics.

  Displays: regression equation, r, r-squared, p-value, and n.

  Parameters
  ----------
  df : pandas.DataFrame
      Input DataFrame.
  feature : str
      Numeric feature (x-axis).
  label : str
      Numeric label (y-axis).
  roundto : int
      Decimal places for statistics.
  linecolor : str
      Color of the regression line.
  title : str or None
      Custom title. Auto-generated if None.
  savepath : str or None
      File path to save the figure.
  show : bool
      If True, display the plot.

  Returns
  -------
  matplotlib.axes.Axes
  """
  import pandas as pd
  from matplotlib import pyplot as plt
  import seaborn as sns
  from scipy import stats

  if feature == label:
    raise ValueError('feature and label must be different.')

  df_temp = df[[feature, label]].dropna()
  if df_temp.shape[0] < 3:
    raise ValueError('Not enough non-missing rows to plot.')

  ax = plt.gca()
  sns.regplot(data=df_temp, x=feature, y=label, ax=ax, line_kws={'color': linecolor})

  m, b, r, p, err = stats.linregress(df_temp[feature], df_temp[label])
  textstr = (f'Regression line:\n'
             f'y = {round(m, roundto)}x + {round(b, roundto)}\n'
             f'r = {round(r, roundto)}\n'
             f'r2 = {round(r**2, roundto)}\n'
             f'p = {round(p, roundto)}\n'
             f'n = {df_temp.shape[0]}')

  if title is None:
    title = f'{feature} vs. {label}'
  ax.set_title(title)
  ax.text(1.02, 0.02, textstr, fontsize=11, transform=ax.transAxes, va='bottom')

  if savepath is not None:
    plt.savefig(savepath, bbox_inches='tight', dpi=200)
  if show:
    plt.show()

  return ax


def bar_chart(df, feature, label, roundto=3, title=None, savepath=None, show=True):
  """
  Create a bar chart for a categorical-numeric relationship with embedded
  ANOVA statistics and Bonferroni-corrected pairwise comparisons.

  Automatically places the categorical variable on the x-axis regardless
  of argument order.

  Parameters
  ----------
  df : pandas.DataFrame
      Input DataFrame.
  feature : str
      One of the two variables (numeric or categorical).
  label : str
      The other variable (numeric or categorical).
  roundto : int
      Decimal places for statistics.
  title : str or None
      Custom title. Auto-generated if None.
  savepath : str or None
      File path to save the figure.
  show : bool
      If True, display the plot.

  Returns
  -------
  matplotlib.axes.Axes
  """
  import pandas as pd
  from scipy import stats
  from matplotlib import pyplot as plt
  import seaborn as sns

  if feature == label:
    raise ValueError('feature and label must be different.')

  is_num_feature = pd.api.types.is_numeric_dtype(df[feature])
  is_num_label = pd.api.types.is_numeric_dtype(df[label])

  if is_num_feature and not is_num_label:
    num, cat = feature, label
  elif not is_num_feature and is_num_label:
    num, cat = label, feature
  else:
    raise ValueError('bar_chart requires one numeric and one categorical variable.')

  df_temp = df[[cat, num]].dropna()
  if df_temp[cat].nunique() < 2:
    raise ValueError('Not enough categories to compare (need at least 2).')

  ax = plt.gca()
  sns.barplot(data=df_temp, x=cat, y=num, ax=ax, errorbar=None)

  groups = list(df_temp[cat].unique())
  group_lists = [df_temp[df_temp[cat] == g][num] for g in groups]
  F_val, p_anova = stats.f_oneway(*group_lists)

  # Pairwise t-tests with Bonferroni correction
  ttests = []
  for i1, g1 in enumerate(groups):
    for i2, g2 in enumerate(groups):
      if i2 > i1:
        vals_1 = df_temp[df_temp[cat] == g1][num]
        vals_2 = df_temp[df_temp[cat] == g2][num]
        t, p = stats.ttest_ind(vals_1, vals_2, equal_var=False, nan_policy='omit')
        ttests.append([f'{g1} - {g2}', round(t, roundto), round(p, roundto)])

  p_threshold = 0.05 / max(1, len(ttests))

  textstr = (f'ANOVA\nF: {round(F_val, roundto)}\n'
             f'p: {round(p_anova, roundto)}\nn: {df_temp.shape[0]}\n\n')

  for ttest in ttests:
    if ttest[2] <= p_threshold:
      if 'Sig. comparisons (Bonferroni)' not in textstr:
        textstr += 'Sig. comparisons (Bonferroni)\n'
      textstr += f'{ttest[0]}: t={ttest[1]}, p={ttest[2]}\n'

  if title is None:
    title = f'{num} by {cat}'
  ax.set_title(title)
  ax.text(1.02, 0.02, textstr, fontsize=10.5, transform=ax.transAxes, va='bottom')

  if savepath is not None:
    plt.savefig(savepath, bbox_inches='tight', dpi=200)
  if show:
    plt.show()

  return ax


def crosstab(df, feature, label, roundto=3, cutoff=0.05, title=None, savepath=None, show=True):
  """
  Create a heatmap for a categorical-categorical relationship with
  embedded chi-square statistics.

  Rare categories are automatically binned before visualization.

  Parameters
  ----------
  df : pandas.DataFrame
      Input DataFrame.
  feature : str
      Categorical feature.
  label : str
      Categorical label.
  roundto : int
      Decimal places for statistics.
  cutoff : float
      Proportion threshold for binning rare categories.
  title : str or None
      Custom title. Auto-generated if None.
  savepath : str or None
      File path to save the figure.
  show : bool
      If True, display the plot.

  Returns
  -------
  matplotlib.axes.Axes
  """
  import pandas as pd
  from scipy.stats import chi2_contingency
  from matplotlib import pyplot as plt
  import seaborn as sns

  if feature == label:
    raise ValueError('feature and label must be different.')

  df_temp = df[[feature, label]].dropna()
  if df_temp.shape[0] == 0:
    raise ValueError('No non-missing rows available for this pair.')

  # Bin rare categories for readability
  df_temp = bin_categories(df_temp, features=[feature], cutoff=cutoff, messages=False)

  ct = pd.crosstab(df_temp[feature], df_temp[label])
  X2, p, dof, expected = chi2_contingency(ct)

  ax = plt.gca()
  sns.heatmap(ct, annot=True, fmt='d', ax=ax)

  if title is None:
    title = f'{feature} vs. {label}'
  ax.set_title(title)

  textstr = (f'X2: {round(X2, roundto)}\n'
             f'p: {round(p, roundto)}\n'
             f'dof: {dof}\n'
             f'n: {df_temp.shape[0]}')
  ax.text(1.02, 0.02, textstr, fontsize=10.5, transform=ax.transAxes, va='bottom')

  if savepath is not None:
    plt.savefig(savepath, bbox_inches='tight', dpi=200)
  if show:
    plt.show()

  return ax


def bivariate(df, label, roundto=4, viz=True):
  """
  Controller function: compute bivariate statistics and generate
  visualizations for all features relative to a label.

  Automatically determines the relationship type (N2N, C2N/N2C, C2C)
  and calls the appropriate statistics and visualization functions.

  Parameters
  ----------
  df : pandas.DataFrame
      Input DataFrame.
  label : str
      Target column name.
  roundto : int
      Decimal places for rounding.
  viz : bool
      If True, generate charts for each feature.

  Returns
  -------
  pandas.DataFrame
      Summary statistics table sorted by p-value.
  """
  import pandas as pd
  from scipy import stats

  output_df = pd.DataFrame(
    columns=['missing', 'p', 'r', 'tau', 'rho', 'y = m(x) + b', 'F', 'X2', 'skew', 'unique', 'values']
  )

  for feature in df.columns:
    if feature == label:
      continue

    df_temp = df[[feature, label]].dropna()
    missing = (df.shape[0] - df_temp.shape[0]) / df.shape[0]
    unique = df_temp[feature].nunique()

    # Bin categories on the temp slice to avoid mutating the full DataFrame
    if not pd.api.types.is_numeric_dtype(df_temp[feature]):
      df_temp = bin_categories(df_temp, features=[feature], messages=False)
    if not pd.api.types.is_numeric_dtype(df_temp[label]):
      df_temp = bin_categories(df_temp, features=[label], messages=False)

    is_num_feat = pd.api.types.is_numeric_dtype(df_temp[feature])
    is_num_label = pd.api.types.is_numeric_dtype(df_temp[label])

    # N2N
    if is_num_feat and is_num_label:
      if len(df_temp) < 3:
        continue
      m, b, r, p, err = stats.linregress(df_temp[feature], df_temp[label])
      tau, _ = stats.kendalltau(df_temp[feature], df_temp[label])
      rho, _ = stats.spearmanr(df_temp[feature], df_temp[label])

      output_df.loc[feature] = [
        f'{missing:.2%}', round(p, roundto), round(r, roundto), round(tau, roundto),
        round(rho, roundto), f'y = {round(m, roundto)}(x) + {round(b, roundto)}',
        '-', '-', round(df_temp[feature].skew(), roundto), unique, '-'
      ]
      if viz:
        scatterplot(df_temp, feature, label, roundto)

    # C2C
    elif not is_num_feat and not is_num_label:
      contingency_table = pd.crosstab(df_temp[feature], df_temp[label])
      if contingency_table.size < 2:
        continue
      X2, p, dof, expected = stats.chi2_contingency(contingency_table)

      output_df.loc[feature] = [
        f'{missing:.2%}', round(p, roundto), '-', '-', '-', '-', '-',
        round(X2, roundto), '-', unique, list(df_temp[feature].unique())
      ]
      if viz:
        crosstab(df_temp, feature, label, roundto)

    # C2N / N2C
    else:
      if is_num_feat:
        num, cat = feature, label
        skew = round(df_temp[feature].skew(), roundto)
      else:
        num, cat = label, feature
        skew = '-'

      groups = df_temp[cat].unique()
      if len(groups) < 2:
        continue
      group_lists = [df_temp[df_temp[cat] == g][num] for g in groups]
      F_val, p = stats.f_oneway(*group_lists)

      output_df.loc[feature] = [
        f'{missing:.2%}', round(p, roundto), '-', '-', '-', '-',
        round(F_val, roundto), '-', skew, unique, list(df_temp[cat].unique())
      ]
      if viz:
        bar_chart(df_temp, cat, num, roundto)

  return output_df.sort_values(by=['p'])


# =============================================================================
# Chapter 9: Regression
# =============================================================================

def fit_regression(df, label, features=None, scale=None, drop_first=True, messages=True):
  """
  Automate the MLR workflow: dummy-code, convert bools, optionally scale,
  add constant, and fit OLS.

  Parameters
  ----------
  df : pandas.DataFrame
      Input DataFrame (not modified).
  label : str
      Name of the target column.
  features : list of str or None
      Columns to include as predictors. If None, all columns except the
      label are used.
  scale : {None, 'standard', 'minmax'}
      Scaling method applied to numeric (non-dummy) columns only.
      None = no scaling. 'standard' = zero-mean, unit-variance.
      'minmax' = scale to [0, 1].
  drop_first : bool
      If True, drop the first dummy column per categorical feature
      to avoid perfect multicollinearity.
  messages : bool
      If True, print a short status report.

  Returns
  -------
  statsmodels.regression.linear_model.RegressionResultsWrapper
      Fitted OLS model.
  """
  import pandas as pd
  import numpy as np
  import statsmodels.api as sm

  out = df.copy()

  # Separate label
  y = out[label]

  # Select features
  if features is not None:
    X = out[features].copy()
  else:
    X = out.drop(columns=[label]).copy()

  # Dummy-code categorical columns
  cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
  if cat_cols:
    X = pd.get_dummies(X, columns=cat_cols, drop_first=drop_first)
    if messages:
      print(f"Dummy-coded: {cat_cols}")

  # Convert boolean columns to int
  bool_cols = X.select_dtypes(bool).columns.tolist()
  if bool_cols:
    X[bool_cols] = X[bool_cols].astype(int)

  # Scale numeric (non-dummy) columns only
  if scale is not None:
    numeric_cols = [c for c in X.columns
                    if pd.api.types.is_numeric_dtype(X[c])
                    and not set(X[c].dropna().unique()).issubset({0, 1})]
    if scale == 'standard':
      from sklearn.preprocessing import StandardScaler
      X[numeric_cols] = StandardScaler().fit_transform(X[numeric_cols])
    elif scale == 'minmax':
      from sklearn.preprocessing import MinMaxScaler
      X[numeric_cols] = MinMaxScaler().fit_transform(X[numeric_cols])
    if messages:
      print(f"Scaled ({scale}): {numeric_cols}")

  # Add constant for intercept
  X = sm.add_constant(X, has_constant='add')

  # Fit OLS
  model = sm.OLS(y, X).fit()

  if messages:
    print(f"Model fit: R²={model.rsquared:.4f}, "
          f"Adj R²={model.rsquared_adj:.4f}, "
          f"features={model.df_model:.0f}")

  return model


def regression_summary(model, y, sort_by='pvalue', roundto=4, messages=True):
  """
  Print key fit metrics and return a sorted coefficient table.

  Parameters
  ----------
  model : statsmodels RegressionResultsWrapper
      A fitted OLS model (e.g., from fit_regression()).
  y : pandas.Series
      The actual label values used to compute error metrics.
  sort_by : {'pvalue', 'coefficient', 'tvalue'}
      Column to sort the coefficient table by (ascending for pvalue,
      descending for coefficient and tvalue).
  roundto : int
      Decimal places for rounding.
  messages : bool
      If True, print fit metrics to the console.

  Returns
  -------
  pandas.DataFrame
      Coefficient table with columns: coef, std_err, t, pvalue,
      ci_lower, ci_upper.
  """
  import pandas as pd
  import numpy as np

  # Fit metrics
  mae = abs(model.fittedvalues - y).mean()
  rmse = ((model.fittedvalues - y) ** 2).mean() ** 0.5

  if messages:
    print(f"R²:        {model.rsquared:.{roundto}f}")
    print(f"Adj R²:    {model.rsquared_adj:.{roundto}f}")
    print(f"MAE:       {mae:.{roundto}f}")
    print(f"RMSE:      {rmse:.{roundto}f}")
    print(f"F-stat:    {model.fvalue:.{roundto}f}  "
          f"(p={model.f_pvalue:.2e})")
    print(f"N:         {int(model.nobs)}  "
          f"Features:  {int(model.df_model)}")
    print("-" * 50)

  # Build coefficient table
  ci = model.conf_int()
  coef_df = pd.DataFrame({
    'coef':      round(model.params, roundto),
    'std_err':   round(model.bse, roundto),
    't':         round(model.tvalues, roundto),
    'pvalue':    round(model.pvalues, roundto),
    'ci_lower':  round(ci.iloc[:, 0], roundto),
    'ci_upper':  round(ci.iloc[:, 1], roundto),
  })

  # Sort
  sort_map = {
    'pvalue':      ('pvalue', True),
    'coefficient': ('coef', False),
    'tvalue':      ('t', False),
  }
  col, asc = sort_map.get(sort_by, ('pvalue', True))
  coef_df = coef_df.sort_values(by=col, ascending=asc)

  return coef_df


# =============================================================================
# Chapter 10: MLR Diagnostics for Causal Inference
# =============================================================================

def regression_diagnostics(model, X, plot=True, messages=True):
  """
  Run the five core regression assumption tests plus Cook's distance.

  Parameters
  ----------
  model : statsmodels RegressionResultsWrapper
      A fitted OLS model.
  X : pandas.DataFrame
      The design matrix used to fit the model (with constant).
  plot : bool
      If True, display a 2x2 diagnostic plot panel.
  messages : bool
      If True, print test results to the console.

  Returns
  -------
  dict
      Keys: 'omnibus_stat', 'omnibus_p', 'durbin_watson',
      'bp_stat', 'bp_p', 'n_high_cooks', 'cooks_threshold',
      'vif' (DataFrame).
  """
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  from statsmodels.stats.stattools import omni_normtest, durbin_watson
  from statsmodels.stats.diagnostic import het_breuschpagan
  from statsmodels.stats.outliers_influence import variance_inflation_factor
  from scipy.stats import probplot

  resid = model.resid
  fitted = model.fittedvalues
  n = int(model.nobs)

  # 1) Normality (Omnibus)
  omni_stat, omni_p = omni_normtest(resid)

  # 2) Autocorrelation (Durbin-Watson)
  dw = durbin_watson(resid)

  # 3) Heteroscedasticity (Breusch-Pagan)
  bp_result = het_breuschpagan(resid, X)
  bp_stat, bp_p = bp_result[0], bp_result[1]

  # 4) Multicollinearity (VIF)
  vif_data = pd.DataFrame({
    "feature": X.columns,
    "VIF": [variance_inflation_factor(X.values, i)
            for i in range(X.shape[1])]
  })
  vif_data = vif_data[vif_data["feature"] != "const"].sort_values(
    "VIF", ascending=False
  )

  # 5) Cook's Distance
  influence = model.get_influence()
  cooks_d = influence.cooks_distance[0]
  threshold = 4 / n
  n_high = int((cooks_d > threshold).sum())

  if messages:
    print("=== Regression Diagnostics ===")
    print(f"Normality (Omnibus):       stat={omni_stat:.3f}, "
          f"p={omni_p:.6f}")
    print(f"Autocorrelation (DW):      {dw:.3f}")
    print(f"Heteroscedasticity (BP):   stat={bp_stat:.3f}, "
          f"p={bp_p:.6f}")
    print(f"Cook's D > {threshold:.4f}:       "
          f"{n_high} of {n} observations")
    print(f"\nVIF (top 5):")
    print(vif_data.head().to_string(index=False))

  if plot:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Regression Diagnostics", fontsize=14)

    # Residual histogram
    axes[0, 0].hist(resid, bins=40, edgecolor="white", alpha=0.7)
    axes[0, 0].set_title("Residual Distribution")
    axes[0, 0].set_xlabel("Residual")

    # Q-Q plot
    probplot(resid, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title("Q-Q Plot")

    # Residuals vs. fitted
    axes[1, 0].scatter(fitted, resid, alpha=0.3, s=10)
    axes[1, 0].axhline(0, color="red", linewidth=0.8)
    axes[1, 0].set_title("Residuals vs. Fitted")
    axes[1, 0].set_xlabel("Fitted Values")
    axes[1, 0].set_ylabel("Residuals")

    # Cook's Distance
    axes[1, 1].stem(range(n), cooks_d, markerfmt=",",
                    basefmt="k-")
    axes[1, 1].axhline(threshold, color="red", linestyle="--",
                       label=f"4/n = {threshold:.4f}")
    axes[1, 1].set_title("Cook's Distance")
    axes[1, 1].set_xlabel("Observation")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

  return {
    "omnibus_stat": omni_stat,
    "omnibus_p": omni_p,
    "durbin_watson": dw,
    "bp_stat": bp_stat,
    "bp_p": bp_p,
    "n_high_cooks": n_high,
    "cooks_threshold": threshold,
    "vif": vif_data
  }


def assumption_checks(model, X, messages=True):
  """
  Quick pass/warning/fail screening for OLS regression assumptions.

  Parameters
  ----------
  model : statsmodels RegressionResultsWrapper
      A fitted OLS model.
  X : pandas.DataFrame
      The design matrix used to fit the model (with constant).
  messages : bool
      If True, print the screening table.

  Returns
  -------
  pandas.DataFrame
      One row per assumption with columns: assumption, test,
      value, threshold, verdict.
  """
  import pandas as pd
  import numpy as np
  from statsmodels.stats.stattools import omni_normtest, durbin_watson
  from statsmodels.stats.diagnostic import het_breuschpagan
  from statsmodels.stats.outliers_influence import variance_inflation_factor

  resid = model.resid
  n = int(model.nobs)

  # Normality
  omni_stat, omni_p = omni_normtest(resid)
  norm_verdict = ("pass" if omni_p > 0.05
                  else "warning" if omni_p > 0.01
                  else "fail")

  # Autocorrelation
  dw = durbin_watson(resid)
  dw_verdict = ("pass" if 1.5 < dw < 2.5
                else "warning" if 1.0 < dw < 3.0
                else "fail")

  # Heteroscedasticity
  bp_result = het_breuschpagan(resid, X)
  bp_p = bp_result[1]
  bp_verdict = ("pass" if bp_p > 0.05
                else "warning" if bp_p > 0.01
                else "fail")

  # Multicollinearity (max VIF, excluding constant)
  vifs = []
  for i, col in enumerate(X.columns):
    if col == "const":
      continue
    vifs.append(variance_inflation_factor(X.values, i))
  max_vif = max(vifs) if vifs else 0
  vif_verdict = ("pass" if max_vif < 5
                 else "warning" if max_vif < 10
                 else "fail")

  # Cook's Distance
  cooks_d = model.get_influence().cooks_distance[0]
  threshold = 4 / n
  pct_high = (cooks_d > threshold).mean() * 100
  cooks_verdict = ("pass" if pct_high < 1
                   else "warning" if pct_high < 5
                   else "fail")

  rows = [
    {"assumption": "Normality", "test": "Omnibus",
     "value": round(omni_p, 6), "threshold": "p > 0.05",
     "verdict": norm_verdict},
    {"assumption": "No Autocorrelation", "test": "Durbin-Watson",
     "value": round(dw, 3), "threshold": "1.5-2.5",
     "verdict": dw_verdict},
    {"assumption": "Homoscedasticity", "test": "Breusch-Pagan",
     "value": round(bp_p, 6), "threshold": "p > 0.05",
     "verdict": bp_verdict},
    {"assumption": "No Multicollinearity", "test": "Max VIF",
     "value": round(max_vif, 2), "threshold": "< 5",
     "verdict": vif_verdict},
    {"assumption": "No Influential Outliers", "test": "Cook's D",
     "value": f"{pct_high:.1f}%", "threshold": "< 1% above 4/n",
     "verdict": cooks_verdict},
  ]

  check_df = pd.DataFrame(rows)

  if messages:
    print("=== Assumption Screening ===")
    print(check_df.to_string(index=False))

  return check_df


def diagnostic_model(df, label, features=None, cat_cols=None,
                     interactions=None, poly_features=None,
                     log_features=None, skew_threshold=1.0,
                     center=True, robust=True, drop_first=True,
                     messages=True):
  """
  Build a diagnostic-adjusted OLS regression model.

  Automates label transformation, nonlinear terms, interaction terms,
  centering, and robust standard errors based on diagnostic patterns.

  Parameters
  ----------
  df : pandas.DataFrame
      Input DataFrame (not modified).
  label : str
      Name of the target column.
  features : list of str or None
      Columns to include as predictors. If None, all columns except
      the label are used.
  cat_cols : list of str or None
      Columns to dummy-code. If None, auto-detects object/category
      dtypes.
  interactions : list of tuple or None
      Pairs of column names to interact, e.g.,
      [('age_c_sq', 'smoker_yes')]. Uses transformed names if
      the base column was transformed.
  poly_features : list of str or None
      Columns to add centered squared terms for.
  log_features : list of str or None
      Columns to add centered log-transformed terms for.
  skew_threshold : float
      If abs(skewness) of the label exceeds this, apply a power
      transform. Set to None to skip auto-detection.
  center : bool
      If True, center numeric features before creating polynomial
      and log terms.
  robust : bool
      If True, fit with HC3 robust standard errors.
  drop_first : bool
      If True, drop the first dummy column per categorical feature.
  messages : bool
      If True, print a summary of transformations applied.

  Returns
  -------
  dict
      Keys:
      - 'model': fitted statsmodels OLS model
      - 'y': transformed label Series
      - 'X': design matrix DataFrame
      - 'transformations': dict describing what was applied
      - 'power_transformer': sklearn PowerTransformer or None
  """
  import pandas as pd
  import numpy as np
  import statsmodels.api as sm

  out = df.copy()
  transformations = {
    "label_transform": None,
    "poly_terms": [],
    "log_terms": [],
    "interactions": [],
    "centered": [],
    "robust_se": robust
  }
  pt_obj = None

  # --- Label transformation ---
  y = out[label].copy()

  if skew_threshold is not None:
    label_skew = y.skew()
    if abs(label_skew) > skew_threshold:
      from sklearn.preprocessing import PowerTransformer
      if (y > 0).all():
        method = "box-cox"
      else:
        method = "yeo-johnson"
      pt_obj = PowerTransformer(method=method, standardize=False)
      y = pd.Series(
        pt_obj.fit_transform(y.values.reshape(-1, 1)).ravel(),
        index=y.index, name=f"{label}_{method.replace('-', '_')}"
      )
      transformations["label_transform"] = {
        "method": method,
        "original_skew": round(label_skew, 4),
        "transformed_skew": round(y.skew(), 4)
      }
      if messages:
        print(f"Label transform: {method} "
              f"(skew {label_skew:.3f} -> {y.skew():.3f})")

  # --- Select features ---
  if features is not None:
    X = out[features].copy()
  else:
    X = out.drop(columns=[label]).copy()

  # --- Dummy-code categoricals ---
  if cat_cols is None:
    cat_cols = X.select_dtypes(
      include=["object", "category"]
    ).columns.tolist()
  if cat_cols:
    X = pd.get_dummies(X, columns=cat_cols, drop_first=drop_first)
    if messages:
      print(f"Dummy-coded: {cat_cols}")

  # Convert booleans to int
  bool_cols = X.select_dtypes(bool).columns.tolist()
  if bool_cols:
    X[bool_cols] = X[bool_cols].astype(int)

  # --- Store column means for centering ---
  col_means = {}

  # --- Polynomial (squared) terms ---
  if poly_features:
    for col in poly_features:
      if col not in X.columns:
        if messages:
          print(f"Warning: '{col}' not found, skipping poly term")
        continue
      if center:
        mean_val = X[col].mean()
        col_means[col] = mean_val
        centered = X[col] - mean_val
        new_col = f"{col}_c_sq"
        X[new_col] = centered ** 2
      else:
        new_col = f"{col}_sq"
        X[new_col] = X[col] ** 2
      transformations["poly_terms"].append(new_col)
      if messages:
        print(f"Polynomial term: {new_col}")

  # --- Log terms ---
  if log_features:
    for col in log_features:
      if col not in X.columns:
        if messages:
          print(f"Warning: '{col}' not found, skipping log term")
        continue
      log_vals = np.log(X[col])
      if center:
        mean_val = log_vals.mean()
        col_means[f"log_{col}"] = mean_val
        new_col = f"{col}_ln_c"
        X[new_col] = log_vals - mean_val
      else:
        new_col = f"{col}_ln"
        X[new_col] = log_vals
      transformations["log_terms"].append(new_col)
      if messages:
        print(f"Log term: {new_col}")

  # --- Interaction terms ---
  if interactions:
    for col_a, col_b in interactions:
      if col_a not in X.columns or col_b not in X.columns:
        if messages:
          print(f"Warning: interaction ({col_a}, {col_b}) — "
                f"column not found, skipping")
        continue
      new_col = f"{col_a}_x_{col_b}"
      X[new_col] = X[col_a] * X[col_b]
      transformations["interactions"].append(new_col)
      if messages:
        print(f"Interaction term: {new_col}")

  if center:
    transformations["centered"] = list(col_means.keys())

  # --- Add constant and fit ---
  X = sm.add_constant(X, has_constant="add")

  if robust:
    model = sm.OLS(y, X).fit(cov_type="HC3")
  else:
    model = sm.OLS(y, X).fit()

  if messages:
    print(f"\nModel fit: R²={model.rsquared:.4f}, "
          f"Adj R²={model.rsquared_adj:.4f}, "
          f"features={model.df_model:.0f}")
    if robust:
      print("Standard errors: HC3 (robust)")

  return {
    "model": model,
    "y": y,
    "X": X,
    "transformations": transformations,
    "power_transformer": pt_obj
  }


# =============================================================================
# Chapter 11: MLR for Predictive Inference
# =============================================================================

def holdout_split(df, label, test_size=0.2, val_size=0.2,
                  random_state=42, messages=True):
  """
  Two-stage split: holdout a test set, then split the remainder
  into training and validation sets.

  Parameters
  ----------
  df : pandas.DataFrame
      Input DataFrame (not modified).
  label : str
      Name of the target column.
  test_size : float
      Fraction of the full dataset reserved for testing.
  val_size : float
      Fraction of the full dataset reserved for validation.
      Internally converted to a fraction of the non-test data.
  random_state : int or None
      Seed for reproducibility.
  messages : bool
      If True, print split sizes.

  Returns
  -------
  tuple
      (X_train, X_val, X_test, y_train, y_val, y_test)
  """
  import pandas as pd
  from sklearn.model_selection import train_test_split

  out = df.copy()
  y = out[label]
  X = out.drop(columns=[label])

  # Stage 1: hold out test set
  X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
  )

  # Stage 2: split remainder into train + validation
  val_frac = val_size / (1 - test_size)
  X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=val_frac, random_state=random_state
  )

  if messages:
    total = len(df)
    print(f"Train: {len(X_train)} ({len(X_train)/total:.0%})  "
          f"Val: {len(X_val)} ({len(X_val)/total:.0%})  "
          f"Test: {len(X_test)} ({len(X_test)/total:.0%})")

  return X_train, X_val, X_test, y_train, y_val, y_test


def predict_and_evaluate(model, X_test, y_test,
                         X_train=None, y_train=None,
                         messages=True):
  """
  Predict on a test set and compute regression performance metrics.
  Optionally compare train vs test metrics to flag overfitting.

  Parameters
  ----------
  model : fitted sklearn Pipeline or estimator
      A model with a .predict() method.
  X_test : pandas.DataFrame or numpy.ndarray
      Test features.
  y_test : pandas.Series or numpy.ndarray
      True test labels.
  X_train : pandas.DataFrame or numpy.ndarray, optional
      Training features. If provided along with y_train,
      train metrics are computed for overfitting comparison.
  y_train : pandas.Series or numpy.ndarray, optional
      True training labels.
  messages : bool
      If True, print metrics and overfitting assessment.

  Returns
  -------
  dict
      Keys: 'test_mae', 'test_rmse', 'test_r2',
      and optionally 'train_mae', 'train_rmse', 'train_r2',
      'overfit_flag' (bool).
  """
  import numpy as np
  from sklearn.metrics import (mean_absolute_error,
                               root_mean_squared_error,
                               r2_score)

  y_pred = model.predict(X_test)
  test_mae = mean_absolute_error(y_test, y_pred)
  test_rmse = root_mean_squared_error(y_test, y_pred)
  test_r2 = r2_score(y_test, y_pred)

  results = {
    "test_mae": test_mae,
    "test_rmse": test_rmse,
    "test_r2": test_r2
  }

  if messages:
    print("=== Test Set Performance ===")
    print(f"MAE:   {test_mae:,.4f}")
    print(f"RMSE:  {test_rmse:,.4f}")
    print(f"R²:    {test_r2:.4f}")

  # Optional overfitting comparison
  if X_train is not None and y_train is not None:
    y_train_pred = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = root_mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    rmse_ratio = test_rmse / train_rmse if train_rmse > 0 else 0
    r2_gap = train_r2 - test_r2
    overfit = rmse_ratio > 1.2 or r2_gap > 0.10

    results.update({
      "train_mae": train_mae,
      "train_rmse": train_rmse,
      "train_r2": train_r2,
      "overfit_flag": overfit
    })

    if messages:
      print(f"\n=== Train vs Test ===")
      print(f"Train RMSE: {train_rmse:,.4f}  "
            f"Test RMSE: {test_rmse:,.4f}  "
            f"Ratio: {rmse_ratio:.3f}")
      print(f"Train R²:   {train_r2:.4f}  "
            f"Test R²:   {test_r2:.4f}  "
            f"Gap: {r2_gap:.4f}")
      if overfit:
        print("Possible overfitting detected "
              "(RMSE ratio > 1.2 or R² gap > 0.10)")
      else:
        print("Generalization looks reasonable.")

  return results


# =============================================================================
# Chapter 12: Decision Trees for Predictive Regression
# =============================================================================

def fit_tree(df, label, features=None, max_depth=None,
             min_samples_leaf=1, min_samples_split=2,
             test_size=0.2, random_state=42, messages=True):
  """
  Build and fit a decision tree regression pipeline.

  Parameters
  ----------
  df : pandas.DataFrame
      Input DataFrame containing features and label.
  label : str
      Name of the target column.
  features : list of str, optional
      Columns to use as predictors. If None, uses all columns
      except the label.
  max_depth : int or None
      Maximum tree depth. None means unlimited.
  min_samples_leaf : int
      Minimum samples required in each leaf node.
  min_samples_split : int
      Minimum samples required to split an internal node.
  test_size : float
      Fraction of data reserved for testing.
  random_state : int
      Random seed for reproducibility.
  messages : bool
      If True, prints summary information.

  Returns
  -------
  dict
      Keys: 'model', 'X_train', 'X_test', 'y_train', 'y_test',
      'feature_names', 'metrics' (dict with mae, rmse, r2).
  """
  import pandas as pd
  import numpy as np
  from sklearn.model_selection import train_test_split
  from sklearn.compose import ColumnTransformer
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import OneHotEncoder
  from sklearn.tree import DecisionTreeRegressor
  from sklearn.metrics import (mean_absolute_error,
                               root_mean_squared_error, r2_score)

  data = df.copy()

  # Separate label
  y = data[label]
  if features is not None:
    X = data[features]
  else:
    X = data.drop(columns=[label])

  # Identify column types
  num_cols = X.select_dtypes(
      include=["int64", "float64"]
  ).columns.tolist()
  cat_cols = X.select_dtypes(
      include=["object", "category"]
  ).columns.tolist()

  # Convert booleans to int
  bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
  if bool_cols:
    X[bool_cols] = X[bool_cols].astype(int)
    num_cols += bool_cols

  # Train/test split
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=test_size, random_state=random_state
  )

  # Build preprocessing
  transformers = []
  if num_cols:
    transformers.append(("num", "passthrough", num_cols))
  if cat_cols:
    transformers.append((
        "cat",
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        cat_cols
    ))

  preprocessor = ColumnTransformer(
      transformers=transformers, remainder="drop"
  )

  # Build pipeline
  tree = DecisionTreeRegressor(
      max_depth=max_depth,
      min_samples_leaf=min_samples_leaf,
      min_samples_split=min_samples_split,
      random_state=random_state
  )

  model = Pipeline(steps=[
      ("prep", preprocessor),
      ("tree", tree)
  ])

  model.fit(X_train, y_train)

  # Build readable feature names
  feature_names = list(num_cols)
  if cat_cols:
    ohe = model.named_steps["prep"].named_transformers_["cat"]
    ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
    feature_names += ohe_names

  # Evaluate
  y_pred = model.predict(X_test)
  mae = mean_absolute_error(y_test, y_pred)
  rmse = root_mean_squared_error(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)

  metrics = {"mae": mae, "rmse": rmse, "r2": r2}

  if messages:
    print(f"Decision Tree Regression: {label}")
    print(f"  Features:       {len(num_cols)} numeric, "
          f"{len(cat_cols)} categorical")
    print(f"  max_depth:      {max_depth}")
    print(f"  min_samples_leaf: {min_samples_leaf}")
    print(f"  Train size:     {len(X_train)}")
    print(f"  Test size:      {len(X_test)}")
    print(f"  Test MAE:       {mae:,.4f}")
    print(f"  Test RMSE:      {rmse:,.4f}")
    print(f"  Test R²:        {r2:.4f}")

  return {
      "model": model,
      "X_train": X_train, "X_test": X_test,
      "y_train": y_train, "y_test": y_test,
      "feature_names": feature_names,
      "metrics": metrics
  }


def tree_summary(result, plot_depth=3, show_tree=True,
                 top_k=10, messages=True):
  """
  Summarize a fitted decision tree model.

  Parameters
  ----------
  result : dict
      Output from fit_tree(). Must contain keys: 'model',
      'X_train', 'X_test', 'y_train', 'y_test',
      'feature_names'.
  plot_depth : int
      Maximum depth shown in the tree visualization.
  show_tree : bool
      If True, renders the tree diagram.
  top_k : int
      Number of top features to show in the importance chart.
  messages : bool
      If True, prints metrics and train/test comparison.

  Returns
  -------
  dict
      Keys: 'metrics' (test mae/rmse/r2, train mae/rmse/r2,
      overfit_flag), 'importances' (pandas Series).
  """
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  from sklearn.tree import plot_tree
  from sklearn.metrics import (mean_absolute_error,
                               root_mean_squared_error, r2_score)

  model = result["model"]
  X_train = result["X_train"]
  X_test = result["X_test"]
  y_train = result["y_train"]
  y_test = result["y_test"]
  feature_names = result["feature_names"]

  # Test metrics
  y_pred = model.predict(X_test)
  test_mae = mean_absolute_error(y_test, y_pred)
  test_rmse = root_mean_squared_error(y_test, y_pred)
  test_r2 = r2_score(y_test, y_pred)

  # Train metrics
  y_train_pred = model.predict(X_train)
  train_mae = mean_absolute_error(y_train, y_train_pred)
  train_rmse = root_mean_squared_error(y_train, y_train_pred)
  train_r2 = r2_score(y_train, y_train_pred)

  # Overfitting check
  rmse_ratio = test_rmse / train_rmse if train_rmse > 0 else 0
  r2_gap = train_r2 - test_r2
  overfit = rmse_ratio > 1.2 or r2_gap > 0.10

  metrics = {
      "test_mae": test_mae, "test_rmse": test_rmse,
      "test_r2": test_r2,
      "train_mae": train_mae, "train_rmse": train_rmse,
      "train_r2": train_r2,
      "overfit_flag": overfit
  }

  if messages:
    print("=== Test Performance ===")
    print(f"MAE:  {test_mae:,.4f}")
    print(f"RMSE: {test_rmse:,.4f}")
    print(f"R²:   {test_r2:.4f}")
    print(f"\n=== Train vs Test ===")
    print(f"Train RMSE: {train_rmse:,.4f}  "
          f"Test RMSE: {test_rmse:,.4f}  "
          f"Ratio: {rmse_ratio:.3f}")
    print(f"Train R²:   {train_r2:.4f}  "
          f"Test R²:   {test_r2:.4f}  "
          f"Gap: {r2_gap:.4f}")
    if overfit:
      print("** Possible overfitting detected.")
    else:
      print("Generalization looks reasonable.")

  # Feature importance
  tree_model = model.named_steps["tree"]
  importances = pd.Series(
      tree_model.feature_importances_,
      index=feature_names
  ).sort_values(ascending=False)

  top_imp = importances.head(top_k)[::-1]
  plt.figure(figsize=(10.5, 4.8))
  plt.barh(top_imp.index, top_imp.values)
  plt.xlabel("Impurity-based importance")
  plt.title("Feature Importance (top features)")
  plt.tight_layout()
  plt.show()

  # Tree visualization
  if show_tree:
    plt.figure(figsize=(16, 6))
    plot_tree(
        tree_model,
        feature_names=feature_names,
        filled=True, rounded=True,
        max_depth=plot_depth, fontsize=8
    )
    plt.title("Decision Tree Structure")
    plt.tight_layout()
    plt.show()

  return {"metrics": metrics, "importances": importances}


def tree_depth_sweep(df, label, depths=None, features=None,
                     min_samples_leaf=1, test_size=0.2,
                     random_state=42, messages=True):
  """
  Train trees at multiple depths and compare train vs test error.

  Parameters
  ----------
  df : pandas.DataFrame
      Input DataFrame containing features and label.
  label : str
      Name of the target column.
  depths : list of int or None
      max_depth values to try. If None, uses
      [2, 3, 4, 5, 6, 8, 10, None].
  features : list of str, optional
      Columns to use as predictors.
  min_samples_leaf : int
      Minimum samples per leaf (held constant across depths).
  test_size : float
      Fraction of data reserved for testing.
  random_state : int
      Random seed for reproducibility.
  messages : bool
      If True, prints the best depth and shows the plot.

  Returns
  -------
  pandas.DataFrame
      One row per depth with columns: max_depth, train_mae,
      train_rmse, train_r2, test_mae, test_rmse, test_r2.
  """
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  from sklearn.model_selection import train_test_split
  from sklearn.compose import ColumnTransformer
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import OneHotEncoder
  from sklearn.tree import DecisionTreeRegressor
  from sklearn.metrics import (mean_absolute_error,
                               root_mean_squared_error, r2_score)

  if depths is None:
    depths = [2, 3, 4, 5, 6, 8, 10, None]

  data = df.copy()
  y = data[label]
  if features is not None:
    X = data[features]
  else:
    X = data.drop(columns=[label])

  num_cols = X.select_dtypes(
      include=["int64", "float64"]
  ).columns.tolist()
  cat_cols = X.select_dtypes(
      include=["object", "category"]
  ).columns.tolist()

  bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
  if bool_cols:
    X[bool_cols] = X[bool_cols].astype(int)
    num_cols += bool_cols

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=test_size, random_state=random_state
  )

  transformers = []
  if num_cols:
    transformers.append(("num", "passthrough", num_cols))
  if cat_cols:
    transformers.append((
        "cat",
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        cat_cols
    ))

  rows = []
  for d in depths:
    preprocessor = ColumnTransformer(
        transformers=transformers, remainder="drop"
    )
    tree = DecisionTreeRegressor(
        max_depth=d,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    model = Pipeline(steps=[
        ("prep", preprocessor),
        ("tree", tree)
    ])
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    rows.append({
        "max_depth": d if d is not None else "None",
        "train_mae": mean_absolute_error(y_train, y_train_pred),
        "train_rmse": root_mean_squared_error(y_train, y_train_pred),
        "train_r2": r2_score(y_train, y_train_pred),
        "test_mae": mean_absolute_error(y_test, y_test_pred),
        "test_rmse": root_mean_squared_error(y_test, y_test_pred),
        "test_r2": r2_score(y_test, y_test_pred)
    })

  results = pd.DataFrame(rows)

  if messages:
    # Find best depth
    best_idx = results["test_rmse"].idxmin()
    best_depth = results.loc[best_idx, "max_depth"]
    best_rmse = results.loc[best_idx, "test_rmse"]
    print(f"Best depth: {best_depth}  "
          f"(Test RMSE: {best_rmse:,.4f})")

    # Plot
    plot_depths = [
        str(d) if d != "None" else "None"
        for d in results["max_depth"]
    ]
    plt.figure(figsize=(8, 5))
    plt.plot(plot_depths, results["train_rmse"],
             marker="o", label="Train RMSE")
    plt.plot(plot_depths, results["test_rmse"],
             marker="o", label="Test RMSE")
    plt.xlabel("max_depth")
    plt.ylabel("RMSE")
    plt.title("Overfitting check: RMSE vs tree depth")
    plt.legend()
    plt.tight_layout()
    plt.show()

  return results


def extract_tree_rules(result, max_rules=None, messages=True):
  """
  Extract human-readable decision rules from a fitted tree pipeline.

  Parameters
  ----------
  result : dict
      Output from fit_tree(). Must contain 'model' (fitted Pipeline)
      and 'feature_names'.
  max_rules : int or None
      Maximum number of rules to return. None returns all leaf rules.
  messages : bool
      If True, prints each rule.

  Returns
  -------
  list of str
      Each string is an IF-THEN rule describing the path from root
      to a leaf node.
  """
  import numpy as np

  model = result["model"]
  feature_names = result["feature_names"]
  tree_model = model.named_steps["tree"]
  tree = tree_model.tree_

  children_left = tree.children_left
  children_right = tree.children_right
  features = tree.feature
  thresholds = tree.threshold
  values = tree.value.flatten()
  n_samples = tree.n_node_samples

  rules = []

  def _recurse(node, conditions):
    if children_left[node] == children_right[node]:
      # Leaf node
      pred = values[node]
      n = n_samples[node]
      if conditions:
        rule = "IF " + " AND ".join(conditions)
      else:
        rule = "IF (root)"
      rule += f" THEN predicted = {pred:,.2f}  (n={n})"
      rules.append(rule)
      return

    feat_name = feature_names[features[node]]
    thresh = thresholds[node]

    # Left child: feature <= threshold
    left_cond = f"{feat_name} <= {thresh:.4f}"
    _recurse(children_left[node], conditions + [left_cond])

    # Right child: feature > threshold
    right_cond = f"{feat_name} > {thresh:.4f}"
    _recurse(children_right[node], conditions + [right_cond])

  _recurse(0, [])

  if max_rules is not None:
    rules = rules[:max_rules]

  if messages:
    print(f"Decision rules ({len(rules)} leaves):\n")
    for i, rule in enumerate(rules, 1):
      print(f"  Rule {i}: {rule}")

  return rules


# ============================================================
# Chapter 13: Classification Modeling
# ============================================================

def fit_classifier(df, label, algorithm="logistic", features=None,
                   test_size=0.2, random_state=42, messages=True,
                   **model_params):
  """
  Build a classification pipeline and fit it.

  Parameters
  ----------
  df : DataFrame
      The full dataset (features + label).
  label : str
      Name of the target column.
  algorithm : str
      One of 'logistic', 'tree', 'knn', 'nb'.
  features : list or None
      Columns to use as predictors. None = all columns except label.
  test_size : float
      Proportion held out for testing.
  random_state : int
      Seed for reproducibility.
  messages : bool
      If True, print status messages.
  **model_params
      Extra keyword arguments passed to the classifier constructor.

  Returns
  -------
  dict with keys: model, X_train, X_test, y_train, y_test,
                  feature_names, y_pred, y_prob, algorithm
  """
  import pandas as pd
  import numpy as np
  from sklearn.model_selection import train_test_split
  from sklearn.compose import ColumnTransformer
  from sklearn.preprocessing import StandardScaler, OneHotEncoder
  from sklearn.pipeline import Pipeline
  from sklearn.linear_model import LogisticRegression
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.naive_bayes import GaussianNB
  from sklearn.preprocessing import FunctionTransformer

  data = df.copy()
  y = data[label].copy()
  if features is not None:
    X = data[features].copy()
  else:
    X = data.drop(columns=[label]).copy()

  # Identify column types
  cat_cols = X.select_dtypes(
    include=["object", "category", "bool"]
  ).columns.tolist()
  num_cols = X.select_dtypes(include=["number"]).columns.tolist()

  # Build preprocessor
  preprocessor = ColumnTransformer(
    transformers=[
      ("num", StandardScaler(), num_cols),
      ("cat", OneHotEncoder(handle_unknown="ignore",
                            sparse_output=False), cat_cols)
    ],
    remainder="drop"
  )

  # Select algorithm
  algorithms = {
    "logistic": LogisticRegression(
      max_iter=2000,
      random_state=random_state,
      **model_params
    ),
    "tree": DecisionTreeClassifier(
      random_state=random_state,
      **model_params
    ),
    "knn": KNeighborsClassifier(**model_params),
    "nb": GaussianNB(**model_params),
  }

  algo_key = algorithm.lower()
  if algo_key not in algorithms:
    raise ValueError(
      f"algorithm must be one of {list(algorithms.keys())}"
    )

  clf = algorithms[algo_key]

  # Build pipeline
  steps = [("prep", preprocessor)]

  # GaussianNB requires dense input
  if algo_key == "nb":
    to_dense = FunctionTransformer(
      lambda X: X.toarray() if hasattr(X, "toarray")
                else np.asarray(X),
      accept_sparse=True
    )
    steps.append(("dense", to_dense))

  steps.append(("clf", clf))
  model = Pipeline(steps=steps)

  # Split and fit
  X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    random_state=random_state,
    stratify=y
  )

  model.fit(X_train, y_train)

  # Predictions
  y_pred = model.predict(X_test)
  y_prob = None
  if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(X_test)

  # Feature names
  feature_names = model.named_steps["prep"].get_feature_names_out()

  if messages:
    print(f"Fitted {algorithm} classifier")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples:     {len(X_test)}")
    print(f"  Features:         {len(feature_names)}")

  return {
    "model": model,
    "X_train": X_train,
    "X_test": X_test,
    "y_train": y_train,
    "y_test": y_test,
    "feature_names": feature_names,
    "y_pred": y_pred,
    "y_prob": y_prob,
    "algorithm": algorithm,
  }


def classification_report_custom(result, sort_by="f1", roundto=4,
                                 show_matrix=True, messages=True):
  """
  Evaluate a fitted classifier and return a per-class metrics DataFrame.

  Parameters
  ----------
  result : dict
      Output from fit_classifier().
  sort_by : str
      Column to sort the per-class table by: 'precision', 'recall',
      'f1', or 'support'.
  roundto : int
      Decimal places for printed metrics.
  show_matrix : bool
      If True, display a confusion matrix plot.
  messages : bool
      If True, print summary metrics and classification report.

  Returns
  -------
  DataFrame with per-class precision, recall, f1-score, and support.
  """
  import pandas as pd
  import numpy as np
  from sklearn.metrics import (accuracy_score, log_loss,
                               classification_report,
                               confusion_matrix,
                               ConfusionMatrixDisplay)

  y_test = result["y_test"]
  y_pred = result["y_pred"]
  y_prob = result["y_prob"]
  algo = result.get("algorithm", "classifier")

  acc = accuracy_score(y_test, y_pred)

  ll = None
  if y_prob is not None:
    ll = log_loss(y_test, y_prob)

  if messages:
    print("=" * 60)
    print(f"  {algo.upper()} — Classification Results")
    print("=" * 60)
    print(f"  Accuracy : {round(acc, roundto)}")
    if ll is not None:
      print(f"  Log Loss : {round(ll, roundto)}")
    else:
      print("  Log Loss : n/a (no predict_proba)")
    print()
    print(classification_report(y_test, y_pred,
                                digits=roundto,
                                zero_division=0))

  if show_matrix and messages:
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm).plot(
      values_format="d", cmap="Blues", ax=ax
    )
    ax.set_title(f"{algo} — Confusion Matrix")
    plt.tight_layout()
    plt.show()

  # Build per-class DataFrame
  report_dict = classification_report(
    y_test, y_pred, output_dict=True, zero_division=0
  )
  class_rows = {
    k: v for k, v in report_dict.items()
    if k not in ("accuracy", "macro avg", "weighted avg")
  }
  metrics_df = pd.DataFrame(class_rows).T
  metrics_df.index.name = "class"

  valid_sorts = ["precision", "recall", "f1-score", "support"]
  sort_col = "f1-score" if sort_by == "f1" else sort_by
  if sort_col in metrics_df.columns:
    metrics_df = metrics_df.sort_values(
      by=sort_col, ascending=False
    )

  return metrics_df


def compare_classifiers(df, label, algorithms=None, features=None,
                        test_size=0.2, random_state=42,
                        plot=True, messages=True):
  """
  Train multiple classifiers and return a comparison table.

  Parameters
  ----------
  df : DataFrame
      The full dataset (features + label).
  label : str
      Name of the target column.
  algorithms : list of str or None
      Algorithm names for fit_classifier().
      Default: ['logistic', 'tree', 'knn', 'nb'].
  features : list or None
      Columns to use as predictors. None = all except label.
  test_size : float
      Proportion held out for testing.
  random_state : int
      Seed for reproducibility.
  plot : bool
      If True, display a grouped bar chart comparing models.
  messages : bool
      If True, print progress messages.

  Returns
  -------
  DataFrame with columns: algorithm, accuracy, log_loss
  """
  import pandas as pd
  import numpy as np
  from sklearn.metrics import accuracy_score, log_loss

  if algorithms is None:
    algorithms = ["logistic", "tree", "knn", "nb"]

  rows = []
  for algo in algorithms:
    result = fit_classifier(
      df, label,
      algorithm=algo,
      features=features,
      test_size=test_size,
      random_state=random_state,
      messages=False
    )

    acc = accuracy_score(result["y_test"], result["y_pred"])
    ll = None
    if result["y_prob"] is not None:
      ll = log_loss(result["y_test"], result["y_prob"])

    rows.append({
      "algorithm": algo,
      "accuracy": round(acc, 4),
      "log_loss": round(ll, 4) if ll is not None else None
    })

    if messages:
      ll_str = f"{round(ll, 4)}" if ll is not None else "n/a"
      print(f"  {algo:12s}  acc={round(acc, 4)}  log_loss={ll_str}")

  comparison = pd.DataFrame(rows)
  comparison = comparison.sort_values(
    by=["log_loss", "accuracy"],
    ascending=[True, False]
  ).reset_index(drop=True)

  if plot:
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = range(len(comparison))
    ax1.bar(x, comparison["accuracy"], color="steelblue",
            alpha=0.8, label="Accuracy")
    ax1.set_ylabel("Accuracy", color="steelblue")
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(x)
    ax1.set_xticklabels(comparison["algorithm"], rotation=30,
                        ha="right")

    ax2 = ax1.twinx()
    ll_vals = comparison["log_loss"].fillna(0)
    ax2.plot(x, ll_vals, color="tomato", marker="o",
             linewidth=2, label="Log Loss")
    ax2.set_ylabel("Log Loss", color="tomato")

    fig.suptitle("Classifier Comparison", fontsize=13)
    fig.tight_layout()
    plt.show()

  return comparison


def threshold_analysis(result, pos_label=1, thresholds=None,
                       plot=True, messages=True):
  """
  Sweep probability thresholds and compute classification metrics.

  Parameters
  ----------
  result : dict
      Output from fit_classifier() (must include y_prob).
  pos_label : int or str
      The positive class label.
  thresholds : list or None
      Thresholds to evaluate. Default: 0.1 to 0.9 by 0.05.
  plot : bool
      If True, plot precision, recall, and F1 vs threshold.
  messages : bool
      If True, print the results table.

  Returns
  -------
  DataFrame with columns: threshold, precision, recall, f1
  """
  import pandas as pd
  import numpy as np
  from sklearn.metrics import precision_score, recall_score, f1_score

  y_test = result["y_test"]
  y_prob = result["y_prob"]

  if y_prob is None:
    raise ValueError("Result must include y_prob (predict_proba).")

  # Find the column index for the positive class
  model = result["model"]
  classes = model.classes_
  pos_idx = list(classes).index(pos_label)
  probs = y_prob[:, pos_idx]

  if thresholds is None:
    thresholds = np.arange(0.10, 0.95, 0.05).round(2).tolist()

  rows = []
  for t in thresholds:
    y_pred_t = (probs >= t).astype(int)
    # Map back to original class labels
    y_pred_labels = np.where(y_pred_t == 1, pos_label,
                             [c for c in classes if c != pos_label][0])

    prec = precision_score(y_test, y_pred_labels,
                           pos_label=pos_label, zero_division=0)
    rec = recall_score(y_test, y_pred_labels,
                       pos_label=pos_label, zero_division=0)
    f1 = f1_score(y_test, y_pred_labels,
                  pos_label=pos_label, zero_division=0)

    rows.append({
      "threshold": t,
      "precision": round(prec, 4),
      "recall": round(rec, 4),
      "f1": round(f1, 4)
    })

  df_out = pd.DataFrame(rows)

  if messages:
    print(df_out.to_string(index=False))

  if plot:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_out["threshold"], df_out["precision"],
            marker="o", label="Precision")
    ax.plot(df_out["threshold"], df_out["recall"],
            marker="s", label="Recall")
    ax.plot(df_out["threshold"], df_out["f1"],
            marker="^", label="F1")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Threshold Analysis (positive class = {pos_label})")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

  return df_out


# ============================================================
# Chapter 14: Ensemble Methods
# ============================================================

def fit_ensemble(df, label, algorithm="random_forest", features=None,
                 test_size=0.2, random_state=42, messages=True,
                 **model_params):
  """
  Build an ensemble classification pipeline and fit it.

  Parameters
  ----------
  df : DataFrame
      The full dataset (features + label).
  label : str
      Name of the target column.
  algorithm : str
      One of 'random_forest', 'bagging', 'adaboost',
      'gradient_boosting', 'xgboost', 'stacking'.
  features : list or None
      Columns to use as predictors. None = all except label.
  test_size : float
      Proportion held out for testing.
  random_state : int
      Seed for reproducibility.
  messages : bool
      If True, print status messages.
  **model_params
      Extra keyword arguments passed to the ensemble constructor.

  Returns
  -------
  dict with keys: model, X_train, X_test, y_train, y_test,
                  feature_names, y_pred, y_prob, algorithm,
                  train_seconds
  """
  import time
  import pandas as pd
  import numpy as np
  from sklearn.model_selection import train_test_split
  from sklearn.compose import ColumnTransformer
  from sklearn.preprocessing import StandardScaler, OneHotEncoder
  from sklearn.pipeline import Pipeline
  from sklearn.ensemble import (RandomForestClassifier,
                                BaggingClassifier,
                                AdaBoostClassifier,
                                GradientBoostingClassifier,
                                StackingClassifier)
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.linear_model import LogisticRegression

  data = df.copy()
  y = data[label].copy()
  if features is not None:
    X = data[features].copy()
  else:
    X = data.drop(columns=[label]).copy()

  cat_cols = X.select_dtypes(
    include=["object", "category", "bool"]
  ).columns.tolist()
  num_cols = X.select_dtypes(include=["number"]).columns.tolist()

  preprocessor = ColumnTransformer(
    transformers=[
      ("num", StandardScaler(), num_cols),
      ("cat", OneHotEncoder(handle_unknown="ignore",
                            sparse_output=False), cat_cols)
    ],
    remainder="drop"
  )

  algorithms = {
    "random_forest": RandomForestClassifier(
      n_estimators=200, random_state=random_state,
      n_jobs=-1, **model_params
    ),
    "bagging": BaggingClassifier(
      estimator=DecisionTreeClassifier(
        max_depth=3, random_state=random_state
      ),
      n_estimators=100, random_state=random_state,
      n_jobs=-1, **model_params
    ),
    "adaboost": AdaBoostClassifier(
      estimator=DecisionTreeClassifier(
        max_depth=1, random_state=random_state
      ),
      n_estimators=100, random_state=random_state,
      **model_params
    ),
    "gradient_boosting": GradientBoostingClassifier(
      n_estimators=200, learning_rate=0.05,
      max_depth=3, random_state=random_state,
      **model_params
    ),
    "stacking": StackingClassifier(
      estimators=[
        ("lr", LogisticRegression(
          max_iter=2000,
          random_state=random_state)),
        ("rf", RandomForestClassifier(
          n_estimators=100,
          random_state=random_state,
          n_jobs=-1)),
      ],
      final_estimator=LogisticRegression(
        max_iter=2000,
        random_state=random_state
      ),
      stack_method="predict_proba",
      cv=5, n_jobs=-1, **model_params
    ),
  }

  algo_key = algorithm.lower()

  # XGBoost handled separately (optional dependency)
  if algo_key == "xgboost":
    from xgboost import XGBClassifier
    clf = XGBClassifier(
      objective="binary:logistic",
      eval_metric="logloss",
      n_estimators=400, learning_rate=0.05,
      max_depth=3, random_state=random_state,
      n_jobs=-1, **model_params
    )
  elif algo_key in algorithms:
    clf = algorithms[algo_key]
  else:
    raise ValueError(
      f"algorithm must be one of "
      f"{list(algorithms.keys()) + ['xgboost']}"
    )

  model = Pipeline(steps=[("prep", preprocessor), ("clf", clf)])

  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size,
    random_state=random_state, stratify=y
  )

  t0 = time.time()
  model.fit(X_train, y_train)
  train_seconds = round(time.time() - t0, 3)

  y_pred = model.predict(X_test)
  y_prob = None
  if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(X_test)

  feature_names = model.named_steps["prep"].get_feature_names_out()

  if messages:
    print(f"Fitted {algorithm} ensemble")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples:     {len(X_test)}")
    print(f"  Features:         {len(feature_names)}")
    print(f"  Training time:    {train_seconds}s")

  return {
    "model": model,
    "X_train": X_train, "X_test": X_test,
    "y_train": y_train, "y_test": y_test,
    "feature_names": feature_names,
    "y_pred": y_pred, "y_prob": y_prob,
    "algorithm": algorithm,
    "train_seconds": train_seconds,
  }


def compare_models(df, label, algorithms=None, features=None,
                   test_size=0.2, random_state=42,
                   plot=True, messages=True):
  """
  Train multiple ensemble/baseline models and compare them.

  Parameters
  ----------
  df : DataFrame
  label : str
  algorithms : list of str or None
      Algorithm names for fit_ensemble().
      Default: ['random_forest', 'bagging', 'adaboost',
                'gradient_boosting'].
  features : list or None
  test_size : float
  random_state : int
  plot : bool
      If True, display dual-axis chart (accuracy + log loss).
  messages : bool

  Returns
  -------
  DataFrame with columns: algorithm, accuracy, log_loss,
                          train_seconds
  """
  import pandas as pd
  from sklearn.metrics import accuracy_score, log_loss

  if algorithms is None:
    algorithms = [
      "random_forest", "bagging", "adaboost",
      "gradient_boosting"
    ]

  rows = []
  for algo in algorithms:
    result = fit_ensemble(
      df, label, algorithm=algo,
      features=features, test_size=test_size,
      random_state=random_state, messages=False
    )

    acc = accuracy_score(result["y_test"], result["y_pred"])
    ll = None
    if result["y_prob"] is not None:
      ll = log_loss(result["y_test"], result["y_prob"])

    rows.append({
      "algorithm": algo,
      "accuracy": round(acc, 4),
      "log_loss": round(ll, 4) if ll is not None else None,
      "train_seconds": result["train_seconds"]
    })

    if messages:
      ll_str = f"{round(ll, 4)}" if ll else "n/a"
      print(f"  {algo:22s}  acc={round(acc, 4)}"
            f"  log_loss={ll_str}"
            f"  time={result['train_seconds']}s")

  comparison = pd.DataFrame(rows).sort_values(
    by=["log_loss", "accuracy"],
    ascending=[True, False]
  ).reset_index(drop=True)

  if plot:
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(9, 5))
    x = range(len(comparison))
    ax1.bar(x, comparison["accuracy"], color="steelblue",
            alpha=0.8, label="Accuracy")
    ax1.set_ylabel("Accuracy", color="steelblue")
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(x)
    ax1.set_xticklabels(comparison["algorithm"],
                        rotation=35, ha="right")

    ax2 = ax1.twinx()
    ll_vals = comparison["log_loss"].fillna(0)
    ax2.plot(x, ll_vals, color="tomato", marker="o",
             linewidth=2, label="Log Loss")
    ax2.set_ylabel("Log Loss", color="tomato")

    fig.suptitle("Ensemble Comparison", fontsize=13)
    fig.tight_layout()
    plt.show()

  return comparison


def ensemble_feature_importance(results, top_k=15,
                                plot=True, messages=True):
  """
  Compare feature importances across tree-based ensembles.

  Parameters
  ----------
  results : list of dict
      Outputs from fit_ensemble() for tree-based algorithms.
  top_k : int
      Number of top features to display.
  plot : bool
      If True, plot grouped horizontal bar chart.
  messages : bool
      If True, print summary.

  Returns
  -------
  DataFrame with feature names and importance per algorithm.
  """
  import pandas as pd
  import numpy as np

  combined = None

  for result in results:
    algo = result["algorithm"]
    model = result["model"]
    clf = model.named_steps["clf"]

    if not hasattr(clf, "feature_importances_"):
      if messages:
        print(f"  Skipping {algo} (no feature_importances_)")
      continue

    feature_names = result["feature_names"]
    importances = clf.feature_importances_

    df_imp = pd.DataFrame({
      "feature": feature_names,
      algo: importances
    }).set_index("feature")

    if combined is None:
      combined = df_imp
    else:
      combined = combined.join(df_imp, how="outer")

  if combined is None:
    if messages:
      print("No tree-based models found.")
    return pd.DataFrame()

  combined = combined.fillna(0)

  # Sort by mean importance across algorithms
  combined["mean"] = combined.mean(axis=1)
  combined = combined.sort_values("mean", ascending=False)
  top = combined.drop(columns=["mean"]).head(top_k)

  if messages:
    print(f"Top {top_k} features by mean importance:")
    print(top.round(4).to_string())

  if plot:
    import matplotlib.pyplot as plt

    top_plot = top.iloc[::-1]  # reverse for horizontal bar
    fig, ax = plt.subplots(figsize=(8, max(4, top_k * 0.35)))
    top_plot.plot.barh(ax=ax, alpha=0.8)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_k} Feature Importances by Algorithm")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

  return top


def ensemble_depth_sweep(df, label, algorithm="random_forest",
                         param_name="n_estimators",
                         param_values=None, features=None,
                         test_size=0.2, random_state=42,
                         plot=True, messages=True):
  """
  Sweep a hyperparameter for an ensemble algorithm.

  Parameters
  ----------
  df : DataFrame
  label : str
  algorithm : str
      Algorithm name for fit_ensemble().
  param_name : str
      Hyperparameter to sweep (e.g., 'n_estimators',
      'max_depth', 'learning_rate').
  param_values : list or None
      Values to try. Default depends on param_name.
  features : list or None
  test_size : float
  random_state : int
  plot : bool
      If True, plot train vs test curves.
  messages : bool

  Returns
  -------
  DataFrame with columns: param_value, train_accuracy,
      test_accuracy, train_log_loss, test_log_loss
  """
  import pandas as pd
  import numpy as np
  from sklearn.metrics import accuracy_score, log_loss

  if param_values is None:
    defaults = {
      "n_estimators": [10, 25, 50, 100, 200, 300, 500],
      "max_depth": [1, 2, 3, 4, 5, 6, 8, 10, 15, None],
      "learning_rate": [0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
    }
    param_values = defaults.get(
      param_name,
      [10, 25, 50, 100, 200, 300, 500]
    )

  rows = []
  for val in param_values:
    kwargs = {param_name: val}
    result = fit_ensemble(
      df, label, algorithm=algorithm,
      features=features, test_size=test_size,
      random_state=random_state, messages=False,
      **kwargs
    )

    model = result["model"]
    X_train = result["X_train"]
    y_train = result["y_train"]
    X_test = result["X_test"]
    y_test = result["y_test"]

    # Train metrics
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    train_ll = None
    if hasattr(model, "predict_proba"):
      y_train_prob = model.predict_proba(X_train)
      train_ll = log_loss(y_train, y_train_prob)

    # Test metrics
    test_acc = accuracy_score(y_test, result["y_pred"])
    test_ll = None
    if result["y_prob"] is not None:
      test_ll = log_loss(y_test, result["y_prob"])

    rows.append({
      "param_value": val,
      "train_accuracy": round(train_acc, 4),
      "test_accuracy": round(test_acc, 4),
      "train_log_loss": round(train_ll, 4) if train_ll else None,
      "test_log_loss": round(test_ll, 4) if test_ll else None,
    })

    if messages:
      label_str = str(val) if val is not None else "None"
      print(f"  {param_name}={label_str:>6s}  "
            f"train_acc={round(train_acc, 4)}  "
            f"test_acc={round(test_acc, 4)}")

  df_out = pd.DataFrame(rows)

  if plot:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x_vals = [str(v) for v in df_out["param_value"]]

    # Accuracy panel
    axes[0].plot(x_vals, df_out["train_accuracy"],
                 marker="o", label="Train")
    axes[0].plot(x_vals, df_out["test_accuracy"],
                 marker="s", label="Test")
    axes[0].set_xlabel(param_name)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy vs " + param_name)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis="x", rotation=45)

    # Log loss panel
    if df_out["train_log_loss"].notna().any():
      axes[1].plot(x_vals, df_out["train_log_loss"],
                   marker="o", label="Train")
      axes[1].plot(x_vals, df_out["test_log_loss"],
                   marker="s", label="Test")
      axes[1].set_xlabel(param_name)
      axes[1].set_ylabel("Log Loss")
      axes[1].set_title("Log Loss vs " + param_name)
      axes[1].legend()
      axes[1].grid(True, alpha=0.3)
      axes[1].tick_params(axis="x", rotation=45)

    fig.suptitle(f"{algorithm} — {param_name} sweep",
                 fontsize=13)
    plt.tight_layout()
    plt.show()

  return df_out


###############################################################################
# Chapter 15: Model Evaluation, Selection & Tuning
###############################################################################

def _get_estimator_map(task, random_state):
  """
  Return a dict mapping algorithm name -> sklearn estimator.

  Parameters
  ----------
  task : str
      'classify' or 'regress'.
  random_state : int
      Random seed for reproducibility.

  Returns
  -------
  dict
      Algorithm name -> unfitted estimator instance.
  """
  from sklearn.linear_model import (LogisticRegression, LinearRegression,
                                    Ridge)
  from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
  from sklearn.ensemble import (RandomForestClassifier,
                                RandomForestRegressor,
                                GradientBoostingClassifier,
                                GradientBoostingRegressor)
  from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
  from sklearn.svm import SVC, SVR
  from sklearn.naive_bayes import GaussianNB

  if task == "classify":
    return {
      "logistic": LogisticRegression(max_iter=3000,
                                     random_state=random_state),
      "tree": DecisionTreeClassifier(random_state=random_state),
      "random_forest": RandomForestClassifier(
        n_estimators=200, random_state=random_state, n_jobs=-1),
      "gradient_boosting": GradientBoostingClassifier(
        n_estimators=200, random_state=random_state),
      "knn": KNeighborsClassifier(),
      "svm": SVC(probability=True, random_state=random_state),
      "naive_bayes": GaussianNB()
    }
  else:
    return {
      "linear": LinearRegression(),
      "ridge": Ridge(random_state=random_state),
      "tree": DecisionTreeRegressor(random_state=random_state),
      "random_forest": RandomForestRegressor(
        n_estimators=200, random_state=random_state, n_jobs=-1),
      "gradient_boosting": GradientBoostingRegressor(
        n_estimators=200, random_state=random_state),
      "knn": KNeighborsRegressor(),
      "svr": SVR()
    }


def algorithm_selector(df, label, algorithms=None, features=None,
                       task="classify", test_size=0.2, cv=5,
                       scoring=None, random_state=42, messages=True):
  """
  Cross-validate multiple algorithms on the same data with consistent
  preprocessing, returning a comparison DataFrame and fitted results.

  Parameters
  ----------
  df : DataFrame
      The full dataset (features + label).
  label : str
      Name of the target column.
  algorithms : list of str, optional
      Algorithm names to compare. Defaults depend on task:
      classify: ['logistic', 'tree', 'random_forest', 'gradient_boosting']
      regress:  ['linear', 'ridge', 'tree', 'random_forest']
  features : list of str, optional
      Columns to use as predictors. If None, uses all columns except label.
  task : str
      'classify' or 'regress'.
  test_size : float
      Fraction held out for the final test set.
  cv : int
      Number of cross-validation folds.
  scoring : dict, optional
      Scoring metrics dict for cross_validate. If None, uses sensible
      defaults per task.
  random_state : int
      Random seed for reproducibility.
  messages : bool
      If True, prints progress and summary information.

  Returns
  -------
  dict with keys:
      'comparison' : DataFrame sorted by primary metric (descending)
      'results'    : dict mapping algorithm name -> fitted pipeline
      'X_train', 'X_test', 'y_train', 'y_test' : train/test splits
  """
  import pandas as pd
  import numpy as np
  import time
  from sklearn.model_selection import train_test_split, cross_validate
  from sklearn.model_selection import StratifiedKFold, KFold
  from sklearn.pipeline import Pipeline
  from sklearn.compose import ColumnTransformer
  from sklearn.preprocessing import StandardScaler, OneHotEncoder
  from sklearn.impute import SimpleImputer

  data = df.copy()

  # --- features and label --------------------------------------------------
  if features is not None:
    data = data[features + [label]]
  y = data[label]
  X = data.drop(columns=[label])

  numeric_cols = X.select_dtypes(include="number").columns.tolist()
  categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

  # --- train / test split --------------------------------------------------
  stratify = y if task == "classify" else None
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state,
    stratify=stratify
  )

  # --- preprocessing pipeline ----------------------------------------------
  numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
  ])
  categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore",
                              sparse_output=False))
  ])
  preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("cat", categorical_pipe, categorical_cols)
  ], remainder="drop")

  # --- default algorithms --------------------------------------------------
  if algorithms is None:
    algorithms = (
      ["logistic", "tree", "random_forest", "gradient_boosting"]
      if task == "classify"
      else ["linear", "ridge", "tree", "random_forest"]
    )

  estimator_map = _get_estimator_map(task, random_state)

  # --- default scoring -----------------------------------------------------
  if scoring is None:
    if task == "classify":
      scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
        "roc_auc_ovr": "roc_auc_ovr"
      }
    else:
      scoring = {
        "r2": "r2",
        "neg_mae": "neg_mean_absolute_error",
        "neg_rmse": "neg_root_mean_squared_error"
      }

  # --- cross-validate each algorithm ---------------------------------------
  cv_obj = (
    StratifiedKFold(n_splits=cv, shuffle=True,
                    random_state=random_state)
    if task == "classify"
    else KFold(n_splits=cv, shuffle=True,
               random_state=random_state)
  )

  rows = []
  fitted = {}
  primary_metric = (list(scoring.keys())[1] if task == "classify"
                    else list(scoring.keys())[0])

  for algo_name in algorithms:
    if algo_name not in estimator_map:
      if messages:
        print(f"  [skip] Unknown algorithm: {algo_name}")
      continue
    model = Pipeline([
      ("preprocessor", preprocessor),
      ("model", estimator_map[algo_name])
    ])
    if messages:
      print(f"  Evaluating {algo_name}...", end=" ")
    t0 = time.time()
    cv_results = cross_validate(
      model, X_train, y_train, cv=cv_obj,
      scoring=scoring, n_jobs=-1, return_train_score=False
    )
    elapsed = time.time() - t0
    row = {"Algorithm": algo_name, "CV Time (sec)": round(elapsed, 2)}
    for key, vals in cv_results.items():
      if key.startswith("test_"):
        metric = key.replace("test_", "")
        row[f"{metric} (mean)"] = round(np.mean(vals), 4)
        row[f"{metric} (std)"] = round(np.std(vals), 4)
    rows.append(row)
    model.fit(X_train, y_train)
    fitted[algo_name] = model
    if messages:
      print(f"done ({elapsed:.1f}s)")

  comparison = pd.DataFrame(rows)
  sort_col = f"{primary_metric} (mean)"
  if sort_col in comparison.columns:
    comparison = comparison.sort_values(sort_col, ascending=False)
  comparison = comparison.reset_index(drop=True)

  if messages:
    print(f"\nComparison sorted by {primary_metric}:")
    print(comparison.to_string(index=False))

  return {
    "comparison": comparison,
    "results": fitted,
    "X_train": X_train, "X_test": X_test,
    "y_train": y_train, "y_test": y_test
  }


def model_comparison_report(comparison, primary_metric="balanced_accuracy",
                            plot=True, roundto=4, messages=True):
  """
  Format and visualize an algorithm comparison table.

  Parameters
  ----------
  comparison : DataFrame
      Output from algorithm_selector()['comparison'].
  primary_metric : str
      Metric name used for sorting and chart emphasis.
  plot : bool
      If True, produces a grouped bar chart.
  roundto : int
      Decimal places for display.
  messages : bool
      If True, prints formatted summary.

  Returns
  -------
  DataFrame sorted by primary_metric (descending).
  """
  import pandas as pd
  import numpy as np

  df = comparison.copy()
  sort_col = f"{primary_metric} (mean)"
  std_col = f"{primary_metric} (std)"

  if sort_col in df.columns:
    df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)

  if messages:
    print("=" * 60)
    print("MODEL COMPARISON REPORT")
    print("=" * 60)
    print(f"Primary metric: {primary_metric}")
    print(f"Number of candidates: {len(df)}")
    print("-" * 60)
    for i, row in df.iterrows():
      score = row.get(sort_col, float("nan"))
      std = row.get(std_col, float("nan"))
      marker = " <-- best" if i == 0 else ""
      print(f"  {row['Algorithm']:<25} "
            f"{score:.{roundto}f} +/- {std:.{roundto}f}"
            f"  ({row.get('CV Time (sec)', 0):.1f}s){marker}")
    print("-" * 60)

    if (len(df) >= 2 and sort_col in df.columns
        and std_col in df.columns):
      top = df.iloc[0][sort_col]
      second = df.iloc[1][sort_col]
      top_std = df.iloc[0][std_col]
      gap = top - second
      if gap < top_std:
        print(f"  Note: top two models differ by {gap:.{roundto}f}, "
              f"which is less than one std ({top_std:.{roundto}f}).")
        print("  Consider choosing the simpler or faster model.")
      else:
        print(f"  The top model leads by {gap:.{roundto}f} "
              f"(> 1 std of {top_std:.{roundto}f}).")
    print("=" * 60)

  if plot:
    import matplotlib.pyplot as plt

    mean_cols = [c for c in df.columns if c.endswith("(mean)")
                 and c != "CV Time (sec)"]
    if mean_cols:
      fig, ax = plt.subplots(figsize=(10, 5))
      x = np.arange(len(df))
      width = 0.8 / len(mean_cols)
      for j, col in enumerate(mean_cols):
        metric_label = col.replace(" (mean)", "")
        bars = ax.bar(x + j * width, df[col], width,
                      label=metric_label, alpha=0.85)
        if metric_label == primary_metric:
          for bar in bars:
            bar.set_edgecolor("black")
            bar.set_linewidth(1.5)
      ax.set_xticks(x + width * (len(mean_cols) - 1) / 2)
      ax.set_xticklabels(df["Algorithm"], rotation=15, ha="right")
      ax.set_ylabel("Score")
      ax.set_title("Algorithm Comparison")
      ax.legend(fontsize=8)
      plt.tight_layout()
      plt.show()

  return df


def learning_curve_report(pipeline, X, y, cv=5,
                          scoring="balanced_accuracy",
                          train_sizes=None, random_state=42,
                          plot=True, messages=True):
  """
  Generate learning curves and diagnose bias/variance patterns.

  Parameters
  ----------
  pipeline : sklearn Pipeline
      A fitted or unfitted pipeline (will be cloned internally).
  X : DataFrame or array
      Feature matrix (training data only).
  y : Series or array
      Target vector (training data only).
  cv : int
      Number of cross-validation folds.
  scoring : str
      Scoring metric for learning_curve.
  train_sizes : array-like, optional
      Fractions or absolute numbers of training examples.
      Defaults to np.linspace(0.1, 1.0, 10).
  random_state : int
      Random seed for reproducibility.
  plot : bool
      If True, produces a learning curve plot with std bands.
  messages : bool
      If True, prints a diagnostic summary.

  Returns
  -------
  dict with keys:
      'train_sizes'  : array of actual training set sizes
      'train_scores' : mean training scores per size
      'val_scores'   : mean validation scores per size
      'diagnosis'    : str ('high_bias', 'high_variance', or 'well_fit')
  """
  import numpy as np
  from sklearn.model_selection import learning_curve

  if train_sizes is None:
    train_sizes = np.linspace(0.1, 1.0, 10)

  sizes, train_scores, val_scores = learning_curve(
    pipeline, X, y, cv=cv, scoring=scoring,
    train_sizes=train_sizes, n_jobs=-1,
    random_state=random_state
  )

  train_mean = np.mean(train_scores, axis=1)
  train_std = np.std(train_scores, axis=1)
  val_mean = np.mean(val_scores, axis=1)
  val_std = np.std(val_scores, axis=1)

  # --- diagnosis -----------------------------------------------------------
  gap = train_mean[-1] - val_mean[-1]
  val_slope = val_mean[-1] - val_mean[len(val_mean) // 2]
  train_level = train_mean[-1]

  if gap < 0.03 and train_level < 0.7:
    diagnosis = "high_bias"
  elif gap > 0.10:
    diagnosis = "high_variance"
  elif gap < 0.03 and val_slope < 0.01:
    diagnosis = "well_fit"
  elif val_slope > 0.02:
    diagnosis = "high_variance"
  else:
    diagnosis = "well_fit"

  if messages:
    print(f"Learning Curve Diagnosis: {diagnosis}")
    print(f"  Final train score: {train_mean[-1]:.4f}")
    print(f"  Final val score:   {val_mean[-1]:.4f}")
    print(f"  Gap:               {gap:.4f}")
    if diagnosis == "high_bias":
      print("  -> Both curves are low and close together.")
      print("     More data alone is unlikely to help.")
      print("     Consider a more complex model or better features.")
    elif diagnosis == "high_variance":
      print("  -> Training score is high but validation lags behind.")
      print("     The model may be overfitting.")
      print("     Consider more data, regularization, or simpler model.")
    else:
      print("  -> Both curves are high and converging.")
      print("     The model appears well-fit for this data.")

  if plot:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(sizes, train_mean - train_std,
                    train_mean + train_std, alpha=0.15, color="blue")
    ax.fill_between(sizes, val_mean - val_std,
                    val_mean + val_std, alpha=0.15, color="orange")
    ax.plot(sizes, train_mean, "o-", color="blue", label="Training")
    ax.plot(sizes, val_mean, "s-", color="orange", label="Validation")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel(scoring)
    ax.set_title(f"Learning Curve ({diagnosis.replace('_', ' ')})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

  return {
    "train_sizes": sizes,
    "train_scores": train_mean,
    "val_scores": val_mean,
    "diagnosis": diagnosis
  }


def tuning_pipeline(df, label, algorithm="gradient_boosting",
                    features=None, task="classify", test_size=0.2,
                    cv=5, scoring=None, budget=50,
                    random_state=42, messages=True):
  """
  Automatically tune an algorithm with sensible hyperparameter search
  spaces, choosing grid or randomized search based on space size.

  Parameters
  ----------
  df : DataFrame
      The full dataset (features + label).
  label : str
      Name of the target column.
  algorithm : str
      Algorithm name (same names as algorithm_selector).
  features : list of str, optional
      Columns to use as predictors. If None, uses all except label.
  task : str
      'classify' or 'regress'.
  test_size : float
      Fraction held out for the test set.
  cv : int
      Number of cross-validation folds.
  scoring : str, optional
      Single scoring metric. Defaults to 'balanced_accuracy' for
      classification or 'r2' for regression.
  budget : int
      Maximum number of fits (controls grid vs random choice).
  random_state : int
      Random seed for reproducibility.
  messages : bool
      If True, prints progress and results.

  Returns
  -------
  dict with keys:
      'best_estimator' : fitted Pipeline with best hyperparameters
      'best_score'     : float, best cross-validated score
      'best_params'    : dict of best hyperparameter values
      'cv_results'     : DataFrame of all evaluated configurations
      'X_train', 'X_test', 'y_train', 'y_test' : train/test splits
  """
  import pandas as pd
  import numpy as np
  from sklearn.model_selection import (train_test_split, GridSearchCV,
                                       RandomizedSearchCV,
                                       StratifiedKFold, KFold)
  from sklearn.pipeline import Pipeline
  from sklearn.compose import ColumnTransformer
  from sklearn.preprocessing import StandardScaler, OneHotEncoder
  from sklearn.impute import SimpleImputer
  from scipy.stats import loguniform, randint, uniform

  data = df.copy()

  # --- features and label --------------------------------------------------
  if features is not None:
    data = data[features + [label]]
  y = data[label]
  X = data.drop(columns=[label])

  numeric_cols = X.select_dtypes(include="number").columns.tolist()
  categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

  # --- train / test split --------------------------------------------------
  stratify = y if task == "classify" else None
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state,
    stratify=stratify
  )

  # --- preprocessing -------------------------------------------------------
  numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
  ])
  categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore",
                              sparse_output=False))
  ])
  preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("cat", categorical_pipe, categorical_cols)
  ], remainder="drop")

  # --- estimator and search space ------------------------------------------
  estimator_map = _get_estimator_map(task, random_state)
  if algorithm not in estimator_map:
    raise ValueError(f"Unknown algorithm: {algorithm}. "
                     f"Options: {list(estimator_map.keys())}")

  pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", estimator_map[algorithm])
  ])

  param_spaces = _get_param_spaces(algorithm, task)

  # --- scoring default -----------------------------------------------------
  if scoring is None:
    scoring = ("balanced_accuracy" if task == "classify" else "r2")

  # --- CV object -----------------------------------------------------------
  cv_obj = (
    StratifiedKFold(n_splits=cv, shuffle=True,
                    random_state=random_state)
    if task == "classify"
    else KFold(n_splits=cv, shuffle=True,
               random_state=random_state)
  )

  # --- choose grid vs random based on space size ---------------------------
  grid_size = 1
  is_grid = True
  for key, vals in param_spaces.items():
    if hasattr(vals, "__len__"):
      grid_size *= len(vals)
    else:
      is_grid = False
      break

  if is_grid and grid_size <= budget:
    if messages:
      print(f"  Grid search ({grid_size} combinations)...")
    searcher = GridSearchCV(
      pipe, param_spaces, cv=cv_obj, scoring=scoring,
      n_jobs=-1, return_train_score=True
    )
  else:
    if messages:
      print(f"  Randomized search (budget={budget})...")
    searcher = RandomizedSearchCV(
      pipe, param_spaces, n_iter=budget, cv=cv_obj,
      scoring=scoring, n_jobs=-1, random_state=random_state,
      return_train_score=True
    )

  searcher.fit(X_train, y_train)

  if messages:
    print(f"  Best {scoring}: {searcher.best_score_:.4f}")
    print(f"  Best params: {searcher.best_params_}")

  cv_results_df = pd.DataFrame(searcher.cv_results_)

  return {
    "best_estimator": searcher.best_estimator_,
    "best_score": searcher.best_score_,
    "best_params": searcher.best_params_,
    "cv_results": cv_results_df,
    "X_train": X_train, "X_test": X_test,
    "y_train": y_train, "y_test": y_test
  }


def _get_param_spaces(algorithm, task):
  """Return default hyperparameter search spaces for common algorithms."""
  from scipy.stats import loguniform, randint

  prefix = "model__"
  spaces = {
    "logistic": {
      f"{prefix}C": loguniform(1e-3, 1e3),
      f"{prefix}l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0]
    },
    "tree": {
      f"{prefix}max_depth": [3, 5, 7, 10, 15, None],
      f"{prefix}min_samples_split": [2, 5, 10, 20],
      f"{prefix}min_samples_leaf": [1, 2, 5, 10]
    },
    "random_forest": {
      f"{prefix}n_estimators": [100, 200, 300],
      f"{prefix}max_depth": [5, 10, 15, None],
      f"{prefix}min_samples_split": [2, 5, 10],
      f"{prefix}min_samples_leaf": [1, 2, 5]
    },
    "gradient_boosting": {
      f"{prefix}n_estimators": [100, 200, 300],
      f"{prefix}max_depth": [3, 5, 7],
      f"{prefix}learning_rate": [0.01, 0.05, 0.1, 0.2],
      f"{prefix}subsample": [0.8, 0.9, 1.0]
    },
    "knn": {
      f"{prefix}n_neighbors": [3, 5, 7, 11, 15, 21],
      f"{prefix}weights": ["uniform", "distance"],
      f"{prefix}metric": ["euclidean", "manhattan"]
    },
    "svm": {
      f"{prefix}C": loguniform(1e-2, 1e2),
      f"{prefix}kernel": ["rbf", "linear"],
      f"{prefix}gamma": ["scale", "auto"]
    },
    "naive_bayes": {
      f"{prefix}var_smoothing": loguniform(1e-12, 1e-6)
    },
    "ridge": {
      f"{prefix}alpha": loguniform(1e-3, 1e3)
    },
    "svr": {
      f"{prefix}C": loguniform(1e-2, 1e2),
      f"{prefix}kernel": ["rbf", "linear"],
      f"{prefix}gamma": ["scale", "auto"]
    }
  }
  return spaces.get(algorithm, {})


###############################################################################
# Chapter 16: Feature Selection
###############################################################################

def feature_selector(df, label, method="embedded", features=None,
                     k=10, task="classify", cv=5, scoring=None,
                     random_state=42, messages=True):
  """
  Apply a feature selection method inside a leakage-safe pipeline
  and return selected feature names with cross-validated performance.

  Parameters
  ----------
  df : DataFrame
      The full dataset (features + label).
  label : str
      Name of the target column.
  method : str
      'filter' (SelectKBest), 'wrapper' (RFECV), or
      'embedded' (SelectFromModel with gradient boosting).
  features : list of str, optional
      Columns to use as predictors. If None, uses all except label.
  k : int
      Number of features to select (used by filter and wrapper).
  task : str
      'classify' or 'regress'.
  cv : int
      Number of cross-validation folds.
  scoring : str, optional
      Scoring metric. Defaults to 'roc_auc' (classify) or 'r2' (regress).
  random_state : int
      Random seed for reproducibility.
  messages : bool
      If True, prints progress and results.

  Returns
  -------
  dict with keys:
      'selected_features' : list of str
      'cv_score_mean'     : float
      'cv_score_std'      : float
      'pipeline'          : fitted Pipeline
  """
  import pandas as pd
  import numpy as np
  from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
  from sklearn.pipeline import Pipeline
  from sklearn.compose import ColumnTransformer
  from sklearn.preprocessing import StandardScaler, OneHotEncoder
  from sklearn.impute import SimpleImputer
  from sklearn.feature_selection import (SelectKBest, f_classif,
                                         f_regression, RFECV,
                                         SelectFromModel)
  from sklearn.ensemble import (GradientBoostingClassifier,
                                GradientBoostingRegressor)

  data = df.copy()
  if features is not None:
    data = data[features + [label]]
  y = data[label]
  X = data.drop(columns=[label])

  numeric_cols = X.select_dtypes(include="number").columns.tolist()
  categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

  # --- preprocessing -------------------------------------------------------
  numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
  ])
  categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore",
                              sparse_output=False))
  ])
  preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("cat", categorical_pipe, categorical_cols)
  ], remainder="drop")

  # --- scoring and CV ------------------------------------------------------
  if scoring is None:
    scoring = "roc_auc" if task == "classify" else "r2"
  cv_obj = (
    StratifiedKFold(n_splits=cv, shuffle=True,
                    random_state=random_state)
    if task == "classify"
    else KFold(n_splits=cv, shuffle=True,
               random_state=random_state)
  )

  # --- estimator -----------------------------------------------------------
  if task == "classify":
    estimator = GradientBoostingClassifier(
      n_estimators=200, max_depth=3, random_state=random_state)
  else:
    estimator = GradientBoostingRegressor(
      n_estimators=200, max_depth=3, random_state=random_state)

  # --- selection step ------------------------------------------------------
  if method == "filter":
    score_func = f_classif if task == "classify" else f_regression
    selector = SelectKBest(score_func, k=k)
  elif method == "wrapper":
    selector = RFECV(estimator, step=1, cv=cv_obj, scoring=scoring,
                     n_jobs=-1, min_features_to_select=5)
  elif method == "embedded":
    selector = SelectFromModel(estimator, threshold="mean")
  else:
    raise ValueError(f"Unknown method: {method}. "
                     "Use 'filter', 'wrapper', or 'embedded'.")

  # --- pipeline ------------------------------------------------------------
  pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("selector", selector),
    ("model", estimator)
  ])

  if messages:
    print(f"  Running {method} selection with {scoring}...", end=" ")

  scores = cross_val_score(pipe, X, y, cv=cv_obj, scoring=scoring,
                           n_jobs=-1)

  # fit once to extract selected features
  pipe.fit(X, y)

  # get feature names after preprocessing
  feature_names = pipe.named_steps["preprocessor"].get_feature_names_out()
  mask = pipe.named_steps["selector"].get_support()
  selected = [f for f, m in zip(feature_names, mask) if m]

  if messages:
    print("done")
    print(f"  CV {scoring}: {scores.mean():.4f} "
          f"(+/- {scores.std():.4f})")
    print(f"  Selected {len(selected)} features:")
    for f in selected:
      print(f"    {f}")

  return {
    "selected_features": selected,
    "cv_score_mean": scores.mean(),
    "cv_score_std": scores.std(),
    "pipeline": pipe
  }


def feature_importance_report(pipeline, X, y, top_k=15,
                              scoring="roc_auc", plot=True,
                              messages=True):
  """
  Compute and compare feature importance metrics for a fitted pipeline.

  Parameters
  ----------
  pipeline : sklearn Pipeline
      A fitted pipeline with a 'preprocessor' and 'model' step.
  X : DataFrame
      Feature matrix (test or validation set recommended).
  y : Series
      Target vector.
  top_k : int
      Number of top features to display.
  scoring : str
      Scoring metric for permutation importance.
  plot : bool
      If True, produces a side-by-side bar chart.
  messages : bool
      If True, prints summary.

  Returns
  -------
  DataFrame with columns for each importance metric, sorted by PFI.
  """
  import pandas as pd
  import numpy as np
  from sklearn.inspection import permutation_importance
  from sklearn.preprocessing import MinMaxScaler

  # get feature names from preprocessor
  feature_names = (
    pipeline.named_steps["preprocessor"].get_feature_names_out()
  )
  model = pipeline.named_steps["model"]

  # --- permutation importance ----------------------------------------------
  pfi = permutation_importance(pipeline, X, y, scoring=scoring,
                               n_repeats=10, random_state=42,
                               n_jobs=-1)
  df = pd.DataFrame({
    "feature": feature_names,
    "pfi_mean": pfi.importances_mean,
    "pfi_std": pfi.importances_std
  })

  # --- built-in importance (tree-based) ------------------------------------
  if hasattr(model, "feature_importances_"):
    df["mdi"] = model.feature_importances_

  # --- coefficients (linear models) ----------------------------------------
  if hasattr(model, "coef_"):
    coefs = (model.coef_.flatten() if model.coef_.ndim > 1
             else model.coef_)
    if len(coefs) == len(feature_names):
      df["abs_coef"] = np.abs(coefs)

  # --- normalize for comparison --------------------------------------------
  scaler = MinMaxScaler()
  importance_cols = [c for c in df.columns
                     if c not in ["feature", "pfi_std"]]
  for col in importance_cols:
    df[col + "_norm"] = scaler.fit_transform(df[[col]])

  df = df.sort_values("pfi_mean", ascending=False).reset_index(drop=True)

  if messages:
    display_cols = (
      ["feature", "pfi_mean", "pfi_std"]
      + [c for c in df.columns if c in ("mdi", "abs_coef")]
    )
    print(f"Top {top_k} features by permutation importance:")
    print(df.head(top_k)[display_cols].to_string(index=False))

  if plot:
    import matplotlib.pyplot as plt

    top = df.head(top_k)
    norm_cols = [c for c in top.columns if c.endswith("_norm")]
    if norm_cols:
      fig, ax = plt.subplots(figsize=(10, 6))
      x = np.arange(len(top))
      width = 0.8 / len(norm_cols)
      for j, col in enumerate(norm_cols):
        label = col.replace("_norm", "").upper()
        ax.barh(x + j * width, top[col], width,
                label=label, alpha=0.85)
      ax.set_yticks(x + width * (len(norm_cols) - 1) / 2)
      ax.set_yticklabels(top["feature"], fontsize=8)
      ax.set_xlabel("Normalized Importance")
      ax.set_title(f"Feature Importance Comparison (top {top_k})")
      ax.legend(fontsize=8)
      ax.invert_yaxis()
      plt.tight_layout()
      plt.show()

  return df


# ============================================================
# Chapter 17: Deploying ML Pipelines
# ============================================================

def save_model(model, path, version="1.0.0", label=None,
               features=None, metrics=None, messages=True):
  """
  Save a trained model, metadata, and optional metrics to disk.

  Parameters
  ----------
  model : object
      Any fitted model or Pipeline that joblib can serialize.
  path : str or Path
      Destination file path for the model artifact (e.g.
      "artifacts/my_model.sav").
  version : str
      Semantic version string stored in metadata.
  label : str or None
      Target column name (recorded in metadata).
  features : list or None
      Feature names used during training (recorded in metadata).
  metrics : dict or None
      Evaluation metrics to persist alongside the model.
  messages : bool
      If True, print confirmation messages.

  Returns
  -------
  dict
      Keys: model_path, metadata_path, metrics_path (or None).
  """
  import joblib, json
  from pathlib import Path
  from datetime import datetime, timezone

  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)

  # Save model artifact
  joblib.dump(model, str(path))

  # Build metadata
  metadata = {
    "model_version": version,
    "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    "model_file": path.name,
  }
  if label:
    metadata["label"] = label
  if features:
    metadata["features"] = list(features)

  meta_path = path.with_name("model_metadata.json")
  with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

  # Optional metrics file
  metrics_path = None
  if metrics:
    metrics_path = path.with_name("metrics.json")
    serializable = {}
    for k, v in metrics.items():
      try:
        json.dumps(v)
        serializable[k] = v
      except TypeError:
        serializable[k] = str(v)
    with open(metrics_path, "w", encoding="utf-8") as f:
      json.dump(serializable, f, indent=2)

  if messages:
    print(f"Model saved  : {path}")
    print(f"Metadata     : {meta_path}")
    if metrics_path:
      print(f"Metrics      : {metrics_path}")

  return {
    "model_path": path,
    "metadata_path": meta_path,
    "metrics_path": metrics_path,
  }


def load_and_predict(model_path, df, features=None,
                     threshold=0.5, messages=True):
  """
  Load a saved model and generate predictions on new data.

  Parameters
  ----------
  model_path : str or Path
      Path to the serialized model file.
  df : DataFrame
      Data to score. Must contain the columns the model expects.
  features : list or None
      Columns to use as predictors. If None, all numeric columns
      are used.
  threshold : float
      Classification threshold for binary models (default 0.5).
  messages : bool
      If True, print summary messages.

  Returns
  -------
  DataFrame
      Copy of df with prediction columns appended:
      - Classifiers: predicted_label, predicted_prob
      - Regressors: predicted_value
  """
  import joblib
  import pandas as pd
  import numpy as np

  model = joblib.load(str(model_path))
  out = df.copy()

  # Determine feature matrix
  if features:
    X = out[features]
  else:
    X = out.select_dtypes(include=[np.number])

  # Classification vs regression
  has_proba = hasattr(model, "predict_proba")

  if has_proba:
    out["predicted_label"] = model.predict(X)
    probs = model.predict_proba(X)
    if probs.shape[1] == 2:
      out["predicted_prob"] = probs[:, 1]
      out["predicted_label"] = (out["predicted_prob"] >= threshold).astype(int)
    else:
      out["predicted_prob"] = probs.max(axis=1)
  else:
    out["predicted_value"] = model.predict(X)

  if messages:
    print(f"Model loaded : {model_path}")
    print(f"Rows scored  : {len(out):,}")
    if has_proba:
      pos = (out["predicted_label"] == 1).sum()
      print(f"Predicted positive: {pos:,} "
            f"({pos / len(out):.1%}) at threshold {threshold}")

  return out


# ============================================================
# Chapter 18: Monitoring and Managing ML Pipelines
# ============================================================

def monitor_drift(df_reference, df_current, y_pred,
                  numeric_cols=None, expected_positive_rate=0.30,
                  psi_threshold=0.25, perf_current=None,
                  perf_baseline=None, messages=True):
  """
  Run a complete drift check across features and predictions.

  Parameters
  ----------
  df_reference : DataFrame
      Training-time data (reference distribution).
  df_current : DataFrame
      Current operational data to compare against reference.
  y_pred : array-like
      Binary predictions from the current model run.
  numeric_cols : list or None
      Columns to check for drift. None = all numeric columns
      in df_reference.
  expected_positive_rate : float
      Baseline positive-class rate for prediction monitoring.
  psi_threshold : float
      PSI value above which a feature is flagged as drifted.
  perf_current : float or None
      Current model performance metric (e.g. F1).
  perf_baseline : float or None
      Baseline performance metric from training.
  messages : bool
      If True, print the drift report.

  Returns
  -------
  dict
      Keys: psi (dict), drifted_features (list),
      positive_rate, prediction_shift, prediction_alert,
      diagnosis (dict or None).
  """
  import pandas as pd
  import numpy as np

  if numeric_cols is None:
    numeric_cols = df_reference.select_dtypes(
      include=[np.number]
    ).columns.tolist()

  # PSI per feature
  psi_results = {}
  drifted = []
  for col in numeric_cols:
    ref = df_reference[col].dropna().values
    cur = df_current[col].dropna().values
    if len(ref) == 0 or len(cur) == 0:
      continue
    bins = np.linspace(min(ref.min(), cur.min()),
                       max(ref.max(), cur.max()), 11)
    ref_hist = np.histogram(ref, bins=bins)[0] / len(ref)
    cur_hist = np.histogram(cur, bins=bins)[0] / len(cur)
    eps = 1e-4
    ref_hist = np.clip(ref_hist, eps, None)
    cur_hist = np.clip(cur_hist, eps, None)
    psi = float(np.sum(
      (cur_hist - ref_hist) * np.log(cur_hist / ref_hist)
    ))
    psi_results[col] = round(psi, 4)
    if psi >= psi_threshold:
      drifted.append(col)

  # Prediction distribution
  pos_rate = float(np.mean(y_pred))
  shift = abs(pos_rate - expected_positive_rate)
  pred_alert = shift > 0.15

  # Diagnosis (if performance metrics provided)
  diagnosis = None
  if perf_current is not None and perf_baseline is not None:
    perf_drop = perf_baseline - perf_current
    has_drift = len(drifted) > 0
    if perf_drop > 0.05 and has_drift:
      diagnosis = {"diagnosis": "DATA DRIFT",
                   "action": "Retrain on recent data."}
    elif perf_drop > 0.05 and not has_drift:
      diagnosis = {"diagnosis": "CONCEPT DRIFT",
                   "action": "Re-examine label definition."}
    elif perf_drop <= 0.05 and has_drift:
      diagnosis = {"diagnosis": "BENIGN DRIFT",
                   "action": "Monitor closely."}
    else:
      diagnosis = {"diagnosis": "STABLE",
                   "action": "No action needed."}

  report = {
    "psi": psi_results,
    "drifted_features": drifted,
    "positive_rate": round(pos_rate, 4),
    "prediction_shift": round(shift, 4),
    "prediction_alert": pred_alert,
    "diagnosis": diagnosis,
  }

  if messages:
    print("=== Drift Report ===")
    for col, val in psi_results.items():
      tag = "[OK]" if val < 0.10 else (
        "[MODERATE]" if val < psi_threshold else "[DRIFT]"
      )
      print(f"  {col:<30s} PSI = {val:.4f}  {tag}")
    if drifted:
      print(f"  Drifted features: {drifted}")
    print(f"\n  Positive rate: {pos_rate:.2%} "
          f"(expected {expected_positive_rate:.2%}, "
          f"shift {shift:.2%})")
    if pred_alert:
      print("  ** Prediction distribution alert **")
    if diagnosis:
      print(f"\n  Diagnosis: {diagnosis['diagnosis']}")
      print(f"  Action:    {diagnosis['action']}")

  return report


def performance_tracker(metrics, model_version, db_path,
                        feature_list=None, f1_threshold=0.70,
                        table="metrics_log", messages=True):
  """
  Log training metrics and evaluate retraining need.

  Parameters
  ----------
  metrics : dict
      Training metrics with keys like 'accuracy', 'f1', 'roc_auc'.
  model_version : str
      Semantic version string for this training run.
  db_path : str or Path
      Path to the SQLite database for the metrics log.
  feature_list : list or None
      Feature names used during training (stored for audit).
  f1_threshold : float
      Minimum acceptable F1 score. Below this triggers a
      retrain recommendation.
  table : str
      Name of the SQLite table for the metrics log.
  messages : bool
      If True, print summary messages.

  Returns
  -------
  dict
      Keys: logged_at, model_version, f1, needs_retrain,
      history (list of recent runs).
  """
  import sqlite3, json
  from datetime import datetime, timezone

  conn = sqlite3.connect(str(db_path))
  cur = conn.cursor()

  cur.execute(f"""
  CREATE TABLE IF NOT EXISTS {table} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at TEXT,
    model_version TEXT,
    accuracy REAL,
    f1 REAL,
    roc_auc REAL,
    features TEXT
  )
  """)

  ts = datetime.now(timezone.utc).isoformat()
  cur.execute(f"""
  INSERT INTO {table}
    (logged_at, model_version, accuracy, f1, roc_auc, features)
  VALUES (?, ?, ?, ?, ?, ?)
  """, (
    ts,
    model_version,
    metrics.get("accuracy"),
    metrics.get("f1"),
    metrics.get("roc_auc"),
    json.dumps(feature_list) if feature_list else None,
  ))
  conn.commit()

  # Retraining decision
  current_f1 = metrics.get("f1", 0)
  needs_retrain = current_f1 < f1_threshold

  # Trend (last 5 runs)
  cur.execute(f"""
  SELECT model_version, f1, logged_at
  FROM {table} ORDER BY id DESC LIMIT 5
  """)
  history = cur.fetchall()
  conn.close()

  result = {
    "logged_at": ts,
    "model_version": model_version,
    "f1": current_f1,
    "needs_retrain": needs_retrain,
    "history": [
      {"version": r[0], "f1": r[1], "logged_at": r[2]}
      for r in history
    ],
  }

  if messages:
    print(f"Metrics logged: v{model_version} | "
          f"F1={current_f1:.4f} | "
          f"{'RETRAIN' if needs_retrain else 'OK'}")
    if len(history) > 1:
      print("  Recent history:")
      for r in history:
        print(f"    v{r[0]}  F1={r[1]:.4f}  {r[2][:10]}")

  return result
