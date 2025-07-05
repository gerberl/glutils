"""
My humble contribution of generic ML utils to support my own projects.

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import get_scorer
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    r2_score
)
from sklearn.model_selection import KFold, RepeatedKFold, cross_validate
from sklearn.base import clone
from inspect import signature

# this is mine - one needs to
# pip install git+https://github.com/gerberl/disfunctools.git
from ..funcitertools import (
    pairwise_lst_el_combinations, pairwise_lst_el_chain,
    pairwise_lst_1st_vs_rest
)
from functools import partial
import re


# Optional swifter integration
try:
    import swifter
    _HAS_SWIFTER = True
except ImportError:
    _HAS_SWIFTER = False

# ## Data normalisation


def min_max_norm(x):
    """could rely on sklearn's implementation, but a bit of faffing involved
    with X and y shapes"""

    x = check_array(x, ensure_2d=False)
    x = (x - x.min()) / (x.max() - x.min())
    return x

def zscore_norm(x):
    """could rely on sklearn's implementation, but a bit of faffing involved
    with X and y shapes"""

    x = check_array(x, ensure_2d=False)
    x = (x - x.mean()) / (x.std())
    return x



# ## Those itertools/functools/numpy utility functions for generating pairs:
#    here, elements are columns of X

def pairwise_X_cols_combinations(X):
    """
    >>> X = np.random.randint(1, 10, (4, 5))
    >>> X
    array([[4, 2, 7, 7, 7],
           [2, 2, 6, 3, 1],
           [9, 4, 6, 7, 7],
           [2, 9, 7, 8, 3]])
    >>> list(pairwise_X_cols_combinations(X))
    [
        (array([4, 2, 9, 2]), array([2, 2, 4, 9])),
        (array([4, 2, 9, 2]), array([7, 6, 6, 7])),
        (array([4, 2, 9, 2]), array([7, 3, 7, 8])),
        (array([4, 2, 9, 2]), array([7, 1, 7, 3])),
        (array([2, 2, 4, 9]), array([7, 6, 6, 7])),
        (array([2, 2, 4, 9]), array([7, 3, 7, 8])),
        (array([2, 2, 4, 9]), array([7, 1, 7, 3])),
        (array([7, 6, 6, 7]), array([7, 3, 7, 8])),
        (array([7, 6, 6, 7]), array([7, 1, 7, 3])),
        (array([7, 3, 7, 8]), array([7, 1, 7, 3]))
    ]
    """
    return pairwise_lst_el_combinations(X.T)

def pairwise_X_cols_chain(X):
    """
    >>> X = np.random.randint(1, 10, (4, 5))
    >>> X
    array([[4, 2, 7, 7, 7],
           [2, 2, 6, 3, 1],
           [9, 4, 6, 7, 7],
           [2, 9, 7, 8, 3]])
    >>> list(pairwise_X_cols_chain(X))
    [
        (array([4, 2, 9, 2]), array([2, 2, 4, 9])),
        (array([2, 2, 4, 9]), array([7, 6, 6, 7])),
        (array([7, 6, 6, 7]), array([7, 3, 7, 8])),
        (array([7, 3, 7, 8]), array([7, 1, 7, 3]))
    ]
    """
    return pairwise_lst_el_chain(X.T)


def pairwise_X_cols_1st_vs_rest(X):
    """
    >>> X = np.random.randint(1, 10, (4, 5))
    >>> X
    array([[4, 2, 7, 7, 7],
           [2, 2, 6, 3, 1],
           [9, 4, 6, 7, 7],
           [2, 9, 7, 8, 3]])
    >>> list(pairwise_X_cols_1st_vs_rest(X))
    [
        (array([4, 2, 9, 2]), array([2, 2, 4, 9])),
        (array([4, 2, 9, 2]), array([7, 6, 6, 7])),
        (array([4, 2, 9, 2]), array([7, 3, 7, 8])),
        (array([4, 2, 9, 2]), array([7, 1, 7, 3]))
    ]
    """
    return pairwise_lst_1st_vs_rest(X.T)




# ## Model Evaluation

def get_names_func_eval_metrics():
    return dict(
        MAE=mean_absolute_error,
        MdAE=median_absolute_error,
        RMSE=root_mean_squared_error,
        MAPE=mean_absolute_percentage_error,
        r2=r2_score
    )

def get_names_of_eval_scores():
    return dict(
        nMAE='neg_mean_absolute_error',
        nMdAE='neg_median_absolute_error',
        nRMSE='neg_root_mean_squared_error',
        nMAPE='neg_mean_absolute_percentage_error',
        r2='r2'
    )


# def evaluate_regr_metrics(
#     est, 
#     X_train, 
#     y_train, 
#     X_test=None, 
#     y=None, 
#     scoring=get_names_of_eval_scores()
# ):
#     """
#     Evaluate regression metrics on train and optional test sets.

#     Parameters
#     ----------
#     est : fitted estimator
#         Trained regression model.
#     X_train : array-like
#         Training features.
#     y_train : array-like
#         Training target.
#     X_test : array-like, optional
#         Test features.
#     y : array-like, optional
#         Test target.
#     scoring : dict, optional
#         Mapping of short metric names to sklearn scorer names
#         (default: get_names_of_eval_scores()).

#     Returns
#     -------
#     pd.DataFrame
#         DataFrame with metrics as rows and 'train', 'test' columns.
#     """
#     # Check that the estimator has been fitted
#     check_is_fitted(est)

#     # Initialise dictionary to collect all scores
#     scores = {}

#     # Predict on training set
#     y_pred_train = est.predict(X_train)

#     # Evaluate scores on training set
#     for short_name, scorer_name in scoring.items():
#         # Retrieve scorer object
#         scorer = get_scorer(scorer_name)
#         # Apply the underlying scoring function
#         score = scorer._score_func(y_train, y_pred_train, **scorer._kwargs)

#         # print(f"Short name: {short_name}, Scorer name: {scorer_name}, "
#         #       f"Raw score: {score:.4f}, "
#         #       f"{'Negating' if scorer_name.startswith('neg_') else 'Keeping'}")

#         # Reverse sign if scorer is a negated loss
#         # No need with _score_func!
#         # if scorer_name.startswith('neg_'):
#         #     score *= -1
#         # Save score with 'train_' prefix and short metric name
#         scores[f"train_{short_name.lstrip('n')}"] = score

#     # Evaluate scores on test set if provided
#     if X_test is not None and y is not None:
#         # Predict on test set
#         y_pred_test = est.predict(X_test)
#         for short_name, scorer_name in scoring.items():
#             scorer = get_scorer(scorer_name)
#             score = scorer._score_func(y, y_pred_test, **scorer._kwargs)
#             # if scorer_name.startswith('neg_'):
#             #     score *= -1
#             scores[f"test_{short_name.lstrip('n')}"] = score

#     # Convert collected scores into a tidy DataFrame
#     scores_df = pd.Series(scores).to_frame(name='score')
#     scores_df = scores_df.reset_index()

#     # Split the index column into 'set' (train/test) and 'metric'
#     scores_df[['set', 'metric']] = scores_df['index'].str.split(
#         '_', n=1, expand=True
#     )

#     # Drop the original concatenated index
#     scores_df = scores_df.drop(columns=['index'])

#     # Pivot: metrics become rows, 'train'/'test' become columns
#     pivot_df = scores_df.pivot(
#         index='metric', 
#         columns='set', 
#         values='score'
#     )

#     # Safely reorder columns if they exist
#     ordered_cols = [c for c in ['train', 'test'] if c in pivot_df.columns]
#     pivot_df = pivot_df[ordered_cols]

#     return pivot_df


import pandas as pd
from sklearn.metrics import get_scorer
from sklearn.utils.validation import check_is_fitted

import pandas as pd
from sklearn.metrics import get_scorer
from sklearn.utils.validation import check_is_fitted

def evaluate_regr_metrics(
    est,
    *data,
    scoring=get_names_of_eval_scores(),
    **labelled_data
):
    """
    Evaluate regression metrics on one or more datasets.

    Parameters
    ----------
    est : fitted estimator
        Trained regression model.
    *data : tuple, optional
        Optional positional dataset (X, y), used if no named datasets provided.
    scoring : dict, optional
        Mapping of short names to scorer names (default: get_names_of_eval_scores()).
        Use short names like 'MAE', 'RMSE', etc., mapped to sklearn scorer strings.
    **labelled_data : dict
        Arbitrary keyword arguments of the form label=(X, y),
        e.g., train=(X_train, y_train), test=(X_test, y_test), etc.

    Returns
    -------
    pd.DataFrame
        DataFrame with metrics as rows and dataset labels as columns.

    Examples
    --------
    Single dataset:

    >>> evaluate_regr_metrics(est, X, y)

    Multiple labelled datasets:

    >>> evaluate_regr_metrics(
    ...     est,
    ...     train=(X_train, y_train),
    ...     test=(X_test, y_test),
    ...     valid=(X_valid, y_valid)
    ... )

    Custom scorer mapping:

    >>> scoring = {
    ...     'MAE': 'neg_mean_absolute_error',
    ...     'RMSE': 'neg_root_mean_squared_error',
    ...     'r2': 'r2'
    ... }
    >>> evaluate_regr_metrics(est, X, y, scoring=scoring)
    """
    # Validate fitted estimator
    check_is_fitted(est)

    # Handle (X, y) positional dataset
    if not labelled_data and data:
        labelled_data = {'data': data}

    if not labelled_data:
        raise ValueError("No data provided for evaluation.")

    results = []

    # Loop over datasets and compute scores
    for label, (X, y) in labelled_data.items():
        y_pred = est.predict(X)
        for short_name, scorer_name in scoring.items():
            scorer = get_scorer(scorer_name)
            score = scorer._score_func(y, y_pred, **scorer._kwargs)
            metric_name = short_name.lstrip('n')  # Clean name
            results.append({
                'set': label,
                'metric': metric_name,
                'score': score
            })

    # Construct tidy DataFrame
    scores_df = pd.DataFrame(results)

    # Pivot to metric-by-set format
    pivot_df = scores_df.pivot(index='metric', columns='set', values='score')

    # Reorder columns according to user-provided label order (if available)
    col_order = [label for label in labelled_data if label in pivot_df.columns]
    pivot_df = pivot_df[col_order]

    return pivot_df




# def cross_validate_it(
#         reg, X, y, 
#         scoring=get_names_of_eval_scores(),
#         cv=KFold(n_splits=5, shuffle=True, random_state=42)
#     ):
#     """
#     e.g.,

#     >>> cv_results_df = cross_validate_it(reg, X, y)
#     >>> cv_results_df

#                     mean       std
#     test_MAE    0.369914  0.056890
#     train_MAE   0.362901  0.014563
#     test_RMSE   0.450043  0.049646
#     train_RMSE  0.445522  0.012497
#     test_MAPE   1.246502  0.785347
#     train_MAPE  1.208433  0.155948
#     test_r2     0.034445  0.090380
#     train_r2    0.101184  0.019556
#     """

#     cv_results = cross_validate(
#         reg, X, y, scoring=scoring, cv=cv, return_train_score=True
#     )
    
#     # from cv_results, I would like to keep all scores/losses, but wouldn't
#     # need any time-related metrics here
#     scores_dict = { k: v for k, v in cv_results.items() if 'time' not in k }
#     scores = pd.DataFrame(scores_dict)
#     # all negative scores (i.e., losses) have their sign reverse (i.e., *-1)
#     # asumption is that they would have been given a prefix `_n`
#     # the sign-reversed ones are concatenated with those passed through
#     scores = pd.concat(
#         [ 
#             scores.filter(regex='_n', axis=1)*-1, 
#             scores.drop(
#                 columns=scores.filter(regex='_n', axis=1).columns, axis=1
#             )
#         ], 
#         axis=1
#     )

#     # keep only the main statistics of the distribution of the scores
#     score_stats = pd.DataFrame(
#         dict(mean=scores.mean(), std=scores.std())
#     )
#     # rename negative scores now, as they are shown as losses (just remove _n)
#     score_stats = score_stats.set_axis(
#         score_stats.index.str.replace('_n', '_'), axis=0
#     )

#     return score_stats



def cross_validate_it(
        reg, X, y, 
        scoring=None,
        cv=None,
        eval_set=None,
        early_stopping_rounds=None,
        step_prefix=None
    ):
    """
    Cross-validation with optional early stopping and pipeline step prefixing.
    """

    if scoring is None:
        scoring = get_names_of_eval_scores()

    if cv is None:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Setup fit parameters
    fit_params = {}
    if eval_set is not None:
        estimator = reg
        if step_prefix:
            estimator = getattr(reg.named_steps, step_prefix)

        monitored_params = {
            'eval_set': eval_set,
            'early_stopping_rounds': early_stopping_rounds,
            'use_best_model': True
        }

        if hasattr(estimator, "fit"):
            sig = signature(estimator.fit)
            for param_name, param_value in monitored_params.items():
                if param_value is not None and param_name in sig.parameters:
                    key = f"{step_prefix}__{param_name}" if step_prefix else param_name
                    fit_params[key] = param_value

    # Perform cross-validation (assuming sklearn ≥1.2)
    cv_results = cross_validate(
        reg, X, y,
        scoring=scoring, cv=cv,
        return_train_score=True,
        params=fit_params if fit_params else None
    )

    # Post-process scores
    scores = pd.DataFrame({
        k: v for k, v in cv_results.items() if 'time' not in k
    })

    # Reverse sign for losses marked with '_n'
    scores = pd.concat(
        [
            scores.filter(regex='_n', axis=1) * -1,
            scores.drop(columns=scores.filter(regex='_n', axis=1).columns, axis=1)
        ],
        axis=1
    )

    # Aggregate mean and std
    score_stats = pd.DataFrame({
        'mean': scores.mean(),
        'std': scores.std()
    })

    # Clean index names (remove '_n')
    score_stats.index = score_stats.index.str.replace('_n', '_')

    return score_stats






def pivot_cross_validated_stats(df):
    """
    https://chatgpt.com/c/dbc13193-7e80-4ad8-91e1-5b041b556d6e
    e.g.,

    >>> pivot_cv_results_df = pivot_cross_validated_results(cv_results_df)
    >>> pivot_cv_results_df

            test_mean  train_mean  test_std  train_std
    metric
    MAE      0.369914    0.362901  0.056890   0.014563
    MAPE     1.246502    1.208433  0.785347   0.155948
    RMSE     0.450043    0.445522  0.049646   0.012497
    r2       0.034445    0.101184  0.090380   0.019556
    """

    # Reset the index to make it a column (assuming it is the metric name)
    df = df.reset_index()
    # Split the index column into 'set' and 'metric'
    df[['set', 'metric']] = df['index'].str.split('_', expand=True)
    # Drop the original 'index' column
    df = df.drop(columns=['index'])
    # Pivot the table
    pivot_df = df.pivot(index='metric', columns='set')
    # Flatten the MultiIndex columns
    pivot_df.columns = ['_'.join(col).strip() for col in pivot_df.columns.values]
    pivot_df = pivot_df.reset_index()
    # Rename columns for clarity
    pivot_df.columns = ['metric', 'test_mean', 'train_mean', 'test_std', 'train_std']
    # Reorder for clarity
    pivot_df = pivot_df[ ['metric', 'train_mean', 'test_mean', 'train_std', 'test_std'] ]
    # Display the reshaped DataFrame
    return pivot_df.set_index('metric')



# def get_CV_train_test_scores(
#         reg, X, y, 
#         scoring=get_names_of_eval_scores(),
#         cv=KFold(n_splits=5, shuffle=True, random_state=42)
#     ):

#     cv_results_df = cross_validate_it(reg, X, y, cv=cv)
#     cv_results_df = pivot_cross_validated_stats(cv_results_df)
#     # I fancy this order of metrics
#     # !!! refactoring needed, as I am hardcoding metrics here
#     cv_results_df = cv_results_df.reindex(
#         ['MAE', 'MdAE', 'RMSE', 'MAPE', 'r2']
#     )
#     return cv_results_df


def get_CV_train_test_scores(
        reg, X, y, 
        scoring=None,
        cv=None,
        eval_set=None,
        early_stopping_rounds=None,
        step_prefix=None 
    ):
    """
    Perform cross-validation and return pivoted scores.
    """

    if scoring is None:
        scoring = get_names_of_eval_scores()

    if cv is None:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Pass optional eval_set and early stopping
    cv_results_df = cross_validate_it(
        reg, X, y,
        scoring=scoring,
        cv=cv,
        eval_set=eval_set,
        early_stopping_rounds=early_stopping_rounds,
        step_prefix=step_prefix
    )

    cv_results_df = pivot_cross_validated_stats(cv_results_df)

    # Safer metric reindexing (do not fail if some are missing)
    preferred_order = ['MAE', 'MdAE', 'RMSE', 'MAPE', 'r2']
    cv_results_df = cv_results_df.reindex(
        [metric for metric in preferred_order if metric in cv_results_df.index]
    )

    return cv_results_df





# def plot_true_vs_predicted(
#         est, 
#         X_train, y_train,
#         X_test, y_test,
#         ax=None,
#         train_style_kws={},
#         test_style_kws={}
#     ):
#     """
#     A few strong assumptions here (e.g., that there is always a test set).
#     Could do with some refactoring, but OK for the moment.
#     """
#     if ax is None:
#         fig, ax = plt.subplots(constrained_layout=True)

#     y_pred_train = est.predict(X_train)
#     y_pred_test = est.predict(X_test)
#     ax.plot(y_train, y_pred_train, '.', label='train', **train_style_kws)
#     ax.plot(y_test, y_pred_test, '.', label='test', **test_style_kws)
#     ax.set_xlabel('True Target')
#     ax.set_ylabel('Predicted Target')

#     # need to make both axis have the same (x,y)-limits
#     all_target_values = np.concatenate([y_pred_train, y_pred_test, y_train, y_test])
#     min_target = min(all_target_values)
#     max_target = max(all_target_values)
#     target_lim = (min_target, max_target)
#     ax.set_xlim(target_lim)
#     ax.set_ylim(target_lim)

#     # "nudge" extremes slightly so that all data points are visible
#     xlim = ax.get_xlim()
#     ylim = ax.get_ylim()
#     x_margin = (xlim[1] - xlim[0]) * 0.01  # 5% margin
#     y_margin = (ylim[1] - ylim[0]) * 0.01  # 5% margin
#     ax.set_xlim(xlim[0] - x_margin, xlim[1] + x_margin)
#     ax.set_ylim(ylim[0] - y_margin, ylim[1] + y_margin)    

#     # the diagnonal line for the idealised space of predictions
#     ax.plot(
#         [0, 1], [0, 1], transform=ax.transAxes, 
#         color='gray', linestyle=':', alpha=0.3
#     )
#     ax.legend()

#     return ax


# def plot_true_vs_predicted(
#         est, 
#         X_train, y_train,
#         X_test=None, y_test=None,
#         ax=None,
#         train_style_kws={},
#         test_style_kws={}
#     ):
#     """
#     Plots true vs predicted values for training data, and optionally for test data.
#     """
#     if ax is None:
#         fig, ax = plt.subplots(constrained_layout=True)

#     # Predict on training data
#     y_pred_train = est.predict(X_train)
#     ax.plot(y_train, y_pred_train, '.', label='train', **train_style_kws)

#     # Predict on test data, if available
#     if X_test is not None and y_test is not None:
#         y_pred_test = est.predict(X_test)
#         ax.plot(y_test, y_pred_test, '.', label='test', **test_style_kws)
#         all_target_values = np.concatenate([y_pred_train, y_train,
#                                             y_pred_test, y_test])
#     else:
#         all_target_values = np.concatenate([y_pred_train, y_train])

#     ax.set_xlabel('True Target')
#     ax.set_ylabel('Predicted Target')

#     # Axis limits and margins
#     min_target = min(all_target_values)
#     max_target = max(all_target_values)
#     target_lim = (min_target, max_target)
#     ax.set_xlim(target_lim)
#     ax.set_ylim(target_lim)

#     xlim = ax.get_xlim()
#     ylim = ax.get_ylim()
#     x_margin = (xlim[1] - xlim[0]) * 0.01
#     y_margin = (ylim[1] - ylim[0]) * 0.01
#     ax.set_xlim(xlim[0] - x_margin, xlim[1] + x_margin)
#     ax.set_ylim(ylim[0] - y_margin, ylim[1] + y_margin)

#     # Reference diagonal
#     ax.plot([0, 1], [0, 1], transform=ax.transAxes,
#             color='gray', linestyle=':', alpha=0.3)
#     ax.legend()

#     return ax


import matplotlib.pyplot as plt
import numpy as np

# def plot_true_vs_predicted(
#     est,
#     *data,  # optional unnamed (X, y) dataset
#     ax=None,
#     style_kws=None,  # optional: dict of label -> style mappings
#     **labelled_data  # labelled datasets like train=(X_train, y_train), etc.
# ):
#     """
#     Plot true vs predicted values for one or more labelled datasets.

#     Parameters
#     ----------
#     est : fitted estimator
#         Trained model.
#     *data : tuple, optional
#         Optional unnamed (X, y) dataset (labelled as 'data').
#     ax : matplotlib axis, optional
#         Axis object to plot on. Creates a new one if None.
#     style_kws : dict, optional
#         Dictionary mapping label to style keyword arguments.
#     **labelled_data : dict
#         Arbitrary keyword arguments of the form label=(X, y).

#     Returns
#     -------
#     ax : matplotlib axis
#         Axis containing the plot.
#     """
#     if ax is None:
#         fig, ax = plt.subplots(constrained_layout=True)

#     # Handle default (X, y) if no labelled data provided
#     if not labelled_data and data:
#         labelled_data = {'data': data}

#     if not labelled_data:
#         raise ValueError("No data provided for plotting.")

#     # Collect all true and predicted values for axis scaling
#     all_target_values = []

#     # Iterate over labelled datasets
#     for label, (X, y) in labelled_data.items():
#         # Predict
#         y_pred = est.predict(X)
        
#         # Determine style for this label
#         style = {}
#         if style_kws and label in style_kws:
#             style = style_kws[label]

#         # Plot true vs predicted points
#         ax.plot(
#             y, y_pred, '.', 
#             label=label, 
#             **style
#         )

#         # Collect values for axis limits
#         all_target_values.append(y)
#         all_target_values.append(y_pred)

#     # Axis labels
#     ax.set_xlabel('True Target')
#     ax.set_ylabel('Predicted Target')

#     # Axis limits and margins
#     all_target_values = np.concatenate(all_target_values)
#     min_target = np.min(all_target_values)
#     max_target = np.max(all_target_values)
#     target_lim = (min_target, max_target)
#     ax.set_xlim(target_lim)
#     ax.set_ylim(target_lim)

#     xlim = ax.get_xlim()
#     ylim = ax.get_ylim()
#     x_margin = (xlim[1] - xlim[0]) * 0.01
#     y_margin = (ylim[1] - ylim[0]) * 0.01
#     ax.set_xlim(xlim[0] - x_margin, xlim[1] + x_margin)
#     ax.set_ylim(ylim[0] - y_margin, ylim[1] + y_margin)

#     # Reference diagonal (ideal predictions)
#     ax.plot(
#         [0, 1], [0, 1], transform=ax.transAxes,
#         color='gray', linestyle=':', alpha=0.3
#     )

#     ax.legend()

#     return ax


import matplotlib.pyplot as plt
import numpy as np

def plot_true_vs_predicted(
    est,
    *data,
    ax=None,
    style_kws=None,
    **labelled_data
):
    """
    Plot true vs predicted values for one or more datasets.

    Parameters
    ----------
    est : fitted estimator
        Trained model.
    *data : tuple, optional
        Optional positional dataset (X, y). Used if no labelled datasets are provided.
    ax : matplotlib axis, optional
        Axis object to plot on. Creates a new one if None.
    style_kws : dict, optional
        Styles to apply:
        - If a single dataset is provided positionally: style_kws is used directly.
        - If multiple labelled datasets are provided: style_kws maps label -> style dict.
    **labelled_data : dict
        Arbitrary keyword arguments like train=(X_train, y_train), test=(X_test, y_test), etc.

    Returns
    -------
    ax : matplotlib axis
        Axis containing the plot.

    Examples
    --------
    Single dataset (positional):

    >>> plot_true_vs_predicted(est, X, y)

    Single dataset with custom style:

    >>> plot_true_vs_predicted(
    ...     est, X, y,
    ...     style_kws={'alpha': 0.5, 'marker': 'x'}
    ... )

    Multiple labelled datasets:

    >>> plot_true_vs_predicted(
    ...     est,
    ...     train=(X_train, y_train),
    ...     test=(X_test, y_test),
    ...     style_kws={
    ...         'train': {'alpha': 0.5, 'color': 'blue'},
    ...         'test': {'alpha': 0.5, 'color': 'red', 'marker': 'x'}
    ...     }
    ... )
    """
    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True)

    # Detect whether using positional or labelled datasets
    positional_data = False
    if not labelled_data and data:
        labelled_data = {'data': data}
        positional_data = True

    if not labelled_data:
        raise ValueError("No data provided for plotting.")

    # Collect all true and predicted values for setting axis limits
    all_target_values = []

    # Iterate over labelled datasets
    for label, (X, y) in labelled_data.items():
        # Predict
        y_pred = est.predict(X)

        # Determine style
        style = {}
        if style_kws:
            if positional_data:
                style = style_kws
            elif label in style_kws:
                style = style_kws[label]

        # Plot true vs predicted points
        ax.plot(
            y, y_pred, '.',
            label=label,
            **style
        )

        # Collect values for axis limits
        all_target_values.append(y)
        all_target_values.append(y_pred)

    # Axis labels
    ax.set_xlabel('True Target')
    ax.set_ylabel('Predicted Target')

    # Set axis limits with margins
    all_target_values = np.concatenate(all_target_values)
    min_target = np.min(all_target_values)
    max_target = np.max(all_target_values)
    target_lim = (min_target, max_target)
    ax.set_xlim(target_lim)
    ax.set_ylim(target_lim)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_margin = (xlim[1] - xlim[0]) * 0.01
    y_margin = (ylim[1] - ylim[0]) * 0.01
    ax.set_xlim(xlim[0] - x_margin, xlim[1] + x_margin)
    ax.set_ylim(ylim[0] - y_margin, ylim[1] + y_margin)

    # Reference diagonal
    ax.plot(
        [0, 1], [0, 1], transform=ax.transAxes,
        color='gray', linestyle=':', alpha=0.3
    )

    ax.legend()

    return ax






def plot_rf_feat_imp_barh(rf, feat_names, ax=None, top_feat_k=10, style_kws={}):
    """ """
    if ax is None:
        fig, ax = plt.subplots()
    
    return pd.Series(
        rf.feature_importances_, 
        index=feat_names
    ).sort_values().tail(top_feat_k).plot.barh(**style_kws)



# ## Hyper-Parameter Search

# def refit_strategy(cv_results):
#     """
#     9-Sep-24: I don't think I need this!

#     To ensure that the best model is that with the best test score, I've
#     found adding `refit=refit_strategy` to my hyps enough to do the job:

#     Example:
#     grid_search = GridSearchCV(
#         estimator=polyb, param_grid=param_grid, cv=5, return_train_score=True,
#         refit=refit_strategy
#     )
#     """
#     df = pd.DataFrame(cv_results)
#     df = df.sort_values(by='mean_test_score', ascending=False)
#     return df.head(1).index.to_numpy().item()

def format_cv_results(grid_search, verbose=False, train_score=True, sort_by_rank=True):
    """
    Return a DataFrame with GridSearchCV results (and the like) in a format that
    I prefer; Only parameters, optionally train and test score means and
    standard deviations, and rank based on test score. `param` and
    (optionally) pipeline prefixes are removed.
    """
    cv_results_full = pd.DataFrame(grid_search.cv_results_)

    cols = [ ]
    cols += cv_results_full.filter(regex=r'param_').columns.tolist()
    
    score_features = [ ]
    if train_score:
        score_features += [ 'mean_train_score', 'std_train_score' ]
    score_features += [ 'mean_test_score', 'std_test_score' ]
    score_features += ['rank_test_score']
    
    cols += score_features

    hs_results = cv_results_full[cols]

    # hopefully this removes the pipeline column name prefixes if asked
    if not verbose:
        hs_results = hs_results.rename(
            columns=lambda col: re.sub('.+__', '', col)
        )

    if sort_by_rank:
        hs_results = hs_results.sort_values('rank_test_score', ascending=True)
    
    return hs_results


def format_optuna_hs_results(
        optuna_search, verbose=False, train_score=True, sort_by_rank=True
    ):

    hs_results_full = pd.DataFrame(optuna_search.trials_dataframe())

    cols = [ ]
    cols += hs_results_full.filter(regex=r'params_').columns.tolist()

    score_features = [ ]
    if train_score:
        score_features += [ 
            'user_attrs_mean_train_score', 'user_attrs_std_train_score'
        ]
    score_features += [ 
        'user_attrs_mean_test_score', 'user_attrs_std_test_score'
    ]
    score_features += ['value']
    
    cols += score_features

    hs_results = hs_results_full[cols]

    # some more managable column names
    hs_results = (hs_results
        .rename(columns=lambda col: re.sub('user_attrs_', '', col))
        .rename(columns=lambda col: re.sub('params_', '', col))
    )

    # hopefully this removes the pipeline column name prefixes if asked
    if not verbose:
        hs_results = hs_results.rename(
            columns=lambda col: re.sub('.+__', '', col)
        )

    if sort_by_rank:
        hs_results = hs_results.sort_values('mean_test_score', ascending=False)
    
    return hs_results
    
