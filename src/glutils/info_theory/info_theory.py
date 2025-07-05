"""
My humble contribution of generic ML utils to support my own projects.

"""
import numpy as np
from sklearn.utils import check_array
from scipy.stats import gaussian_kde
from scipy import stats
from scipy.integrate import quad, dblquad
from functools import partial

from ..funcitertools import (
    pairwise_lst_el_combinations, pairwise_lst_el_chain,
    pairwise_lst_1st_vs_rest
)

from ..ml import (
    min_max_norm, zscore_norm,
    pairwise_X_cols_combinations,
    pairwise_X_cols_chain,
    pairwise_X_cols_1st_vs_rest
)


# ## Unary/binary information theoretical metrics evaluated on KDEs on a
#    [x_min, x_max] interval


def entropy_cp(p_kde, min_x, max_x):
    """
    This is entropy estimated by integration via a KDE estimator p_kde
    to be evaluated in the (min_x, max_x) interval.
    >>> x = np.random.normal(0, 1, 100)
    >>> entropy_cp(gaussian_kde(x), x.min(), x.max())
    2.0320850681500757
    """
    def integrand(x):
        p_x = p_kde(x)[0]
        # safeguarding...
        if p_x==0:
            return 0
        return -p_x * np.log(p_x)
    
    # Calculate entropy through numerical integration
    entropy_value, _ = quad(integrand, min_x, max_x)

    return entropy_value


def KL_div_ccp(p_kde, q_kde, min_x, max_x):
    """
    This is KL-divergence estimated by integration via a KDE estimators 
    p_kde and q_kde to be avaluated in the (min_x, max_x) interval.

    There is an expectation here that p and q are on the same scale and that [min_x, max_x] makes sense for both. If not, probably a good idea to normalise the data distributions before the KDE inference.

    >>> X = np.random.normal(0, 1, (100, 2)) + np.random.normal(0.1, 0.1,
        (100, 2))
    >>> p, q = X[:, 0], X[:, 1]
    >>> KL_div_ccp(gaussian_kde(p), gaussian_kde(q), X.min(), X.max())
    0.036801765689964315
    """
    def integrand(x):
        p_x = p_kde(x)[0]
        p_y = q_kde(x)[0]
        # safeguarding...
        if p_x==0 or p_y==0:
            return 0
        return p_x * (np.log(p_x) - np.log(p_y))

    KL_div_value, _ = quad(integrand, min_x, max_x)

    return KL_div_value


def JS_div_ccp(p_kde, q_kde, min_x, max_x):
    """
    I asked ChatGPT to adapt my implementation above to JS divergence - it seems
    to have done a good job... "This is JS-divergence estimated by integration
    via KDE estimators p_kde and q_kde to be evaluated in the (min_x, max_x)
    interval."

    >>> X = np.random.normal(0, 1, (100, 2)) + np.random.normal(0.1, 0.1,
        (100, 2))
    >>> p, q = X[:, 0], X[:, 1]
    >>> JS_div_ccp(gaussian_kde(p), gaussian_kde(q), X.min(), X.max())
    0.006430630813541132
    """
    
    # (LG comment) This is the average distribution m out of p and q. Some sort
    # of blending going on here; interesting approach; the value returned by
    # m_kde is just a scalar, not a KDE object. I guess it is not an issue, as
    # anything that returns probabilities as a function of x should work
    def m_kde(x):
        return 0.5 * (p_kde(x) + q_kde(x))
    
    # Calculate KL(P || M)
    kl_p_m = KL_div_ccp(p_kde, m_kde, min_x, max_x)
    
    # Calculate KL(Q || M)
    kl_q_m = KL_div_ccp(q_kde, m_kde, min_x, max_x)
    
    # JS divergence is the average of the above two KL divergences
    js_div_value = 0.5 * kl_p_m + 0.5 * kl_q_m
    
    return js_div_value



def mutual_information_ccp(p_kde, q_kde, pq_kde, min_x, max_x):
    """
    https://en.wikipedia.org/wiki/Mutual_information#In_terms_of_PDFs_for_continuous_distributions

    >>> mvnorm = stats.multivariate_normal(mean=[0, 0], cov=[[1., 0.5],
                                                             [0.5, 1.]])
    >>> mvnorm = stats.multivariate_normal(mean=[0, 0], cov=[[1., 0.9],
                                                             [0.9, 1.]])
    >>> mvnorm = stats.multivariate_normal(mean=[0, 0], cov=[[1., 0.99],
                                                             [0.99, 1.]])
    >>> X = mvnorm.rvs(100)
    >>> p, q = X[:, 0], X[:, 1]
    
    >>> mutual_information_ccp(gaussian_kde(p), gaussian_kde(q), gaussian_kde(np.vstack([p,q])), X.min(), X.max())
    0.08438896497693056
    
    # the mutual information of p with itself should be its entropy?
    >>> mutual_information_ccp(gaussian_kde(p), gaussian_kde(p), 
        gaussian_kde(np.vstack([p,p])), X.min(), X.max())
    # a division by 0 is being captured here
    0.0
    # and sometimes:
    # LinAlgError: The data appears to lie in a lower-dimensional subspace of
      the space in which it is expressed. This has resulted in a singular data
      covariance matrix, which cannot be treated using the algorithms
      implemented in `gaussian_kde`. Consider performing principle component
      analysis / dimensionality reduction and using `gaussian_kde` with the
      transformed data.

    >>> entropy_cp(gaussian_kde(p), x.min(), x.max())
    1.3495217295612263
    
    >>> mutual_info_regression(X[:, [0]], X[:, 1])
    array([0.06182941])
    >>> mutual_info_regression(X[:, [0]], X[:, 0])
    array([3.34404418]) 
    """
    def integrand(x, y):
        """here x and y are distributions and p prefix is for density values"""
        p_xy = pq_kde((x, y))[0]
        p_x = p_kde(x)[0]
        p_y = q_kde(y)[0]
        # safeguarding...
        if p_xy==0 or p_x==0 or p_y==0:
            return 0
        return p_xy * np.log(p_xy / (p_x * p_y))

    # evaluation of the double integral assumes that the two distributions work
    # with similar ranges as defined by [min_x, max_x]. If not, probably best
    # to normalise the data before KDEfying it.
    mi_value, _ = dblquad(
        integrand, min_x, max_x, min_x, max_x
    )

    return mi_value



# ## Vectorised unary/binary metrics on KDEs via map/apply

def apply_unary_metric_cP(
        kdes, min_x, max_x, map_cp_f=entropy_cp 
    ):
    """ 
    map_cp_f needs to have `min_x` and `max_x` as arguments (see below).

    >>> X = np.random.normal(0, 1, (100, 10)) + np.random.normal(0.1, 0.1, (100, 10))
    >>> apply_unary_metric_cP(np.apply_along_axis(gaussian_kde, axis=0, arr=X).tolist(), X.min(), X.max(), map_cp_f=entropy_cp)
    (
        1.4575602223521167,
        0.06652463760373395,
        array([1.45408819, 1.41921518, 1.4579623 , 1.31619232, 1.48479202,
           1.37978728, 1.47583918, 1.52436324, 1.52136113, 1.54200138])
    )
    """
    
    # kdes = check_array(kdes, ensure_2d=False)
    IT_metric = partial(map_cp_f, min_x=min_x, max_x=max_x)
    # apply_along_axis not working for me at the moment here
    # values = np.apply_along_axis(IT_metric, axis=0, arr=X)
    values = np.array([ IT_metric(kde) for kde in kdes ])

    return (values.mean(), values.std(), values)


def apply_binary_metric_cP(
        kdes, min_x, max_x, map_ccp_f=KL_div_ccp, pairing_approach='chain'
    ):
    """ 
    map_ccp_f needs to have `min_x` and `max_x` as arguments (see below).

    >>> X = np.random.normal(0, 1, (100, 10)) + np.random.normal(0.1, 0.1, (100, 10))
    >>> apply_binary_metric_cP(np.apply_along_axis(gaussian_kde, axis=0, arr=X), X.min(), X.max(), map_ccp_f=KL_div_ccp)
    (
        0.03826030373622435,
        0.01645189162855585,
        array([0.06455031, 0.01637664, 0.03930873, 0.04382795, 0.02248477,
           0.04465308, 0.05252211, 0.0127844 , 0.04783474])
    )
    """

    # kdes = check_array(kdes, ensure_2d=False)
    IT_metric = partial(map_ccp_f, min_x=min_x, max_x=max_x)
    
    pairing_functions = dict(
        combinations=pairwise_lst_el_combinations,
        chain=pairwise_lst_el_chain,
        first_vs_rest=pairwise_lst_1st_vs_rest
    ) 
    pairing_f = pairing_functions[pairing_approach]

    # the KDEs of pairs of distributions are evaluated by the binary metric
    values = np.array([ 
        IT_metric(kde1, kde2) for kde1, kde2 in pairing_f(kdes) 
    ])

    return (values.mean(), values.std(), values)


# ## Vectorised unary/binary metrics on data distributions via map/apply

def apply_unary_metric_cX(
        X, normalise=False, map_cp_f=entropy_cp,
        norm_f=zscore_norm, kde_f=gaussian_kde, 
        min_x=None, max_x=None
    ):
    """
    >>> X = np.random.normal(0, 1, (100, 10)) + np.random.normal(0.1, 0.1, (100, 10))
    >>> apply_unary_metric_cX(X)
    (
        1.4496119285466365,
        0.04595327330743104,
        array([1.39400634, 1.44540142, 1.43433972, 1.4254306 , 1.56314876,
           1.44090902, 1.40834574, 1.44664193, 1.43925347, 1.49864228]),
        [
            <scipy.stats._kde.gaussian_kde object at 0x1031e09d0>,
            :
            <scipy.stats._kde.gaussian_kde object at 0x121e217d0>
        ]
    )
    """
    X = check_array(X, ensure_2d=True)
    if normalise:
        X = np.apply_along_axis(norm_f, axis=0, arr=X)

    # the min/max across columns (after normalisation, if applicable)
    # otherwise I'll not have comparable distributions/data spaces
    # from the data points inferred by the KDEs
    X_flat = X.ravel()
    if not min_x:
        min_x = min(X_flat)
    if not max_x:
        max_x = max(X_flat)

    # estimation of the KDE given the kde_f estimator argument
    kdes = np.apply_along_axis(kde_f, axis=0, arr=X).tolist()

    return apply_unary_metric_cP(
        kdes, min_x, max_x, map_cp_f=map_cp_f
    ) + (kdes,)


def apply_binary_metric_cX(
        X, normalise=False, map_ccp_f=KL_div_ccp, 
        pairing_approach='chain',
        norm_f=zscore_norm, kde_f=gaussian_kde, 
        min_x=None, max_x=None
    ):
    """
    >>> X = np.random.normal(0, 1, (100, 10)) + np.random.normal(0.1, 0.1, (100, 10))
    >>> apply_binary_metric_cX(X)
    (
        0.03826030373622435,
        0.01645189162855585,
        array([0.06455031, 0.01637664, 0.03930873, 0.04382795, 0.02248477,
           0.04465308, 0.05252211, 0.0127844 , 0.04783474]),
        [
            <scipy.stats._kde.gaussian_kde object at 0x124a6dc10>,
            :
            <scipy.stats._kde.gaussian_kde object at 0x1031e0e90>
        ]
    )
    """
    X = check_array(X, ensure_2d=True)
    if normalise:
        X = np.apply_along_axis(norm_f, axis=0, arr=X)

    # the min/max across columns (after normalisation, if applicable)
    # otherwise I'll not have comparable distributions/data spaces
    # from the data points inferred by the KDEs
    X_flat = X.ravel()
    if not min_x:
        min_x = min(X_flat)
    if not max_x:
        max_x = max(X_flat)

    # estimation of the KDE given the kde_f estimator argument
    kdes = np.apply_along_axis(kde_f, axis=0, arr=X).tolist()

    return apply_binary_metric_cP(
        kdes, min_x, max_x, map_ccp_f=map_ccp_f, 
        pairing_approach=pairing_approach
    ) + (kdes,)



def apply_unary_metric_cd(x, map_cp_f=entropy_cp, **kwargs):
    """
    Version for a single sampled continuous data distribution where x is
    array-like.
    
    - kwargs are those for apply_unary_metric_cX

    >>> apply_unary_metric_cd(X[:, 0], map_cp_f=entropy_cp)
    (1.2935955160614747, <scipy.stats._kde.gaussian_kde object at 0x1249bd110>)
    """
    
    x = check_array(x, ensure_2d=False)
    # if not a unidimensional array has been passed on
    if x.ndim>1:
        # I'll make it 1d by picking the first column
        x = x[:, 0]
    # x is reshaped to a column vector here to work with the vectorised
    # entropy_cX

    mu, sigma, values, kdes = apply_unary_metric_cX(
        x.reshape(-1, 1), map_cp_f=map_cp_f, **kwargs
    )

    # we have only one distribution, so let us pick the individual elements here
    return (values[0], kdes[0])


"""
>>> apply_binary_metric_ccd(X[:, 0], X[:, 1], map_ccp_f=KL_div_ccp)
(0.05672761511583152, <scipy.stats._kde.gaussian_kde object at 0x124b59e50>)
"""
def apply_binary_metric_ccd(x1, x2, map_ccp_f=KL_div_ccp, **kwargs):
    x1 = check_array(x1, ensure_2d=False)
    x2 = check_array(x2, ensure_2d=False)
    mu, sigma, values, kdes = apply_binary_metric_cX(
        np.column_stack([x1,x2]), map_ccp_f=map_ccp_f, **kwargs
    )
    return (values[0], kdes[0])



# ## Wrappers on apply_unary and apply_binary on KDEs or Xs
# 
# Up to me now - we can have entropy_cX, entropy_cP, entropy_cx... and KL_div_ccX, KL_div_ccP, KL_div_ccx1x2

"""
Entropy version for a matrix of sampled continuous data distributions; each column is seen as one distribution; kwargs are those for apply_unary_metric_cX

>>> entropy_cX(X[:, :3])
(
    1.4004475495250401,
    0.031737845609194015,
    array([1.35568537, 1.41996568, 1.4256916 ]),
    [
        <scipy.stats._kde.gaussian_kde object at 0x124b14cd0>,
        <scipy.stats._kde.gaussian_kde object at 0x121d58290>,
        <scipy.stats._kde.gaussian_kde object at 0x121d59150>
    ]
)    
"""
entropy_cX = partial(apply_unary_metric_cX, map_cp_f=entropy_cp)

"""
Entropy version for a single sampled continuous data distribution where x is
array-like. kwargs are those for apply_unary_metric_cd
>>> entropy_cd(X[:, 0])
(1.2935955160614747, <scipy.stats._kde.gaussian_kde object at 0x124ac2490>)
"""
entropy_cd = partial(apply_unary_metric_cd, map_cp_f=entropy_cp)


"""
>>> KL_div_ccX(X, pairing_approach='first_vs_rest')
(
    0.06298909760344443,
    0.02397769826813846,
    array([0.06455031, 0.10600712, 0.07819521, 0.06254719, 0.05038961,
       0.02996968, 0.08939705, 0.05592753, 0.02991817]),
    [
        <scipy.stats._kde.gaussian_kde object at 0x121d5acd0>,
        :
        <scipy.stats._kde.gaussian_kde object at 0x124d69bd0>
    ]
)
"""
KL_div_ccX = partial(apply_binary_metric_cX, map_ccp_f=KL_div_ccp)

# >>> KL_div_ccd(X[:, 0], X[:, 1])
# (0.05672761511583152, <scipy.stats._kde.gaussian_kde object at 0x124b40e50>)
KL_div_ccd = partial(apply_binary_metric_ccd, map_ccp_f=KL_div_ccp)

"""
>>> JS_div_ccX(X[:, :3])
(
    0.009375100617187684,
    0.005299236870704633,
    array([0.01467434, 0.00407586]),
    [
        <scipy.stats._kde.gaussian_kde object at 0x102e04e90>,
        <scipy.stats._kde.gaussian_kde object at 0x124a2fe10>,
        <scipy.stats._kde.gaussian_kde object at 0x124a2d4d0>
    ]
)
"""
JS_div_ccX = partial(apply_binary_metric_cX, map_ccp_f=JS_div_ccp)

# >>> JS_div_ccd(X[:, 0], X[:, 1])
# (0.014674337487892316, <scipy.stats._kde.gaussian_kde object at 0x1249be410>)
JS_div_ccd = partial(apply_binary_metric_ccd, map_ccp_f=JS_div_ccp)