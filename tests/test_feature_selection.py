import numpy as np
import pytest
import pandas as pd

from feature_selection.feature_selection import (
    whiten_E0,
    std_X,
    standardize_X,
    Edgeworth_t1_t2_t3,
    H_whiten,
    H_normal,
    H_EdgeworthApprox,
    MI_EdgeworthApprox,
    CMI_EdgeworthApprox,
    EAcmi_framework_tol,
)

#####################
# Helper data fixtures
#####################

@pytest.fixture
def small_data():
    """Returns a small (n_samples x n_features) dataset and a target."""
    X = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 2.1, 2.9],
        [1.5, 2.0, 3.1],
        [2.5, 2.05, 2.95],
    ])
    Y = np.array([10, 11, 10.5, 11.5])
    return X, Y

@pytest.fixture
def random_data():
    """Returns random X, Y arrays for a slightly bigger test."""
    np.random.seed(42)  # for repeatability
    X = np.random.randn(100, 5)  # 100 samples, 5 features
    Y = np.random.randn(100)     # 100 samples, 1-dim target
    return X, Y

#####################
# Tests for helpers
#####################

def test_whiten_E0(small_data):
    X, _ = small_data
    Xw = whiten_E0(X)
    # After whitening, each column should have mean ~ 0
    col_means = np.mean(Xw, axis=0)
    assert np.allclose(col_means, 0.0, atol=1e-7)

def test_std_X(small_data):
    X, _ = small_data
    sds = std_X(X)
    # Check length and positivity
    assert sds.shape == (3,)
    for val in sds:
        assert val >= 0.0

def test_standardize_X(small_data):
    X, _ = small_data
    Xs = standardize_X(X)
    # Means should be 0, stdevs should be ~1
    means = np.mean(Xs, axis=0)
    sds = np.std(Xs, axis=0, ddof=1)
    assert np.allclose(means, 0.0, atol=1e-7)
    # Some columns might not have huge variation, but we expect stdev ~ 1
    # We'll allow a small tolerance
    assert np.allclose(sds, 1.0, atol=1e-7)

def test_Edgeworth_t1_t2_t3(small_data):
    X, _ = small_data
    # Let's just check they return floats
    Xs = standardize_X(X)
    terms = Edgeworth_t1_t2_t3(Xs)
    assert "t1" in terms and "t2" in terms and "t3" in terms
    assert isinstance(terms["t1"], float)
    assert isinstance(terms["t2"], float)
    assert isinstance(terms["t3"], float)

def test_H_whiten(small_data):
    X, _ = small_data
    sds = std_X(X)
    hw = H_whiten(sds)
    # Just check it's a float
    assert isinstance(hw, float)

def test_H_normal(small_data):
    X, _ = small_data
    Xs = standardize_X(X)
    val = H_normal(Xs)
    assert isinstance(val, float)
    # Entropy of standardized data won't be negative
    assert val > 0.0

def test_H_EdgeworthApprox(small_data):
    X, _ = small_data
    val = H_EdgeworthApprox(X)
    assert isinstance(val, float)
    # Edgeworth approximation can be negative for small data
    assert np.isfinite(val)

def test_MI_EdgeworthApprox(small_data):
    X, Y = small_data
    # Now that shapes are fixed in the function, no dimension error
    mi_val = MI_EdgeworthApprox(Y, X)
    # MI is clamped to >= 0
    assert mi_val >= 0.0

def test_CMI_EdgeworthApprox(small_data):
    X, Y = small_data
    Z = X[:, [0]]  # shape (4,1)
    cmi_val = CMI_EdgeworthApprox(Y, X, Z)
    # It's also clamped to >= 0
    assert cmi_val >= 0.0


############################
# Test the main stepwise function
############################

def test_EAcmi_framework_tol_small(small_data):
    X, Y = small_data
    thresh = 0.05  # fairly small threshold
    df_results = EAcmi_framework_tol(X, Y, thresh, silent=True)
    # Check that we get a Pandas DataFrame
    assert isinstance(df_results, pd.DataFrame)
    # The DataFrame can be empty if all MIs are <= 0, 
    # but usually we expect 1 or more selected features:
    if not df_results.empty:
        assert "Iteration" in df_results.columns
        assert "Input" in df_results.columns
        assert "CMI" in df_results.columns

def test_EAcmi_framework_tol_random(random_data):
    X, Y = random_data
    thresh = 0.02
    df_results = EAcmi_framework_tol(X, Y, thresh, silent=True)
    # We only test shape and type here
    assert isinstance(df_results, pd.DataFrame)
    # With random data, we expect at least 1 row selected
    # but let's just ensure no errors
    # We can do mild checks, e.g. the 'Iteration' is in ascending order
    if not df_results.empty:
        iters = df_results["Iteration"].values
        assert np.all(iters == sorted(iters))
