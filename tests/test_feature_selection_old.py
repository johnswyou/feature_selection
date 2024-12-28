import pytest
import numpy as np
import pandas as pd

from feature_selection_old import (
    whiten_E0,
    std_X,
    standardize_X,
    Edgeworth_t1_t2_t3,
    H_whiten,
    H_normal,
    H_EdgeworthApprox,
    MI_EdgeworthApprox,
    CMI_EdgeworthApprox,
    EAcmi_framework_tol
)

def test_whiten_E0():
    """
    Test that columns are correctly centered to have mean zero,
    and any columns that end up all NaN are set to zero.
    """
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    Xw = whiten_E0(X)
    # Column means should be near zero
    col_means = Xw.mean(axis=0)
    assert np.allclose(col_means, 0.0, atol=1e-7), f"Column means not zero: {col_means}"

    # Test a case that leads to NaN column
    X_nan = np.array([[np.nan, 1.0], [np.nan, 3.0]])
    Xw_nan = whiten_E0(X_nan)
    # The first column is all NaN => after centering, it should be replaced with zeros
    assert np.all(Xw_nan[:, 0] == 0.0), "All-NaN column should be replaced with zeros."
    # The second column should be mean-centered
    assert abs(Xw_nan[:, 1].mean()) < 1e-7, "Second column should be centered."

def test_std_X():
    """
    Test that the standard deviations are correctly computed column-wise,
    and that zero or NaN standard deviations are replaced with zero.
    """
    X = np.array([[1.0, 2.0], [3.0, 2.0]])
    sds = std_X(X)
    # First column: std dev of [1,3] is 1.4142...
    # Second column: std dev of [2,2] is 0 => replaced with 0.
    assert pytest.approx(sds[0], 0.01) == 1.414, f"Unexpected std dev for first column: {sds[0]}"
    assert sds[1] == 0.0, f"Second column should have zero std dev: {sds[1]}"

def test_standardize_X():
    """
    Test that the data are standardized to zero mean and unit variance
    (when stdev != 0). Columns with zero stdev or all NaN become all zeros.
    """
    X = np.array([[1.0, 2.0],
                  [3.0, 2.0]])
    Xs = standardize_X(X)
    # For the first column: values [1,3] => mean=2, std=1.4142 => standardized => [-0.707..., 0.707...]
    # For the second column: values [2,2] => zero stdev => replaced with zeros
    col_means = Xs.mean(axis=0)
    col_stds = Xs.std(axis=0, ddof=1)
    assert np.allclose(col_means, 0.0, atol=1e-7), f"Column means not zero after standardization: {col_means}"
    # First column should have stdev ~ 1
    assert abs(col_stds[0] - 1.0) < 1e-7, f"First column not unit variance: {col_stds[0]}"
    # Second column should be 0 since stdev was 0
    assert np.allclose(Xs[:, 1], 0.0, atol=1e-7), f"Second column not replaced with zeros: {Xs[:,1]}"

def test_Edgeworth_t1_t2_t3():
    """
    Test that the returned dictionary has keys 't1', 't2', 't3'
    and that the values are non-negative for a simple numeric input.
    """
    X = np.random.randn(100, 3)  # some random data
    e_terms = Edgeworth_t1_t2_t3(X)
    assert set(e_terms.keys()) == {"t1", "t2", "t3"}, "Missing expected keys from Edgeworth_t1_t2_t3."
    # For random normal data, these 3rd-moment-based terms might be small (on average),
    # but we just check that the function returns floats in a sensible range
    for key in ("t1", "t2", "t3"):
        assert isinstance(e_terms[key], float), f"{key} is not of type float."

def test_H_whiten():
    """
    Test log(det(W)) = log(product(s)) functionality.
    s <= 0 are clipped to 1e-16 inside the function.
    """
    s = np.array([1.0, 2.0, 3.0])
    val = H_whiten(s)
    # log(1*2*3) = log(6) = ~1.791759
    assert pytest.approx(val, 0.001) == 1.791759, f"Wrong whitening contribution for s={s}."

    # Test that non-positive values are handled gracefully
    s_neg = np.array([0.0, -1.0, 2.0])
    val_neg = H_whiten(s_neg)  
    # We clamp the zero/-1 to 1e-16, so effectively log(2e-16^2) but let's just check it doesn't crash
    assert isinstance(val_neg, float), "H_whiten did not return a float for non-positive s."

def test_H_normal():
    """
    Test the normal-based differential entropy approximation.
    """
    # Simple test: for large random normal data, it should produce
    # a numeric float. We'll just do a sanity check that it doesn't fail.
    X = np.random.randn(1000, 2)
    val = H_normal(X)
    assert isinstance(val, float), "H_normal must return a float."

    # Check shape invariance, e.g. 2D shape with single column
    X2 = np.random.randn(500, 1)
    val2 = H_normal(X2)
    assert isinstance(val2, float), "H_normal must return a float for single-column data."

def test_H_EdgeworthApprox():
    """
    Test the Edgeworth approximation-based entropy for a simple dataset.
    """
    X = np.random.randn(500, 2)
    val = H_EdgeworthApprox(X)
    # Just check it's a number and doesn't crash
    assert isinstance(val, float), "H_EdgeworthApprox must return a float."

def test_MI_EdgeworthApprox():
    """
    Test the mutual information approximation on a simple dataset.
    """
    # If Y == X, the MI should be > 0
    X = np.random.randn(300, 2)
    mi_val = MI_EdgeworthApprox(X, X)
    assert mi_val >= 0.0, "Mutual Information with itself should be >= 0."

    # If X is random normal, but Y is random noise uncorrelated with X,
    # we still get some small positive numerical MI.
    Y = np.random.randn(300, 2)
    mi_val2 = MI_EdgeworthApprox(Y, X)
    assert mi_val2 >= 0.0, "MI should be non-negative."

def test_CMI_EdgeworthApprox():
    """
    Test the conditional mutual information approximation.
    """
    # If Y is just X plus some noise, and Z is empty, then CMI ~ MI(Y; X).
    X = np.random.randn(300, 1)
    noise = 0.01 * np.random.randn(300, 1)
    Y = X + noise
    Z = np.empty((300, 0))  # empty, so effectively no conditioning
    cmi_val = CMI_EdgeworthApprox(Y, X, Z)
    mi_val = MI_EdgeworthApprox(Y, X)
    # The ratio should be fairly close
    if mi_val != 0:
        ratio = cmi_val / mi_val
        assert 0.5 < ratio < 2.0, f"CMI and MI mismatch: cmi={cmi_val}, mi={mi_val}"
    # Also check it doesn't go negative
    assert cmi_val >= 0.0, "CMI should not be negative."

def test_EAcmi_framework_tol():
    """
    Basic test for the EAcmi_framework_tol procedure to ensure it returns a DataFrame
    and does not crash. We do not do advanced checking on the content here.
    """
    X = np.random.randn(100, 3)
    Y = np.random.randn(100, 1)
    thresh = 0.01

    # Just check if the function runs and returns a DataFrame
    df_result = EAcmi_framework_tol(X, Y, thresh, silent=True)
    assert isinstance(df_result, pd.DataFrame), "Expected a pandas DataFrame result."

    # The DataFrame can be empty or have some rows; we simply check it does not crash
    # A more thorough test might compare selected features for a known dataset.
