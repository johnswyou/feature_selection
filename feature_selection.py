import time
import numpy as np
import pandas as pd

def whiten_E0(X):
    """
    Centers each column of X to have zero mean.
    If any column becomes all NaN, it is set to 0.
    """
    X = np.array(X, dtype=float)
    means = np.nanmean(X, axis=0)
    Xw = X - means
    nan_cols = np.isnan(Xw).all(axis=0)
    Xw[:, nan_cols] = 0.0
    return Xw

def std_X(X):
    """
    Computes the column-wise standard deviation of X (unbiased estimator, ddof=1).
    Returns 0 if column is constant or all-NaN.
    """
    X = np.array(X, dtype=float)
    sds = np.nanstd(X, axis=0, ddof=1)
    sds = np.nan_to_num(sds, nan=0.0)
    return sds

def standardize_X(X):
    """
    Standardizes each column to have zero mean and unit variance.
    Columns with stdev=0 or all-NaN become 0.
    """
    X = np.array(X, dtype=float)
    means = np.nanmean(X, axis=0)
    sds = np.nanstd(X, axis=0, ddof=1)
    sds[sds == 0] = 1.0
    Xs = (X - means) / sds
    Xs = np.nan_to_num(Xs, nan=0.0)
    return Xs

def Edgeworth_t1_t2_t3(X):
    """
    Computes t1, t2, t3 used in the Edgeworth correction for entropy estimation.
    """
    X = np.array(X, dtype=float)
    n, d = X.shape

    # t1: sum of the squares of the 3rd moments (univariate)
    t1 = 0.0
    for i in range(d):
        kappa_iii = np.mean(X[:, i] ** 3)
        t1 += (kappa_iii ** 2)

    # t2: 3 * sum of squares of cross-moments E[x_i^2 * x_j]
    t2_sum = 0.0
    if d > 1:
        for i in range(d - 1):
            for j in range(i + 1, d):
                kappa_iij = np.mean(X[:, i]**2 * X[:, j])
                t2_sum += (kappa_iij ** 2)
                # The code doubles it by looping j<i again,
                # so we can replicate that:
                kappa_jjj = np.mean(X[:, j]**2 * X[:, i])
                t2_sum += (kappa_jjj ** 2)
    t2 = 3.0 * t2_sum

    # t3: (1/6) * sum of squares of E[x_i * x_j * x_k] for i<j<k
    t3_sum = 0.0
    if d > 2:
        for i in range(d - 2):
            for j in range(i + 1, d - 1):
                for k in range(j + 1, d):
                    kappa_ijk = np.mean(X[:, i] * X[:, j] * X[:, k])
                    t3_sum += (kappa_ijk ** 2)
    t3 = t3_sum / 6.0

    return {"t1": t1, "t2": t2, "t3": t3}

def H_whiten(s):
    """
    s is a vector of scaling factors (e.g., stdevs).
    Returns log of their product, with small epsilon if any are non-positive.
    """
    s_nonneg = np.where(s > 0, s, 1e-16)
    return float(np.log(np.prod(s_nonneg)))

def H_normal(X):
    """
    Differential entropy of a normal with sample covariance of X:
      0.5*log(det(Cov)) + (d/2)*log(2*pi) + d/2
    """
    X = np.array(X, dtype=float)
    _, d = X.shape
    cov_mat = np.cov(X, rowvar=False)
    det_cov = np.linalg.det(cov_mat + 1e-16*np.eye(d))
    return 0.5*np.log(det_cov) + (d/2)*np.log(2*np.pi) + (d/2)

def H_EdgeworthApprox(X):
    """
    Edgeworth Approximation for differential entropy of X.
    """
    Xw = whiten_E0(X)  # zero-center
    s = std_X(Xw)      # stdev of zero-centered data
    Y = standardize_X(X)  # fully standardized
    H_w = H_whiten(s)
    H_n = H_normal(Y)
    ea_terms = Edgeworth_t1_t2_t3(Y)
    corr = (ea_terms["t1"] + ea_terms["t2"] + ea_terms["t3"]) / 12.0
    return (H_n - corr) + H_w

def MI_EdgeworthApprox(Y, X):
    # Explicitly force shape to (n, d)
    Y = np.array(Y)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    HY = H_EdgeworthApprox(Y)
    HX = H_EdgeworthApprox(X)
    HXY = H_EdgeworthApprox(np.column_stack([Y, X]))
    MI_ea_val = HY + HX - HXY
    return max(0.0, MI_ea_val)

def CMI_EdgeworthApprox(Y, X, Z):
    Y = np.array(Y)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    Z = np.array(Z)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    HYZ = H_EdgeworthApprox(np.column_stack([Y, Z]))
    HXZ = H_EdgeworthApprox(np.column_stack([X, Z]))
    HZ = H_EdgeworthApprox(Z)
    HYXZ = H_EdgeworthApprox(np.column_stack([Y, X, Z]))

    CMI_ea_val = HYZ + HXZ - HZ - HYXZ
    return max(0.0, CMI_ea_val)

def EAcmi_framework_tol(x, y, thresh, silent=False):
    """
    Stepwise selection of features from x to explain y, 
    stopping when CMI/MI <= thresh or other conditions hold.
    
    Args:
        x (np.ndarray): shape (n_samples, n_features), the input data
        y (np.ndarray): shape (n_samples,) or (n_samples, 1), the target
        thresh (float): threshold for the ratio (CMI / MI)
        silent (bool): if True, suppress printing iteration details
    
    Returns:
        pd.DataFrame with columns:
          Iteration, Input, CMI, MI, CMI.MI.ratio, CMIevals, CPUtime, ElapsedTime
        Each row corresponds to one new selected feature.
    """
    # Track time
    cpu_time_start = time.process_time()
    wall_time_start = time.time()

    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float).reshape(-1, 1)  # ensure 2D

    # Standardize everything
    x = standardize_X(x)
    y = standardize_X(y)

    n_samples, n_features = x.shape
    input_indices = list(range(n_features))  # track columns that remain unselected

    # For naming columns: if desired, could pass original col names
    # here we use "Input_0", "Input_1", ...
    feature_names = [f"Input_{i}" for i in range(n_features)]

    # We'll store the subset of selected columns in z_in
    z_in = np.empty((n_samples, 0))  # no columns initially
    n_selected = 0
    n_cmi_evals = 0

    scores_list = []  # store iteration results

    def record_and_print(iter_num, chosen_idx, cmi_val, mi_val, ratio):
        """Helper to record one iteration’s results and optionally print."""
        # Build the row
        row_dict = {
            "Iteration": iter_num,
            "Input": feature_names[chosen_idx],
            "CMI": round(cmi_val, 4),
            "MI": round(mi_val, 4),
            "CMI.MI.ratio": round(ratio, 4),
            "CMIevals": n_cmi_evals,
            "CPUtime": round(time.process_time() - cpu_time_start, 4),
            "ElapsedTime": round(time.time() - wall_time_start, 4),
        }
        scores_list.append(row_dict)
        if not silent:
            print(row_dict)

    max_iter = n_features  # we cannot select more features than we have

    for it in range(1, max_iter + 1):
        if z_in.shape[1] == 0:
            # First iteration: pick the predictor that maximizes MI(Y; x_j)
            mi_vals = []
            for j in range(len(input_indices)):
                col_j = x[:, j].reshape(-1, 1)
                val = MI_EdgeworthApprox(y, col_j)
                mi_vals.append(val)
                n_cmi_evals += 1

            best_mi = np.max(mi_vals)
            best_idx_in_mi_vals = int(np.argmax(mi_vals))
            if best_mi <= 0:
                # all MIs <= 0 => no improvement
                break

            chosen_idx_global = input_indices[best_idx_in_mi_vals]
            chosen_col = x[:, best_idx_in_mi_vals].reshape(-1, 1)

            # Move this chosen col into z_in
            z_in = np.hstack([z_in, chosen_col])
            n_selected += 1

            # Evaluate MI of the augmented set (just 1 col so far)
            mi_aug = MI_EdgeworthApprox(y, z_in)
            ratio_val = (best_mi / mi_aug) if mi_aug != 0 else 0.0

            record_and_print(it, chosen_idx_global, best_mi, mi_aug, ratio_val)

            # Remove from the candidate set
            del input_indices[best_idx_in_mi_vals]
            x = np.delete(x, best_idx_in_mi_vals, axis=1)

        else:
            # For subsequent iterations, pick the predictor that maximizes CMI(Y; x_j | z_in)
            cmi_vals = []
            for j in range(len(input_indices)):
                col_j = x[:, j].reshape(-1, 1)
                val = CMI_EdgeworthApprox(y, col_j, z_in)
                cmi_vals.append(val)
                n_cmi_evals += 1

            best_cmi = np.max(cmi_vals)
            best_idx_in_cmi_vals = int(np.argmax(cmi_vals))
            if best_cmi <= 0:
                # no improvement
                break

            chosen_idx_global = input_indices[best_idx_in_cmi_vals]
            chosen_col = x[:, best_idx_in_cmi_vals].reshape(-1, 1)

            # Evaluate MI of the augmented set
            z_in_test = np.hstack([z_in, chosen_col])
            mi_aug = MI_EdgeworthApprox(y, z_in_test)
            ratio_val = (best_cmi / mi_aug) if mi_aug != 0 else 0.0

            # Record iteration
            n_selected += 1
            record_and_print(it, chosen_idx_global, best_cmi, mi_aug, ratio_val)

            # Stopping checks
            if (it >= 2) and (ratio_val <= thresh or np.isnan(ratio_val)):
                break

            # Check if the ratio has strictly increased 3 times in a row
            if len(scores_list) > 2:
                r1 = scores_list[-1]["CMI.MI.ratio"]
                r2 = scores_list[-2]["CMI.MI.ratio"]
                r3 = scores_list[-3]["CMI.MI.ratio"]
                if (r1 > r2) and (r2 > r3):
                    break

            # Otherwise accept the chosen feature
            z_in = z_in_test
            del input_indices[best_idx_in_cmi_vals]
            x = np.delete(x, best_idx_in_cmi_vals, axis=1)

    if not silent:
        print("\nEA_CMI_TOL ROUTINE COMPLETED\n")

    # Convert results to DataFrame
    scores_df = pd.DataFrame(scores_list)

    # Show final table if not silent
    if not scores_df.empty and not silent:
        with pd.option_context('display.precision', 4):
            print(scores_df)

    # Return the entire DataFrame
    return scores_df
