import time
import numpy as np
import pandas as pd

def whiten_E0(X):
    """
    Centers each column of X to have zero mean, 
    i.e., subtract the column mean from each column.
    If any columns become all NaN, they are set to 0.
    """
    X = np.array(X, dtype=float)
    means = np.nanmean(X, axis=0)
    Xw = X - means
    # Replace columns that are all NaN with zeros
    # (np.isnan(Xw).all(0) checks for all-NaN by column)
    nan_cols = np.isnan(Xw).all(axis=0)
    Xw[:, nan_cols] = 0.0
    return Xw

def std_X(X):
    """
    Computes the column-wise standard deviation of X.
    If the data are all identical in a column, the standard deviation
    is 0; in that case, this function returns 0 for that column.
    """
    X = np.array(X, dtype=float)
    # ddof=1 makes Python std() match R's default unbiased estimator
    sds = np.nanstd(X, axis=0, ddof=1)
    # Where sds is NaN (extreme cases), return 0
    sds = np.nan_to_num(sds, nan=0.0)
    return sds

def standardize_X(X):
    """
    Standardizes each column of X to have zero mean and unit variance.
    Where the column stdev=0 or the column is all-NaN, that column is set to 0.
    """
    X = np.array(X, dtype=float)
    means = np.nanmean(X, axis=0)
    sds = np.nanstd(X, axis=0, ddof=1)
    # Avoid division by zero
    sds[sds == 0] = 1.0
    Xs = (X - means) / sds
    # Replace any NaN with zeros
    Xs = np.nan_to_num(Xs, nan=0.0)
    return Xs

def Edgeworth_t1_t2_t3(X):
    """
    Computes the three kappa_ijk := E[x_i x_j x_k]-based terms (t1, t2, t3)
    in the multivariate Edgeworth expansion for entropy estimation.
    
    :param X: np.ndarray of shape (n_samples, d)
    :return: dict with keys 't1', 't2', 't3'
    """
    X = np.array(X, dtype=float)
    n, d = X.shape

    # t1: sum of the squares of the 3rd moments (one dimension)
    t1 = 0.0
    for i in range(d):
        kappa_iii = np.mean(X[:, i] ** 3)
        t1 += kappa_iii**2

    # t2: 3 * the sum of squares of the cross-moments E[x_i^2 x_j]
    # The R implementation sums over i<j and j<i, effectively doubling the count, 
    # then multiplies by 3. We replicate that approach:
    t2_sum = 0.0
    if d > 1:
        # i<j
        for i in range(d - 1):
            for j in range(i+1, d):
                kappa_iij = np.mean((X[:, i]**2) * X[:, j])
                t2_sum += kappa_iij**2
        # j<i
        for j in range(d - 1):
            for i2 in range(j+1, d):
                kappa_iij = np.mean((X[:, i2]**2) * X[:, j])
                t2_sum += kappa_iij**2
    t2 = 3.0 * t2_sum

    # t3: (1/6) * sum of squares of E[x_i x_j x_k] for i<j<k
    t3_sum = 0.0
    if d > 2:
        for i in range(d - 2):
            for j in range(i+1, d - 1):
                for k in range(j+1, d):
                    kappa_ijk = np.mean(X[:, i] * X[:, j] * X[:, k])
                    t3_sum += kappa_ijk**2
    t3 = t3_sum / 6.0

    return {"t1": t1, "t2": t2, "t3": t3}

def H_whiten(s):
    """
    Applies the high-level 'whitening' transform contribution to Entropy:
    H(Wz) = log(|det(W)|) => log(product of scaling factors s).
    """
    # s should be a vector of stdev or scaling factors
    # If any s <= 0, we handle gracefully:
    s_nonneg = np.where(s > 0, s, 1e-16)
    return float(np.log(np.prod(s_nonneg)))

def H_normal(X):
    """
    Computes the differential entropy of a normal variable 
    with the sample covariance of X.
    That is: H(X) = 0.5 * log(det(cov(X))) + (d/2) * (1 + log(2*pi)).
    
    However, in the R code, there's an extra + d/2. 
    That stems from log(det(Cov)) / 2 + d/2 * log(2*pi) + d/2
    This matches the R approach: H_w = log(det(stats::cov(X)))/2 
                                 + (d/2 * log(2*pi)) + d/2
    """
    X = np.array(X, dtype=float)
    _, d = X.shape
    # rowvar=False => columns are variables, same as R
    cov_mat = np.cov(X, rowvar=False)
    # Ensure cov is not singular
    det_cov = np.linalg.det(cov_mat + 1e-16 * np.eye(d))
    # R logic: log(det(Cov)) / 2 + (d/2 * log(2*pi)) + d/2
    return 0.5 * np.log(det_cov) + (d / 2.0) * np.log(2.0 * np.pi) + (d / 2.0)

def H_EdgeworthApprox(X):
    """
    Computes the Edgeworth Approximation (EA)-based differential Shannon entropy
    for a dataset X, matching the R version in EAcmi_utils.R:
      1) Whiten the data (zero-mean, though the function name is "whiten_E0"),
      2) Then get s = std_X(Xw),
      3) standardize the original data fully (zero mean, unit sd),
      4) H_w = H_whiten(s),
      5) H_n = H_normal(Y), where Y is the standardized data,
      6) Edgeworth corrections from t1, t2, t3, 
      7) combine them as H_n - (t1 + t2 + t3)/12 + H_w.
    """
    Xw = whiten_E0(X)
    s = std_X(Xw)
    Y = standardize_X(X)
    H_w = H_whiten(s)
    H_n = H_normal(Y)
    
    ea_terms = Edgeworth_t1_t2_t3(Y)
    corr = (ea_terms["t1"] + ea_terms["t2"] + ea_terms["t3"]) / 12.0
    H_ea = (H_n - corr) + H_w
    return H_ea

def MI_EdgeworthApprox(Y, X):
    """
    Computes the Edgeworth Approximation (EA)-based Shannon Mutual Information:
      MI(Y; X) = H(Y) + H(X) - H(Y, X)
    Then clamps the result to be non-negative.
    """
    # Make sure Y and X are 2D
    Y = np.array(Y, ndmin=2)
    X = np.array(X, ndmin=2)
    # If Y.shape[1] or X.shape[1] ends up 0, handle that gracefully
    HX = H_EdgeworthApprox(X)
    HY = H_EdgeworthApprox(Y)
    HXY = H_EdgeworthApprox(np.column_stack([Y, X]))
    # R logic: MI_ea = max(0, Re(MI_ea))
    # In Python, we'll just ensure no negative results:
    MI_ea_val = (HY + HX) - HXY
    return max(0.0, MI_ea_val)

def CMI_EdgeworthApprox(Y, X, Z):
    """
    Computes the Edgeworth Approximation (EA)-based Shannon Conditional 
    Mutual Information: I(Y; X | Z) = H(Y, Z) + H(X, Z) - H(Z) - H(Y, X, Z).
    Then clamps the result to be non-negative.
    """
    Y = np.array(Y, ndmin=2)
    X = np.array(X, ndmin=2)
    Z = np.array(Z, ndmin=2)

    HYZ = H_EdgeworthApprox(np.column_stack([Y, Z]))
    HXZ = H_EdgeworthApprox(np.column_stack([X, Z]))
    HZ = H_EdgeworthApprox(Z)
    HYXZ = H_EdgeworthApprox(np.column_stack([Y, X, Z]))

    CMI_ea_val = HYZ + HXZ - HZ - HYXZ
    return max(0.0, CMI_ea_val)

def EAcmi_framework_tol(x, y, thresh, silent=False):
    """
    Implements the stepwise EAcmi_tol procedure in Python, mirroring the functionality
    of the R implementation. Utilizes the ratio of conditional mutual information (CMI)
    over mutual information (MI) to determine the most significant inputs.

    Args:
        x (np.ndarray): 2D numpy array of model inputs, shape (n_samples, n_features).
        y (np.ndarray): 1D or 2D numpy array for the target variable(s),
                        shape (n_samples,) or (n_samples, 1).
        thresh (float): Stopping threshold for the CMI/MI ratio.
        silent (bool, optional): If True, suppresses printing intermediate results.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the iterative selection details:
            - Input: Which input was selected each iteration.
            - CMI: Conditional mutual information at selection time.
            - MI: MI of the augmented set.
            - CMI.MI.ratio: Ratio between new CMI and the augmented MI.
            - CMIevals: Number of CMI (or MI) evaluations so far.
            - CPUtime: CPU time consumed so far.
            - ElapsedTime: Wall-clock time elapsed so far.

    Dependencies:
        - NumPy
        - Pandas

    Expects the following helper functions (to be provided by the user or
    placed in the same module):
        - standardize_X(X) -> np.ndarray:
            Standardizes each column to have zero mean and unit variance.
        - MI_EdgeworthApprox(Y, X) -> float:
            Computes EA-based mutual information.
        - CMI_EdgeworthApprox(Y, X, Z) -> float:
            Computes EA-based conditional mutual information.

    References:
        Quilty, J., Adamowski, J., Khalil, B., & Rathinasamy, M. (2016).
        Kugiumtzis, D. (2013).
        Tsimpiris, A., Vlachos, I., & Kugiumtzis, D. (2012).
    """
    # Track start for CPU and wall time
    cpu_time_start = time.process_time()
    wall_time_start = time.time()

    # Ensure x and y are numpy arrays
    x = np.array(x, copy=True)
    y = np.array(y, copy=True).reshape(-1, 1)  # ensure 2D for standardization

    # Number of inputs
    n_inputs = x.shape[1]
    n_data = x.shape[0]

    # If there are column names, store; otherwise, index-based names
    # (In Python, you might supply them from a DataFrame; here we just keep indices.)
    inp_names = [f"Input_{i}" for i in range(n_inputs)]

    # Record number of CMI (or MI) calls
    nfevals_2 = 0

    # Standardize output data with zero mean unit variance
    y_stand = standardize_X(y)
    y = y_stand

    # Standardize input data with zero mean unit variance
    z_stand = standardize_X(x)
    x = z_stand

    # Track which inputs remain unselected
    input_tracker = list(range(n_inputs))
    n_selected = 0
    z_in = np.empty((n_data, 0))  # empty 2D array to store selected inputs

    # Prepare for iterative procedure
    scores_list = []  # will store dicts, converted to DF later
    max_iter = n_inputs + 1

    # Main iteration loop
    for iter_1 in range(1, max_iter + 1):
        if n_selected > 0:
            if n_inputs == 0:
                # No unselected inputs remain
                break

            # Now compute CMI for each unselected input
            cmi_vals = []
            for current_inp in range(n_inputs):
                z = x[:, current_inp].reshape(-1, 1)
                current_cmi = CMI_EdgeworthApprox(y, z, z_in)
                nfevals_2 += 1
                cmi_vals.append(current_cmi)

            # Identify best input
            best_cmi = np.max(cmi_vals)
            if best_cmi <= 0:
                # If all CMI <= 0, we end
                break
            tag = int(np.argmax(cmi_vals))

            # Current best input as a column
            best_input_col = x[:, tag].reshape(-1, 1)
            # Add best input to the set of selected inputs
            z_in = np.hstack([z_in, best_input_col])
            n_selected += 1

            # Compute MI over augmented set
            mi_aug = MI_EdgeworthApprox(y, z_in)
            cmi_mi_ratio = best_cmi / mi_aug if mi_aug != 0 else 0.0

            # Create scoreboard row
            sel_dict = {
                "Input": input_tracker[tag],
                "CMI": round(best_cmi, 4),
                "MI": round(mi_aug, 4),
                "CMI.MI.ratio": round(cmi_mi_ratio, 4),
                "CMIevals": nfevals_2,
                "CPUtime": round(time.process_time() - cpu_time_start, 4),
                "ElapsedTime": round(time.time() - wall_time_start, 4),
            }
            scores_list.append(sel_dict)

            # Optionally print the just-selected row
            if not silent:
                print(sel_dict)

            # Check stopping conditions
            # 1) Reached maximum iteration
            if iter_1 == max_iter:
                break
            # 2) Ratio <= threshold
            if iter_1 > 2:
                # cmi_mi_ratio might be NaN check:
                if (np.isnan(cmi_mi_ratio)) or (cmi_mi_ratio <= thresh):
                    break
            # 3) If ratio keeps increasing for the last 3 picks
            if len(scores_list) > 2:
                # Compare last three ratio entries
                if (
                    scores_list[-1]["CMI.MI.ratio"] >
                    scores_list[-2]["CMI.MI.ratio"]
                ) and (
                    scores_list[-2]["CMI.MI.ratio"] >
                    scores_list[-3]["CMI.MI.ratio"]
                ):
                    break

            # Remove selected input from remaining pool
            del input_tracker[tag]
            x = np.delete(x, tag, axis=1)
            n_inputs -= 1

        else:
            # n_selected == 0, compute MI for each input
            mi_vals = []
            for current_inp in range(n_inputs):
                z = x[:, current_inp].reshape(-1, 1)
                current_mi = MI_EdgeworthApprox(y, z)
                nfevals_2 += 1
                mi_vals.append(current_mi)

            # Identify best input (using MI now)
            best_mi = np.max(mi_vals)
            if best_mi <= 0:
                # If all MIs <= 0, we end
                break
            tag = int(np.argmax(mi_vals))

            # Current best input
            best_input_col = x[:, tag].reshape(-1, 1)
            z_in = np.hstack([z_in, best_input_col])
            n_selected += 1

            # On first pick, there's no "CMI ratio" relative to an augmented set,
            # so we can do a simpler scoreboard row
            mi_aug = MI_EdgeworthApprox(y, z_in)
            cmi_mi_ratio = best_mi / mi_aug if mi_aug != 0 else 0.0

            sel_dict = {
                "Input": input_tracker[tag],
                "CMI": round(best_mi, 4),  # conceptually it's "MI" for first pick
                "MI": round(mi_aug, 4),
                "CMI.MI.ratio": round(cmi_mi_ratio, 4),
                "CMIevals": nfevals_2,
                "CPUtime": round(time.process_time() - cpu_time_start, 4),
                "ElapsedTime": round(time.time() - wall_time_start, 4),
            }
            scores_list.append(sel_dict)

            if not silent:
                print(sel_dict)

            del input_tracker[tag]
            x = np.delete(x, tag, axis=1)
            n_inputs -= 1

    # End of main loop
    print("\nEA_CMI_TOL ROUTINE COMPLETED\n")

    # Convert scores list to DataFrame
    scores_df = pd.DataFrame(scores_list)

    # Print the final table with classic formatting
    if not scores_df.empty:
        with pd.option_context('display.precision', 4):
            print(scores_df)

    # Return the DataFrame up to (iter_1 - 2) rows, if that slice is valid
    # For safety, we clip negative indexing
    end_index = max(0, len(scores_df) - 2)
    return scores_df.iloc[:end_index]
