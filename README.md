# `feature_selection`

This repository provides code for performing stepwise feature selection using an Edgeworth-Approximation-based Mutual Information (MI) and Conditional Mutual Information (CMI) approach. The primary entry point is the EAcmi_framework_tol function, located in feature_selection.py. Below is an overview of the repository structure, setup instructions, and basic usage examples.

## Repository Structure

• **feature_selection.py**
  - Contains the core feature selection function (EAcmi_framework_tol) and associated helper functions (e.g., for approximating differential entropy, mutual information, and conditional mutual information).

• **tests/test_feature_selection.py**
  - Contains Pytest-based unit tests for the code in feature_selection.py.  
  - Validates helper functions (e.g., standardize_X, MI_EdgeworthApprox) and the main EAcmi_framework_tol procedure.

• **notebooks/testing.ipynb**
  - Example Jupyter notebook demonstrating how to run EAcmi_framework_tol on sample data sets, including scikit-learn’s diabetes dataset.

• **old/feature_selection_old.py**
  - An older or alternative implementation of the feature selection routine, kept for archival or reference purposes.

## Installation and Requirements

This project requires Python 3.7+ (tested on Python 3.10+). The main libraries used are:

• numpy  
• pandas  
• pytest (for testing)  
• scikit-learn (used in some tests to load sample data)

You can install them with pip:
```bash
pip install numpy pandas pytest scikit-learn
```

## Running Tests

To run all tests with Pytest, navigate to the repository root and run:
```bash
pytest
```

You may see tests for both the “feature_selection.py” file and the older “old/feature_selection_old.py”. All tests should pass if everything is installed correctly.

## Basic Usage

Below is a minimal Python example illustrating how to call EAcmi_framework_tol:

```python
import numpy as np
import pandas as pd
from feature_selection import EAcmi_framework_tol

# Generate random data
X = np.random.randn(100, 5)  # 100 samples, 5 features
y = np.random.randn(100)     # 100 samples (target)

# Set a threshold for CMI/MI ratio
threshold = 0.01

# Run feature selection
df_results = EAcmi_framework_tol(X, y, thresh=threshold, silent=True)

# Inspect the resulting DataFrame
print(df_results)
```

• X must be a 2D NumPy array of shape (n_samples, n_features).  
• y can be 1D or 2D, representing the target array of shape (n_samples,).  
• thresh is the stopping criterion for the ratio of conditional mutual information to mutual information.  
• silent (bool) controls whether iteration details are printed.

## Additional Notes

1. The Edgeworth Approximation is used internally to estimate entropy. This approach can be sensitive to data distribution, especially for small sample sizes or data with strong non-normality.  
2. The stepwise procedure selects features until certain stopping criteria are met (e.g., ratio ≤ thresh, no further improvement in CMI, or three consecutive increases in CMI/MI).

## Contributing

Feel free to open an issue or make a pull request if you find a bug or have a feature request. All contributions, from bug fixes to improvements, are welcome!
