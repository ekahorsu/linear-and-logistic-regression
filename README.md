# Linear and Logistic Regression: From Scratch and Validated

A from-first-principles implementation of linear and logistic regression using only NumPy, validated against scikit-learn on real public datasets.

> **Note on viewing the notebook:** GitHub's notebook preview is currently experiencing a platform-wide rendering outage. The key results and plots are reproduced below in this README. To view the full notebook, open `linear_and_logistic_regression.ipynb` locally in Jupyter.

## Overview

This project implements linear and logistic regression from first principles, including the cost functions, gradients, batch gradient descent, L2 regularization, and polynomial feature mapping. Each implementation is then validated against scikit-learn on real public datasets.

The objective is not to outperform scikit-learn, but to demonstrate a thorough understanding of the mechanics beneath the library API: the underlying mathematics, the optimization procedure, and the engineering decisions that make these algorithms work in practice.

## Contents

| Part | Topic | Dataset |
|------|-------|---------|
| 1 | Linear regression from scratch with scikit-learn validation | California Housing (20,640 districts, 8 features) |
| 2 | Logistic regression from scratch with scikit-learn validation | Breast Cancer Wisconsin (569 samples, 30 features) |
| 3 | Regularized logistic regression with polynomial features | Synthetic 2D non-linearly-separable data |
| 4 | Reflection and lessons learned | - |

## Part 1: Linear Regression

Linear regression is implemented with vectorized NumPy operations: the cost function, the gradient, and batch gradient descent. The analysis begins with a single feature (median income) to allow a visual sanity check, then extends to all eight features.

![Linear regression fit and cost convergence](linear_and_logistic_regression_files/linear_and_logistic_regression_9_0.png)

The left panel shows the fitted line through the median-income versus house-value data. The right panel shows the cost decreasing monotonically across gradient descent iterations, which confirms the gradient implementation is correct.

**Validation against scikit-learn (all eight features):**

| Metric | From-scratch | scikit-learn |
|--------|-------------|--------------|
| RMSE | 0.7456 | 0.7456 |
| R-squared | 0.5758 | 0.5758 |

Every parameter matched scikit-learn's closed-form solution to four decimal places. An R-squared of approximately 0.58 is consistent with the established performance ceiling of linear regression on this dataset; the remaining variance is attributable to non-linear effects that a linear model cannot capture.

## Part 2: Logistic Regression

Binary classification of breast tumor biopsies (malignant versus benign), implemented from scratch with a numerically stable sigmoid, binary cross-entropy loss, and L2 regularization.

![Logistic regression cost and feature importance](linear_and_logistic_regression_files/linear_and_logistic_regression_23_0.png)

The left panel shows the cross-entropy loss converging. The right panel shows the most influential features by weight magnitude, which align with established clinical indicators (worst concave points, worst perimeter, mean concavity).

**Result:** Test accuracy of approximately 97%, consistent with scikit-learn's `LogisticRegression`.

## Part 3: Regularization and the Bias-Variance Tradeoff

To demonstrate the effect of regularization, the notebook generates synthetic 2D data with a circular decision boundary and added label noise, maps it to degree-6 polynomial features, and trains models at three regularization strengths.

![Synthetic non-linear data](linear_and_logistic_regression_files/linear_and_logistic_regression_26_0.png)

![Decision boundaries at different regularization strengths](linear_and_logistic_regression_files/linear_and_logistic_regression_28_0.png)

- **Low regularization:** the boundary conforms to individual points, including noise, indicating overfitting.
- **Moderate regularization:** the boundary closely approximates the true underlying circle, representing a well-balanced fit.
- **High regularization:** the boundary becomes overly simplified, indicating underfitting.

Together these illustrate the bias-variance tradeoff, with the regularization strength serving as the control that moves the model between the two extremes.

## How to Run

```bash
git clone https://github.com/ekahorsu/linear-and-logistic-regression.git
cd linear-and-logistic-regression
pip install -r requirements.txt
jupyter notebook linear_and_logistic_regression.ipynb
```

Tested with Python 3.10. All datasets are downloaded automatically by scikit-learn.

## Tech Stack

- **NumPy** for the from-scratch implementations
- **Matplotlib** for visualizations
- **scikit-learn** for datasets, preprocessing, and validation baselines
- **Jupyter** for the notebook environment

## Key Takeaways

1. The mathematical core of these algorithms is compact, while the surrounding engineering, including data loading, feature scaling, train/test splitting, validation, and visualization, constitutes the majority of the work in a complete analysis.
2. Feature scaling is essential for gradient descent. Without standardization, gradient descent on features of differing magnitudes either diverges or converges very slowly.
3. Regularization strength is a parameter that must be tuned rather than fixed arbitrarily. Both extremes produced inferior models relative to a moderate setting, and in practice the strength should be selected through cross-validation.
4. Validation against a trusted baseline is among the most valuable practices in machine learning engineering. Agreement with scikit-learn to four decimal places provides strong evidence that the from-scratch implementation is correct.

## Challenges and Solutions

### Dataset availability in restricted environments

**Challenge.** The California Housing dataset is retrieved over the network on first use rather than bundled with scikit-learn. In a restricted cloud notebook environment, this download was blocked, returning a `403 Forbidden` error and halting the analysis.

**Solution.** The dependency on network access was identified as the root cause and documented explicitly. The notebook runs without issue in any environment with standard internet access, and the requirement is now noted so the constraint is clear to anyone reproducing the work.

### Feature scaling for stable gradient descent

**Challenge.** The eight housing features span very different magnitudes; for example, median income is on the order of single digits while population is in the thousands. Running gradient descent on the raw features produced slow and unstable convergence, because a single learning rate cannot suit features of such different scales simultaneously.

**Solution.** Each feature was standardized to zero mean and unit variance using `StandardScaler`. The scaler was fit on the training data only and then applied to the test data, which prevents information from the test set leaking into the preprocessing step. After scaling, gradient descent converged smoothly across all features.

### Selecting regularization strength

**Challenge.** A single train/test accuracy figure does not reveal whether a model is overfitting or underfitting, which makes it difficult to choose an appropriate regularization strength on its own.

**Solution.** Models were trained across a range of regularization strengths and their decision boundaries visualized side by side. This exposed the bias-variance tradeoff directly: weak regularization produced an overfit boundary that traced noise, while strong regularization produced an underfit boundary that oversimplified the structure. The visualization makes the case for selecting the strength through systematic comparison rather than a single arbitrary value.

### Numerical stability of the sigmoid function

**Challenge.** The sigmoid function involves an exponential term that can overflow for large-magnitude inputs, producing invalid values that propagate through the cost and gradient computations.

**Solution.** The input to the exponential was clipped to a bounded range before evaluation. This preserves the function's behavior across the meaningful input range while preventing overflow, keeping the cost and gradient calculations numerically stable throughout training.
