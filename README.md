# Linear and Logistic Regression: From Scratch and Validated

A from-first-principles implementation of linear and logistic regression using only NumPy, validated against scikit-learn on real public datasets.

## What this project is

Most ML practitioners reach for `sklearn.LinearRegression().fit(X, y)` without thinking about what's happening underneath. This project implements both algorithms from scratch by using cost functions, gradients, batch gradient descent, L2 regularization, polynomial feature mapping, and then validates the implementations against scikit-learn on real data.

The point isn't that NumPy beats scikit-learn. The point is to demonstrate a working understanding of the layer below the API: the math, the optimization, and the engineering decisions that make these algorithms practical.

## What's inside

| Part | Topic | Dataset |
|------|-------|---------|
| 1 | Linear regression from scratch + scikit-learn validation | California Housing (20,640 districts × 8 features) |
| 2 | Logistic regression from scratch + scikit-learn validation | Breast Cancer Wisconsin (569 samples × 30 features) |
| 3 | Regularized logistic regression with polynomial features | Synthetic 2D non-linearly-separable data |
| 4 | Reflection and lessons learned | — |

## Results

**Linear regression on California Housing:**
- From-scratch test RMSE matches scikit-learn to 3 decimal places
- Test R² ≈ 0.59 on all 8 features (consistent with known dataset characteristics)

**Logistic regression on Breast Cancer Wisconsin:**
- Test accuracy ≈ 97%, identical to scikit-learn
- Top-weighted features align with established clinical indicators (worst concave points, worst perimeter, mean concavity)

**Regularization sweep on synthetic non-linear data:**
- λ = 0: overfits to label noise with a wiggly boundary
- λ = 1: cleanly recovers the underlying circular boundary
- λ = 100: underfits to nearly linear

The bias-variance tradeoff visualized in one figure.

## How to run

```bash
git clone https://github.com/ekahorsu/linear-and-logistic-regression.git
cd linear-and-logistic-regression
pip install -r requirements.txt
jupyter notebook linear_and_logistic_regression.ipynb
```

Tested with Python 3.10. All datasets are downloaded automatically by scikit-learn.

## Tech stack

- **NumPy** — all from-scratch math
- **Matplotlib** — visualizations
- **scikit-learn** — datasets, preprocessing, and validation baselines
- **Jupyter** — notebook environment
