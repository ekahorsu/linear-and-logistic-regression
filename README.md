# ML Foundations: Linear and Logistic Regression

> Project 1 of 5 in the ML Foundations portfolio. Implements linear and logistic regression from first principles using only NumPy, then validates the implementations against scikit-learn on real public datasets.

## What this project is

Most ML practitioners reach for `sklearn.LinearRegression().fit(X, y)` without thinking about what's happening underneath. This project demonstrates I understand the layer below the API by implementing both algorithms from scratch — cost functions, gradients, batch gradient descent, L2 regularization, polynomial feature mapping — and then checking my work against scikit-learn on real data.

## Why this matters

Job interviews and ML engineering work both reward people who can debug models from first principles. A `sklearn` user who can't tell you why their gradient descent diverged is much less valuable than one who can.

## What's inside

| Part | Topic | Dataset |
|------|-------|---------|
| 1 | Linear regression from scratch + scikit-learn validation | California Housing (20,640 districts × 8 features) |
| 2 | Logistic regression from scratch + scikit-learn validation | Breast Cancer Wisconsin (569 samples × 30 features) |
| 3 | Regularized logistic regression with polynomial features | Synthetic 2D non-linearly-separable data |
| 4 | Reflection and lessons learned | — |

## Results summary

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

This is the bias-variance tradeoff visualized in one figure.

## How to run

```bash
git clone https://github.com/ekahorsu/ml-foundations-regression.git
cd ml-foundations-regression
pip install -r requirements.txt
jupyter notebook linear_and_logistic_regression.ipynb
```

Tested with Python 3.10. All datasets are downloaded automatically by scikit-learn — nothing to install manually.

## Tech stack

- **NumPy** — all from-scratch math
- **Matplotlib** — visualizations
- **scikit-learn** — datasets, preprocessing, and validation baselines
- **Jupyter** — notebook environment

## What I'd extend next

- Mini-batch / stochastic gradient descent for scalability
- Cross-validation for principled λ selection
- Multinomial (softmax) logistic regression for multi-class problems

## About this portfolio

This is one project in a 5-project ML Foundations portfolio built around the Andrew Ng Machine Learning Specialization (Coursera), with each project reframed around real public datasets and extended with validation, regularization, and discussion that go beyond the original lab scope.

Author: **Etornam Kwasi Ahorsu** — PhD student in Electrical Engineering at the University of Nevada, Reno. Research focuses on optimization under uncertainty for power systems planning. [LinkedIn](#) · [GitHub](https://github.com/ekahorsu)
