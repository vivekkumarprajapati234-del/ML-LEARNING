# ML Learning

A collection of machine learning implementations built while studying core ML algorithms — covering regression, classification, and model evaluation, implemented with scikit-learn.

## Structure

### `regression/`
| Notebook | Description |
|---|---|
| [`boston_housing_regression.ipynb`](regression/boston_housing_regression.ipynb) | Linear, Ridge, and Lasso regression on the Boston housing dataset, with hyperparameter tuning via GridSearchCV and model evaluation (R², MSE) |

### `classification/`
| Notebook | Description |
|---|---|
| [`breast_cancer_logistic_regression.ipynb`](classification/breast_cancer_logistic_regression.ipynb) | Logistic regression on the breast cancer dataset, with feature scaling, accuracy scoring, and a confusion matrix |
| [`iris_naive_bayes.ipynb`](classification/iris_naive_bayes.ipynb) | Gaussian Naive Bayes classifier on the Iris dataset |
| [`iris_knn.ipynb`](classification/iris_knn.ipynb) | K-Nearest Neighbours (k=5) classifier on the Iris dataset, with feature scaling |
| [`iris_decision_tree.ipynb`](classification/iris_decision_tree.ipynb) | Decision Tree classifier (Gini criterion) on the Iris dataset, with tree visualization |

### `notes/`
Reserved for conceptual notes and derivations (formulas, metric explanations, etc.) that don't belong inside a specific model notebook.

## Setup

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Notes

Each notebook is self-contained (imports, data loading, training, and evaluation) and can be run independently in Jupyter or VS Code.
