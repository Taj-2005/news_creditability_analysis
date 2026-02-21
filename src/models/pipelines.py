"""
Sklearn pipelines for news credibility classification.

TF-IDF vectorization + classifier. Two models: Logistic Regression
and Decision Tree, both with class_weight='balanced' for imbalance.
"""

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

# Default TF-IDF and model hyperparameters (used in training and CV)
TFIDF_MAX_FEATURES_LR = 15000
TFIDF_MAX_FEATURES_DT = 10000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 3
TFIDF_MAX_DF = 0.90


def build_lr_pipeline(
    max_features: int = TFIDF_MAX_FEATURES_LR,
    max_iter: int = 1000,
    C: float = 1.0,
    random_state: int = 42,
) -> Pipeline:
    """
    Build TF-IDF + Logistic Regression pipeline.

    Logistic Regression is interpretable and works well with high-dimensional
    sparse TF-IDF features. class_weight='balanced' handles class imbalance.

    Args:
        max_features: Maximum vocabulary size for TF-IDF.
        max_iter: Maximum iterations for LR solver.
        C: Inverse regularization strength.
        random_state: Random seed for reproducibility.

    Returns:
        Fitted sklearn Pipeline (not fitted; call .fit(X, y)).
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=TFIDF_NGRAM_RANGE,
                    sublinear_tf=True,
                    min_df=TFIDF_MIN_DF,
                    max_df=TFIDF_MAX_DF,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=max_iter,
                    C=C,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_dt_pipeline(
    max_features: int = TFIDF_MAX_FEATURES_DT,
    max_depth: int = 25,
    min_samples_split: int = 10,
    random_state: int = 42,
) -> Pipeline:
    """
    Build TF-IDF + Decision Tree pipeline.

    Decision trees offer interpretability and no feature scaling need.
    class_weight='balanced' addresses class imbalance.

    Args:
        max_features: Maximum vocabulary size for TF-IDF.
        max_depth: Maximum tree depth to limit overfitting.
        min_samples_split: Minimum samples required to split a node.
        random_state: Random seed for reproducibility.

    Returns:
        Unfitted sklearn Pipeline.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=TFIDF_NGRAM_RANGE,
                    sublinear_tf=True,
                    min_df=TFIDF_MIN_DF,
                    max_df=TFIDF_MAX_DF,
                ),
            ),
            (
                "clf",
                DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
