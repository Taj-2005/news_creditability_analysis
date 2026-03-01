"""
Sklearn pipelines for news credibility classification.

TF-IDF vectorization + classifier. Models: Logistic Regression,
Decision Tree, Naive Bayes, Random Forest, and optional SVM.
All use class_weight='balanced' where applicable for imbalance.
"""

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

# Default TF-IDF and model hyperparameters (tuned for Kaggle Fake and Real News dataset)
TFIDF_MAX_FEATURES_LR = 25000
TFIDF_MAX_FEATURES_DT = 15000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.92
LR_C = 2.0  # Inverse regularization; slightly lower for strong performance


def build_lr_pipeline(
    max_features: int = TFIDF_MAX_FEATURES_LR,
    max_iter: int = 2000,
    C: float = LR_C,
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


def build_nb_pipeline(
    max_features: int = 20000,
    random_state: int = 42,
) -> Pipeline:
    """Build TF-IDF + Multinomial Naive Bayes pipeline."""
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
            ("clf", MultinomialNB(alpha=0.1)),
        ]
    )


def build_rf_pipeline(
    max_features: int = 20000,
    n_estimators: int = 200,
    max_depth: int = 30,
    min_samples_split: int = 5,
    random_state: int = 42,
) -> Pipeline:
    """Build TF-IDF + Random Forest pipeline."""
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
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_svm_pipeline(
    max_features: int = 20000,
    C: float = 1.0,
    max_iter: int = 2000,
    random_state: int = 42,
) -> Pipeline:
    """Build TF-IDF + Linear SVM pipeline."""
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
                LinearSVC(
                    C=C,
                    class_weight="balanced",
                    max_iter=max_iter,
                    random_state=random_state,
                    dual=False,
                ),
            ),
        ]
    )
