# src/features.py
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

NUM_COLS = ["loan_amnt","int_rate","emp_length","annual_inc","dti","delinq_2yrs",
            "inq_last_6mths","open_acc","pub_rec","revol_util","total_acc","mort_acc",
            "fico_range_low","fico_range_high"]
CAT_COLS = ["term","home_ownership","verification_status","purpose","addr_state","application_type"]

class QuantileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, q_low=1, q_high=99):
        self.q_low = q_low
        self.q_high = q_high
        self.low_ = None
        self.high_ = None

    def fit(self, X, y=None):
        self.low_  = np.nanpercentile(X, self.q_low,  axis=0)
        self.high_ = np.nanpercentile(X, self.q_high, axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.low_, self.high_)

def make_preprocessor():
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", QuantileClipper(q_low=1, q_high=99)),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01))
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUM_COLS),
            ("cat", cat_pipe, CAT_COLS),
        ],
        sparse_threshold=0.3
    )
