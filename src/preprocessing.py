"""Centralized preprocessing pipeline for hotel cancellation prediction.

Provides a reusable class that encapsulates:
- Categorical handling strategy (currently: drop)
- Numeric scaling (StandardScaler)
- Feature ordering preservation
- Artifact persistence / loading

Future extension points:
- onehot / target / hybrid categorical strategies
- numeric imputation strategies
- feature selection masks

Usage:
    pipeline = PreprocessingPipeline(categorical_strategy='drop', scale=True)
    X_train_proc = pipeline.fit_transform(X_train)
    X_test_proc = pipeline.transform(X_test)
    pipeline.save('models/preprocessor.pkl')

    # Later / inference
    pipeline = PreprocessingPipeline.load('models/preprocessor.pkl')
    X_new = pipeline.transform(X_incoming)
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import os
import numpy as np


@dataclass
class PreprocessingState:
    categorical_strategy: str
    scaled_numeric: List[str]
    dropped_columns: List[str]
    feature_order: List[str]
    scale: bool
    # One-hot specific
    onehot_categories: Optional[Dict[str, List[str]]] = None
    # Target encoding specific
    target_mappings: Optional[Dict[str, Dict[str, float]]] = None
    target_global_mean: Optional[float] = None
    target_encoded_columns: Optional[List[str]] = None


class PreprocessingPipeline:
    def __init__(self, categorical_strategy: str = 'drop', scale: bool = True, target_min_samples: int = 5, target_smoothing: float = 10.0):
        self.categorical_strategy = categorical_strategy
        self.scale = scale
        self._scaler: Optional[StandardScaler] = None
        self.state: Optional[PreprocessingState] = None
        # target encoding hyperparams
        self.target_min_samples = target_min_samples
        self.target_smoothing = target_smoothing

    def _apply_onehot_fit(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        cat_cols = [c for c in X.columns if X[c].dtype == 'object' or pd.api.types.is_categorical_dtype(X[c])]
        categories: Dict[str, List[str]] = {}
        transformed_parts = [X[[c]] for c in X.columns if c not in cat_cols]
        for c in cat_cols:
            cats = sorted([str(v) for v in X[c].dropna().unique()])
            categories[c] = cats
            for val in cats:
                col_name = f"{c}__{val}"
                transformed_parts.append((X[c].astype(str) == val).astype(int).to_frame(col_name))
        X_new = pd.concat(transformed_parts, axis=1)
        return X_new, categories

    def _apply_onehot_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.state and self.state.onehot_categories
        cat_schema = self.state.onehot_categories
        out_parts = []
        # Numeric / other passthrough first (original columns that were not categorical at fit time)
        for c in self.state.feature_order:
            # original feature_order contains post-onehot columns already; skip here
            pass
        # Reconstruct expected columns deterministically
        for base_col, cats in cat_schema.items():
            series = X[base_col].astype(str) if base_col in X.columns else pd.Series([None]*len(X), index=X.index)
            for val in cats:
                col_name = f"{base_col}__{val}"
                out_parts.append((series == val).astype(int).rename(col_name))
        # Add any numeric columns (those not in cat_schema keys)
        numeric_like = [c for c in X.columns if c not in cat_schema]
        for c in numeric_like:
            if c not in self.state.feature_order and any(c.startswith(f"{k}__") for k in cat_schema):
                # skip inadvertent collision
                continue
            if c in cat_schema:
                continue
            if pd.api.types.is_numeric_dtype(X[c]):
                out_parts.append(X[c])
        X_new = pd.concat(out_parts, axis=1)
        # Align to stored feature order
        missing = [c for c in self.state.feature_order if c not in X_new.columns]
        for m in missing:
            X_new[m] = 0  # unseen category -> all zeros
        X_new = X_new[self.state.feature_order]
        return X_new

    def _compute_target_encoding(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, Dict[str, Dict[str,float]], float, List[str]]:
        cat_cols = [c for c in X.columns if X[c].dtype == 'object' or pd.api.types.is_categorical_dtype(X[c])]
        mappings: Dict[str, Dict[str, float]] = {}
        global_mean = float(y.mean())
        X_encoded = X.copy()
        encoded_cols: List[str] = []
        for c in cat_cols:
            stats = y.groupby(X[c]).agg(['mean','count'])
            # smoothing: (count*mean + smoothing*global) / (count + smoothing)
            smooth = (stats['count'] * stats['mean'] + self.target_smoothing * global_mean) / (stats['count'] + self.target_smoothing)
            mapping = smooth.to_dict()
            mappings[c] = mapping
            new_col = f"{c}__te"
            encoded_cols.append(new_col)
            X_encoded[new_col] = X[c].map(mapping).fillna(global_mean)
        # Drop original categorical columns
        X_encoded = X_encoded.drop(columns=cat_cols)
        return X_encoded, mappings, global_mean, encoded_cols

    def _apply_target_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.state and self.state.target_mappings is not None
        global_mean = self.state.target_global_mean
        X_new = X.copy()
        # For each mapping, create encoded column
        for col, mapping in self.state.target_mappings.items():
            new_col = f"{col}__te"
            series = X_new[col] if col in X_new.columns else pd.Series([None]*len(X_new), index=X_new.index)
            X_new[new_col] = series.map(mapping).fillna(global_mean)
        # Drop raw categorical cols
        X_new = X_new.drop(columns=list(self.state.target_mappings.keys()))
        # Align order / add any missing
        missing = [c for c in self.state.feature_order if c not in X_new.columns]
        for m in missing:
            X_new[m] = 0.0
        X_new = X_new[self.state.feature_order]
        return X_new

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'PreprocessingPipeline':
        X = X.copy()
        dropped: List[str] = []
        onehot_categories: Optional[Dict[str, List[str]]] = None
        target_mappings: Optional[Dict[str, Dict[str, float]]] = None
        target_global_mean: Optional[float] = None
        target_encoded_cols: Optional[List[str]] = None

        if self.categorical_strategy == 'drop':
            non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
            if non_numeric:
                X = X.drop(columns=non_numeric)
                dropped = non_numeric
        elif self.categorical_strategy == 'onehot':
            X, onehot_categories = self._apply_onehot_fit(X)
        elif self.categorical_strategy == 'target':
            if y is None:
                raise ValueError("Target series y must be provided for target encoding strategy.")
            X, target_mappings, target_global_mean, target_encoded_cols = self._compute_target_encoding(X, y)
        else:
            raise NotImplementedError(f"Categorical strategy '{self.categorical_strategy}' not implemented.")

        numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        if self.scale and numeric_cols:
            self._scaler = StandardScaler()
            self._scaler.fit(X[numeric_cols])
        self.state = PreprocessingState(
            categorical_strategy=self.categorical_strategy,
            scaled_numeric=numeric_cols if self.scale else [],
            dropped_columns=dropped,
            feature_order=list(X.columns),
            scale=self.scale,
            onehot_categories=onehot_categories,
            target_mappings=target_mappings,
            target_global_mean=target_global_mean,
            target_encoded_columns=target_encoded_cols
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.state is None:
            raise RuntimeError("Pipeline not fitted.")
        X = X.copy()
        if self.state.categorical_strategy == 'drop':
            for col in self.state.dropped_columns:
                if col in X.columns:
                    X = X.drop(columns=col)
            missing = [c for c in self.state.feature_order if c not in X.columns]
            if missing:
                raise ValueError(f"Incoming data missing columns required by preprocessor: {missing}")
            X = X[self.state.feature_order]
        elif self.state.categorical_strategy == 'onehot':
            X = self._apply_onehot_transform(X)
        elif self.state.categorical_strategy == 'target':
            X = self._apply_target_transform(X)
        else:
            raise NotImplementedError(f"Unknown strategy {self.state.categorical_strategy}")
        if self.scale and self._scaler is not None:
            # Ensure float dtype prior to scaling assignment to avoid pandas FutureWarning
            for col in self.state.scaled_numeric:
                if not pd.api.types.is_float_dtype(X[col]):
                    X[col] = X[col].astype('float64')
            X.loc[:, self.state.scaled_numeric] = self._scaler.transform(X[self.state.scaled_numeric])
        return X

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload: Dict[str, Any] = {
            'state': asdict(self.state) if self.state else None,
            'categorical_strategy': self.categorical_strategy,
            'scale': self.scale,
            'scaler': self._scaler,
            'target_min_samples': self.target_min_samples,
            'target_smoothing': self.target_smoothing
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str) -> 'PreprocessingPipeline':
        payload = joblib.load(path)
        pipe = cls(
            categorical_strategy=payload.get('categorical_strategy', 'drop'),
            scale=payload.get('scale', True),
            target_min_samples=payload.get('target_min_samples', 5),
            target_smoothing=payload.get('target_smoothing', 10.0)
        )
        state_dict = payload.get('state')
        if state_dict:
            pipe.state = PreprocessingState(**state_dict)
        pipe._scaler = payload.get('scaler')
        return pipe

    def to_metadata(self) -> Dict[str, Any]:
        return asdict(self.state) if self.state else {}

"""Helper for future extension: registration of new categorical strategies.
Currently omitted for brevity."""
