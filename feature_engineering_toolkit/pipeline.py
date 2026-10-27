"""Feature Engineering Pipeline — create, select, and transform features.

Supports both pandas and PySpark DataFrames via a backend abstraction.
Part of the data-science pipeline: data cleaning → feature engineering →
preprocessing → model building.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List, Any, Union
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from .backends import get_backend, DataFrameBackend

class FeatureEngineeringPipeline:
    def __init__(self, df=None, config=None, target_col=None):
        from .config import DEFAULT_CONFIG
        self.config = config or DEFAULT_CONFIG.copy()
        self.df = df.copy() if df is not None else None
        self.target_col = target_col or self.config["data_input"]["target_variable"]
        self.backend: DataFrameBackend = get_backend(self.df)
        self.created_features = []
        self.selected_features = []
        self.transformed_features = []
        self.cleaned_features = []
        self.model = None
        self.predictions = None
        self.log = []
        self.original_shape = (0, 0)
        self.accuracy = None

    def _log(self, msg):
        entry = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        self.log.append(entry); print(msg)
        return self

    def clean_data(self) -> "FeatureEngineeringPipeline":
        """Data-cleaning stage: remove duplicates, fix types, drop empty rows.
        Works for both pandas and PySpark via backend."""
        self.backend = get_backend(self.df)
        # Remove fully duplicate rows (pandas native; PySpark via dropDuplicates)
        if self.backend.is_pandas:
            before = len(self.df)
            self.df = self.df.drop_duplicates()
            after = len(self.df)
            if before != after:
                self._log(f"Clean: removed {before - after} duplicate rows")
            self.df = self.df.reset_index(drop=True)
        else:
            # PySpark: dropDuplicates is native
            self.df = self.df.dropDuplicates()
            self._log("Clean: dropDuplicates applied (PySpark)")
            self.cleaned_features.extend(["duplicate_removal"])
        return self

    def preprocess(self) -> "FeatureEngineeringPipeline":
        """Preprocessing stage: split train/test, prepare for modelling.
        Converts to pandas for sklearn-based model building if needed."""
        # For sklearn evaluation / model building we need pandas
        if self.backend.is_pyspark:
            self._log("Preprocess: converting PySpark -> pandas for sklearn")
            pandas_df = self.backend.to_pandas(self.df)
            self.df = pandas_df
            self.backend = get_backend(self.df)
        feature_cols = [c for c in self.df.columns if c != self.target_col]
        if len(feature_cols) < 2 or self.target_col not in self.df.columns:
            self._log("Preprocess: insufficient features/target")
            return self
        self._log(f"Preprocess: {len(feature_cols)} features ready for modelling")
        return self

    def build_model(self) -> "FeatureEngineeringPipeline":
        """Model-building stage: train logistic regression, predict, store results."""
        # sklearn/scipy steps need pandas
        if self.backend.is_pyspark:
            self._log("build_model: converting PySpark -> pandas")
            self.df = self.backend.to_pandas(self.df)
            self.backend = get_backend(self.df)
        feature_cols = [c for c in self.df.columns if c != self.target_col and np.issubdtype(self.df[c].dtype, np.number)]
        if len(feature_cols) < 2 or self.target_col not in self.df.columns:
            self._log("Model build: not enough numeric features/target")
            return self
        X = self.df[feature_cols].fillna(0)
        y = self.df[self.target_col]
        if y.dtype == object or str(y.dtype).startswith("category"):
            y = pd.Categorical(y).codes
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.config.get("advanced", {}).get("random_state", 42))
        self.model = LogisticRegression(solver="lbfgs", max_iter=1000)
        self.model.fit(X_train, y_train)
        self.predictions = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, self.predictions)
        self._log(f"Model built — Accuracy: {self.accuracy:.3f}")
        return self

    def run_full(self, data_path=None, df=None):
        """Complete data-science pipeline.

        Parameters
        ----------
        data_path : str or Path, optional
            Path to any CSV/Parquet dataset. If None, falls back to config.
        df : DataFrame, optional
            Direct DataFrame input (pandas or PySpark). Overrides data_path.
        """
        if df is not None:
            self.df = df.copy() if hasattr(df, "copy") else df
            self.backend = get_backend(self.df)
            self._log(f"Using provided {self.backend.name} DataFrame; shape={self.df.shape}")
        elif data_path is not None:
            self.load(data_path)
        else:
            import warnings
            warnings.warn(
                "No df or data_path given; falling back to bundled sample data."
            )
            self.load()
        self.clean_data()          # 1. cleaning
        self.handle_missing()      # 2. missing-value handling (cleaning sub-step)
        self.create_features()     # 3. feature engineering
        self.transform()           # 4. feature transformation
        self.select_features()     # 5. feature selection
        self.encode_advanced()     # 6. encoding
        self.evaluate()            # 7. evaluation (feature-only accuracy)
        self.preprocess()          # 8. preprocessing for model
        self.build_model()         # 9. model building
        self.save_report()
        self.save_data()
        self._log("=== FULL DS PIPELINE COMPLETE ===")
        return self

    def load(self, path=None, fmt=None):
        """Load a dataset from any path. Falls back to the bundled sample data
        if no path is given.

        Parameters
        ----------
        path : str or Path, optional
            Path to your CSV / Parquet file. If omitted, uses the configured
            path (falls back to the bundled sample data).
        fmt : str, optional
            File format override: ``"csv"`` or ``"parquet"``. If None,
            inferred from the file extension or configuration.
        """
        path = path or self.config["data_input"]["data_path"]
        p = Path(path)
        # Search: given path -> data/ in cwd -> bundled sample in package data/
        candidates = [p, Path("data") / p.name, Path(__file__).parent.parent / "data" / p.name]
        found = None
        for cand in candidates:
            if cand.exists() and cand.is_file():
                found = cand; break
        if found is None:
            # Last resort: try the bare path (may be a URL or non-local location)
            if p.exists() and p.is_file():
                found = p
        if found is None:
            raise FileNotFoundError(f"Could not find data file: {path}")
        p = found
        # Format priority: explicit fmt arg > file extension > config file_type
        if fmt:
            file_fmt = fmt
        elif str(p).lower().endswith((".parquet", ".pq")):
            file_fmt = "parquet"
        elif str(p).lower().endswith((".csv", ".tsv")):
            file_fmt = "csv"
        else:
            file_fmt = self.config["data_input"].get("file_type", "csv")
        encoding = self.config["data_input"].get("encoding", "utf-8")
        if file_fmt == "parquet":
            self.df = pd.read_parquet(str(p))
        else:
            self.df = pd.read_csv(str(p), encoding=encoding)
        self.original_shape = self.df.shape
        self.backend = get_backend(self.df)
        self._log(f"Loaded: {self.df.shape[0]}x{self.df.shape[1]} from {p}")
        return self

    def handle_missing(self) -> "FeatureEngineeringPipeline":
        """Impute missing values (mean for numeric, mode for categorical)."""
        # sklearn/scipy steps need pandas; convert PySpark lazily
        if self.backend.is_pyspark:
            self._log("handle_missing: converting PySpark -> pandas")
            self.df = self.backend.to_pandas(self.df)
            self.backend = get_backend(self.df)
        if self.config["advanced"]["handle_missing"]:
            numeric = self.df.select_dtypes(include=[np.number]).columns
            strategy = self.config["advanced"]["imputation_strategy"]
            for col in numeric:
                if self.df[col].isnull().sum() > 0:
                    fill = self.df[col].mean() if strategy == "mean" else self.df[col].median() if strategy == "median" else 0
                    self.df[col] = self.df[col].fillna(fill)
                    self._log(f"Imputed '{col}' with {strategy}: {fill:.2f}")
        return self

    def create_features(self):
        if not self.config["feature_creation"]["enabled"]:
            self._log("Feature creation disabled"); return self
        df = self.df
        techniques = self.config["feature_creation"]["techniques"]
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if "polynomial_features" in techniques:
            for col in numeric[:3]:
                df[f"{col}_sq"] = df[col] ** 2
                self.created_features.append(f"{col}_sq")
                self._log(f"Created polynomial: {col}_sq")
        if "interaction_terms" in techniques and len(numeric) >= 2:
            df["interaction"] = df[numeric[0]] * df[numeric[1]]
            self.created_features.append("interaction")
            self._log("Created interaction term")
        if "aggregation_features" in techniques:
            gb = self.config["feature_creation"]["aggregation_features"]["group_by"]
            agg_funcs = self.config["feature_creation"]["aggregation_features"]["aggregations"]
            if gb in df.columns and agg_funcs:
                for func in agg_funcs:
                    new_col = f"{gb}_{func}"
                    try:
                        if func == "mean":
                            df[new_col] = df.groupby(gb)[gb].transform("mean")
                        elif func == "sum":
                            df[new_col] = df.groupby(gb)[gb].transform("sum")
                        elif func == "count":
                            df[new_col] = df.groupby(gb)[gb].transform("count")
                        else:
                            continue
                        self.created_features.append(new_col)
                        self._log(f"Created aggregation: {new_col}")
                    except Exception as e:
                        self._log(f"Aggregation failed for {new_col}: {e}")
        if "datetime_features" in techniques:
            for col in self.config["feature_creation"]["datetime_features"]["datetime_columns"]:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col])
                        df[f"{col}_year"] = df[col].dt.year
                        self.created_features.append(f"{col}_year")
                        self._log(f"Created datetime: {col}_year")
                    except Exception as e:
                        self._log(f"Datetime failed for {col}: {e}")
        self.df = df
        return self

    def transform(self):
        if not self.config["feature_transformation"]["enabled"]:
            self._log("Transformation disabled"); return self
        numeric = self.df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric if c != self.target_col]
        method = self.config["feature_transformation"]["method"]
        if method == "standard_scaler":
            s = StandardScaler(); self.df[feature_cols] = s.fit_transform(self.df[feature_cols])
            self.transformed_features.extend(feature_cols); self._log("StandardScaler applied")
        elif method == "min_max_scaler":
            s = MinMaxScaler(feature_range=self.config["feature_transformation"]["min_max_scaler"]["feature_range"])
            self.df[feature_cols] = s.fit_transform(self.df[feature_cols]); self.transformed_features.extend(feature_cols); self._log("MinMaxScaler applied")
        elif method == "robust_scaler":
            s = RobustScaler(quantile_range=self.config["feature_transformation"]["robust_scaler"]["quantile_range"])
            self.df[feature_cols] = s.fit_transform(self.df[feature_cols]); self.transformed_features.extend(feature_cols); self._log("RobustScaler applied")
        elif method == "power_transformer":
            pt = PowerTransformer(method=self.config["feature_transformation"]["power_transformer"]["method"])
            self.df[feature_cols] = pt.fit_transform(self.df[feature_cols]); self.transformed_features.extend(feature_cols); self._log("PowerTransformer applied")
        elif method == "quantile_transformer":
            qt = QuantileTransformer(n_quantiles=self.config["feature_transformation"]["quantile_transformer"]["n_quantiles"], output_distribution=self.config["feature_transformation"]["quantile_transformer"]["output_distribution"])
            self.df[feature_cols] = qt.fit_transform(self.df[feature_cols]); self.transformed_features.extend(feature_cols); self._log("QuantileTransformer applied")
        else:
            self._log(f"Unknown method: {method}")
        return self

    def select_features(self):
        if not self.config["feature_selection"]["enabled"]:
            self._log("Selection disabled"); return self
        method = self.config["feature_selection"]["method"]
        feature_df = self.df.select_dtypes(include=[np.number]).drop(columns=[self.target_col], errors="ignore")
        # Drop constant columns (zero variance) — they cause SelectKBest warnings
        non_const = feature_df.columns[feature_df.nunique() > 1].tolist()
        dropped = [c for c in feature_df.columns if c not in non_const]
        if dropped:
            self._log(f"Dropped constant columns: {dropped}")
        X = feature_df[non_const] if non_const else feature_df
        y = self.df[self.target_col] if self.target_col in self.df.columns else pd.Series([0]*len(self.df))
        if y.dtype == object: y = pd.Categorical(y).codes
        if method == "select_k_best":
            k = min(self.config["feature_selection"]["select_k_best"]["k"], X.shape[1])
            s = SelectKBest(score_func=f_classif, k=k)
            s.fit(X, y); mask = s.get_support()
            self.selected_features = X.columns[mask].tolist()
            self.df = self.df[self.selected_features + ([self.target_col] if self.target_col in self.df.columns else [])]
            self._log(f"SelectKBest selected {len(self.selected_features)} (k={k})")
        elif method == "recursive_feature_elimination":
            n = min(self.config["feature_selection"]["recursive_feature_elimination"]["n_features_to_select"], X.shape[1])
            est = LogisticRegression(solver="lbfgs", max_iter=1000)
            s = RFE(est, n_features_to_select=n, step=1)
            s.fit(X, y); mask = s.support_
            self.selected_features = X.columns[mask].tolist()
            self.df = self.df[self.selected_features + ([self.target_col] if self.target_col in self.df.columns else [])]
            self._log(f"RFE selected {len(self.selected_features)}")
        elif method == "variance_threshold":
            from sklearn.feature_selection import VarianceThreshold
            thresh = self.config["feature_selection"]["variance_threshold"]["threshold"]
            s = VarianceThreshold(threshold=thresh)
            s.fit(X); mask = s.get_support()
            self.selected_features = X.columns[mask].tolist()
            self.df = self.df[self.selected_features + ([self.target_col] if self.target_col in self.df.columns else [])]
            self._log(f"VarianceThreshold selected {len(self.selected_features)}")
        else:
            self._log(f"Unknown selection: {method}")
        return self

    def encode_advanced(self):
        self._log("Advanced encoding: placeholder (install feature-engine for target encoding)")
        return self

    def evaluate(self):
        feature_cols = [c for c in self.df.columns if c != self.target_col and np.issubdtype(self.df[c].dtype, np.number)]
        if len(feature_cols) < 2 or self.target_col not in self.df.columns:
            self._log("Not enough features/target for evaluation")
            return self
        X = self.df[feature_cols].fillna(0)
        y = self.df[self.target_col]
        if y.dtype == object: y = pd.Categorical(y).codes
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.config["advanced"]["random_state"])
        model = LogisticRegression(solver="lbfgs", max_iter=1000)
        model.fit(X_train, y_train)
        self.accuracy = accuracy_score(y_test, model.predict(X_test))
        self._log(f"Accuracy with engineered features: {self.accuracy:.3f}")
        return self

    def generate_report(self):
        lines = ["="*60, "DATA SCIENCE PIPELINE REPORT", "="*60,
                 f"Original: {self.original_shape}",
                 f"Final: {self.df.shape if self.df is not None else 'N/A'}",
                 f"Backend: {self.backend.name}",
                 f"Cleaned: {len(self.cleaned_features)}",
                 f"Created: {len(self.created_features)}",
                 f"Selected: {len(self.selected_features)}",
                 f"Transformed: {len(self.transformed_features)}",
                 f"Accuracy: {self.accuracy or 'N/A'}",
                 ""]
        for e in self.log: lines.append(f"  - {e}")
        lines.append("="*60)
        return "\n".join(lines)

    def save_report(self, path=None):
        path = path or self.config["output"]["report_path"]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f: f.write(f"<pre>{self.generate_report()}</pre>")
        self._log(f"Report saved to {path}")
        return self

    def save_data(self, path=None, fmt=None):
        path = path or self.config["output"]["output_path"]
        fmt = fmt or self.config["output"]["output_format"]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if self.backend.is_pyspark:
            if fmt == "csv":
                self.df.coalesce(1).write.mode("overwrite").option("header", "true").csv(path)
            else:
                self.df.write.mode("overwrite").parquet(path)
        else:
            (self.df.to_csv(path, index=False) if fmt == "csv" else self.df.to_parquet(path, index=False))
        self._log(f"Data saved to {path}")
        return self

    def get_engineered_data(self):
        if self.df is None:
            return pd.DataFrame()
        if self.backend.is_pyspark:
            return self.backend.to_pandas(self.df)
        return self.df.copy()

    def run(self, data_path=None, df=None):
        """Run the full pipeline on a dataset.

        Parameters
        ----------
        data_path : str or Path, optional
            Path to any CSV/Parquet dataset.
        df : DataFrame, optional
            Direct DataFrame input (pandas or PySpark). If provided, bypasses
            data loading and uses this frame directly.

        Returns
        -------
        self
        """
        if df is not None:
            self.df = df.copy() if hasattr(df, "copy") else df
            self.backend = get_backend(self.df)
            self._log(f"Using provided {self.backend.name} DataFrame; shape={self.df.shape}")
        elif data_path is not None:
            self.load(data_path)
        else:
            import warnings
            warnings.warn("No df or data_path given; using bundled sample.")
            self.load()
        return (self.handle_missing().create_features().transform().select_features().encode_advanced().evaluate().save_report().save_data())
