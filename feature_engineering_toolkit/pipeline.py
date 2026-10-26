"""Feature Engineering Pipeline — create, select, and transform features."""
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

class FeatureEngineeringPipeline:
    def __init__(self, df=None, config=None, target_col=None):
        from .config import DEFAULT_CONFIG
        self.config = config or DEFAULT_CONFIG.copy()
        self.df = df.copy() if df is not None else None
        self.target_col = target_col or self.config["data_input"]["target_variable"]
        self.created_features = []
        self.selected_features = []
        self.transformed_features = []
        self.log = []
        self.original_shape = (0, 0)
        self.accuracy = None

    def _log(self, msg):
        entry = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        self.log.append(entry); print(msg)
        return self

    def load(self, path=None):
        from .config import load_config
        path = path or self.config["data_input"]["data_path"]
        p = Path(path)
        if not p.exists():
            alt = Path("data") / p.name
            if alt.exists(): p = alt
        fmt = self.config["data_input"]["file_type"]
        self.df = pd.read_csv(str(p)) if fmt == "csv" else pd.read_parquet(str(p))
        self.original_shape = self.df.shape
        self._log(f"Loaded: {self.df.shape[0]}x{self.df.shape[1]} from {p}")
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
        X = feature_df
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
        feature_cols = [c for c in self.df.columns if c != self.target_col and self.df[c].dtype in [np.float64, np.int64]]
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
        lines = ["="*60, "FEATURE ENGINEERING REPORT", "="*60, f"Original: {self.original_shape}", f"Final: {self.df.shape if self.df is not None else 'N/A'}", f"Created: {len(self.created_features)}", f"Selected: {len(self.selected_features)}", f"Transformed: {len(self.transformed_features)}", f"Accuracy: {self.accuracy or 'N/A'}", ""]
        for e in self.log: lines.append(f"  - {e}")
        lines.append("="*60)
        return "\n".join(lines)

    def save_report(self, path=None):
        path = path or self.config["output"]["report_path"]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f: f.write(f"<pre>{self.generate_report()}</pre>")
        self._log(f"Report saved to {path}")

    def save_data(self, path=None, fmt=None):
        path = path or self.config["output"]["output_path"]
        fmt = fmt or self.config["output"]["output_format"]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        (self.df.to_csv(path, index=False) if fmt=="csv" else self.df.to_parquet(path, index=False))
        self._log(f"Data saved to {path}")

    def get_engineered_data(self):
        return self.df.copy() if self.df is not None else pd.DataFrame()

    def run(self):
        return (self.load().create_features().transform().select_features().encode_advanced().evaluate().save_report().save_data())
