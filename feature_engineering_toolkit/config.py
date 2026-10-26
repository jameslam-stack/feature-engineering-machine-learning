"""Configuration loading for the Feature Engineering Toolkit."""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union


# Default configuration matching the skill's configuration_template.yaml
DEFAULT_CONFIG: Dict[str, Any] = {
    "data_input": {
        "data_path": "data/input.csv",
        "file_type": "csv",
        "csv_separator": ",",
        "csv_quotechar": '"',
        "encoding": "utf-8",
        "target_variable": "target",
        "index_column": None,
    },
    "feature_creation": {
        "enabled": True,
        "techniques": ["polynomial_features", "interaction_terms", "aggregation_features", "datetime_features"],
        "polynomial_features": {"degree": 2, "include_bias": False},
        "interaction_terms": {"interaction_degree": 2},
        "aggregation_features": {
            "group_by": "user_id",
            "aggregations": ["mean", "sum"],
        },
        "datetime_features": {
            "datetime_columns": ["date"],
            "features_to_extract": ["year", "month", "dayofweek"],
        },
    },
    "feature_selection": {
        "enabled": True,
        "method": "select_k_best",
        "variance_threshold": {"threshold": 0.0},
        "select_k_best": {"k": 10, "score_func": "f_classif"},
        "recursive_feature_elimination": {"n_features_to_select": 10, "step": 1},
    },
    "feature_transformation": {
        "enabled": True,
        "method": "standard_scaler",
        "standard_scaler": {},
        "min_max_scaler": {"feature_range": [0, 1]},
        "robust_scaler": {"quantile_range": [25.0, 75.0]},
        "power_transformer": {"method": "yeo-johnson", "standardize": True},
        "quantile_transformer": {"n_quantiles": 100, "output_distribution": "uniform"},
    },
    "output": {
        "output_path": "data/processed.csv",
        "output_format": "csv",
        "save_report": True,
        "report_path": "reports/feature_engineering_report.html",
        "save_model": False,
        "model_path": "models/feature_selection_model.pkl",
    },
    "advanced": {
        "n_jobs": -1,
        "random_state": 42,
        "verbosity": 1,
        "handle_missing": True,
        "imputation_strategy": "mean",
        "imputation_constant": 0,
    },
}


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load a feature engineering configuration from a YAML file.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to a YAML config file. If None, returns the default config.

    Returns
    -------
    dict
        The merged configuration dictionary.

    Examples
    --------
    >>> cfg = load_config('config.yaml')
    >>> cfg = load_config()  # use defaults
    """
    if config_path is None:
        return DEFAULT_CONFIG.copy()

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        user_config = yaml.safe_load(f) or {}

    # Merge user config with defaults (shallow merge per section)
    merged = DEFAULT_CONFIG.copy()
    for section, section_cfg in user_config.items():
        if section in merged and isinstance(merged[section], dict):
            merged[section] = {**merged[section], **section_cfg}
        else:
            merged[section] = section_cfg

    return merged


def save_config(config: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """Save a configuration dictionary to a YAML file.

    Parameters
    ----------
    config : dict
        Configuration dictionary to save.
    output_path : str or Path
        Destination file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Config saved to {output_path}")