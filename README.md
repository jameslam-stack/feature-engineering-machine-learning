# Feature Engineering Toolkit

A feature engineering pipeline for machine learning — create, select, and
transform features to improve model performance. Handles scaling, encoding,
interaction terms, and importance analysis.

## Overview

This toolkit implements feature engineering for ML models. It automates:
- **Creating** new features (polynomial, interaction, aggregation, datetime)
- **Selecting** the most relevant ones (SelectKBest, RFE, variance threshold)
- **Transforming** them (scaling, power transforms, quantile transforms)
- **Evaluating** impact on model accuracy

## Installation

Install directly from GitHub (clone the repo, then install in editable mode):

```bash
# 1. Clone the repo
git clone https://github.com/jameslam-stack/feature-engineering-machine-learning.git
cd feature-engineering-toolkit

# 2. Install (editable mode — no PyPI required)
pip install -e .

# 3. Install dependencies
pip install scikit-learn pandas numpy pyyaml
```

> **Note:** This package is not on PyPI. Use the GitHub install above.

### Requirements

- Python 3.8+
- `pandas`, `numpy`, `scikit-learn`, `pyyaml` (installed automatically)
- `polars` (optional)

## Quick Start

```python
import pandas as pd
from feature_engineering_toolkit import FeatureEngineeringPipeline

df = pd.read_csv('your_data.csv')

pipeline = FeatureEngineeringPipeline(df, target_col='target')
pipeline.run()  # create → transform → select → evaluate

print(pipeline.generate_report())
engineered = pipeline.get_engineered_data()
```

## Usage

### Step by Step

```python
pipeline = FeatureEngineeringPipeline(df, target_col='target')

pipeline.create_features()     # polynomial, interaction, aggregation, datetime
pipeline.transform()           # scaling, power, quantile
pipeline.select_features()     # SelectKBest, RFE, variance threshold
pipeline.encode_advanced()     # target encoding (placeholder)
pipeline.evaluate()            # accuracy score
```

### Configuration

Edit `config_template.yaml` to customize:

```yaml
feature_creation:
  enabled: true
  techniques:
    - "polynomial_features"
    - "interaction_terms"
    - "aggregation_features"
    - "datetime_features"

feature_transformation:
  method: "standard_scaler"  # or min_max_scaler, robust_scaler, power_transformer

feature_selection:
  method: "select_k_best"    # or recursive_feature_elimination, variance_threshold
  select_k_best:
    k: 10
    score_func: "f_classif"
```

## Features

| Feature | Description |
|---------|-------------|
| **Feature Creation** | Polynomial terms, interaction terms, aggregations, datetime extraction |
| **Feature Selection** | SelectKBest, RFE, variance threshold |
| **Feature Transformation** | StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer |
| **Evaluation** | Model accuracy comparison |
| **Configuration** | YAML-based config with sensible defaults |

## Project Structure

```
feature-engineering-toolkit/
├── feature_engineering_toolkit/   # Source package
│   ├── __init__.py                # Package exports
│   ├── pipeline.py                # Main pipeline
│   └── config.py                  # Config loading
├── tests/                         # Test suite
├── examples/                      # Tutorial notebooks
├── data/                          # Sample data
├── models/                        # Saved models
├── reports/                       # Generated reports
├── README.md
├── pyproject.toml
└── config_template.yaml
```

## Examples

See `examples/teach_features.ipynb` for a beginner-friendly walkthrough.

## License

MIT License — see LICENSE file.
