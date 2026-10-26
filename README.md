# Feature Engineering Toolkit

A feature engineering pipeline for machine learning — create, select, and
transform features to improve model performance. Handles scaling, encoding,
interaction terms, and importance analysis.

## Overview

This toolkit implements the full data-science pipeline:

**data cleaning → feature engineering → preprocessing → model building**

It automates:
- **Cleaning** duplicates and missing values
- **Creating** new features (polynomial, interaction, aggregation, datetime)
- **Selecting** the most relevant ones (SelectKBest, RFE, variance threshold)
- **Transforming** them (scaling, power transforms, quantile transforms)
- **Preprocessing** train/test splits for modelling
- **Building** a model and evaluating accuracy

Works with **pandas** and **PySpark** DataFrames — the backend is detected
automatically from whatever you pass in.

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
> PySpark is optional — the pipeline falls back to pandas automatically.

### Requirements

- Python 3.8+
- `pandas`, `numpy`, `scikit-learn`, `pyyaml` (installed automatically)
- `pyspark` (optional, for PySpark DataFrames)
- `polars` (optional)

## Quick Start

```python
import pandas as pd
from feature_engineering_toolkit import FeatureEngineeringPipeline

df = pd.read_csv('your_data.csv')

# Full DS pipeline: cleaning → feature engineering → preprocessing → model
pipeline = FeatureEngineeringPipeline(df, target_col='target')
pipeline.run_full()

print(pipeline.generate_report())
engineered = pipeline.get_engineered_data()
```

Or run the feature-engineering stage only:

```python
pipeline.run()  # create → transform → select → evaluate
```

## Usage

### Step by Step

```python
pipeline = FeatureEngineeringPipeline(df, target_col='target')

pipeline.clean_data()          # remove duplicates
pipeline.handle_missing()      # impute missing values
pipeline.create_features()     # polynomial, interaction, aggregation, datetime
pipeline.transform()           # scaling, power, quantile
pipeline.select_features()     # SelectKBest, RFE, variance threshold
pipeline.encode_advanced()     # target encoding (placeholder)
pipeline.evaluate()            # accuracy score
pipeline.preprocess()          # train/test split
pipeline.build_model()         # train + predict + accuracy
```

### Full pipeline (one call)

```python
pipeline.run_full('/path/to/your_data.csv')
# cleaning → feature engineering → preprocessing → model building
```

### PySpark DataFrames

```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('fe').getOrCreate()
df = spark.read.csv('data.csv', header=True)

pipeline = FeatureEngineeringPipeline(df, target_col='target')
pipeline.run_full()  # backend detected automatically
```

sklearn-based steps (imputation, scaling, model building) convert the
PySpark frame to pandas internally and write results back to the
PySpark format on save.

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
| **Data Cleaning** | Duplicate removal, missing-value imputation |
| **Feature Creation** | Polynomial terms, interaction terms, aggregations, datetime extraction |
| **Feature Selection** | SelectKBest, RFE, variance threshold |
| **Feature Transformation** | StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer |
| **Preprocessing** | Train/test split for modelling |
| **Model Building** | Logistic regression with accuracy evaluation |
| **Backends** | pandas + PySpark, auto-detected |
| **Configuration** | YAML-based config with sensible defaults |

## Project Structure

```
feature-engineering-toolkit/
├── feature_engineering_toolkit/   # Source package
│   ├── __init__.py                # Package exports
│   ├── pipeline.py                # Main pipeline
│   ├── backends.py                # pandas / PySpark backend abstraction
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
