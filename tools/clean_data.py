from crewai_tools import tools
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler


@tools('Data Cleaner')
def preprocess_df(df, 
                   handle_missing_values="median", 
                   target_variable=None, 
                   categorical_handling="auto", 
                   scaling="standard") -> pd.DataFrame:
  """
  Preprocesses a dataframe for model training with more options. It understands the dataset and based on its parameters, clean the dataset for
  model training.

  Args:
      df (str): the dataframe
      handle_missing_values (str, optional): Strategy for handling missing values. Defaults to "median". Valid options are "median", "mean", "knn" (K-Nearest Neighbors).
      target_variable (str, optional): Name of the target variable (if any). Excluding it assumes it's not part of preprocessing. Defaults to None.
      categorical_handling (str, optional): Approach for handling categorical features. Defaults to "auto". Valid options are "auto" (automatic based on cardinality), "label_encoder", "onehot_encoder".
      scaling (str, optional): Scaling method for numerical features. Defaults to "standard". Valid options are "standard", "minmax".

  Returns:
      pd.DataFrame: The preprocessed DataFrame.
  """

  # Handle missing values
  if handle_missing_values == "knn":
    imputer = KNNImputer()  # Use KNN for potentially better imputation (adjust n_neighbors if needed)
  else:
    imputer_strategy = "median" if handle_missing_values in ("median", "mean") else handle_missing_values
    imputer = SimpleImputer(strategy=imputer_strategy)
  df = imputer.fit_transform(df)

  # Separate target variable (if specified)
  if target_variable:
    y = df[target_variable]
    X = df.drop(target_variable, axis=1)
  else:
    X = df

  # Handle categorical features
  categorical_cols = X.select_dtypes(include=[object]).columns
  if categorical_handling == "auto":
    categorical_handling = "label_encoder" if X[categorical_cols].nunique().max() == 1 else "onehot_encoder"
  for col in categorical_cols:
    if categorical_handling == "label_encoder":
      le = LabelEncoder()
      X[col] = le.fit_transform(X[col])
    elif categorical_handling == "onehot_encoder":
      encoder = OneHotEncoder(sparse=False)
      encoded_cols = pd.DataFrame(encoder.fit_transform(X[[col]]))
      encoded_cols.columns = [f"{col}_{i}" for i in range(encoded_cols.shape[1])]
      X = pd.concat([X, encoded_cols], axis=1).drop(col, axis=1)

  # Handle numerical features (optional: scaling)
  numerical_cols = X.select_dtypes(include=[np.number]).columns
  if scaling == "standard":
    scaler = StandardScaler()
  elif scaling == "minmax":
    scaler = MinMaxScaler()
  else:
    scaler = None  # No scaling applied
  if scaler:
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

  # Re-combine target variable if applicable
  if target_variable:
    return X, y
  else:
    return X
