import os
import json
import inspect
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, roc_auc_score, 
                             confusion_matrix, roc_curve, 
                             balanced_accuracy_score,
                             classification_report)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def get_data():
    """
    Load and process consumer, account, and transaction data, merging it with a category mapping file.
    Automatically finds the data directory based on the script's location.

    Returns:
    --------
    tuple
        A tuple of three pandas DataFrames:
        - consumer_df : Consumer data.
        - account_df : Account data.
        - transaction_df : Processed transaction data merged with the category mapping.
    """
    try:
        # Determine the data directory path dynamically
        current_dir = os.path.dirname(os.path.abspath(__file__))  # Current script directory
        data_dir = os.path.join(os.path.dirname(current_dir), 'data', 'raw_data')

        # Define file paths
        consumer_file = os.path.join(data_dir, 'q2-ucsd-consDF.pqt')
        account_file = os.path.join(data_dir, 'q2-ucsd-acctDF.pqt')
        transaction_file = os.path.join(data_dir, 'q2-ucsd-trxnDF.pqt')
        category_map_file = os.path.join(data_dir, 'q2-ucsd-cat-map.csv')

        # Check if files exist
        for file_path in [consumer_file, account_file, transaction_file, category_map_file]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")

        # Load data
        consumer_df = pd.read_parquet(consumer_file)
        account_df = pd.read_parquet(account_file)
        transaction_df = pd.read_parquet(transaction_file)
        category_map_df = pd.read_csv(category_map_file)

        # Merge transaction data with category mapping
        transaction_df = transaction_df.merge(
            category_map_df, 
            how='left', 
            left_on='category', 
            right_on='category_id'
        )
        
        # Clean up columns
        transaction_df.drop(columns=['category_x', 'category_id'], inplace=True)
        transaction_df.rename(columns={'category_y': 'category'}, inplace=True)
            
        # Convert 'posted_date' to datetime format
        transaction_df['posted_date'] = pd.to_datetime(transaction_df['posted_date'])

        # Convert 'balance_date' to datetime format
        account_df['balance_date'] = pd.to_datetime(account_df['balance_date'])

        print("Data successfully loaded and processed.")
        return consumer_df, account_df, transaction_df

    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

def split_data(consumer_df, test_size=0.25, val_size=None, random_state=123):
    """
    Splits the consumer data into training, validation (optional), and test sets 
    after removing rows with null DQ_TARGET.

    Parameters:
        consumer_df (pd.DataFrame):
        test_size (float): The proportion of the dataset to include in the test split.
        val_size (float or None): The proportion of the training dataset to allocate for validation.
                                  If None, no validation set will be created (default is None).
        random_state (int):

    Returns:
        train_df (pd.DataFrame): Training dataframe.
        val_df (pd.DataFrame or None): Validation dataframe (if val_size is specified, else None).
        test_df (pd.DataFrame): Test dataframe.
    """
    # Remove rows with null DQ_TARGET
    clean_df = consumer_df.dropna(subset=['DQ_TARGET'])

    # Split data into training and testing sets
    train_df, test_df = train_test_split(clean_df, test_size=test_size, random_state=random_state)

    # Further split the training data into training and validation sets if val_size is specified
    val_df = None
    if val_size is not None:
        val_size_adjusted = val_size / (1 - test_size)  # Adjust validation size based on the remaining data
        train_df, val_df = train_test_split(train_df, test_size=val_size_adjusted, random_state=random_state)

    # Print the sizes of each set
    print(f"Training set size: {len(train_df)}")
    if val_df is not None:
        print(f"Validation set size: {len(val_df)}")
    else:
        print("No validation set created.")
    print(f"Test set size: {len(test_df)}")

    return train_df, val_df, test_df

def merge_features(dataframes):
    """
    Merge multiple DataFrames on a 'prism_consumer_id' using an inner join.

    Parameters:
    -----------
    dataframes : list of pd.DataFrame
        A list of pandas DataFrames to be merged. All DataFrames must contain the 'prism_consumer_id' column.

    Returns:
    --------
    pd.DataFrame
        A single DataFrame resulting from the inner join of all input DataFrames on the 'prism_consumer_id' column.

    Prints:
    -------
    Shape of the resulting DataFrame after the merge.
    """
    result = dataframes[0]
    for df in dataframes[1:]:
        result = pd.merge(result, df, on='prism_consumer_id', how='inner')  # Use 'inner' to keep consistent rows
    print("\nTotal Created Features: ", result.shape[1] - 1)
    return result

class MetadataManager:
    """
    Manages feature function metadata and caching of feature computation results.
    """
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.temp_dir = os.path.join(self.base_dir, 'data', 'temporary_data')
        os.makedirs(self.temp_dir, exist_ok=True)
        
    def get_metadata_path(self):
        """Get the path to the metadata JSON file."""
        return os.path.join(self.temp_dir, 'feature_metadata.json')
    
    def get_cache_path(self):
        """Get the path to the feature cache directory."""
        cache_dir = os.path.join(self.temp_dir, 'feature_cache')
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
    
    def get_function_metadata(self, fn):
        """
        Extract metadata for a given function.
        
        Parameters:
            fn: Function object to analyze
        
        Returns:
            dict: Metadata including source code hash and last modified time
        """
        source = inspect.getsource(fn)
        hash_object = hashlib.md5(source.encode())
        return {
            'hash': hash_object.hexdigest(),
            'last_modified': datetime.now().isoformat(),
            'name': fn.__name__
        }
    
    def load_metadata(self):
        """Load feature function metadata from JSON file."""
        metadata_path = self.get_metadata_path()
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}
    
    def save_metadata(self, metadata):
        """Save feature function metadata to JSON file."""
        metadata_path = self.get_metadata_path()
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_cached_result(self, feature_name):
        """Load cached feature computation result if it exists."""
        cache_path = os.path.join(self.get_cache_path(), f"{feature_name}.pkl")
        if os.path.exists(cache_path):
            try:
                return pd.read_pickle(cache_path)
            except Exception as e:
                print(f"Warning: Failed to load cache for {feature_name}: {e}")
                return None
        return None
    
    def save_cached_result(self, feature_name, result):
        """Save feature computation result to cache."""
        cache_path = os.path.join(self.get_cache_path(), f"{feature_name}.pkl")
        try:
            result.to_pickle(cache_path)
        except Exception as e:
            print(f"Warning: Failed to cache {feature_name}: {e}")

def execute_selected_features(selected_features, account_df, transaction_df):
    """
    Execute only new or modified feature functions and return their results.
    
    Parameters:
        selected_features: List of feature functions to potentially execute
    
    Returns:
        list: List of DataFrames containing feature results
    """
    metadata_manager = MetadataManager()
    metadata = metadata_manager.load_metadata()
    
    # Analyze which functions need to be executed
    new_features = []
    updated_features = []
    unchanged_features = []
    
    for fn in selected_features:
        current_metadata = metadata_manager.get_function_metadata(fn)
        if fn.__name__ not in metadata:
            new_features.append(fn)
        elif metadata[fn.__name__]['hash'] != current_metadata['hash']:
            updated_features.append(fn)
        else:
            unchanged_features.append(fn)
    
    # Report execution plan
    total_features = len(selected_features)
    print(f"\nFeature Execution Plan:")
    print(f"- Total feature creation functions: {total_features}")
    
    if len(new_features) > 0:
        if len(new_features) <= 5:
            print(f"- New features ({len(new_features)}): {', '.join(f.__name__ for f in new_features)}")
        else:
            print(f"- New features: {len(new_features)}")
    
    if len(updated_features) > 0:
        if len(updated_features) <= 5:
            print(f"- Updated features ({len(updated_features)}): {', '.join(f.__name__ for f in updated_features)}")
        else:
            print(f"- Updated features: {len(updated_features)}")
    
    print(f"- Unchanged feature creation functions: {len(unchanged_features)} (using cached results)")
    
    # Execute only new and updated features
    feature_dataframes = []
    features_to_execute = new_features + updated_features
    
    if features_to_execute:
        print("\nExecuting features...")
        for fn in features_to_execute:
            print(f"- Running {fn.__name__}...")
            result = fn(account_df, transaction_df)
            feature_dataframes.append(result)
            
            # Update metadata and cache for this function
            metadata[fn.__name__] = metadata_manager.get_function_metadata(fn)
            metadata_manager.save_cached_result(fn.__name__, result)
    
        # Save updated metadata
        metadata_manager.save_metadata(metadata)
    else:
        print("\nNo feature creation functions need to be executed.")
    
    # For unchanged features, load from cache or recalculate if cache missing
    for fn in unchanged_features:
        cached_result = metadata_manager.load_cached_result(fn.__name__)
        if cached_result is not None:
            print(f"- Using cached result for {fn.__name__}")
            feature_dataframes.append(cached_result)
        else:
            print(f"- Cache missing for {fn.__name__}, recalculating...")
            result = fn(account_df, transaction_df)
            metadata_manager.save_cached_result(fn.__name__, result)
            feature_dataframes.append(result)
    
    return feature_dataframes

class FeatureSelector:
    """
    A class for selecting the most important features.
    """
    def __init__(self, train_df, feature_columns, forced_features=None, base_dir=None):
        self.X = train_df[feature_columns]
        self.y = train_df['DQ_TARGET']
        
        self.forced_features = forced_features if forced_features is not None else []
        # Validate that each forced feature exists in the dataset
        for feature in self.forced_features:
            if feature not in self.X.columns:
                raise ValueError(f"Forced feature '{feature}' not found in the dataset columns.")
        
        if base_dir is None:
            self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        else:
            self.base_dir = base_dir
        self.results_dir = os.path.join(self.base_dir, 'data', 'temporary_data', 'feature_selection')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Internal dictionary to store results for each method
        self.results = {}
        
    def _convert_to_serializable(self, obj):
        """
        Convert numpy types to native Python types for JSON serialization.
        """
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(x) for x in obj]
        elif isinstance(obj, np.generic):
            return obj.item()
        else:
            return obj
        
    def _compute_importance_scores(self, method):
        """
        Compute feature importance scores using the specified method.
        
        Parameters:
            method (str): The feature selection method ('mutual_info', 'random_forest', 'xgboost')
        
        Returns:
            dict: A dictionary mapping feature names to importance scores.
        """
        if method == 'mutual_info':
            scores = mutual_info_classif(self.X, self.y)
        elif method == 'random_forest':
            rf = RandomForestClassifier(n_estimators=100, random_state=123)
            rf.fit(self.X, self.y)
            scores = rf.feature_importances_
        elif method == 'xgboost':
            xgb = XGBClassifier(random_state=123, eval_metric='logloss')
            xgb.fit(self.X, self.y)
            scores = xgb.feature_importances_
        else:
            raise ValueError(f"Unknown method: {method}")
            
        importance_dict = dict(zip(self.X.columns, scores))
        return importance_dict
    
    def _apply_forced_features(self, sorted_features, n_features):
        """
        Combine forced features with the remaining top features until the desired count is reached.
        
        Parameters:
            sorted_features (list of tuples): List of (feature, importance) sorted in descending order.
            n_features (int): Desired total number of features.
        
        Returns: List of selected feature names.
        """
        # Start with forced features
        selected = list(self.forced_features)
        for feature, _ in sorted_features:
            if feature not in selected:
                selected.append(feature)
            if len(selected) >= n_features:
                break
        return selected
    
    def _save_feature_selection_results(self, importance_dict, selected_features, method):
        """
        Save feature selection results to a JSON file.
        The file is overwritten each time for the given method.
        """
        results = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'method': method,
            'importance_scores': self._convert_to_serializable(importance_dict),
            'selected_features': selected_features
        }
        
        filepath = os.path.join(self.results_dir, f'feature_selection_{method}.json')
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
    
    def _select_features_for_method(self, method, n_features):
        """
        Compute importance scores and select top features (including forced features) for a given method.
        
        Parameters:
            method (str): The feature selection method.
            n_features (int): The number of features to select.
        
        Returns:
            tuple: (selected_features, importance_dict)
        """
        # Compute importance scores
        importance_dict = self._compute_importance_scores(method)
        
        # Sort features by importance (descending)
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Merge forced features with top-ranked features
        selected_features = self._apply_forced_features(sorted_features, n_features)
        
        # Save results to file (overwriting previous file for this method)
        self._save_feature_selection_results(importance_dict, selected_features, method)
        
        return selected_features, importance_dict
    
    def run_feature_selection(self, methods, n_features=50):
        """
        Run feature selection for a list of methods.
        
        Parameters:
            methods: List of feature selection method names (e.g., ['xgboost', 'random_forest']).
            n_features: Number of features to select.
        """
        for method in methods:
            selected_features, importance_dict = self._select_features_for_method(method, n_features)
            self.results[method] = {
                'selected_features': selected_features,
                'importance_scores': importance_dict
            }
    
    def get_selected_features(self, method):
        """
        Retrieve the selected features for the specified method.
        
        Parameters: Feature selection method (str).
        
        Returns: List of selected feature names.
        """
        if method in self.results:
            return self.results[method]['selected_features']
        else:
            raise ValueError(f"No results found for method '{method}'. Run feature selection first.")
    
    def get_importance_scores(self, method):
        """
        Retrieve the importance scores for the specified method.
        
        Parameters: Feature selection method (str).
        
        Returns: Dictionary of feature importance scores.
        """
        if method in self.results:
            return self.results[method]['importance_scores']
        else:
            raise ValueError(f"No results found for method '{method}'. Run feature selection first.")
    
    def plot_feature_importance(self, method, top_n=20):
        """
        Plot the feature importance scores for the specified method.
        
        Parameters:
            method (str): The feature selection method whose importance scores will be plotted.
            top_n (int): Number of top features to display in the plot.
        """
        if method not in self.results:
            raise ValueError(f"No results found for method '{method}'. Run feature selection first.")
            
        importance_dict = self.results[method]['importance_scores']
        # Sort features by importance (descending) and select top_n
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        
        features, scores = zip(*top_features)
        plt.figure(figsize=(12, 12))
        plt.barh(range(len(features)), scores)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_n} Most Important Features ({method})')
        plt.gca().invert_yaxis()  # Highest importance at the top
        plt.tight_layout()
        plt.show()

def train_model(train_df, feature_columns, model_name="logistic", random_state=123):
    """
    Trains a specified model (LogisticRegression, RandomForest ...) on the training dataframe
    using a predefined set of hyperparameters.

    Parameters:
    -----------
    train_df : pd.DataFrame
        Training dataframe containing the features and the target column 'DQ_TARGET'.
    feature_columns : list
        The feature columns to be used for training.
    model_name : str (default='logistic')
        Identifier for which model to train. Can be extended to other model names.

    Returns:
    --------
    model : sklearn model
        The trained model instance.
    scaler : StandardScaler
        The fitted scaler used to transform the training (and later test) data.
    """

    # Predefined model parameters for each model_name
    # Adjust or extend as needed for your specific use case
    model_params_dict = {
        "logistic": {
            "class": LogisticRegression,
            "params": {
                "class_weight": "balanced",
                "max_iter": 1000
            }
        },
        "random_forest": {
            "class": RandomForestClassifier,
            "params": {
                "class_weight": "balanced",
                "n_estimators": 100
            }
        },
        "gradient_boosting": {
            "class": GradientBoostingClassifier,
            "params": {
                "n_estimators": 100,
                "max_depth": 3
            }
        },
        "xgboost": {
            "class": XGBClassifier,
            "params": {
                "eval_metric": "logloss",
                "scale_pos_weight": 19, # (#negatives / #positives) 95/15
                "n_estimators": 100,
                "learning_rate": 0.1,       # Lower learning rates can help reduce overfitting.
                "max_depth": 3,            # Deeper trees can overfit;
                "subsample": 0.8,          # Row subsampling: helps reduce variance.
                "colsample_bytree": 0.8,   # Feature subsampling: also helps reduce variance.
                "reg_lambda": 1.0         # L2 regularization (default=1)
            }
        },
        "lightgbm": {
            "class": LGBMClassifier,
            "params": {
                "class_weight": "balanced"
                # For heavily imbalanced data, you can also tune "is_unbalance" or adjust "scale_pos_weight".
            }
        }
    }

    # Check if the requested model is in our dictionary
    if model_name not in model_params_dict:
        raise ValueError(
            f"Unknown model_name '{model_name}'. "
            f"Valid options are: {list(model_params_dict.keys())}"
        )

    chosen_model_class = model_params_dict[model_name]["class"]
    chosen_model_params = model_params_dict[model_name]["params"]

    # Inject the random_state into the model’s parameters (if it’s relevant)
    chosen_model_params["random_state"] = random_state

    # Initialize model with the predefined parameters
    model = chosen_model_class(**chosen_model_params)

    # Define features (X) and target (y) for training
    X_train = train_df[feature_columns]
    y_train = train_df['DQ_TARGET']

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train the model
    model.fit(X_train_scaled, y_train)

    return model, scaler

def predict_and_analyze_model(model, scaler, train_df, test_df, feature_columns):
    """
    Predicts and evaluates the model using accuracy, ROC AUC score, and a detailed confusion matrix.
    
    Parameters:
    -----------
    model : (e.g., LogisticRegression, RandomForest).
    scaler : The fitted scaler used during training.
    train_df : Training dataframe containing features and the target column "DQ_TARGET".
    test_df : Testing dataframe containing features and the target column "DQ_TARGET".
    feature_columns : List of column names to use as features.
    """

    # ===== Training Performance =====
    X_train = train_df[feature_columns]
    y_train = train_df['DQ_TARGET']
    X_train_scaled = scaler.transform(X_train)

    # Make predictions on training data
    y_pred_train = model.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    train_balanced_acc = balanced_accuracy_score(y_train, y_pred_train)

    # Some models might not have predict_proba
    y_pred_proba_train = model.predict_proba(X_train_scaled)[:, 1] if hasattr(model, "predict_proba") else None
    train_roc_auc = roc_auc_score(y_train, y_pred_proba_train) if y_pred_proba_train is not None else None

    # ===== Testing Performance =====
    X_test = test_df[feature_columns]
    y_test = test_df['DQ_TARGET']
    X_test_scaled = scaler.transform(X_test)

    y_pred_test = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred_test)
    balanced_acc = balanced_accuracy_score(y_test, y_pred_test)
    
    y_pred_proba_test = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
    roc_auc = roc_auc_score(y_test, y_pred_proba_test) if y_pred_proba_test is not None else None
    
    conf_matrix = confusion_matrix(y_test, y_pred_test)
    conf_matrix_df = pd.DataFrame(
        conf_matrix,
        index=['Actual Negative', 'Actual Positive'],
        columns=['Predicted Negative', 'Predicted Positive']
    )

    # ===== Visualizations =====
    # Confusion Matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()

    # ROC Curve
    if y_pred_proba_test is not None:
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_test)
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, label='Model')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Random Chance')
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.show()
    else:
        print("ROC Curve not available for the chosen model.")

    # ===== Print metrics =====
    print("=== TEST METRICS ===")
    print("Accuracy:", accuracy)
    if roc_auc is not None:
        print("ROC AUC Score:", roc_auc)
    print("Balanced Accuracy:", balanced_acc)
    print("Classification Report:\n", classification_report(y_test, y_pred_test))

    print("=== TRAINING METRICS ===")
    print("Train Accuracy:", train_accuracy)
    if train_roc_auc is not None:
        print("Train ROC AUC:", train_roc_auc)
    print("Train Balanced Accuracy:", train_balanced_acc)

def filter_unknown(consumer_df, account_df, transaction_df):
    """
    Filter out consumers who do not have either account or transaction information.

    This function identifies consumers present in the consumer DataFrame who are missing in 
    the transaction or account DataFrames. It then filters these consumers out from all three 
    DataFrames so that only consumers with both account and transaction data remain.

    Parameters:
        consumer_df (pd.DataFrame): DataFrame containing consumer-level data.
        account_df (pd.DataFrame): DataFrame containing account-level data.
        transaction_df (pd.DataFrame): DataFrame containing transaction-level data.

    Returns:
        tuple: A tuple of three DataFrames:
            - filtered_consumer_df: Consumer data with only consumers having both account and transaction info.
            - filtered_account_df: Account data filtered by consumers with both account and transaction info.
            - filtered_transaction_df: Transaction data filtered by consumers with both account and transaction info.
    """
    # Identify consumer IDs missing in the transaction DataFrame
    transaction_unknown = set(consumer_df['prism_consumer_id']) - set(transaction_df['prism_consumer_id'])
    
    # Identify consumer IDs missing in the account DataFrame
    account_unknown = set(consumer_df['prism_consumer_id']) - set(account_df['prism_consumer_id'])
    
    # Combine the two sets to get all consumer IDs with missing account or transaction info
    total_unknown = account_unknown.union(transaction_unknown)
    
    # Filter the consumer DataFrame to retain only those consumers who have both account and transaction data
    filtered_consumer_df = consumer_df[~consumer_df['prism_consumer_id'].isin(total_unknown)]
    
    # Filter the account DataFrame to remove data for consumers with missing info
    filtered_account_df = account_df[~account_df['prism_consumer_id'].isin(total_unknown)]
    
    # Filter the transaction DataFrame to remove data for consumers with missing info
    filtered_transaction_df = transaction_df[~transaction_df['prism_consumer_id'].isin(total_unknown)]

    return filtered_consumer_df, filtered_account_df, filtered_transaction_df

def apply_smote_to_train_df(train_df, label_col, random_state=42):
    """
    Apply SMOTE to balance the training dataset contained in a single DataFrame.
    
    This function extracts the label column from the training DataFrame, applies SMOTE to 
    balance the classes, and then recombines the resampled features and labels back into a 
    new DataFrame.
    
    Args:
        train_df (pd.DataFrame): Training DataFrame containing both features and the label.
        label_col (str): Column name representing the target label.
        random_state (int): Random state for reproducibility.
        
    Returns:
        pd.DataFrame: Resampled training DataFrame with balanced classes.
    """
    from imblearn.over_sampling import SMOTE

    # Separate features (X) and label (y)
    X = train_df.drop(columns=[label_col])
    y = train_df[label_col]
    
    # Apply SMOTE to the feature matrix and label vector
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Recombine the resampled features and labels into a DataFrame
    train_resampled_df = X_resampled.copy()
    train_resampled_df[label_col] = y_resampled
    
    print("\nTraining set class distribution after SMOTE:")
    print(train_resampled_df[label_col].value_counts(normalize=True))
    
    return train_resampled_df
