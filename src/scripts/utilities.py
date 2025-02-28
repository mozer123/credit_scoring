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
import shap

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


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
        data_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'data')

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
        self.temp_dir = os.path.join(self.base_dir, 'temporary_data')
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
        self.results_dir = os.path.join(self.base_dir, 'temporary_data', 'feature_selection')
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

def get_model_params(model_name, use_optimized=False, config_path='model_config.json'):
    """
    Retrieves the model class and parameters from the JSON configuration file.

    Parameters:
    -----------
    model_name : The key in the JSON file corresponding to the desired model.
    use_optimized : Whether to use the optimized parameters if they are available.
    config_path : Path to the JSON configuration file.

    Returns:
    --------
    model_class : The model class (e.g., LogisticRegression).
    params : Dictionary of parameters to instantiate the model.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    if model_name not in config:
        raise ValueError(f"Unknown model_name '{model_name}'. Valid options are: {list(config.keys())}")

    model_info = config[model_name]
    # Use optimized parameters if requested and available; otherwise, fallback to default.
    params = model_info.get("optimized_params") if use_optimized and model_info.get("optimized_params") else model_info.get("default_params")
    model_class_str = model_info.get("class")

    # Mapping from string names to actual Python classes.
    class_mapping = {
        "LogisticRegression": LogisticRegression,
        "RandomForestClassifier": RandomForestClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "XGBClassifier": XGBClassifier,
        "LGBMClassifier": LGBMClassifier,
        "SequentialNN": SequentialNN
    }

    if model_class_str not in class_mapping:
        raise ValueError(f"Unknown model class '{model_class_str}' for model '{model_name}'.")

    model_class = class_mapping[model_class_str]
    return model_class, params

def train_model(train_df, feature_columns, model_name, use_optimized=False, config_path='model_config.json'):
    """
    Trains the specified model using parameters loaded from a JSON configuration file.

    Parameters:
    -----------
    train_df : Training dataframe containing the features and target column 'DQ_TARGET'.
    feature_columns : List of feature column names used for training.
    model_name : Key corresponding to the model configuration in the JSON file.
    use_optimized : Whether to use the optimized parameters from the config file.
    config_path : Path to the JSON configuration file.
    
    Returns:
    --------
    model : The trained model instance.
    scaler : The fitted scaler used to transform the training data.
    """
    # Retrieve model class and parameters
    model_class, model_params = get_model_params(model_name, use_optimized, config_path)
    
    # Define features and target
    X_train = train_df[feature_columns]
    y_train = train_df['DQ_TARGET']

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Initialize and train the model
    model = model_class(**model_params)
    model.fit(X_train_scaled, y_train)

    return X_train, X_train_scaled, model, scaler

def predict_and_analyze_model(model, scaler, train_df, test_df, feature_columns,
                              show_metrics=True,
                              show_classification_report=True,
                              show_confusion_matrix=True,
                              show_roc_curve=True):
    """
    Predicts and evaluates a classification model on both training and test data.
    Returns a dictionary containing computed metrics and figure objects.
    
    Parameters:
    -----------
    model : sklearn-like model with .predict() and .predict_proba() methods.
    scaler : Fitted scaler to transform features.
    train_df : Training dataframe containing features and target column "DQ_TARGET".
    test_df : Testing dataframe containing features and target column "DQ_TARGET".
    feature_columns : List of column names used as features.
    
    Optional Display Flags:
    -------------------------
    show_metrics : If True, displays a metrics table graph for overall accuracy, balanced accuracy, and ROC AUC for both train and test.
    show_classification_report : If True, prints the classification report for both train and test side by side.
    show_confusion_matrix : If True, displays the confusion matrix plots for train and test side by side.
    show_roc_curve : If True, displays the ROC curve plot for the test set.
        
    Returns:
    --------
    results : dict
        A dictionary with the following keys:
            - 'test': Dictionary containing test metrics and ROC data.
            - 'train': Dictionary containing train metrics.
            - 'figures': Dictionary with figure objects (if any plots were generated).
    """
    results = {}
    figures = {}
    
    # --------------------------
    # Testing Performance
    # --------------------------
    X_test = test_df[feature_columns]
    y_test = test_df['DQ_TARGET']
    X_test_scaled = scaler.transform(X_test)
    
    y_pred_test = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_balanced_acc = balanced_accuracy_score(y_test, y_pred_test)
    y_pred_proba_test = model.predict_proba(X_test_scaled)[:, 1]
    test_roc_auc = roc_auc_score(y_test, y_pred_proba_test)
    
    # ROC curve data
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_pred_proba_test)
    
    # Confusion matrix
    conf_matrix_test = confusion_matrix(y_test, y_pred_test)
    conf_matrix_test_df = pd.DataFrame(
        conf_matrix_test,
        index=['Actual Negative', 'Actual Positive'],
        columns=['Predicted Negative', 'Predicted Positive']
    )
    
    # Save test metrics
    results['test'] = {
        'accuracy': test_accuracy,
        'balanced_accuracy': test_balanced_acc,
        'roc_auc': test_roc_auc,
        'roc_curve': {
            'fpr': fpr_test,
            'tpr': tpr_test,
            'thresholds': thresholds_test
        },
        'confusion_matrix': conf_matrix_test,
        'classification_report': classification_report(y_test, y_pred_test, output_dict=True)
    }
    
    # --------------------------
    # Training Performance
    # --------------------------
    X_train = train_df[feature_columns]
    y_train = train_df['DQ_TARGET']
    X_train_scaled = scaler.transform(X_train)
    
    y_pred_train = model.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    train_balanced_acc = balanced_accuracy_score(y_train, y_pred_train)
    y_pred_proba_train = model.predict_proba(X_train_scaled)[:, 1]
    train_roc_auc = roc_auc_score(y_train, y_pred_proba_train)
    
    # ROC curve data
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_pred_proba_train)
    
    # Confusion matrix
    conf_matrix_train = confusion_matrix(y_train, y_pred_train)
    conf_matrix_train_df = pd.DataFrame(
        conf_matrix_train,
        index=['Actual Negative', 'Actual Positive'],
        columns=['Predicted Negative', 'Predicted Positive']
    )
    
    results['train'] = {
        'accuracy': train_accuracy,
        'balanced_accuracy': train_balanced_acc,
        'roc_auc': train_roc_auc,
        'roc_curve': {
            'fpr': fpr_train,
            'tpr': tpr_train,
            'thresholds': thresholds_train
        },
        'confusion_matrix': conf_matrix_train,
        'classification_report': classification_report(y_train, y_pred_train, output_dict=True)
    }

    # --------------------------
    # Generate Metrics Table as a Figure
    # --------------------------
    metrics_df = pd.DataFrame({
        'Train': [train_accuracy, train_balanced_acc, train_roc_auc],
        'Test': [test_accuracy, test_balanced_acc, test_roc_auc]
    }, index=["Accuracy", "Balanced Accuracy", "ROC AUC"])

    fig_metrics, ax_metrics = plt.subplots(figsize=(6, 2.5))
    ax_metrics.set_axis_off()

    tbl = ax_metrics.table(cellText=metrics_df.round(3).values,
                        rowLabels=metrics_df.index,
                        colLabels=metrics_df.columns,
                        cellLoc='center',
                        loc='center')

    # Customize the table appearance
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.2, 1.2)

    # Style header and row label cells
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor('black')
        # Header cells: top row and left-most row labels (col < 0)
        if row == 0 or col < 0:
            cell.set_facecolor('#40466e')
            cell.set_text_props(color='white', weight='bold')
        else:
            cell.set_facecolor('white')

    ax_metrics.set_title("Overall Metrics (Train vs Test)", fontweight="bold", fontsize=14)
    fig_metrics.tight_layout()
    figures['metrics_table'] = fig_metrics

    if show_metrics:
        plt.show()
    else:
        plt.close(fig_metrics)
    
    # --------------------------
    # Display Classification Reports Side by Side
    # --------------------------
    if show_classification_report:
        # Convert classification report dictionaries to DataFrames
        cr_train_df = pd.DataFrame(results['train']['classification_report']).T
        cr_test_df = pd.DataFrame(results['test']['classification_report']).T
        
        combined_cr = pd.concat([cr_train_df, cr_test_df], axis=1, keys=["Train", "Test"])
        
        print("\n=== Classification Report (Train vs Test) ===")
        print(combined_cr.to_string())
    
    # --------------------------
    # Visualizations: Confusion Matrix for both Train and Test side by side
    # --------------------------
    # Create matrices for Train
    total_train = conf_matrix_train.sum()
    annot_train = np.empty_like(conf_matrix_train, dtype=object)
    for i in range(conf_matrix_train.shape[0]):
        for j in range(conf_matrix_train.shape[1]):
            count = conf_matrix_train[i, j]
            perc = count / total_train * 100 if total_train > 0 else 0
            annot_train[i, j] = f"{count}\n({perc:.1f}%)"
    
    # Create matrices for Test
    total_test = conf_matrix_test.sum()
    annot_test = np.empty_like(conf_matrix_test, dtype=object)
    for i in range(conf_matrix_test.shape[0]):
        for j in range(conf_matrix_test.shape[1]):
            count = conf_matrix_test[i, j]
            perc = count / total_test * 100 if total_test > 0 else 0
            annot_test[i, j] = f"{count}\n({perc:.1f}%)"
    
    # Generate the confusion matrix figure
    fig_cm, (ax_cm_train, ax_cm_test) = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.heatmap(conf_matrix_train_df, annot=annot_train, fmt="", cmap="Blues", cbar=False, ax=ax_cm_train)
    ax_cm_train.set_title("Confusion Matrix (Train)")
    ax_cm_train.set_ylabel("Actual")
    ax_cm_train.set_xlabel("Predicted")
    
    sns.heatmap(conf_matrix_test_df, annot=annot_test, fmt="", cmap="Blues", cbar=False, ax=ax_cm_test)
    ax_cm_test.set_title("Confusion Matrix (Test)")
    ax_cm_test.set_ylabel("Actual")
    ax_cm_test.set_xlabel("Predicted")
    
    plt.tight_layout()
    if show_confusion_matrix:
        plt.show()
    else:
        plt.close(fig_cm)
    figures['confusion_matrix'] = fig_cm
    
    # --------------------------
    # Visualizations: ROC Curve for Test Set only
    # --------------------------
    # Generate the ROC curve figure
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    ax_roc.plot(fpr_test, tpr_test, label='Test ROC')
    ax_roc.plot([0, 1], [0, 1], linestyle='--', label='Random Chance')
    ax_roc.set_title("ROC Curve (Test)")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend()
    if show_roc_curve:
        plt.show()
    else:
        plt.close(fig_roc)
    figures['roc_curve'] = fig_roc
    
    results['figures'] = figures
    
    return X_test, X_test_scaled, y_test, y_pred_test, results

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

def evaluate_feature_model_combinations(train_df, test_df, all_feature_columns,
                                          feature_selection_methods=['all_features', 'mutual_info', 'random_forest', 'xgboost'],
                                          model_types=['xgboost', 'gradient_boosting', 'random_forest', 'logistic']):
    """
    Evaluates model performance for various combinations of feature selection methods
    and model types. For each combination, a model is trained, predictions are made,
    and test performance metrics (accuracy, balanced accuracy, ROC AUC) are computed.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training data containing feature columns and target column "DQ_TARGET".
    test_df : pd.DataFrame
        Testing data containing feature columns and target column "DQ_TARGET".
    all_feature_columns : list
        List of all available feature column names.
    feature_selection_methods : list, optional
        List of feature selection approaches to try. Use 'all_features' to skip feature selection.
        Default is ['all_features', 'mutual_info', 'random_forest', 'xgboost'].
    model_types : list, optional
        List of model types to train. Acceptable values are 
        ['lightgbm', 'xgboost', 'gradient_boosting', 'random_forest', 'logistic'].
    
    Returns:
    --------
    pd.DataFrame
        A DataFrame summarizing test accuracy, balanced accuracy, and ROC AUC for every combination.
    """
    import pandas as pd
    
    results_list = []
    
    # Loop over each feature selection method.
    for fs_method in feature_selection_methods:
        if fs_method == 'all_features':
            selected_features = all_feature_columns
        else:
            # Initialize the FeatureSelector with the full set of features.
            fs = FeatureSelector(train_df, all_feature_columns)
            # Run feature selection with the current method, selecting top 50 features.
            fs.run_feature_selection(methods=[fs_method], n_features=50)
            selected_features = fs.get_selected_features(fs_method)
        
        # Loop over each model type.
        for model_type in model_types:
            print(f"Evaluating: Feature Selection = {fs_method}, Model = {model_type}")
            
            # Train the model using the selected features.
            _, _, model, scaler = train_model(train_df, selected_features, model_type)
            
            # Evaluate the model (plots and additional displays are disabled).
            _, _, _, _, results = predict_and_analyze_model(
                model, scaler, train_df, test_df, selected_features,
                show_metrics=False,
                show_classification_report=False,
                show_confusion_matrix=False,
                show_roc_curve=False
            )
            
            test_metrics = results['test']
            results_list.append({
                "Feature Selection": fs_method,
                "Model": model_type,
                "Test Accuracy": round(test_metrics['accuracy'], 2),
                "Test Balanced Accuracy": round(test_metrics['balanced_accuracy'], 2),
                "Test ROC AUC": round(test_metrics['roc_auc'], 2)
            })
    
    # Create and return a summary DataFrame.
    summary_df = pd.DataFrame(results_list)
    return summary_df

def shap_plots(model, X, X_scaled):
    scaled_feats = pd.DataFrame(X_scaled)
    scaled_feats.columns = X.columns

    explainer = shap.Explainer(model)

    shap_values = explainer(scaled_feats)

    shap.summary_plot(shap_values, scaled_feats)

    return scaled_feats, explainer, shap_values

class SequentialNN:
    def __init__(self, hidden_layers=[128, 64, 32], dropout_rate=0.3, 
                 learning_rate=0.001, batch_size=32, epochs=100, 
                 early_stopping_patience=10, random_state=123):
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        
        # Set random seeds for reproducibility
        tf.random.set_seed(random_state)
        self.model = None
        
    def build_model(self, input_dim):
        model = Sequential()
        
        # First hidden layer
        model.add(Dense(self.hidden_layers[0], activation='relu', 
                       input_dim=input_dim))
        model.add(Dropout(self.dropout_rate))
        
        # Additional hidden layers
        for units in self.hidden_layers[1:]:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                     loss='binary_crossentropy',
                     metrics=['accuracy', 'AUC'])
        
        self.model = model
        return self
    
    def fit(self, X, y):
        if self.model is None:
            self.build_model(X.shape[1])
            
        # Print model summary
        print("\nModel Architecture:")
        self.model.summary()
        
        # Print class distribution
        print("\nClass Distribution:")
        print(pd.Series(y).value_counts(normalize=True))
        
        # Calculate class weights
        n_samples = len(y)
        n_positive = sum(y)
        n_negative = n_samples - n_positive
        weight_for_0 = (1 / n_negative) * (n_samples / 2)
        weight_for_1 = (1 / n_positive) * (n_samples / 2)
        class_weight = {0: weight_for_0, 1: weight_for_1}
        
        print("\nClass Weights:")
        print(class_weight)
            
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train with verbose=1 to see progress
        history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            class_weight=class_weight,
            verbose=1
        )
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        return self
    
    def predict(self, X):
        predictions = self.model.predict(X, verbose=0)
        # Print prediction distribution
        print("\nPrediction Distribution:")
        print(pd.Series((predictions > 0.5).flatten().astype(int)).value_counts(normalize=True))
        return (predictions > 0.5).astype(int).flatten()
    
    def predict_proba(self, X):
        probs = self.model.predict(X, verbose=0).flatten()
        return np.vstack((1 - probs, probs)).T

def identify_thin_files(transaction_df, account_df, consumer_df, 
                       min_transactions=10,
                       min_transaction_months=3,
                       min_balance_records=5):
    """
    Identifies consumers with insufficient data for reliable scoring.
    
    Parameters:
    -----------
    transaction_df : DataFrame
        Transaction data
    account_df : DataFrame
        Account balance data
    consumer_df : DataFrame
        Consumer information
    min_transactions : int
        Minimum number of transactions required
    min_transaction_months : int
        Minimum number of months with transaction activity
    min_balance_records : int
        Minimum number of balance records
    
    Returns:
    --------
    DataFrame
        Original consumer_df with additional columns indicating data sufficiency
    """
    # Calculate transaction metrics
    transaction_counts = transaction_df.groupby('prism_consumer_id').agg({
        'posted_date': ['count', 'nunique']
    }).reset_index()
    transaction_counts.columns = ['prism_consumer_id', 'total_transactions', 'unique_transaction_dates']
    
    # Calculate number of months with transactions
    transaction_df['transaction_month'] = transaction_df['posted_date'].dt.to_period('M')
    months_with_transactions = transaction_df.groupby('prism_consumer_id')['transaction_month'].nunique().reset_index()
    months_with_transactions.columns = ['prism_consumer_id', 'months_with_transactions']
    
    # Calculate balance record metrics
    balance_counts = account_df.groupby('prism_consumer_id').agg({
        'balance_date': 'count'
    }).reset_index()
    balance_counts.columns = ['prism_consumer_id', 'balance_records']
    
    # Merge all metrics with consumer data
    result_df = consumer_df.merge(transaction_counts, on='prism_consumer_id', how='left')
    result_df = result_df.merge(months_with_transactions, on='prism_consumer_id', how='left')
    result_df = result_df.merge(balance_counts, on='prism_consumer_id', how='left')
    
    # Fill NaN values with 0 for consumers with no data
    result_df = result_df.fillna({
        'total_transactions': 0,
        'unique_transaction_dates': 0,
        'months_with_transactions': 0,
        'balance_records': 0
    })
    
    # Create flags for different types of thin files
    result_df['insufficient_transactions'] = result_df['total_transactions'] < min_transactions
    result_df['insufficient_transaction_history'] = result_df['months_with_transactions'] < min_transaction_months
    result_df['insufficient_balance_records'] = result_df['balance_records'] < min_balance_records
    
    # Create overall thin file flag
    result_df['is_thin_file'] = (
        result_df['insufficient_transactions'] |
        result_df['insufficient_transaction_history'] |
        result_df['insufficient_balance_records']
    )
    
    return result_df

def analyze_thin_files(df):
    """
    Analyzes the characteristics and default rates of thin vs thick files.
    
    Parameters:
    -----------
    df : DataFrame
        Output from identify_thin_files function
    """
    print("Data Sufficiency Analysis")
    print("-" * 50)
    
    # Overall statistics
    print("\nOverall Thin File Rate:")
    print(df['is_thin_file'].value_counts(normalize=True).round(3))
    
    # Breakdown by type
    print("\nInsufficiency Reasons:")
    for col in ['insufficient_transactions', 'insufficient_transaction_history', 'insufficient_balance_records']:
        print(f"\n{col}:")
        print(df[col].value_counts(normalize=True).round(3))
    
    # Default rates comparison
    print("\nDefault Rates Comparison:")
    thin_default_rate = df[df['is_thin_file']]['DQ_TARGET'].mean()
    thick_default_rate = df[~df['is_thin_file']]['DQ_TARGET'].mean()
    print(f"Thin Files Default Rate: {thin_default_rate:.3f}")
    print(f"Thick Files Default Rate: {thick_default_rate:.3f}")
    
    # Distribution of data metrics
    print("\nData Metrics Distribution:")
    metrics = ['total_transactions', 'months_with_transactions', 'balance_records']
    for metric in metrics:
        print(f"\n{metric}:")
        print(df[metric].describe().round(2))

def analyze_exclusion_impact(original_df, filtered_df):
    """
    Analyzes the impact of thin file exclusions on the dataset.
    
    Parameters:
    -----------
    original_df : DataFrame
        Original consumer dataset
    filtered_df : DataFrame
        Dataset after excluding thin files
    """
    print("Exclusion Impact Analysis")
    print("-" * 50)
    
    # Sample size impact
    print("\nSample Size Impact:")
    print(f"Original samples: {len(original_df)}")
    print(f"Filtered samples: {len(filtered_df)}")
    print(f"Reduction: {(1 - len(filtered_df)/len(original_df))*100:.1f}%")
    
    # Default rate impact
    print("\nDefault Rate Impact:")
    original_default = original_df['DQ_TARGET'].mean()
    filtered_default = filtered_df['DQ_TARGET'].mean()
    print(f"Original default rate: {original_default:.3f}")
    print(f"Filtered default rate: {filtered_default:.3f}")
    
    # Population stability analysis
    if 'credit_score' in original_df.columns:
        print("\nCredit Score Distribution:")
        print("Original:")
        print(original_df['credit_score'].describe().round(2))
        print("\nFiltered:")
        print(filtered_df['credit_score'].describe().round(2))
