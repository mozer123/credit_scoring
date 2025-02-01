import os
import json
import inspect
import hashlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


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
    print("Shape is: ", result.shape)
    return result

def train_logistic_regression(train_df, feature_columns):
    """
    Trains a LogisticRegression model using the training dataframe.

    Parameters:
        train_df (pd.DataFrame): Training dataframe containing features and the target column "DQ_TARGET".

    Returns:
        LogisticRegression: The trained LogisticRegression model.
        scaler: To be used again before making predictions.
    """

    # Define features (X) and target (y) for training
    X_train = train_df[feature_columns]
    y_train = train_df['DQ_TARGET']

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Initialize and train the logistic regression model
    model = LogisticRegression(random_state=123, class_weight="balanced", max_iter=1000)
    model.fit(X_train_scaled, y_train)

    return model, scaler

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
    
    print(f"- Unchanged features: {len(unchanged_features)} (using cached results)")
    
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
        print("\nNo features need to be executed.")
    
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

def predict_and_analyze_model(model, scaler, train_df, test_df, feature_columns):
    """
    Predicts and evaluates the model using accuracy, ROC AUC score, and a detailed confusion matrix.
    
    Parameters:
        model (LogisticRegression): The trained logistic regression model.
        scaler (StandardScaler): The fitted scaler used during training.
        train_df (pd.DataFrame): Training dataframe containing features and the target column "DQ_TARGET".
        test_df (pd.DataFrame): Testing dataframe containing features and the target column "DQ_TARGET".
        feature_columns (list): List of column names to use as features.
    """
    # Get raw train features, then scale them with the same scaler
    X_train = train_df[feature_columns]
    X_train_scaled = scaler.transform(X_train)
    y_train = train_df['DQ_TARGET']

    # Use the scaled array for predictions
    y_pred_proba_train = model.predict_proba(X_train_scaled)[:, 1]
    y_pred_train = model.predict(X_train_scaled)
    
    # Calculate training set metrics
    train_accuracy = accuracy_score(y_train, y_pred_train)
    train_roc_auc = roc_auc_score(y_train, y_pred_proba_train)

    # Get raw test features, then scale them
    X_test = test_df[feature_columns]
    X_test_scaled = scaler.transform(X_test)
    y_test = test_df['DQ_TARGET']

    # Use scaled test data for predictions
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)

    # Calculate accuracy and ROC AUC for the test set
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Create a confusion matrix for the test set
    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_matrix_df = pd.DataFrame(
        conf_matrix,
        index=['Actual Negative', 'Actual Positive'],
        columns=['Predicted Negative', 'Predicted Positive']
    )

    # Plot the confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()

    # Plot ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Chance')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

    # Print the evaluation metrics for the test set
    print("=== TEST METRICS ===")
    print("Accuracy:", accuracy)
    print("ROC AUC Score:", roc_auc)
    print("Confusion Matrix:\n", conf_matrix_df)

    # Print training metrics
    print("=== TRAINING METRICS ===")
    print("Train Accuracy:", train_accuracy)
    print("Train ROC AUC:", train_roc_auc)
    print()