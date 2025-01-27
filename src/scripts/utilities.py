import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


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
    """

    # Define features (X) and target (y) for training
    X_train = train_df[feature_columns]
    y_train = train_df['DQ_TARGET']

    # Initialize and train the logistic regression model
    model = LogisticRegression(random_state=123, class_weight="balanced")
    model.fit(X_train, y_train)

    return model

def predict_and_analyze_model(model, test_df, feature_columns):
    """
    Predicts and evaluates the model using accuracy, ROC AUC score, and a detailed confusion matrix.

    Parameters:
        model (LogisticRegression): The trained logistic regression model.
        test_df (pd.DataFrame): Testing dataframe containing features and the target column "DQ_TARGET".

    Returns:
        dict: A dictionary containing accuracy, ROC AUC score, and the confusion matrix.
    """

    # Extract features (X_test) and target (y_test)
    X_test = test_df[feature_columns]
    y_test = test_df['DQ_TARGET']

    # Predict probabilities and class labels
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability for the positive class
    y_pred = model.predict(X_test)  # Predicted class labels

    # Calculate accuracy and ROC AUC score
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Create a detailed confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_matrix_df = pd.DataFrame(
        conf_matrix,
        index=['Actual Negative', 'Actual Positive'],
        columns=['Predicted Negative', 'Predicted Positive']
    )

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()

    # Print the evaluation metrics and confusion matrix
    print("Accuracy:", accuracy)
    print("ROC AUC Score:", roc_auc)
    print("Confusion Matrix:\n", conf_matrix_df)
