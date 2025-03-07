import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import shap

from utilities import get_data, identify_thin_files, execute_selected_features, merge_features, train_model
from features import (
    average_transaction_amount,
    net_monthly_cash_flow,
    average_account_balance,
    time_based_average_transaction_amounts,
    descriptive_stats_by_category,
    outflow_stats,
    outflow_over_time_fix,
    balance_over_time,
    credit_score
)

def main():
    # Load data
    print("Loading data...")
    consumer_df, account_df, transaction_df = get_data()
    
    # Identify the heldout set (consumers with null DQ_TARGET)
    heldout_df = consumer_df[consumer_df['DQ_TARGET'].isna()].copy()
    training_df = consumer_df[~consumer_df['DQ_TARGET'].isna()].copy()
    
    print(f"Heldout set size: {len(heldout_df)}")
    print(f"Training set size: {len(training_df)}")
    
    # Apply thin file identification to heldout set
    print("Identifying thin files...")
    enriched_heldout = identify_thin_files(
        transaction_df[transaction_df['prism_consumer_id'].isin(heldout_df['prism_consumer_id'])],
        account_df[account_df['prism_consumer_id'].isin(heldout_df['prism_consumer_id'])],
        heldout_df,
        min_transactions=3,
        min_transaction_months=1,
        min_balance_records=1
    )
    
    # Split into thick and thin files
    thick_file_ids = enriched_heldout[~enriched_heldout['is_thin_file']]['prism_consumer_id']
    thin_file_ids = enriched_heldout[enriched_heldout['is_thin_file']]['prism_consumer_id']
    
    print(f"Thick files: {len(thick_file_ids)}")
    print(f"Thin files: {len(thin_file_ids)}")
    
    # Get the reason for thin files
    thin_file_reasons = {}
    for idx, row in enriched_heldout[enriched_heldout['is_thin_file']].iterrows():
        consumer_id = row['prism_consumer_id']
        if row['insufficient_transactions']:
            thin_file_reasons[consumer_id] = "insufficient_transactions"
        elif row['insufficient_transaction_history']:
            thin_file_reasons[consumer_id] = "insufficient_transaction_history"
        elif row['insufficient_balance_records']:
            thin_file_reasons[consumer_id] = "insufficient_balance_records"
        else:
            thin_file_reasons[consumer_id] = "unknown_reason"
    
    # Define feature functions
    selected_features = [
        average_transaction_amount,
        net_monthly_cash_flow,
        average_account_balance,
        time_based_average_transaction_amounts,
        descriptive_stats_by_category,
        outflow_stats,
        outflow_over_time_fix,
        balance_over_time,
        credit_score
    ]
    
    # Calculate features for training data
    print("Calculating features for training data...")
    train_feature_dataframes = execute_selected_features(
        selected_features, 
        training_df, 
        account_df[account_df['prism_consumer_id'].isin(training_df['prism_consumer_id'])],
        transaction_df[transaction_df['prism_consumer_id'].isin(training_df['prism_consumer_id'])]
    )
    train_features = merge_features(train_feature_dataframes)
    
    # Merge with consumer data
    train_ml_features = training_df.merge(train_features, how="left", on="prism_consumer_id")
    train_ml_features = train_ml_features.apply(lambda col: col.fillna(0) if col.name != "DQ_TARGET" else col)
    
    # Define feature columns (with and without credit score)
    exclude_columns = ['prism_consumer_id', 'evaluation_date', 'DQ_TARGET']
    all_feature_columns = [col for col in train_ml_features.columns if col not in exclude_columns]
    
    # Check credit score column name
    credit_score_col = 'credit_score' if 'credit_score' in train_ml_features.columns else 'credit_score_x'
    print(f"Credit score column found: {credit_score_col}")
    
    # Create feature sets with and without credit score
    with_cs_features = all_feature_columns
    without_cs_features = [col for col in all_feature_columns if col != credit_score_col]
    
    # Train XGBoost models using the existing train_model function
    print("Training XGBoost models...")
    
    # Model with credit score
    _, _, model_with_cs, scaler_with_cs = train_model(train_ml_features, with_cs_features, "xgboost")
    
    # Model without credit score
    _, _, model_without_cs, scaler_without_cs = train_model(train_ml_features, without_cs_features, "xgboost")
    
    # Calculate features for all consumers in heldout set
    print("Calculating features for heldout set...")
    heldout_feature_dataframes = execute_selected_features(
        selected_features, 
        heldout_df, 
        account_df[account_df['prism_consumer_id'].isin(heldout_df['prism_consumer_id'])],
        transaction_df[transaction_df['prism_consumer_id'].isin(heldout_df['prism_consumer_id'])]
    )
    heldout_features = merge_features(heldout_feature_dataframes)
    
    # Merge with consumer data
    heldout_ml_features = heldout_df.merge(heldout_features, how="left", on="prism_consumer_id")
    heldout_ml_features = heldout_ml_features.apply(lambda col: col.fillna(0) if col.name != "DQ_TARGET" else col)
    
    # Generate predictions for all heldout consumers
    print("Generating predictions for heldout set...")
    X_heldout_with_cs = heldout_ml_features[with_cs_features]
    X_heldout_without_cs = heldout_ml_features[without_cs_features]
    
    X_heldout_scaled_with_cs = scaler_with_cs.transform(X_heldout_with_cs)
    X_heldout_scaled_without_cs = scaler_without_cs.transform(X_heldout_without_cs)
    
    # Get probabilities
    heldout_probs_with_cs = model_with_cs.predict_proba(X_heldout_scaled_with_cs)[:, 1]
    heldout_probs_without_cs = model_without_cs.predict_proba(X_heldout_scaled_without_cs)[:, 1]
    
    # Get SHAP values for reason codes
    print("Calculating reason codes...")
    
    # Create SHAP explainers
    explainer_with_cs = shap.Explainer(model_with_cs)
    explainer_without_cs = shap.Explainer(model_without_cs)
    
    # Calculate SHAP values
    shap_values_with_cs = explainer_with_cs(X_heldout_scaled_with_cs)
    shap_values_without_cs = explainer_without_cs(X_heldout_scaled_without_cs)
    
    # Create DataFrames with SHAP values
    shap_df_with_cs = pd.DataFrame(shap_values_with_cs.values, columns=with_cs_features)
    shap_df_without_cs = pd.DataFrame(shap_values_without_cs.values, columns=without_cs_features)
    
    # Function to get top feature names based on SHAP values
    def get_top_feature_names(shap_row, feature_names, top_n=3):
        top_indices = np.argsort(np.abs(shap_row))[-top_n:][::-1]  
        return [feature_names[i] for i in top_indices]
    
    # Get top features for each consumer
    top_features_with_cs = shap_df_with_cs.apply(
        lambda row: get_top_feature_names(row, with_cs_features), axis=1
    )
    
    top_features_without_cs = shap_df_without_cs.apply(
        lambda row: get_top_feature_names(row, without_cs_features), axis=1
    )
    
    # Create results dataframe
    results = pd.DataFrame({
        'prism_consumer_id': heldout_ml_features['prism_consumer_id'],
        'score_with_cs': heldout_probs_with_cs,
        'score_without_cs': heldout_probs_without_cs
    })
    
    # Add reason codes
    results['reason1_with_cs'] = [features[0] for features in top_features_with_cs]
    results['reason2_with_cs'] = [features[1] for features in top_features_with_cs]
    results['reason3_with_cs'] = [features[2] for features in top_features_with_cs]
    
    results['reason1_without_cs'] = [features[0] for features in top_features_without_cs]
    results['reason2_without_cs'] = [features[1] for features in top_features_without_cs]
    results['reason3_without_cs'] = [features[2] for features in top_features_without_cs]
    
    # For thin files, set scores to NA and update reason codes
    for consumer_id in thin_file_ids:
        reason = thin_file_reasons.get(consumer_id, "unknown_reason")
        mask = results['prism_consumer_id'] == consumer_id
        
        # Set scores to NA
        results.loc[mask, ['score_with_cs', 'score_without_cs']] = np.nan
        
        # Set reason codes to the thin file reason
        for col in ['reason1_with_cs', 'reason2_with_cs', 'reason3_with_cs',
                   'reason1_without_cs', 'reason2_without_cs', 'reason3_without_cs']:
            results.loc[mask, col] = reason
    
    # Add is_thin_file column
    results['is_thin_file'] = results['prism_consumer_id'].isin(thin_file_ids)
    
    # Save results
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'predictions.csv')
    results.to_csv(output_path, index=False)
    
    print(f"Predictions saved to {output_path}")
    print(f"Total predictions: {len(results)}")
    print(f"Thick files: {len(thick_file_ids)}")
    print(f"Thin files: {len(thin_file_ids)}")
    
    # Print summary statistics
    print("\nScore Statistics (Thick Files):")
    thick_file_stats = results[~results['is_thin_file']][['score_with_cs', 'score_without_cs']].describe()
    print(thick_file_stats)
    
    # Print most common reason codes
    print("\nMost Common Reason Codes (With Credit Score):")
    reason_counts = pd.concat([
        results[~results['is_thin_file']]['reason1_with_cs'],
        results[~results['is_thin_file']]['reason2_with_cs'],
        results[~results['is_thin_file']]['reason3_with_cs']
    ]).value_counts().head(10)
    print(reason_counts)
    
    print("\nMost Common Reason Codes (Without Credit Score):")
    reason_counts = pd.concat([
        results[~results['is_thin_file']]['reason1_without_cs'],
        results[~results['is_thin_file']]['reason2_without_cs'],
        results[~results['is_thin_file']]['reason3_without_cs']
    ]).value_counts().head(10)
    print(reason_counts)

if __name__ == "__main__":
    main() 