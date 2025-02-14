# Example of a feature creation function
def average_transaction_amount(accountransaction_df, transaction_df):
    """
    Calculates the average transaction amount per consumer and returns a dataframe
    with the `prism_consumer_id` and the new feature.

    Guidelines for creating feature functions:
    1. Always accept `accountransaction_df` and `transaction_df` in this specific order.
    2. Do not modify the input dataframes.
    3. Return a dataframe with only `prism_consumer_id` and the calculated features.
    4. Ensure the function is self-contained and easy to test.

    Parameters:
        accountransaction_df (pd.DataFrame): Account-level dataframe (not used in this function but included for consistency).
        transaction_df (pd.DataFrame): Transaction-level dataframe used for calculations.

    Returns:
        pd.DataFrame: A dataframe with `prism_consumer_id` and the new feature column `average_transaction_amount`.
    """

    # Step 1: Start with a unique list of consumers from transaction_df
    result = transaction_df[['prism_consumer_id']].drop_duplicates().reset_index(drop=True)

    # Step 2: Perform calculations
    # Calculate total transaction amounts per consumer
    transaction_sums = transaction_df.groupby('prism_consumer_id')['amount'].sum()

    # Calculate the total number of transactions per consumer
    transaction_counts = transaction_df.groupby('prism_consumer_id')['prism_transaction_id'].count()

    # Calculate the average transaction amount
    average_transaction_amount = transaction_sums / transaction_counts

    # Step 3: Map the calculated averages back to the result dataframe
    result['average_transaction_amount'] = result['prism_consumer_id'].map(average_transaction_amount)

    # Step 5: Return the final dataframe
    return result

def net_monthly_cash_flow(accountransaction_df, transaction_df):
    """
    Calculates the net monthly cash flow per consumer.
    """
    # Extract month and year without modifying the original dataframe
    month_year = transaction_df['posted_date'].dt.to_period('M')

    # Compute inflows (CREDIT) and outflows (DEBIT) as a temporary series
    adjusted_amounts = transaction_df['amount'] * transaction_df['credit_or_debit'].apply(
        lambda x: 1 if x == 'CREDIT' else -1
    )

    # Aggregate monthly net cash flow per consumer
    monthly_net = transaction_df.assign(month_year=month_year, adjusted_amount=adjusted_amounts).groupby(
        ['prism_consumer_id', 'month_year']
    )['adjusted_amount'].sum()

    # Summarize over all months for each consumer
    net_cash_flow = monthly_net.groupby('prism_consumer_id').sum().reset_index()
    net_cash_flow.rename(columns={'adjusted_amount': 'net_monthly_cash_flow'}, inplace=True)

    return net_cash_flow

def average_account_balance(accountransaction_df, transaction_df):
    """
    Calculates the average account balance per consumer using accountransaction_df.
    """
    # Aggregate the average balance per consumer
    average_balance = accountransaction_df.groupby('prism_consumer_id')['balance'].mean()

    # Create a result dataframe
    result = accountransaction_df[['prism_consumer_id']].drop_duplicates().reset_index(drop=True)
    result['average_account_balance'] = result['prism_consumer_id'].map(average_balance)

    return result

def time_based_average_transaction_amounts(accountransaction_df, transaction_df):
    """
    Creates features related to average transaction amounts over monthly, weekly, and yearly periods per consumer.
    """
    # Extract date-related components from the transaction data
    transaction_df['year'] = transaction_df['posted_date'].dt.year
    transaction_df['month'] = transaction_df['posted_date'].dt.month
    transaction_df['week'] = transaction_df['posted_date'].dt.isocalendar().week

    # Group transactions by different time periods and aggregate totals per consumer
    monthly_totals = (
        transaction_df.groupby(['prism_consumer_id', 'year', 'month'])['amount']
        .sum()
        .groupby('prism_consumer_id')
        .mean()
    )
    weekly_totals = (
        transaction_df.groupby(['prism_consumer_id', 'year', 'week'])['amount']
        .sum()
        .groupby('prism_consumer_id')
        .mean()
    )
    yearly_totals = (
        transaction_df.groupby(['prism_consumer_id', 'year'])['amount']
        .sum()
        .groupby('prism_consumer_id')
        .mean()
    )

    # Create a result dataframe with unique consumer IDs
    result = transaction_df[['prism_consumer_id']].drop_duplicates().reset_index(drop=True)
    result['avg_monthly_transaction'] = result['prism_consumer_id'].map(monthly_totals)
    result['avg_weekly_transaction'] = result['prism_consumer_id'].map(weekly_totals)
    result['avg_yearly_transaction'] = result['prism_consumer_id'].map(yearly_totals)

    return result

def descriptive_stats_by_category(accountransaction_df, transaction_df):
    """
    Creates descriptive statistics based on amounts spent/going into consumers' accounts
    """
    import pandas as pd

    feats = transaction_df.groupby(['prism_consumer_id', 'category']).agg({'amount': ['count', 'sum', 'std', 'mean', 'median']})
    feats = feats.unstack(level=1)
    feats.columns = ['_'.join(col).strip() for col in feats.columns.values]
    feats = feats.fillna(0)

    result = transaction_df[['prism_consumer_id']].drop_duplicates().reset_index(drop=True)

    mapped_feats = pd.DataFrame({col: result['prism_consumer_id'].map(feats[col]) for col in feats.columns})
    result = pd.concat([result, mapped_feats], axis=1)

    return result

def outflow_stats(accountransaction_df, transaction_df):
    """
    Features based on spending habits over time
    """
    outflows = transaction_df[transaction_df.credit_or_debit == 'DEBIT']

    avg_spending = outflows.groupby('prism_consumer_id')['amount'].mean()
    outflows.loc[:, 'year'] = outflows['posted_date'].dt.year
    outflows.loc[:, 'month'] = outflows['posted_date'].dt.month
    outflows.loc[:, 'week'] = outflows['posted_date'].dt.isocalendar().week
    monthly_totals = outflows.groupby(['prism_consumer_id', 'year', 'month'])['amount'].sum().groupby('prism_consumer_id').mean()
    weekly_totals  = outflows.groupby(['prism_consumer_id', 'year', 'week'])['amount'].sum().groupby('prism_consumer_id').mean()
    yearly_totals  = outflows.groupby(['prism_consumer_id', 'year', 'year'])['amount'].sum().groupby('prism_consumer_id').mean()

    result = transaction_df[['prism_consumer_id']].drop_duplicates().reset_index(drop=True)
    result['avg_spending'] = result['prism_consumer_id'].map(avg_spending).fillna(0)
    result['avg_monthly_outflow'] = result['prism_consumer_id'].map(monthly_totals).fillna(0)
    result['avg_weekly_outflow'] = result['prism_consumer_id'].map(weekly_totals).fillna(0)
    result['avg_yearly_outflow'] = result['prism_consumer_id'].map(yearly_totals).fillna(0)

    return result

def outflow_over_time(accountransaction_df, transaction_df):
    import pandas as pd

    outflows = transaction_df[transaction_df.credit_or_debit == 'DEBIT']
    
    outflows['posted_date'] = pd.to_datetime(outflows['posted_date'])
    spending_over_time = outflows.sort_values(['prism_consumer_id', 'posted_date'])

    initial_dates = outflows.groupby('prism_consumer_id')['posted_date'].min()
    spending_over_time = spending_over_time.merge(initial_dates, on='prism_consumer_id', how='left', suffixes=('', '_initial'))
    spending_over_time = spending_over_time.rename(columns={'posted_date_initial': 'initial_date'})

    spending_over_time['days_between'] = spending_over_time['posted_date'] - spending_over_time['initial_date']

    spending_over_time['months_between'] = (
        (spending_over_time['posted_date'].dt.year - spending_over_time['initial_date'].dt.year) * 12 +
        (spending_over_time['posted_date'].dt.month - spending_over_time['initial_date'].dt.month)
    ).abs()

    for weeks in range(7, 53, 7):    
        spending_over_time[f'first_{weeks}_weeks'] = spending_over_time['days_between'].astype('int64') <= weeks

    for months in range(3, 13, 3):    
        spending_over_time[f'first_{months}_months'] = spending_over_time['months_between'] <= months

    month_aggs = transaction_df[['prism_consumer_id']].drop_duplicates().reset_index(drop=True)

    for months in range(3, 13, 3):  
        months_df = spending_over_time[spending_over_time[f'first_{months}_months']]
        
        agg_df = (
            months_df
            .groupby('prism_consumer_id')
            .agg(amount_sum=('amount', 'sum'), amount_std=('amount', 'std'), amount_mean=('amount', 'mean'))
            .reset_index()
        )

        month_aggs = month_aggs.merge(
            agg_df, on='prism_consumer_id', how='left', suffixes=('', f'_first_{months}_months')
        )

    result = transaction_df[['prism_consumer_id']].drop_duplicates().reset_index(drop=True)
    spending_feats = pd.DataFrame({col: result['prism_consumer_id'].map(month_aggs[col]) for col in month_aggs.columns})
    result = pd.concat([result, spending_feats], axis=1)
    result = result.loc[:,~result.columns.duplicated()].copy()
    return result

def outflow_over_time_fix(accounttransaction_df, transaction_df):
    import pandas as pd

    # Filter and Prepare Transactions
    outflows = transaction_df[transaction_df.credit_or_debit == 'DEBIT'].copy()
    
    # Convert the posted_date column to datetime
    outflows['posted_date'] = pd.to_datetime(outflows['posted_date'])
    
    # Sort transactions by consumer and date
    spending_over_time = outflows.sort_values(['prism_consumer_id', 'posted_date']).copy()
    
    # Compute Time Differences per Consumer
    initial_dates = spending_over_time.groupby('prism_consumer_id')['posted_date'].min().reset_index()
    initial_dates = initial_dates.rename(columns={'posted_date': 'initial_date'})
    
    # Merge the initial_date back into the transactions
    spending_over_time = spending_over_time.merge(initial_dates, on='prism_consumer_id', how='left')
    
    # Compute the difference in time from the initial transaction
    spending_over_time['days_between'] = spending_over_time['posted_date'] - spending_over_time['initial_date']
    spending_over_time['months_between'] = (
        (spending_over_time['posted_date'].dt.year - spending_over_time['initial_date'].dt.year) * 12 +
        (spending_over_time['posted_date'].dt.month - spending_over_time['initial_date'].dt.month)
    ).abs()
    
    # Create Weekly and Monthly Indicator Columns
    for weeks in range(7, 53, 7):    
        spending_over_time[f'first_{weeks}_weeks'] = spending_over_time['days_between'].dt.days <= (weeks * 7)
    
    # Check if a transaction occurred within a given number of months
    for months in range(3, 13, 3):    
        spending_over_time[f'first_{months}_months'] = spending_over_time['months_between'] <= months

    # Aggregate Spending Features over Time Windows
    month_aggs = transaction_df[['prism_consumer_id']].drop_duplicates().reset_index(drop=True)
    
    # For each monthly window (3, 6, 9, 12 months), calculate aggregated spending statistics
    for months in range(3, 13, 3):  
        # Filter transactions within the first 'months' months
        months_df = spending_over_time[spending_over_time[f'first_{months}_months']]
        
        # Aggregate spending statistics per consumer
        agg_df = (
            months_df
            .groupby('prism_consumer_id')
            .agg(
                **{
                    f'amount_sum_first_{months}_months': ('amount', 'sum'),
                    f'amount_std_first_{months}_months': ('amount', 'std'),
                    f'amount_mean_first_{months}_months': ('amount', 'mean')
                }
            )
            .reset_index()
        )
        
        # Merge the new aggregates into the month_aggs dataframe
        month_aggs = month_aggs.merge(agg_df, on='prism_consumer_id', how='left')


    # Create a dataframe of unique consumers for the final result
    result = transaction_df[['prism_consumer_id']].drop_duplicates().reset_index(drop=True)
    
    # Merge the aggregated features directly into result
    result = result.merge(month_aggs, on='prism_consumer_id', how='left')
    
    return result

def balance_over_time(accounttransaction_df, transaction_df):
    """
    Calculate monthly balance information for each consumer and extract multiple features
    based on the monthly balance data.

    The function:
      1. Computes monthly net inflow based on transaction data.
      2. Computes cumulative net inflow per consumer.
      3. Merges the cumulative inflow with account balance data to determine a 'start_balance'
         (the real starting balance) per consumer.
      4. Computes the monthly balance as the sum of cumulative inflow and start_balance.
      5. For each consumer, extracts summary statistics and additional features including:
         - Standard deviation of monthly balance.
         - Mean monthly balance.
         - Median monthly balance.
         - Maximum monthly balance.
         - Minimum monthly balance.
         - Range (max minus min).
         - Count of months with balance information.
         - Coefficient of variation (std/mean).
         - Trend (slope) of monthly balance over time.
         
    Parameters:
        accounttransaction_df (pd.DataFrame): DataFrame with account-level data (including 'balance' and 'balance_date').
        transaction_df (pd.DataFrame): DataFrame with transaction-level data (including 'posted_date', 'amount', and 'credit_or_debit').

    Returns:
        pd.DataFrame: A DataFrame indexed by 'prism_consumer_id' containing the additional balance features.
    """
    import pandas as pd
    import numpy as np

    # Helper function to calculate the real starting balance for each consumer.
    def find_real_start_balance(grp):
        """
        For a given consumer, find the earliest available account balance (the "anchor")
        and compute the starting balance by subtracting inflows that occurred before that anchor.
        """
        # 'year_month_y' comes from account balance; sort by it
        anchor_rows = grp.dropna(subset=["balance"]).sort_values("year_month_y")
        if anchor_rows.empty:
            return pd.Series({"start_balance": 0.0})
        
        anchor_balance = anchor_rows["balance"].iloc[0]
        anchor_month   = anchor_rows["year_month_y"].iloc[0]
        
        # Sum inflows from the transaction side (year_month_x) before or equal to the anchor month.
        inflow_before_anchor = grp.loc[grp["year_month_x"] <= anchor_month, "monthly_net_inflow"].sum()
        
        start_balance = anchor_balance - inflow_before_anchor
        
        return pd.Series({"start_balance": start_balance})

    # --- Process Transaction Data ---
    txn_df = transaction_df.copy()
    # Create a monthly period column based on transaction posted_date.
    txn_df['year_month'] = transaction_df['posted_date'].dt.to_period('M')
    
    # Aggregate transactions: calculate monthly inflow and outflow.
    txn_aggregated = txn_df.groupby(['prism_consumer_id', 'year_month']).agg(
        monthly_inflow=('amount', lambda x: x[txn_df.loc[x.index, 'credit_or_debit'] == 'CREDIT'].sum()),
        monthly_outflow=('amount', lambda x: x[txn_df.loc[x.index, 'credit_or_debit'] == 'DEBIT'].sum())
    ).reset_index()
    
    # Calculate net inflow per month.
    txn_aggregated['monthly_net_inflow'] = txn_aggregated.apply(
        lambda row: row['monthly_inflow'] - row['monthly_outflow'], axis=1)
    txn_aggregated = txn_aggregated[['prism_consumer_id', 'year_month', 'monthly_net_inflow']]
    
    # Calculate cumulative net inflow for each consumer.
    txn_aggregated['cumulative_inflow'] = txn_aggregated.groupby('prism_consumer_id')['monthly_net_inflow'].cumsum()
    
    # --- Process Account Data ---
    acct_df = accounttransaction_df.copy()
    # Create a monthly period column from the balance_date.
    acct_df['year_month'] = pd.to_datetime(acct_df['balance_date']).dt.to_period('M')
    # Aggregate account balances by consumer and month.
    balance_df = acct_df.groupby(['prism_consumer_id', 'year_month']).agg(
        balance=('balance', 'sum')
    ).reset_index()
    
    # --- Merge Transaction and Account Data ---
    # Merge aggregated transaction data with account balance data.
    # Use different suffixes for the 'year_month' columns to differentiate source.
    merged = pd.merge(
        txn_aggregated, 
        balance_df[['prism_consumer_id', 'balance', 'year_month']], 
        on='prism_consumer_id', 
        how='left', 
        suffixes=('_x', '_y')
    )
    
    # Group by consumer and determine the starting balance using the helper function.
    real_starts = merged.groupby("prism_consumer_id", group_keys=False)\
                        .apply(find_real_start_balance)\
                        .reset_index()
    
    # Merge the starting balance back into the aggregated transaction data.
    merged = pd.merge(txn_aggregated, real_starts[['prism_consumer_id','start_balance']], 
                      on='prism_consumer_id', how='left')
    
    # Calculate monthly balance as cumulative inflow plus the real starting balance.
    merged['monthly_balance'] = merged['cumulative_inflow'] + merged['start_balance']
    monthly_balance_df = merged[['prism_consumer_id', 'year_month', 'monthly_balance']]
    
    # --- Extract Additional Features from Monthly Balance ---
    def extract_balance_features(df):
        """
        Compute various summary statistics and trend features from monthly balance data.
        """
        # Aggregate basic statistics per consumer.
        features = df.groupby('prism_consumer_id').agg(
            balance_std=('monthly_balance', 'std'),
            balance_mean=('monthly_balance', 'mean'),
            balance_median=('monthly_balance', 'median'),
            balance_max=('monthly_balance', 'max'),
            balance_min=('monthly_balance', 'min'),
            balance_count=('monthly_balance', 'count')
        ).reset_index()
        # Calculate balance range and coefficient of variation.
        features['balance_range'] = features['balance_max'] - features['balance_min']
        features['balance_cv'] = features['balance_std'] / features['balance_mean']
        
        # Compute the trend (slope) of monthly balance over time.
        def compute_slope(group):
            # Sort by time (year_month). For simplicity, use the order index as a proxy for time.
            group_sorted = group.sort_values('year_month')
            x = np.arange(len(group_sorted))
            y = group_sorted['monthly_balance'].values
            if len(x) > 1:
                slope, _ = np.polyfit(x, y, 1)
            else:
                slope = 0.0
            return slope
        
        trend = df.groupby('prism_consumer_id').apply(compute_slope).reset_index().rename(columns={0: 'balance_trend'})
        features = features.merge(trend, on='prism_consumer_id', how='left')
        
        return features

    # Get all additional balance features.
    balance_features_df = extract_balance_features(monthly_balance_df)
    
    return balance_features_df