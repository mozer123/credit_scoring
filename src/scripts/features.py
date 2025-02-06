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

def outflow_stats(transaction_df):
    """
    Features based on spending habits over time
    """
    outflows = transaction_df[transaction_df.credit_or_debit == 'DEBIT']

    avg_spending = outflows.groupby('prism_consumer_id')['amount'].mean()
    outflows['year'] = outflows['posted_date'].dt.year
    outflows['month'] = outflows['posted_date'].dt.month
    outflows['week'] = outflows['posted_date'].dt.isocalendar().week
    monthly_totals = outflows.groupby(['prism_consumer_id', 'year', 'month'])['amount'].sum().groupby('prism_consumer_id').mean()
    weekly_totals  = outflows.groupby(['prism_consumer_id', 'year', 'week'])['amount'].sum().groupby('prism_consumer_id').mean()
    yearly_totals  = outflows.groupby(['prism_consumer_id', 'year', 'year'])['amount'].sum().groupby('prism_consumer_id').mean()

    result = transaction_df[['prism_consumer_id']].drop_duplicates().reset_index(drop=True)
    result['avg_spending'] = result['prism_consumer_id'].map(avg_spending).fillna(0)
    result['avg_monthly_outflow'] = result['prism_consumer_id'].map(monthly_totals).fillna(0)
    result['avg_weekly_outflow'] = result['prism_consumer_id'].map(weekly_totals).fillna(0)
    result['avg_yearly_outflow'] = result['prism_consumer_id'].map(yearly_totals).fillna(0)

    return result