# Predicting Loan Default Probability Using Transaction Data

Mert Ozer, Brandon Dioneda, Qianjin Zhou

## 1. Introduction

- Traditional credit scoring models exclude individuals without credit history, limiting financial access.
- This project develops a Cash Score, an alternative credit measure using financial behavior.
- It analyzes bank transactions, account activity, and income patterns for better credit assessment.
- With advancements in data infrastructure and open banking, we now have the technology to efficiently leverage financial data, making this the ideal moment to redefine credit assessment.
- This allows us to extend loans to more newcomers, including immigrants and students, while also generating greater profits for our partners.

## 2. Research Question

How can machine learning be applied to develop a "Cash Score" that accurately reflects financial behavior and equal access to credit?

## 3. Data Overview

### Sample of Consumer Data

| prism_consumer_id | evaluation_date | credit_score | DQ_TARGET |
| ----------------- | --------------- | ------------ | --------- |
| 0                 | 2021-09-01      | 726.0        | 0.0       |
| 1                 | 2021-07-01      | 626.0        | 0.0       |
| ...               | ...             | ...          | ...       |

---

### Sample of Account Data

| prism_consumer_id | prism_account_id | account_type | balance_date | balance |
| ----------------- | ---------------- | ------------ | ------------ | ------- |
| 3023              | 0                | SAVINGS      | 2021-08-31   | 90.57   |
| 3023              | 1                | CHECKING     | 2021-08-31   | 225.95  |
| ...               | ...              | ...          | ...          | ...     |

---

### Sample Transaction Data

| prism_consumer_id | amount | credit_or_debit | posted_date | category        |
| ----------------- | ------ | --------------- | ----------- | --------------- |
| 3023              | 0.05   | CREDIT          | 2021-04-16  | MISCELLANEOUS   |
| 10533             | 4.96   | DEBIT           | 2021-03-11  | BILLS_UTILITIES |
| ...               | ...    | ...             | ...         | ...             |

---

- **Consumer Data**: States if a consumer credit defaulted
- **Account Data**: Record of consumers' bank accounts
- **Transaction Data**: Tracks consumers' bank transactions

## 4. Feature Engineering

We created hundreds of features based on attributes in our datasets. Our features fall under 3 types concerned with:

Bank Balance Features: Measure account balance trends, including current balance, balance changes over time, and average account balance.

Income Features:Capture income-related metrics like average transaction amounts over different time frames, net monthly cash flow, and detailed statistics on transaction categories.

Spending Features: Analyze spending patterns through outflow statistics over different time periods, including yearly, monthly, and weekly trends.

While developing these features, we had to ensure our model remained unbiased. In the financial services industry, compliance with the Equal Credit Opportunity Act (ECOA) is essential. This meant removing certain features—not only based on their impact on model performance but also to prevent unintentional bias toward specific demographics.

## 5. Feature Selection

<iframe src="figures/mutual_info_top15.png" width="100%" height="500px" frameBorder=0></iframe>

## 6. Model Evaluation

## 7. Results

<iframe src="figures/comparison_table.png" width="100%" height="500px" frameBorder=0></iframe>

<iframe src="figures/model_confusion_matrix.png" width="100%" height="500px" frameBorder=0></iframe>

## 8. Reason Codes

## 9. Evaluating Our Cash Scores Against Traditional Credit Scores

## 10. Future Work

- Our "Cash Score" provides a more inclusive credit evaluation.
- Real-time transaction data enhances creditworthiness assessment.
- Future Work:

## 11. Acknowledgments & References

- We sincerely thank our mentors and PrismData for providing datasets.
- Literature: AI in credit scoring, fairness in ML-based finance.
