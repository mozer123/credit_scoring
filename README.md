# Cash Score System: Predicting Loan Default Probability Using Bank Transaction Data

Mert Ozer, Brandon Dioneda, Qianjin Zhou

## Introduction

Traditional credit scoring models (e.g. FICO Score) often fail to account for individuals lacking a conventional credit history.

Our project leverages alternative financial data, such as categorized bank transactions and income predictions, to develop a fairer creditworthiness assessment with advanced machine learning (ML) models.

## Research Question

How can ML be applied to develop a "Cash Score" that accurately reflects financial behavior and equal access to credit? Can we predict if someone will credit default or not based on their behavior?

## Data Overview

Bank transaction data (2017-2023) provided  by PrismData. Includes categorized inflow and outflow transactions sufficient to extract features like: income levels, spending habits, balance changes, and other consumer-level financial behaviors.

## Feature Engineering

We created hundreds of features based on attributes in our datasets. Our features fall under 3 types concerned with:

Bank Balance Features: Measure account balance trends, including current balance, balance changes over time, and average account balance.

Income Features:Capture income-related metrics like average transaction amounts over different time frames, net monthly cash flow, and detailed statistics on transaction categories.

Spending Features: Analyze spending patterns through outflow statistics over different time periods, including yearly, monthly, and weekly trends.

While developing these features, we had to ensure our model remained unbiased. In the financial services industry, compliance with the Equal Credit Opportunity Act (ECOA) is essential. This meant removing certain features—not only based on their impact on model performance but also to prevent unintentional bias toward specific demographics.

## Feature Selection


<iframe src="figures/mutual_info_top15.png" width="100%" height="500px" frameBorder=0></iframe>

## Machine Learning Models


## Reason Codes


## Evaluating Our Scores Against the FICO Score


## Results

<iframe src="figures/comparison_table.png" width="100%" height="500px" frameBorder=0></iframe>

<iframe src="figures/model_confusion_matrix.png" width="100%" height="500px" frameBorder=0></iframe>

## Conclusion

- Our "Cash Score" provides a more inclusive credit evaluation.
- Real-time transaction data enhances creditworthiness assessment.
- Future Work:

## Acknowledgments & References

- We sincerely thank our mentors and PrismData for providing datasets.
- Literature: AI in credit scoring, fairness in ML-based finance.
