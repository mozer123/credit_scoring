# Predicting Loan Default Probability Using Transaction Data

Mert Ozer, Brandon Dioneda, Qianjin Zhou

## Website

**Live Website:** [Cash Score Project](https://mozer123.github.io/credit_scoring/)

## Project Overview

This project develops a "Cash Score" as an alternative credit assessment measure for individuals with limited or no traditional credit history. By analyzing bank transaction data, account activity, and income patterns, we've created a machine learning model that predicts loan default probability with an ROC-AUC of 0.81. Our approach enables more inclusive financial access while maintaining accurate risk assessment, potentially extending credit opportunities to underserved populations such as immigrants and students.

## Project Content Summary

Our website presents a comprehensive analysis of using transaction data to predict loan default probability:

- **Introduction**: We address the limitations of traditional credit scoring models and propose an alternative approach using financial behavior data.

- **Research Question**: We explore how machine learning can be applied to develop a "Cash Score" that accurately reflects financial behavior while promoting equal access to credit.

- **Data Overview**: The project utilizes bank transaction data, account information, and consumer credit histories provided by PrismData, covering the period from 2017-2023.

- **Feature Engineering**: We developed over 200 features across three main categories:
  - Bank Balance Features (account balance trends, changes over time)
  - Income Features (transaction amounts, cash flow, category statistics)
  - Spending Features (outflow patterns across different time periods)

- **Model Development**: After testing multiple approaches, XGBoost emerged as our best-performing model with an ROC-AUC of 0.81, demonstrating strong predictive power for loan default risk.

- **Results & Findings**: Our Cash Score effectively complements traditional credit scores, particularly for individuals with limited credit history, while maintaining compliance with fair lending regulations.

Visit our website for detailed visualizations, methodology, and complete findings.

## Repository Structure

```
credit_scoring/
├── figures/              # Data visualizations and charts
├── logos/                # Project and partner logos
├── index.html            # Main website content
├── styles.css            # CSS styling for the website
├── script.js             # JavaScript functionality
└── README.md             # Project documentation
```

## Build and Deployment Instructions

To run this website locally:

1. Clone the repository:
   ```
   git clone https://github.com/mozer123/credit_scoring.git
   cd credit_scoring
   git checkout gh-pages
   ```

2. Open the website:
   - Option 1: Simply open the `index.html` file in your web browser
   - Option 2: Use a local server (recommended for full functionality)
     ```
     # Using Python
     python -m http.server
     # Then visit http://localhost:8000 in your browser
     ```

3. For deployment:
   - The website is automatically deployed through GitHub Pages
   - Any changes pushed to the gh-pages branch will be reflected on the live site

## Credits and Acknowledgments

This project was developed as part of the DSC 180AB Capstone sequence at UC San Diego's Halıcıoğlu Data Science Institute.

- **Team Members:** Mert Ozer, Brandon Dioneda, Qianjin Zhou
- **Faculty Advisor:** Brian Duke (PrismData), Kyle Nero (PrismData), Berk Ustun (UCSD)
- **Industry Partner:** PrismData
- **Special Thanks:** We extend our gratitude to our mentors who provided guidance throughout this project, and to PrismData for providing the datasets that made this research possible.

The website design utilizes Bootstrap 5 framework and incorporates interactive visualization elements to effectively communicate our findings.
