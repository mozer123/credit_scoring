# Predicting Loan Default Probability Using Transaction Data

A huge thank you 🙌 to the capstone teaching team at the Halıcıoğlu Data Science Institute at UC San Diego 🎓 and our mentors at Prism Data for their invaluable guidance and support. Your expertise and mentorship have been instrumental in bringing this project to life! 🌟

## Overview

This project focuses on developing a credit scoring model 💳 by leveraging transaction data and applying natural language processing (NLP) techniques 🧠. The primary objective is to enhance credit risk assessment ⚖️ by analyzing transaction details and extracting insights from unstructured text data found in transaction memos.

The project involves data preprocessing ⚙️, feature engineering 🔧, and model development 📈 to predict loan default probabilities. Transaction data is used to create meaningful features that improve predictive performance. For reproducibility, a subset of the data is provided.

For NLP-based transaction categorization, refer to the `category_classification` branch. This branch (`default_prediction`) focuses on feature engineering and building models to predict loan default probabilities.

### Data Privacy

The dataset used in this project is derived from Prism Data, which contains sensitive and proprietary information about financial transactions. Due to confidentiality agreements and privacy regulations, the full dataset cannot be shared publicly. Prism Data includes detailed financial information that, if exposed, could compromise the privacy of individuals and the intellectual property of the data provider.

To allow for reproducibility of the methods and analysis presented in this project while respecting these confidentiality constraints, we are providing a small, representative subset of 5000 rows. This subset captures the essential characteristics of the full dataset but does not reveal any sensitive or proprietary details. By working with this sample, other researchers can replicate the data preprocessing, feature engineering, and modeling steps without requiring access to the complete, confidential dataset.

### File Structure

After cloning this repository, your project directory should look like this:

```
📂 src/
┣ 📂 scripts/
┃ ┣ 📜 features.py - Contains functions for feature engineering and feature creation.
┃ ┣ 📜 utilities.py - Utility functions used across the project.
┣ 📜 main.ipynb - The main Jupyter Notebook to execute the project workflow.
┣ 📂 data/
┃ ┣ 📂 temporary_data/ - Stores temporary or intermediate data during processing.
┃ ┣ 📂 raw_data/ - Stores original, unprocessed data files.
┣ 📜 requirements.txt
┣ 📜 README.md
```

### Reproducing Results

To ensure a smooth reproduction of results, follow these steps on a Windows machine. Before proceeding, make sure you have Python installed: [Download Python](https://www.python.org/downloads/).

1. **Clone this Repository**  
   Open a terminal or command prompt and run:

   ```sh
   git clone https://github.com/mozer123/credit_scoring.git
   cd credit_scoring
   ```

2. **Create a Virtual Environment**  
   Run the following command in the project directory:

   ```sh
   python -m venv env
   ```

3. **Activate the Virtual Environment**

   ```sh
   env\Scripts\activate
   ```

4. **Install Dependencies**  
   Install the required libraries using:

   ```sh
   pip install -r requirements.txt
   ```

5. **Launch Jupyter Notebook**  
   Start a Jupyter Notebook by running:

   ```sh
   jupyter lab
   ```

6. **Run the Notebook**
   - A browser window will open.
   - Navigate to `main.ipynb` and open it.
   - Run the cells in order to reproduce the results.

Following these steps ensures that all necessary dependencies are installed, and the environment is correctly set up for executing the notebook. 🚀
