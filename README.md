# 🏦 Bank Loan Default Risk Analysis

This project analyzes the likelihood of loan default using datasets from loan applicants and their previous applications. The aim is to understand patterns, clean the data, and prepare it for machine learning models or business insights.

## 📁 Dataset

- `application_data.csv`: Contains information about current loan applicants.
- `previous_application.csv`: Contains information about applicants' previous loan applications.

## 🔧 Technologies Used

- Python
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Missingno

## 📊 Key Steps

### 1. Data Loading
Both datasets are read using `pandas.read_csv`.

### 2. Data Cleaning
- Checked for null values using `isnull()` and visualized with `missingno`.
- Dropped columns with ≥40% missing values.
- Removed redundant or low-value columns such as document flags and contact flags.
- Dropped non-correlated features based on heatmaps.

### 3. Feature Engineering
- Converted negative day values (e.g., `DAYS_BIRTH`, `DAYS_EMPLOYED`) to positive.
- Created categorical bins for income, credit amount, age, and employment years.
- Converted appropriate columns to categorical types.

### 4. Imputation
- Used mode or median to fill missing values in categorical and numerical columns.
- Created a new "Unknown" category where necessary.

### 5. Visualizations
- Plotted distributions for age, income, employment, and document flags.
- Correlation heatmaps were created to identify features strongly related to loan default.

## 📈 Insights

- A large number of applicants had income in the range of 100k–200k.
- Most applicants were over 40 years of age.
- Many applicants had not submitted most of the `FLAG_DOCUMENT_*` fields except for `FLAG_DOCUMENT_3`.

## 📦 Requirements

```bash
pip install pandas numpy matplotlib seaborn missingno
