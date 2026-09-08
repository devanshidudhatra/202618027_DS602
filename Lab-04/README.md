# Medical Insurance — DS602 Lab-4

An end-to-end statistical modeling project based on the **Medical Insurance Costs** dataset. It implements the lab requirements in Python and provides an interactive Streamlit dashboard.

## 🚀 Live Demo

[👉 Open the Streamlit App](https://medical-insurance-lab.streamlit.app/)

## What is included

### Part 1 — Exploratory Data Analysis & Hypothesis Testing
- Mean, median, standard deviation, IQR, skewness, and kurtosis for numerical variables.
- Histograms and bivariate scatter plots.
- Numerical correlation matrix.
- **Hypothesis Test 1:** compare medical charges between two groups. The code checks Shapiro-Wilk normality and Levene's equal-variance assumption, then uses an independent two-sample t-test (or Welch t-test when variances differ) when normality is acceptable; otherwise it uses Mann-Whitney U.
- **Hypothesis Test 2:** both tests are performed, even though the lab says to pick one:
  1. Chi-square test of independence: smoking status vs region.
  2. One-way ANOVA: medical charges across regions.
- All hypothesis tests use α = 0.05.

### Part 2 — Statistical Modeling & Diagnostics
OLS multiple linear regression:

`charges ~ age + bmi + children + C(sex) + C(smoker) + C(region)`

The project reports:
- coefficients
- p-values
- 95% confidence intervals
- R² and adjusted R²
- residuals vs fitted plot
- Q-Q plot
- Jarque-Bera and Omnibus normality tests
- VIF for continuous predictors

### Streamlit dashboard
Three tabs are provided as required by the lab:
1. **Data Exploration** — sidebar filters, summary statistics, interactive distribution/scatter/correlation plots.
2. **Hypothesis Testing Lab** — dynamic group/metric selectors, Test 1, plus both Chi-square and ANOVA for Test 2.
3. **Live Prediction & Diagnostics** — interactive patient inputs, predicted charges with 95% confidence/prediction intervals, coefficient table, residual plots, normality tests, and VIF.

## Dataset

The assignment recommends the Medical Insurance Costs dataset and describes features including age, sex, BMI, children, smoker, region, and medical charges.

This project uses the common `insurance.csv` version. The app first checks `data/insurance.csv`. If it is absent, it downloads the public CSV from:

https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv

**Recommended:** keep a local copy at `data/insurance.csv` in your GitHub repository so the repository is self-contained.

## Project structure

```text
medical-insurance-lab/
├── app.py
├── 202618027_Assignment04.ipynb
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
└── data/
    ├── insurance.csv        
    └── README.md
```

## 1. Run on your computer

### Step 1 — Install Python

Use a supported Python 3 version. A virtual environment is recommended.

### Step 2 — Download/clone the repository

```bash
git clone https://github.com/devanshidudhatra/202618027_DS602/tree/main/Lab-04
cd medical-insurance-lab
```

If you downloaded the repository as a ZIP, extract it and open a terminal in the extracted folder.

### Step 3 — Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — Run the Part 1 + Part 2 analysis

```bash
python 202618027_DS602.ipynb
```

This prints the descriptive statistics, Hypothesis Test 1, **both Hypothesis Test 2 tests**, OLS summary, coefficients, model fit, normality diagnostics, and VIF.

### Step 6 — Run Streamlit

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit, normally:

`http://localhost:8501`

Always run the command from the repository root so the local paths match Streamlit Community Cloud behavior.

## 2. Push the project to GitHub

Create a new GitHub repository, then from the project folder:

```bash
git init
git add .
git commit -m "Add medical insurance statistical modeling lab"
git branch -M main
git remote add origin https://github.com/devanshidudhatra/202618027_DS602/tree/main/Lab-04
git push -u origin main
```

Before pushing, confirm that `requirements.txt` and `app.py` are in the repository root. If possible, also add `data/insurance.csv`.

## 3. Deploy to Streamlit Community Cloud

1. Create/sign in to a Streamlit Community Cloud account using GitHub.
2. Connect/authorize your GitHub account.
3. Push this project to GitHub.
4. Open your Streamlit Community Cloud workspace.
5. Click **Create app**.
6. Select **Yup, I have an app**.
7. Select your GitHub repository, branch (`main`), and entrypoint file (`app.py`).
8. If desired, choose a custom app URL.
9. Deploy the app.
10. Wait for the build to finish and open the generated `streamlit.app` URL.

### Important deployment notes

- Keep `requirements.txt` in the repository root because `app.py` is in the root.
- The Streamlit app uses only Python packages listed in `requirements.txt`.
- If you add a local dataset, commit `data/insurance.csv` to GitHub.
- If you modify `requirements.txt`, push the change; Community Cloud detects dependency changes and rebuilds the environment.
- Test locally with the same Python version you plan to use in the cloud.

## 4. Updating the deployed app

After deployment, GitHub is the source of the app. Make changes locally, then:

```bash
git add .
git commit -m "Update analysis/dashboard"
git push
```

Community Cloud will detect the GitHub update and refresh the deployed app.

## Statistical conclusion template

The exact p-values are printed by `202618027_DS602.ipynb` and shown interactively in the dashboard. At α = 0.05:

- If p-value < 0.05: **Reject H0**.
- If p-value ≥ 0.05: **Fail to reject H0**.

Do not write a conclusion based only on the size of a test statistic; use the p-value and α = 0.05.

## Academic note

This dashboard is an educational statistical modeling application. The medical insurance data are used for statistical analysis and prediction practice; the predicted charge is not a medical or insurance decision recommendation.
