[README_salary_predictor.md](https://github.com/user-attachments/files/32622689/README_salary_predictor.md)

# Data Science Salary Predictor & Job Market Segmentation

Predicting data science salaries and uncovering natural job-market segments using the Kaggle "Data Science Job Salaries" dataset.

## Problem Statement

As I go through my own job search, I wanted to understand what actually drives compensation in data science roles — and whether meaningful, natural groupings exist in the job market beyond simple averages. This project combines a supervised prediction model with unsupervised clustering to explore both questions.

## Dataset

- **Source:** [Data Science Job Salaries](https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries) (Kaggle, via ruchi798)
- **Description:** 607 data science job postings (2020–2022) with experience level, employment type, job title, salary (standardized to USD), remote work ratio, company size, and company location.

## Approach

### Exploratory Data Analysis
- Salaries range widely ($2.8K–$600K), with a median around $101K and a long right tail of high earners.
- Salary rises clearly and monotonically with experience level (Entry → Mid → Senior → Executive).
- Contract roles pay more on average than full-time roles, likely offsetting the lack of benefits/security.
- Counterintuitively, hybrid (50% remote) roles had the *lowest* average pay of any remote-ratio group — lower than both fully remote and fully in-office roles.

### Salary Prediction (Supervised Learning)
1. One-hot encoded categorical features. An initial pass one-hot encoded `job_title` with 50+ unique values, which — combined with a small (607-row) dataset — diluted the model's ability to learn (R² of only 0.183). Fixed by grouping rare titles into an "Other" category.
2. That alone didn't meaningfully help (R² 0.167), pointing to a bigger issue: the initial feature set excluded `company_location`. Adding it back nearly doubled performance (R² 0.310), confirming location is a major, previously-missing salary driver.
3. Tuned a Random Forest Regressor via GridSearchCV (`n_estimators`, `max_depth`, `min_samples_split`), improving results further.

**Final results:**

| Metric | Value |
|---|---|
| MAE | $32,548 |
| RMSE | $48,540 |
| R² | 0.385 |

**Top predictive features:** whether a company is US-based (by far the single strongest signal, importance 0.47), followed by Executive-level experience (0.15) and Senior-level experience (0.07).

An R² of 0.385 means the model captures real, meaningful signal but leaves a majority of salary variation unexplained. This is expected: salary is also driven by factors this dataset doesn't capture — specific technical skills, company funding stage/industry, and individual negotiation — none of which are available here. This mirrors a common real-world limitation: structured job-posting metadata alone is a useful but incomplete salary predictor.

### Job Market Segmentation (Unsupervised Learning)
Used K-Means clustering (k=4, chosen via the elbow method) on salary, remote ratio, and experience level to find natural groupings:

| Cluster | Avg Salary | Remote Ratio | Experience | Profile |
|---|---|---|---|---|
| 0 | $244,352 | 90.6% | Senior | Elite remote earners |
| 1 | $72,177 | 85.4% | Entry–Mid | Junior remote workers |
| 2 | $100,194 | 0.4% | Mid | In-office mid-level |
| 3 | $118,327 | 92.6% | Senior | Standard senior remote |

Notably, Clusters 0 and 3 have similar experience and remote-work levels but more than a 2x difference in average salary — suggesting job title or location (not captured directly in the clustering features) further separates "elite" from "standard" senior remote roles.

## Limitations & Future Improvements

- No skills data (e.g., Python, SQL, cloud platforms) — likely the single biggest missing predictor.
- No company-level context (industry, funding stage, size in revenue).
- Relatively small dataset (607 rows) for the number of categories involved.
- Could try clustering on the full one-hot encoded feature set, or add skill-tag data via web scraping, in a future iteration.

## Tech Stack

Python, pandas, scikit-learn, matplotlib

## How to Run

1. Open the notebook in Google Colab.
2. Add your Kaggle API credentials (`kaggle.json`) to download the dataset.
3. Run all cells in order.
