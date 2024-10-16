#  A/B Testing with Propensity Score Matching for Causal Inference
![img](abtesting.webp)
Credits: [Hyperion 360](https://hyperion360.com/blog/how-to-set-up-ab-testing-for-your-online-business/)

<a href="https://www.python.org/"> <img alt="Python" src="https://img.shields.io/badge/Python-3B7EAB?logo=python&logoColor=white"> </a> <a href="https://pandas.pydata.org/"> <img alt="Pandas" src="https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white"> </a> <a href="https://numpy.org/"> <img alt="NumPy" src="https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white"> </a> <a href="https://matplotlib.org/"> <img alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-3C3F4D?logo=matplotlib&logoColor=white"> </a> <a href="https://seaborn.pydata.org/"> <img alt="Seaborn" src="https://img.shields.io/badge/Seaborn-FF6F20?logo=seaborn&logoColor=white"> </a> <a href="https://scikit-learn.org/"> <img alt="Scikit-learn" src="https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white"> </a> <a href="https://www.scipy.org/"> <img alt="SciPy" src="https://img.shields.io/badge/SciPy-3b211f.svg?logo=scipy&logoColor=white"> </a> <a href="https://en.wikipedia.org/wiki/Chi-squared_test"> <img alt="Chi-squared Test" src="https://img.shields.io/badge/Chi--squared%20Test-ffcc00.svg?logo=apache&logoColor=black"> </a> <a href="https://en.wikipedia.org/wiki/Propensity_score_matching"> <img alt="Propensity Score Matching" src="https://img.shields.io/badge/Propensity%20Score%20Matching-007acc.svg?logo=chart.js&logoColor=white"> </a> <a href="https://en.wikipedia.org/wiki/A/B_testing"> <img alt="A/B Testing" src="https://img.shields.io/badge/A/B%20Testing-28a745.svg?logo=chart.js&logoColor=white"> </a> <a href="https://en.wikipedia.org/wiki/Marketing"> <img alt="Marketing" src="https://img.shields.io/badge/Marketing-ff69b4.svg?logo=chart.js&logoColor=white"> </a> <a href="https://en.wikipedia.org/wiki/Statistics"> <img alt="Statistics" src="https://img.shields.io/badge/Statistics-ffcc00.svg?logo=apache&logoColor=black"> </a>

## Overview

- In the dynamic landscape of marketing, companies strive to execute effective campaigns amidst a plethora of options. To make informed decisions, they often employ A/B testing—a method that involves presenting different variations of a marketing element (such as a webpage, advertisement, or banner) to distinct groups of individuals simultaneously. This technique allows businesses to identify which version yields the best results and optimizes key performance indicators.

- This project aims to explore the application of propensity score matching to mitigate potential biases within the dataset while gaining practical insights into the execution of A/B tests for marketing strategy development. The dataset utilized for this analysis is sourced from Kaggle ([Marketing A/B Testing Dataset](https://www.kaggle.com/datasets/faviovaz/marketing-ab-testing?datasetId=1660669)).

## Dataset Description

The dataset encompasses several important fields:

- **Index**: Unique identifier for each row
- **user_id**: Distinct identifier for each user
- **test_group**: Indicates whether the user was shown an advertisement ("ad") or a Public Service Announcement ("psa")
- **converted**: Indicates whether the user made a purchase (True) or not (False)
- **total_ads**: Total number of ads viewed by the user
- **most_ads_day**: The day on which the user encountered the highest number of ads
- **most_ads_hour**: The hour during which the user saw the most ads

## Objectives

The primary objectives of this analysis are two-fold:

1. To assess the overall success of the marketing campaign.
2. To quantify the extent to which the success can be attributed to the advertisements.

## Steps Taken

### 1. Data Loading

**Reason**: The first step is to import the necessary libraries and load the dataset to begin the analysis.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('marketing_AB.csv')
```
### 2. Data Cleaning

**Reason:** Data cleaning is essential to ensure that the dataset is in a usable format. This includes removing unnecessary columns and renaming columns for better clarity.

```python
# Drop the Unnamed: 0 column as it is the same as index
df.drop('Unnamed: 0', axis=1, inplace=True)

# Renaming columns for better data handling
df.rename(columns={
    "user id": "user_id",
    "test group": "test_group",
    "converted": "converted",
    "total ads": "total_ads",
    "most ads day": "most_ads_day",
    "most ads hour": "most_ads_hour"
}, inplace=True)
```

### 3. Data Exploration

**Reason:** Exploratory Data Analysis (EDA) helps in understanding the dataset better. This includes checking for missing values, data types, and basic statistics to identify any anomalies or patterns.

```python
# Check the shape and info of the dataset
print(df.shape)
print(df.info())

# Summary statistics
print(df.describe())

# Check for missing values
print(df.isna().sum())

# Visualize the distribution of conversion rates
sns.countplot(x='converted', data=df)
plt.title('Conversion Distribution')
plt.show()
```
#### Conversion Rate Analysis

**Reason:** To evaluate the effectiveness of the marketing campaign, we compare the conversion rates between the treatment group (ad) and the control group (psa).

```python
# Compare the conversion rate between the two test groups
conversion_rates = df.groupby("test_group")["converted"].value_counts(normalize=True)
print(conversion_rates)
```
### 4. Hypothesis Testing

**Reason:** To determine if there is a statistically significant difference in conversion rates between the two groups, we define null and alternative hypotheses and perform a Chi-squared test.

```python
# Define the null and alternative hypotheses
# Null Hypothesis (H₀): There is no difference in the conversion rates between the test groups.
# Alternative Hypothesis (H₁): There is a difference in the conversion rates between the test groups.

# Perform Chi-squared test
from scipy.stats import chi2_contingency

contingency_table = pd.crosstab(df['test_group'], df['converted'])
chi2, p, dof, expected = chi2_contingency(contingency_table)

print(f"Chi-squared statistic: {chi2}, p-value: {p}")
```

### 5. Confounder Analysis

**Reason**: To ensure that the observed differences in conversion rates are not influenced by confounding variables, we perform additional Chi-squared tests on categorical covariates.

```python
# Chi-squared test for 'most ads day'
contingency_table_day = pd.crosstab(df['most_ads_day'], df['test_group'])
chi2_day, p_day, dof_day, expected_day = chi2_contingency(contingency_table_day)
print(f"Chi-squared test for 'most ads day' p-value: {p_day}")

# Chi-squared test for 'most ads hour'
contingency_table_hour = pd.crosstab(df['most_ads_hour'], df['test_group'])
chi2_hour, p_hour, dof_hour, expected_hour = chi2_contingency(contingency_table_hour)
print(f"Chi-squared test for 'most ads hour' p-value: {p_hour}")
```

Further looking into total ads as well 

**Reason:** To assess whether the mean values of numerical data (like total ads) significantly differ between the two groups, we perform both a t-test and a Mann-Whitney U test. The t-test assumes normal distribution, while the Mann-Whitney U test is a non-parametric alternative that does not require this assumption.

```python
from scipy.stats import ttest_ind, mannwhitneyu

# T-test
t_stat, p_t = ttest_ind(df[df['test_group'] == 'ad']['total_ads'],
                         df[df['test_group'] == 'psa']['total_ads'])
print(f"T-test p-value: {p_t}")

# Mann-Whitney U Test
u_stat, p_u = mannwhitneyu(df[df['test_group'] == 'ad']['total_ads'],
                            df[df['test_group'] == 'psa']['total_ads'])
print(f"Mann-Whitney U test p-value: {p_u}")
```
### 6. Propensity Score Matching

**Reason:** Given the significant differences in covariates between the groups, we employ propensity score matching (PSM) to adjust for potential confounding variables. PSM helps create comparable treatment and control groups based on their propensity scores, which are the probabilities of being assigned to the treatment group given their observed covariates.

```python
import statsmodels.api as sm

# Create dummy variables for categorical variables
df_dummy = pd.get_dummies(df, columns=['most_ads_day', 'most_ads_hour'], drop_first=True, dtype=int)

# Define X and y with the updated DataFrame
X = df_dummy.drop(['user_id', 'test_group', 'converted'], axis=1)
y = df_dummy['test_group'].apply(lambda x: 1 if x == 'ad' else 0)

# Add a constant to the model for the intercept
X = sm.add_constant(X)

# Fit the logistic regression model
model = sm.Logit(y, X)
result = model.fit()
propensity_scores = result.predict(X)
```
#### Nearest Neighbors Matching

**Reason:** After estimating the propensity scores, we perform nearest neighbors matching to create a matched dataset that balances the treatment and control groups.

```python
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Define the treatment and control groups
treatment_group = df_dummy[df_dummy['test_group'] == 'ad']
control_group = df_dummy[df_dummy['test_group'] == 'psa']

# Standardize the covariates
scaler = StandardScaler()
treatment_covariates_scaled = scaler.fit_transform(treatment_group.drop(['user_id', 'test_group', 'converted'], axis=1))
control_covariates_scaled = scaler.transform(control_group.drop(['user_id', 'test_group', 'converted'], axis=1))

# Perform Nearest Neighbors Matching
nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(control_covariates_scaled)
distances, indices = nbrs.kneighbors(treatment_covariates_scaled)

# Create a DataFrame of matched pairs
matched_pairs = pd.DataFrame({
    'treatment_index': treatment_group.index,
    'control_index': control_group.index[indices.flatten()]
})

# Remove duplicate control indices
matched_pairs = matched_pairs.drop_duplicates(subset='control_index', keep='first')

# Create the matched treatment and control groups
matched_treatment_group = treatment_group.loc[matched_pairs['treatment_index']]
matched_control_group = control_group.loc[matched_pairs['control_index']]

# Combine the matched treatment and control groups
matched_df = pd.concat([matched_treatment_group, matched_control_group])
print("Number of matched pairs:", len(matched_df))
```
### 7. Final Analysis and Chi-Squared Test After Matching

**Reason**: After performing propensity score matching, we need to assess whether the conversion rates between the matched treatment and control groups are significantly different. This is done by conducting a Chi-squared test on the matched dataset to determine if the observed differences in conversion rates are statistically significant.

```python
# Chi-squared test for conversion after matching
contingency_table_matched = pd.crosstab(matched_df['converted'], matched_df['test_group'])
chi2_matched, p_matched, dof_matched, expected_matched = chi2_contingency(contingency_table_matched)

print(f"Chi-squared test for conversion after matching p-value: {p_matched}")
print(f"Chi-squared statistic: {chi2_matched}")
print(f"Degrees of freedom: {dof_matched}")
print(f"Expected frequencies: {expected_matched}")
```
Based on the Chi-squared test results, there is no statistically significant difference in the conversion rates between the treatment (ad) and control (psa) groups.

The p-value of 0.2489 is greater than the typical significance level of 0.05. This indicates that we fail to reject the null hypothesis.

#### Implications of the Findings
- **Effect of Confounders:**
The initial significant result suggests that confounding variables were indeed influencing the outcome. This highlights the importance of controlling for these variables in observational studies to avoid misleading conclusions.

- **Causal Inference:**
The lack of significance after matching suggests that the ads may not have a direct causal effect on conversion rates when accounting for confounding factors. This is a critical insight for decision-making, as it indicates that the observed effect in the unadjusted analysis was likely due to these confounders rather than the ads themselves.

- **Reevaluation of Marketing Strategy:**
Given the results after matching, the company may need to reevaluate its marketing strategy. If the ads do not significantly increase conversion rates when controlling for confounders, it may be necessary to explore other marketing tactics or improve the ad content and targeting.

### 8. Conclusion
In summary, this project illustrates the necessity of using robust statistical techniques to evaluate marketing strategies effectively. It serves as a reminder that while initial findings may suggest a certain level of effectiveness, a deeper analysis that accounts for confounding variables is essential for making informed business decisions. The insights gained from this analysis can guide marketing companies in optimizing their campaigns and improving conversion rates, ultimately leading to more successful marketing strategies.