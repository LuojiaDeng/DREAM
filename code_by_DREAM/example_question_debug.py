# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from scipy.stats import spearmanr, pearsonr
import statsmodels.api as sm
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('./FHS.csv')

# Step 1: Data Preprocessing

# Handle Missing Data
# Use KNN imputer for missing value imputation
imputer = KNNImputer(n_neighbors=5)
df_imputed = imputer.fit_transform(df)
df_imputed = pd.DataFrame(df_imputed, columns=df.columns)

# Data Cleaning
# Ensure that there are no inconsistencies in data types
df_imputed = df_imputed.astype(df.dtypes.to_dict())

# Unique Record per Participant
# Collapse records to a single entry per participant using the most recent record (highest 'PERIOD')
df_imputed = df_imputed.sort_values('PERIOD').groupby('RANDID').last().reset_index()

# Step 2: Descriptive Analysis
# Compile statistics (mean, median, mode, range) for 'SYSBP', 'DIABP', and potentially confounding variables ('AGE', 'BMI', 'DIABETES')
print(df_imputed[['SYSBP', 'DIABP', 'AGE', 'BMI', 'DIABETES']].describe())

# Step 3: Correlation Analysis
# Calculate Pearson correlation coefficients between 'SYSBP'/'DIABP' and outcome variables ('PREVHYP', 'HYPERTEN', 'ANYCHD', 'STROKE', 'CVD')
correlation_variables = ['SYSBP', 'DIABP', 'PREVHYP', 'HYPERTEN', 'ANYCHD', 'STROKE', 'CVD']
correlation_matrix = df_imputed[correlation_variables].corr(method='pearson')
print(correlation_matrix)

# Step 4: Logistic Regression Analysis
# Model Building
# Construct logistic regression models to explore the association of 'SYSBP' and 'DIABP' (independently and adjusted) with binary outcomes such as 'ANYCHD', 'STROKE', and 'CVD'
# Adjust for Confounders
# Include 'AGE', 'BMI', and 'DIABETES' as covariates in the models to control their confounding effect
# Output
# Estimate odds ratios with 95% confidence intervals to determine the strength of associations
outcome_variables = ['ANYCHD', 'STROKE', 'CVD']
for outcome in outcome_variables:
    model = sm.Logit(df_imputed[outcome], df_imputed[['SYSBP', 'DIABP', 'AGE', 'BMI', 'DIABETES']])
    result = model.fit()
    print(result.summary2())  # Use summary2() to avoid encoding issues

# Step 5: Cox Proportional Hazards Model
# For outcomes with corresponding time variables ('TIMECHD', 'TIMESTRK', 'TIMECVD'), use a Cox Proportional Hazards model to assess the impact of 'SYSBP'/'DIABP' on the time to event, adjusting for the same confounders as in step 4
outcome_time_variables = [('ANYCHD', 'TIMECHD'), ('STROKE', 'TIMESTRK'), ('CVD', 'TIMECVD')]
for outcome, time in outcome_time_variables:
    cph = CoxPHFitter()
    df_cox = df_imputed[['SYSBP', 'DIABP', 'AGE', 'BMI', 'DIABETES', outcome, time]].dropna()
    df_cox = df_cox.rename(columns={outcome: 'event', time: 'duration'})
    cph.fit(df_cox, 'duration', event_col='event')
    cph.print_summary()

# Step 6: Analysis of Variance (ANOVA)
# If necessary, use ANOVA to compare mean 'SYSBP'/'DIABP' across differing levels of categorical outcomes (like different categories of 'educ' or 'PERIOD') to see if there are statistically significant differences
anova_variables = ['educ', 'PERIOD']
for var in anova_variables:
    model = sm.OLS(df_imputed['SYSBP'], df_imputed[var])
    result = model.fit()
    print(result.summary2())  # Use summary2() to avoid encoding issues
    model = sm.OLS(df_imputed['DIABP'], df_imputed[var])
    result = model.fit()
    print(result.summary2())  # Use summary2() to avoid encoding issues

# Step 7: Interpretation and Report Writing
# Interpret the outcomes of the analyses, focusing on the size of the effect measures and the statistical significance. Discuss potential causal pathways if the method and data support causal inference.
# Prepare a comprehensive report detailing the methodology, results, limitations due to confounders or missing data, and potential clinical or public health implications of the findings.
# This step is beyond the scope of this code and would be performed outside the Python environment.