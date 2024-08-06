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

# Set pandas options
pd.set_option('display.width', 300)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Set numpy options
np.set_printoptions(threshold=np.inf)

# Step 1: Data Preprocessing

# Handle missing data using KNN imputer
imputer = KNNImputer(n_neighbors=5)
df_imputed = imputer.fit_transform(df)
df_imputed = pd.DataFrame(df_imputed, columns=df.columns)

# Ensure data type consistency
for col in df.columns:
    if df[col].dtype == 'object':
        df_imputed[col] = df_imputed[col].astype(str)
    else:
        df_imputed[col] = df_imputed[col].astype(df[col].dtype)

# Subset the data to the most recent period for each participant
df_latest = df_imputed.sort_values('PERIOD').groupby('RANDID').last()

# Step 2: Descriptive Analysis
print(df_latest[['SYSBP', 'DIABP', 'AGE', 'BMI', 'DIABETES']].describe())

# Step 3: Correlation Analysis
correlation_sysbp = spearmanr(df_latest['SYSBP'], df_latest['ANYCHD'])
correlation_diabp = spearmanr(df_latest['DIABP'], df_latest['ANYCHD'])
print(f"Correlation between SYSBP and ANYCHD: {correlation_sysbp}")
print(f"Correlation between DIABP and ANYCHD: {correlation_diabp}")

# Step 4: Logistic Regression Analysis
# Define independent and dependent variables
X = df_latest[['SYSBP', 'DIABP', 'AGE', 'BMI', 'DIABETES']]
y = df_latest['ANYCHD']

# Add constant to the independent variables
X = sm.add_constant(X)

# Fit the logistic regression model
model = sm.Logit(y, X)
result = model.fit()
print(result.summary())

# Step 5: Cox Proportional Hazards Model
# Define the CoxPHFitter object
cph = CoxPHFitter()

# Fit the model
cph.fit(df_latest[['TIMECHD', 'SYSBP', 'DIABP', 'AGE', 'BMI', 'DIABETES', 'ANYCHD']], duration_col='TIMECHD', event_col='ANYCHD')
cph.print_summary()

# Step 6: Analysis of Variance (ANOVA)
# Perform ANOVA between 'SYSBP' and 'educ'
anova_sysbp = sm.stats.anova_lm(sm.OLS(df_latest['SYSBP'], sm.add_constant(df_latest['educ'])).fit())
print(anova_sysbp)

# Perform ANOVA between 'DIABP' and 'educ'
anova_diabp = sm.stats.anova_lm(sm.OLS(df_latest['DIABP'], sm.add_constant(df_latest['educ'])).fit())
print(anova_diabp)

# Step 7: Interpretation and Report Writing
# This step involves interpretation of the results and writing the report, which is not covered in this code.