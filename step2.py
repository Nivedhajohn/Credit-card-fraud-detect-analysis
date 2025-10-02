import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
df = pd.read_csv("creditcard.csv")  
print(df.head())         # Preview first few rows
print(df.columns)        # See column names

# Check for missing values
print(df.isnull().sum())

# Class distribution (adjust column name!)
sns.countplot(x='is_fraud', data=df)   # <-- Change here
plt.title("Fraud vs Non-Fraud Distribution")
plt.show()

# Correlation heatmap (optional: sample for speed)
# Select only numeric columns
numeric_df = df.select_dtypes(include=['number'])

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.show()

