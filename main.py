# eda_quick.py
import pandas as pd

df = pd.read_csv("creditcard.csv")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nTarget 'is_fraud' counts:")
print(df["is_fraud"].value_counts())
print("\nTarget proportions:")
print(df["is_fraud"].value_counts(normalize=True))
print("\nAmount stats by target (count, mean, median, std):")
print(df.groupby("is_fraud")["amt"].agg(["count","mean","median","std"]))
print("\nMissing values per column:")
print(df.isnull().sum().sort_values(ascending=False).head(20))
print("\nSample rows:")
print(df.head(10).to_string(index=False))
