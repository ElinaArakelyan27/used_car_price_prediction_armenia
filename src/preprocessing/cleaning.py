import pandas as pd
import os 

RAW_PATH = "data/raw/car_listings_full.csv"
CLEAN_PATH = "data/processed/car_listings_clean.csv"


df = pd.read_csv(RAW_PATH)

print(" Original Shape")
print(df.shape)
print("Missing values before cleaning")
print(df.isna().sum())

df = df.drop_duplicates(subset="car_id")
df = df[df["price_type"] != "contractual"]
df = df.dropna(subset = ["price_usd"])
df = df.dropna(subset=["transmission"])
df = df.drop(columns=["price_type", "drive", "engine_l", "body_type"])
df = df[(df["year"] >= 1980) & (df["year"] <= 2026)]
df = df[(df["price_usd"] >= 1000) & (df["price_usd"] <= 500000)]
df = df[(df["mileage_km"] >= 0) & (df["mileage_km"] <= 1000000)]
df["fuel"] = df["fuel"].fillna("Unknown")
df["condition"] = df["color"].str.extract(r'Վիճակը\s+(\S+)').fillna("Unknown")
df["engine_l"] = df["color"].str.extract(r'Շարժիչի ծավալը\s+(\S+)').fillna("Unknown")
df["color"] = (df["color"].fillna("Unknown").astype(str).str.split().str[0])
print("Clean shape")
print(df.shape)
print("Missing values after cleaning")
print(df.isna().sum())
df.to_csv(CLEAN_PATH, index=False)
print(df["condition"].value_counts().head())
print(df["color"].value_counts().head(20))
print(df.dtypes)
os.makedirs("data/processed", exist_ok=True)
print("Clean CSV saved successfully!")