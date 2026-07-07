import pandas as pd

df1 = pd.read_csv("data/raw/car_listings_final_scraping.csv")
df2 = pd.read_csv("data/raw/car_listings_raw.csv")
df3 = pd.read_csv("data/raw/car_listings_raw_new.csv")

df = pd.concat([df1, df2, df3], ignore_index=True)

print("Rows before removing duplicates:", len(df))

df = df.drop_duplicates()

print("Rows after removing duplicates:", len(df))

df.to_csv("data/raw/car_listings_full.csv", index=False)

print("Merged file saved.") 