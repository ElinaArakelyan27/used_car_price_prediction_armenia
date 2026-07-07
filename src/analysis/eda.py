import os
import pandas as pd
import matplotlib.pyplot as plt

CLEAN_PATH = "data/processed/car_listings_clean.csv"
OUTPUT_DIR = "reports/figures"


def load_data(path):
    df = pd.read_csv(path)
    return df


def basic_overview(df):
    print("Shape")
    print(df.shape)

    print("Columns")
    print(df.columns)

    print("First 5 rows")
    print(df.head())

    print("Missing Values")
    print(df.isna().sum())

    print("Summary Statistics")
    print(df.describe())


def brand_analysis(df):
    print("Top 10 popular brands")
    print(df["brand"].value_counts().head(10))

    print("Top 10 most expensive brands by average price")
    print(
        df.groupby("brand")["price_usd"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

def most_common_models(df):
    print("Top 10 popular models")
    model_counts = df["model"].value_counts()
    top_models = model_counts.head(20)
    print(top_models)

    
def most_common_models_percent(df):
    print("Most Common Models (Percantage)")
    model_percent = df["model"].value_counts(normalize=True) *100
    print(model_percent.head(20))

def most_common_models_table(df):
    print("Models table")
    models_table = (
        df["model"]
        .value_counts()
        .reset_index()
    )
    models_table.columns = ["model", "count"]
    print(models_table.head(20))

def fuel_transmission_analysis(df):
    print("Fuel and Transmission crosstab")
    crosstab = pd.crosstab(df["fuel"], df["transmission"])
    crosstab["total"] = crosstab.sum(axis=1)
    print(crosstab)
    return crosstab

def final_insights(df):
    print("FINAL INSIGHTS")
    print("Top 5 brands:")
    print(df["brand"].value_counts().head(5))

    print("Newest cars trend:")
    print(df.groupby("year")["price_usd"].mean().tail(5))

def plot_fuel_transmission(df):
    fig, axes = plt.subplots(1,2, figsize=(12,5))
    df["fuel"].value_counts().plot(kind="bar", ax =axes[0])
    axes[0].set_title("Listings by Fuel Type")
    axes[0].set_xlabel("Fuel")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis ="x", rotation = 45)
    df["transmission"].value_counts().plot(kind="bar", ax=axes[1])
    axes[1].set_title("Listings by Transmission")
    axes[1].set_xlabel("Transmission")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fuel_transmission.png")
    plt.show()
    plt.close()

def plot_price_by_fuel(df):
    fuel_groups = [group["price_usd"].values for _, group in df.groupby("fuel")]
    fuel_labels = df["fuel"].unique()

    plt.figure(figsize=(10, 6))
    plt.boxplot(fuel_groups, labels=fuel_labels, patch_artist=True)
    plt.title("Price Distribution by Fuel Type")
    plt.xlabel("Fuel Type")
    plt.ylabel("Price (USD)")
    plt.yscale("log")  
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/price_by_fuel.png")
    plt.show()
    plt.close()

def plot_year_analysis(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df["year"], bins=30)
    axes[0].set_title("Listings by Year")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Count")

    avg_price_by_year = df.groupby("year")["price_usd"].mean()
    axes[1].plot(avg_price_by_year.index, avg_price_by_year.values, marker="o", markersize=3)
    axes[1].set_title("Average Price by Year")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Avg Price (USD)")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/year_analysis.png")
    plt.show()
    plt.close()

def plot_mileage_vs_year(df):
    plt.figure(figsize=(8, 5))
    plt.scatter(df["year"], df["mileage_km"], alpha=0.3, s=10)
    plt.title("Mileage vs Year")
    plt.xlabel("Year")
    plt.ylabel("Mileage (km)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/mileage_vs_year.png")
    plt.show()
    plt.close()

def plot_color_distribution(df):
    plt.figure(figsize=(12, 5))
    df["color"].value_counts().plot(kind="bar")
    plt.title("Listings by Color")
    plt.xlabel("Color")
    plt.ylabel("Count")
    plt.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/color_distribution.png")
    plt.show()
    plt.close()

def price_outlier_analysis(df):
    p01 = df["price_usd"].quantile(0.01)
    p99 = df["price_usd"].quantile(0.99)
    outliers = df[df["price_usd"] > p99]

    print(f"1st percentile price:  ${p01:,.0f}")
    print(f"99th percentile price: ${p99:,.0f}")
    print(f"Outliers above 99th percentile ({len(outliers)} rows):")
    print(outliers[["brand", "model", "year", "mileage_km", "price_usd"]].sort_values("price_usd", ascending=False))

def plot_price_distribution_log(df):
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df["price_usd"], bins=40)
    axes[0].set_title("Price Distribution (linear)")
    axes[0].set_xlabel("Price (USD)")
    axes[0].set_ylabel("Count")

    axes[1].hist(np.log1p(df["price_usd"]), bins=40)
    axes[1].set_title("Price Distribution (log scale)")
    axes[1].set_xlabel("log(Price + 1)")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/price_distribution_log.png")
    plt.show()
    plt.close()

def plot_correlation_heatmap(df):
    import numpy as np
    numeric_cols = ["year", "price_usd", "mileage_km"]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
    ax.set_yticklabels(numeric_cols)

    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=11)

    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png")
    plt.show()
    plt.close()

def plot_price_distribution(df):
    plt.figure(figsize=(8, 5))
    plt.hist(df["price_usd"], bins=20)
    plt.title("Price Distribution")
    plt.xlabel("Price (USD)")
    plt.ylabel("Number of Cars")
    plt.savefig(f"{OUTPUT_DIR}/price_distribution.png")
    plt.show()
    plt.close()

def plot_mileage_distribution(df):
    plt.figure(figsize=(8, 5))
    plt.hist(df["mileage_km"], bins=20)
    plt.title("Mileage Distribution")
    plt.xlabel("Mileage (km)")
    plt.ylabel("Number of Cars")
    plt.savefig(f"{OUTPUT_DIR}/mileage_distribution.png")
    plt.show()
    plt.close()


def plot_price_vs_year(df):
    plt.figure(figsize=(8, 5))
    plt.scatter(df["year"], df["price_usd"])
    plt.title("Price vs Year")
    plt.xlabel("Year")
    plt.ylabel("Price (USD)")
    plt.savefig(f"{OUTPUT_DIR}/price_vs_year.png")
    plt.show()
    plt.close()

def plot_price_vs_mileage(df):
    plt.figure(figsize=(8, 5))
    plt.scatter(df["mileage_km"], df["price_usd"])
    plt.title("Price vs Mileage")
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price (USD)")
    plt.savefig(f"{OUTPUT_DIR}/price_vs_mileage.png")
    plt.show()
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data(CLEAN_PATH)

    basic_overview(df)
    brand_analysis(df)
    most_common_models(df)
    most_common_models_percent(df)
    most_common_models_table(df)
    fuel_transmission_analysis(df)
    price_outlier_analysis(df)
    final_insights(df)

    plot_fuel_transmission(df)
    plot_price_by_fuel(df)
    plot_year_analysis(df)
    plot_mileage_vs_year(df)
    plot_color_distribution(df)
    plot_correlation_heatmap(df)
    plot_price_distribution_log(df)
    plot_price_distribution(df)
    plot_mileage_distribution(df)
    plot_price_vs_year(df)
    plot_price_vs_mileage(df)
    
   


if __name__ == "__main__":
    main()