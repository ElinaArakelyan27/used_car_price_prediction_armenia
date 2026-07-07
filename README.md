# Used Car Price Prediction in Armenia

An end-to-end machine learning project that predicts the market price of used cars in Armenia. Instead of relying on an existing dataset, I built a complete data pipeline by scraping real vehicle listings from **auto.am**, cleaning and analyzing the data, training multiple regression models, and deploying the best-performing model for real-time price prediction.

The project also integrates **Google Gemini 2.5 Flash**, allowing users to describe a vehicle in natural language. Gemini automatically extracts structured vehicle information, which is then processed by the trained machine learning model to estimate the vehicle's market price.


# Project Highlights

- Built a custom web scraper for **auto.am**
- Collected approximately **12,000** used car listings
- Cleaned and preprocessed raw vehicle data
- Performed exploratory data analysis (EDA)
- Engineered new features to improve prediction accuracy
- Compared multiple regression algorithms
- Optimized models using **RandomizedSearchCV**
- Achieved **84.7% R²** using **CatBoost Regressor**
- Integrated **Google Gemini 2.5 Flash** for natural language vehicle descriptions
- Built an end-to-end prediction pipeline


# Project Structure

```text
used_car_price_prediction_armenia/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── reports/
│   └── figures/
│
├── src/
│   ├── analysis/
│   ├── modeling/
│   ├── prediction/
│   ├── preprocessing/
│   ├── scraping/
│   └── main.py
│
├── requirements.txt
├── README.md
└── .env
```


# Data Collection

To build the dataset, I developed a custom web scraper for **auto.am**, one of Armenia's largest online vehicle marketplaces.

One challenge during data collection was the website's inconsistent HTML structure, which made traditional page-by-page scraping unreliable.

After analyzing the website, I identified the underlying vehicle ID pattern and designed a scraper that accessed listings directly by their IDs. This approach significantly improved scraping speed, reliability, and coverage.

To further accelerate data collection, I divided the scraper into **three independent Python scripts**, each responsible for scraping a different range of vehicle IDs. After scraping, the datasets were merged into a single dataset containing approximately **12,000 vehicle listings**.

### Libraries Used

- Requests
- BeautifulSoup
- Pandas
- Regular Expressions
- os
- time


# Data Cleaning & Preprocessing

After collecting the raw listings, extensive preprocessing was performed before model training.

The preprocessing pipeline included:

- Removing duplicate vehicle listings
- Handling missing values
- Converting columns to appropriate data types
- Standardizing inconsistent brand names
- Extracting numerical values from text fields
- Removing unnecessary columns
- Filtering unrealistic prices and mileage values
- Grouping infrequent vehicle brands into an **Other** category
- Preparing categorical variables for machine learning

Most preprocessing tasks were completed using **Pandas**.


# Exploratory Data Analysis (EDA)

Exploratory data analysis was conducted to better understand the Armenian used car market and identify relationships between vehicle characteristics and price.

The analysis included visualizations such as:

- Vehicle price distribution
- Mileage distribution
- Manufacturing year distribution
- Price vs. Mileage
- Price vs. Manufacturing Year
- Fuel type distribution
- Transmission distribution
- Vehicle color distribution
- Correlation heatmap
- Feature importance
- Model comparison

## Key Insights

- Vehicle price generally increases with newer manufacturing years.
- Higher mileage is associated with lower market value.
- Engine size has a measurable effect on price.
- Fuel type and transmission influence resale value.
- Luxury brands consistently have higher average prices than economy brands.


# Feature Engineering

Several additional features were created to improve model performance.

These include:

- **Car Age**
- **Kilometers Driven Per Year**
- **Log-Transformed Target Variable**

Grouping rare brands into an **Other** category also helped reduce noise during training.


#  Machine Learning Models

The following regression algorithms were trained and evaluated:

- Linear Regression
- Ridge Regression
- Random Forest Regressor
- XGBoost Regressor
- CatBoost Regressor

All models were implemented using **Scikit-learn Pipelines**, ensuring identical preprocessing during both training and inference.


# Hyperparameter Tuning

To improve model performance, hyperparameter optimization was performed using **RandomizedSearchCV**.

The following models were tuned:

- Ridge Regression
- Random Forest
- XGBoost
- CatBoost

Each search used **5-fold cross-validation** to reduce overfitting and improve generalization.


# Model Performance

| Model | Test R² | Cross Validation R² |
| **CatBoost** | **0.847** | **0.836** |
| **XGBoost** | **0.843** | **0.834** |
| Linear Regression | 0.833 | — |
| Random Forest | 0.830 | 0.801 |
| Ridge Regression | 0.815 | 0.821 |

Among all evaluated models, **CatBoost Regressor** achieved the highest predictive performance and was selected as the final model.


# Model Deployment

The final prediction system combines **Google Gemini 2.5 Flash** with the trained **CatBoost** machine learning pipeline.

The prediction workflow is:

```
Natural Language Input
        │
        ▼
Google Gemini 2.5 Flash
        │
        ▼
Structured Vehicle Features
        │
        ▼
Feature Engineering
(car_age, km_per_year)
        │
        ▼
CatBoost Prediction Pipeline
        │
        ▼
Estimated Vehicle Price (USD)
```

The trained pipeline was saved using **Joblib**, allowing predictions without repeating preprocessing manually.


# Natural Language Prediction

Unlike traditional prediction systems that require manually entering every feature, this project accepts natural language descriptions.

Example:

```text
White 2020 Toyota Camry automatic with 80,000 km
```

or

```text
Black Mercedes-Benz C-Class 2019 gasoline automatic
```

Google Gemini extracts structured vehicle information, including:

- Brand
- Model
- Manufacturing Year
- Mileage
- Fuel Type
- Transmission
- Color
- Vehicle Condition
- Engine Size

Missing values are automatically filled with predefined defaults.

Additional engineered features are then generated before the trained CatBoost pipeline estimates the vehicle's market price.


# Demo

```text
$ python src/prediction/prediction.py

Car Price Predictor

Enter your car details:

2020 Toyota Camry white automatic with 80,000 km

[1. Gemini Translated Data]

{
  "year": 2020,
  "brand": "Toyota",
  "model": "Camry",
  "mileage_km": 80000,
  "fuel": "Բենզին",
  "transmission": "Ավտոմատ",
  "color": "Սպիտակ",
  "condition": "Լավ",
  "engine_l": 2.5
}

[2. ML Model Prediction]

Estimated Price is $22,845.37 USD
```


# Technologies Used

### Data Collection

- Python
- Requests
- BeautifulSoup

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib

### Machine Learning

- Scikit-learn
- XGBoost
- CatBoost
- Joblib

### Natural Language Processing

- Google Gemini 2.5 Flash
- Pydantic
- python-dotenv


# Environment Variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```


# Running the Project

## Clone the repository

```bash
git clone https://github.com/ElinaArakelyan27/used_car_price_prediction_armenia.git

cd used_car_price_prediction_armenia
```

## (Optional) Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run the prediction application

```bash
python src/prediction/predict.py
```

Describe a vehicle in natural language and the application will estimate its market price.


# Future Improvements

- Expand the dataset with additional vehicle listings.
- Develop a web application for public use.
- Automate periodic data collection.
- Incorporate additional vehicle specifications such as trim level and optional equipment.
- Explore advanced hyperparameter optimization techniques such as Optuna.


# Conclusion

This project demonstrates a complete end-to-end machine learning workflow, from data collection to deployment. I designed and implemented a custom web scraper, collected and cleaned approximately **12,000 used car listings**, performed exploratory data analysis, engineered meaningful features, trained and optimized multiple regression models, and deployed the best-performing model as a reusable prediction pipeline.

By integrating **Google Gemini 2.5 Flash** with the trained **CatBoost** model, the application enables users to estimate used car prices using simple natural language descriptions, making the prediction process both intuitive and accessible.

