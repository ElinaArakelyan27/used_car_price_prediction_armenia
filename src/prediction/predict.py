import os
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field
from typing import Optional


load_dotenv()

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/car_price_pipeline.joblib"))
print(f"Loading ML model from: {MODEL_PATH}...")

loaded_data = joblib.load(MODEL_PATH)

if isinstance(loaded_data, dict):
    model = loaded_data.get('model') or loaded_data.get('pipeline') or list(loaded_data.values())[0]
else:
    model = loaded_data

class CarDetails(BaseModel):
    year: Optional[int] = Field(description="The manufacturing year of the car")
    brand: Optional[str] = Field(description="The brand or make of the car, e.g., 'Toyota', 'Mercedes-Benz'")
    model: Optional[str] = Field(description="The specific model name, e.g., 'Camry', 'X5'")
    mileage_km: Optional[int] = Field(description="The mileage converted completely to kilometers as an integer number")
    fuel: Optional[str] = Field(description="Translate fuel type to Armenian. Use 'Բենզին' for Petrol/Gasoline, 'Գազ' for Gas, 'Դիզελ' for Diesel, 'Հիբրիդ' for Hybrid, or 'Unknown'")
    transmission: Optional[str] = Field(description="Translate transmission to Armenian. Use 'Ավտոմատ' for Automatic, 'Մեխանիկական' for Manual, or 'Վարիատոր' for CVT")
    color: Optional[str] = Field(description="Translate color to Armenian. E.g., 'Սպիտակ' for White, 'Սև' for Black")
    condition: Optional[str] = Field(description="Translate condition to Armenian. E.g., 'Գերազանց' for Excellent, 'Լավ' for Good")
    # Changed to float to match the StandardScaler configuration in your pipeline
    engine_l: Optional[float] = Field(description="The engine size in liters as a float number, e.g., 2.0, 2.5, 3.0, 4.4. Leave as None if unknown.")

client = genai.Client()

print("Car Price Predictor")

CURRENT_YEAR = 2026

while True:
    user_prompt = input("Enter your car details or type 'exit': ")
    if user_prompt.lower() == 'exit':
        break
        
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=f"Translate this car query into structured features: {user_prompt}",
            config={
                'response_mime_type': 'application/json',
                'response_schema': CarDetails,
            },
        )
        
        car_dict = response.parsed.model_dump()
        
        fallback_defaults = {
            'year': 2020,
            'brand': 'Toyota',       
            'model': 'Camry',
            'mileage_km': 80000,      
            'fuel': 'Բենզին',         
            'transmission': 'Ավտոմատ', 
            'color': 'Սպիտակ',        
            'condition': 'Լավ',       
            'engine_l': 2.0  # Must be a purely numerical float value!       
        }
    
        for key, default_val in fallback_defaults.items():
            if car_dict.get(key) is None:
                car_dict[key] = default_val
        
        car_age = max(CURRENT_YEAR - car_dict['year'], 1)  
        km_per_year = car_dict['mileage_km'] / car_age
        
        car_dict['car_age'] = car_age
        car_dict['km_per_year'] = km_per_year
            

        ordered_features = [
            'car_age', 'mileage_km', 'km_per_year', 'engine_l', 
            'brand', 'model', 'fuel', 'transmission', 'condition', 'color'
        ]
        

        input_df = pd.DataFrame([car_dict])[ordered_features]
        
       
        predicted_log_value = model.predict(input_df)[0] 
        real_predicted_price = np.expm1(predicted_log_value)
        
        print(f"[1. Gemini Translated Data]:\n{car_dict}")
        print(f"[2. ML Model Prediction]: Estimated Price is ${real_predicted_price} USD")
        
    except Exception as e:
        print(f"An error occurred: {e}")