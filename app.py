import streamlit as st
import joblib
import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

class AQIDataPreparation:
    def __init__(self, lookback=24, forecast_horizon=4):
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.scaler = MinMaxScaler()

    def prepare_for_prediction(self, current_data):
        """Prepare recent data for prediction"""
        # Create a DataFrame with 24 hours of data (repeating current values)
        dates = pd.date_range(end=datetime.now(), periods=self.lookback, freq='H')
        df = pd.DataFrame({
            'From Date': dates,
            'PM2.5': [current_data['pm2_5']] * self.lookback,
            'PM10': [current_data['pm10']] * self.lookback,
            'NO': [current_data['no2']/2] * self.lookback,  # Estimated NO as NO2/2
            'NO2': [current_data['no2']] * self.lookback,
            'SO2': [current_data['so2']] * self.lookback,
            'CO': [current_data['co']] * self.lookback,
            'O3': [current_data['o3']] * self.lookback
        })

        # Add time-based features
        df['From_Date'] = pd.to_datetime(df['From Date'])
        df['hour'] = df['From_Date'].dt.hour
        df['day_of_week'] = df['From_Date'].dt.dayofweek
        df['month'] = df['From_Date'].dt.month

        features = [
            'PM2.5', 'PM10', 'NO', 'NO2', 'SO2', 'CO', 'O3',
            'hour', 'day_of_week', 'month'
        ]

        # Scale the features
        scaled_data = self.scaler.fit_transform(df[features])
        
        # Reshape for LSTM: [samples, timesteps, features]
        return scaled_data.reshape(1, self.lookback, len(features))

def get_air_quality_data(city):
    """Get air quality data from OpenWeatherMap API"""
    API_KEY = "00026a375cd6ca8d46b3e710d17ed47b"
    
    try:
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={API_KEY}"
        geo_response = requests.get(geo_url)
        
        if geo_response.status_code == 200:
            location_data = geo_response.json()
            if location_data:
                lat = location_data[0]['lat']
                lon = location_data[0]['lon']
                
                air_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
                air_response = requests.get(air_url)
                
                if air_response.status_code == 200:
                    air_data = air_response.json()
                    return air_data['list'][0]['components']
                    
        return get_sample_data()
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return get_sample_data()

def get_sample_data():
    """Return sample air quality data"""
    return {
        'pm2_5': 35.5,
        'pm10': 65.2,
        'no2': 45.8,
        'so2': 15.3,
        'co': 8.2,
        'o3': 42.1
    }

def predict_aqi(model, data_prep, components):
    """Make AQI predictions"""
    try:
        # Prepare input data
        X = data_prep.prepare_for_prediction(components)
        
        # Make prediction
        predictions = model.predict(X, verbose=0)
        
        # If model outputs multiple values (4 hours), return them all
        if len(predictions.shape) > 1 and predictions.shape[1] == 4:
            return predictions[0]  # Return all 4 predictions
        else:
            return predictions  # Return single prediction
            
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None

def get_aqi_category(aqi):
    """Categorize AQI value"""
    if aqi <= 50:
        return 'Good'
    elif aqi <= 100:
        return 'Moderate'
    elif aqi <= 150:
        return 'Unhealthy for Sensitive Groups'
    elif aqi <= 200:
        return 'Unhealthy'
    elif aqi <= 300:
        return 'Very Unhealthy'
    else:
        return 'Hazardous'

# Streamlit UI
st.title("Real Time Air Quality Index (AQI) Prediction System")

# City selection
city = st.text_input("Enter city name", "Nagpur")

# Initialize data preparation
data_prep = AQIDataPreparation(lookback=24, forecast_horizon=4)

# Load the model
@st.cache_resource
def load_model():
    try:
        return joblib.load('lstm_model.joblib')
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

if st.button("Get Predictions"):
    if model is None:
        st.error("Model could not be loaded. Please check if the model file exists.")
    else:
        with st.spinner('Fetching current air quality data...'):
            components = get_air_quality_data(city)
            
        if components:
            st.success("Data fetched successfully!")
            
            # Display current parameters
            st.subheader("Current Air Quality Parameters")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("PM2.5", f"{components['pm2_5']:.2f} μg/m³")
                st.metric("NO2", f"{components['no2']:.2f} ppb")
            with col2:
                st.metric("PM10", f"{components['pm10']:.2f} μg/m³")
                st.metric("SO2", f"{components['so2']:.2f} ppb")
            with col3:
                st.metric("CO", f"{components['co']:.2f} ppm")
                st.metric("O3", f"{components['o3']:.2f} ppb")
            
            # Make predictions
            predictions = predict_aqi(model, data_prep, components)
            
            if predictions is not None:
                st.subheader("Predicted AQI for Next 4 Hours")
                
                # Create timestamps
                times = [(datetime.now() + timedelta(hours=i)).strftime("%H:%M") 
                        for i in range(1, 5)]
                
                # Create prediction DataFrame
                pred_df = pd.DataFrame({
                    'Time': times,
                    'Predicted AQI': [round(p, 2) for p in predictions],
                    'Category': [get_aqi_category(p) for p in predictions]
                })
                
                # Display predictions
                st.table(pred_df)
                
                # Display chart
                chart_df = pred_df[['Time', 'Predicted AQI']].copy()
                st.line_chart(chart_df.set_index('Time'))
                
                # Display interpretation
                st.subheader("Interpretation")
                for _, row in pred_df.iterrows():
                    st.write(f"At {row['Time']}: AQI will be {row['Predicted AQI']:.1f} ({row['Category']})")

# Add sidebar information
st.sidebar.markdown("""
### About This App
    
    This application provides real-time air quality predictions using advanced machine learning techniques.
    
     Features:
    - Real-time air quality data
    - 4-hour AQI predictions
    - Multiple pollutant monitoring
    - Trend analysis
    
     AQI Categories:
    🟢 0-50: Good  
    🟡 51-100: Moderate  
    🟠 101-150: Unhealthy for Sensitive Groups  
    🔴 151-200: Unhealthy  
    ⚫ 201-300: Very Unhealthy  
    🟣 >300: Hazardous
    
     Data Sources:
    - OpenWeatherMap API
    - LSTM Deep Learning Model
""")