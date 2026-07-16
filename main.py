import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Page Settings
st.set_page_config(
    page_title="AI Weather Forecast",
    page_icon="🌦️",
    layout="centered"
)

# Title
st.title("🌦️ AI Weather Forecast")
st.write("Welcome to the AI Weather Forecast & Data Analysis Project")

# User Input
city = st.text_input("🏙️ Enter City Name")

# Button
if st.button("🔍 Get Weather"):

    api_key = "6b35f0e3a31f64978fda432ed5c99e84"

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    response = requests.get(url)
    data = response.json()

    # Success
    if str(data["cod"]) == "200":

        st.success("✅ Weather Found Successfully!")

        st.subheader("🌤 Current Weather")

        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        wind = data["wind"]["speed"]

        st.write("🌍 City :", data["name"])
        st.write("🌡 Temperature :", temp, "°C")
        st.write("💧 Humidity :", humidity, "%")
        st.write("🌬 Wind Speed :", wind, "m/s")
        st.write("☁ Weather :", data["weather"][0]["description"])

        # Pandas DataFrame
        weather = pd.DataFrame({
            "Temperature": [temp],
            "Humidity": [humidity],
            "Wind Speed": [wind]
        })

        st.subheader("📋 Weather Data")
        st.dataframe(weather)

        # NumPy Analysis
        values = np.array([temp, humidity, wind])

        st.subheader("📊 NumPy Analysis")
        st.write("Average :", np.mean(values))
        st.write("Maximum :", np.max(values))
        st.write("Minimum :", np.min(values))

        # AI/ML
        x = np.array([[1], [2], [3]])
        y = np.array([temp, temp + 1, temp + 2])

        model = LinearRegression()
        model.fit(x, y)

        prediction = model.predict([[4]])

        st.subheader("🤖 AI Temperature Prediction")
        st.write("Predicted Temperature :", round(prediction[0], 2), "°C")

        # Graph
        plt.figure(figsize=(5,3))
        plt.plot(["Temp", "Humidity", "Wind"], [temp, humidity, wind], marker="o")
        plt.title("Weather Graph")
        plt.xlabel("Weather Data")
        plt.ylabel("Values")
        st.pyplot(plt)

    else:
        st.error("❌ " + data["message"])