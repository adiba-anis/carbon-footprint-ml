import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Carbon Footprint Calculator", page_icon="🌱", layout="wide")

st.title("🌱 Personal Carbon Footprint Estimator")
st.write("Predict your monthly carbon footprint using Machine Learning.")

st.divider()

# Load Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'carbon_model.pkl')

@st.cache_resource
def load_carbon_model():
    if not os.path.exists(MODEL_PATH):
        return None, f"Model file not found at: {MODEL_PATH}"
    try:
        return joblib.load(MODEL_PATH), None
    except Exception as e:
        return None, str(e)

model, error_msg = load_carbon_model()

if error_msg:
    st.error("⚠️ Model Loading Error")
    st.info(error_msg)
    st.stop()

# Form Inputs for all 19 columns
st.subheader("Enter Your Lifestyle Habits")

col1, col2, col3 = st.columns(3)

with col1:
    sex = st.selectbox("Sex", ["female", "male"])
    body_type = st.selectbox("Body Type", ["normal", "overweight", "obese", "underweight"])
    diet = st.selectbox("Dietary Pattern", ["omnivore", "vegan", "vegetarian", "pescatarian"])
    how_often_shower = st.selectbox("How Often Shower", ["daily", "less frequently", "more frequently", "twice daily"])
    heating_energy = st.selectbox("Heating Energy Source", ["natural gas", "electricity", "wood", "coal"])
    transport = st.selectbox("Transport Mode", ["public", "private", "walk/bicycle"])
    vehicle_type = st.selectbox("Vehicle Type", ["None", "petrol", "diesel", "hybrid", "electric", "lpg"])

with col2:
    vehicle_km = st.number_input("Vehicle Monthly Distance (km)", min_value=0.0, value=300.0)
    air_travel = st.selectbox("Frequency of Air Travel", ["never", "rarely", "frequently", "very frequently"])
    waste_bag_size = st.selectbox("Waste Bag Size", ["small", "medium", "large", "extra large"])
    waste_bag_count = st.number_input("Waste Bag Weekly Count", min_value=0, value=2)
    recycling = st.selectbox("Recycling Level", ["Low", "Medium", "High"])
    cooking_with = st.selectbox("Cooking With", ["gas", "electricity", "microwave", "stove"])

with col3:
    grocery_bill_inr = st.number_input("Monthly Grocery Bill (₹)", min_value=0.0, value=15000.0, step=500.0)
    clothes_monthly = st.number_input("New Clothes Monthly", min_value=0, value=2)
    tv_pc_hours = st.number_input("Daily TV/PC Hours", min_value=0.0, value=4.0)
    internet_hours = st.number_input("Daily Internet Hours", min_value=0.0, value=5.0)
    energy_eff = st.selectbox("Energy Efficiency", ["No", "Yes", "Sometimes"])
    social_activity = st.selectbox("Social Activity Level", ["never", "often", "sometimes"])

st.divider()

if st.button("Calculate Footprint", type="primary"):
    # Convert INR input to the value expected by your model if trained on USD equivalent (approx 1 USD = 83 INR)
    # If your dataset was originally in USD, we convert INR -> USD here before passing to model:
    grocery_bill_model_val = grocery_bill_inr / 83.0  # adjust conversion if your original dataset was already in INR

    # Match the exact 19 column names expected by your trained model
    input_df = pd.DataFrame([{
        'Sex': sex,
        'Diet': diet,
        'How Often Shower': how_often_shower,
        'Heating Energy Source': heating_energy,
        'Transport': transport,
        'Vehicle Type': vehicle_type,
        'Social Activity': social_activity,
        'Energy efficiency': energy_eff,
        'Recycling': recycling,
        'Cooking_With': cooking_with,
        'Body Type': body_type,
        'Frequency of Traveling by Air': air_travel,
        'Waste Bag Size': waste_bag_size,
        'Waste Bag Weekly Count': waste_bag_count,
        'How Many New Clothes Monthly': clothes_monthly,
        'Monthly Grocery Bill': grocery_bill_model_val,
        'Vehicle Monthly Distance Km': vehicle_km,
        'How Long TV PC Daily Hour': tv_pc_hours,
        'How Long Internet Daily Hour': internet_hours
    }])
    
    try:
        prediction = model.predict(input_df)[0]
        
        st.subheader("Results")
        st.metric(label="Predicted Monthly Footprint", value=f"{prediction:.1f} kg CO₂e")
        
        if prediction > 2000:
            st.warning("⚠️ **High Footprint:** Consider reducing air travel, optimizing vehicle usage, or improving home energy efficiency.")
        elif prediction > 1000:
            st.info("💡 **Moderate Footprint:** You are within typical limits. Incremental habits can help lower it further.")
        else:
            st.success("🎉 **Low Footprint:** Excellent sustainability habits!")

    except Exception as err:
        st.error(f"Prediction Error: {err}")