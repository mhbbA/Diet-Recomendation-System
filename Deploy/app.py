import pandas as pd
import streamlit as st
import numpy as np
import joblib  # For loading the model
import openai
import requests
import joblib
from streamlit_chat import message  # For chatbot UI
from dotenv import load_dotenv
import os
import time

load_dotenv()

AZURE_OPENAI_ENDPOINT=st.secrets["AZURE_OPENAI_ENDPOINT"]   
API_KEY= st.secrets["MY_API_KEY"]


# Load the trained model and scaler
Model = joblib.load("pkl/model.pkl")  # Ensure this file exists in your directory
Scaler = joblib.load("pkl/scaler.pkl")  # Ensure this file exists in your directory

# Load your cleaned DataFrame (assumed to contain recipe data)
@st.cache_data
def load_data():
    # Load individual DataFrames
    df1 = pd.read_csv('datasets/split_1.csv')
    df2 = pd.read_csv('datasets/split_2.csv')
    df3 = pd.read_csv('datasets/split_3.csv')
    df4 = pd.read_csv('datasets/split_4.csv')
    df5 = pd.read_csv('datasets/split_5.csv')
    
    # Concatenate all DataFrames into one
    df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
    return df
df = load_data()

@st.cache_data
def load_recommendations():
    return joblib.load("pkl/recommendations_compressed.pkl")
recommendations_dict = load_recommendations()
# Mapping for activity intensity and gender (for encoding)
intensity_multipliers = {
        'sedentary': 1.2,
        'lightly_active': 1.375,
        'moderately_active': 1.55,
        'very_active': 1.725,
        'extra_active': 1.9
    }

    # Define goal adjustments based on objective
objective_adjustments = {
        'weight_loss': 0.8,
        'muscle_gain': 1.2,
        'health_maintenance': 1
    }

category_map = {
    "Male": 1,
    "Female": 0
}
# BMR and caloric intake calculations
def compute_bmr(gender, body_weight, body_height, age):
 
    if gender == 'male':
        # For Men: BMR = (10 x weight in kg) + (6.25 x height in cm) - (5 x age in years) + 5
        bmr_value = 10 * body_weight + 6.25 * body_height - 5 * age + 5
    elif gender == 'female':
        # For Women: BMR = (10 x weight in kg) + (6.25 x height in cm) - (5 x age in years) - 161
        bmr_value = 10 * body_weight + 6.25 * body_height - 5 * age - 161
    else:
        raise ValueError("Invalid gender. Please choose 'male' or 'female'.")
    return bmr_value
def compute_daily_caloric_intake(bmr, activity_intensity, objective):
   
    # Define activity multipliers based on intensity
    intensity_multipliers = {
        'sedentary': 1.2,
        'lightly_active': 1.375,
        'moderately_active': 1.55,
        'very_active': 1.725,
        'extra_active': 1.9
    }

    # Define goal adjustments based on objective
    objective_adjustments = {
        'weight_loss': 0.8,
        'muscle_gain': 1.2,
        'health_maintenance': 1
    }

    # Calculate maintenance calories based on activity intensity
    maintenance_calories = bmr * intensity_multipliers[activity_intensity]

    # Adjust maintenance calories based on personal objective
    total_caloric_intake = maintenance_calories * objective_adjustments[objective]

    return round(total_caloric_intake)

# The main recommendation function
def suggest_recipes(gender, body_weight, body_height, age, activity_intensity, objective):
   
    # Calculate the Basal Metabolic Rate (BMR) for the user
    bmr = compute_bmr(gender, body_weight, body_height, age)

    # Calculate the total daily caloric intake based on activity intensity and dietary objective
    total_calories = compute_daily_caloric_intake(bmr, activity_intensity, objective)

    # Prepare input data for the model with desired total calories
    user_input_features = np.array([[total_calories, 0,0, 0, 0, 0, 0, 0, 0]])
    

    # Scale the input data to match the model's training scale
    scaled_input_features = Scaler.transform(user_input_features)

    # Predict latent features for the input data
    predicted_latent_features = Model.predict(scaled_input_features)

    # Find the index with the highest prediction probability
    top_prediction_index = np.argmax(predicted_latent_features.flatten())

    # Retrieve recommended recipes based on the highest prediction
    similar_recipe_indices = np.array(recommendations_dict[top_prediction_index])
    recommended_recipes = df.iloc[similar_recipe_indices[:, 1].astype(int)][['Name', 'Calories']]

    return recommended_recipes.head(5)  # Return the top 5 recommended recipes

def get_response_from_openai(prompt):
    headers = {
        'Content-Type': 'application/json',
        'api-key': API_KEY
    }
    data = {
        'messages': [
            {"role": "system", "content": "Act as a senior Nutritionist and you suggest diet system and take in your mind its medical effect.Your role is to provide personalized food recommendations, suggest healthy diets, and address disease-specific food queries. You should also help users manage allergies and give recommendations based on their health conditions.Feel free to request any additional information if needed."},
            {"role": 'user', 'content': prompt}
        ],
        'max_tokens': 200,
        'temperature': 0.7
    }

    response = requests.post(AZURE_OPENAI_ENDPOINT, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()['choices'][0]["message"]["content"]
    else:
        return 'Error: unable to connect to Azure service for chatting.'
   
def delayed_slider(label, min_val, max_val, value, key, delay=5):
    # Display the slider and update session state
    slider_val = st.slider(label, min_val, max_val, value, key=key)
    
    # Wait for the specified delay if the slider value has changed
    if st.session_state.get(f"{key}_prev") != slider_val:
        time.sleep(delay)
    
    # Update the previous value in session state
    st.session_state[f"{key}_prev"] = slider_val
    
    return slider_val

# Streamlit UI Code
st.title("Diet Recommender Web App")
st.image("Images/logo.png",width=300)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Diet Page", "Disease Page", "Chatbot"])

# Home Page
if page == "Home":
    st.header("Welcome to the Diet Recommender!")

# Diet Page
elif page == "Diet Page":
    st.header("Diet Page")
    st.image("Images/diet.png", width=300)

    gender = st.radio("Gender", ["Male", "Female"], key="gender_input").lower()
    weight = st.number_input("Weight (kg)", min_value=1.0, key="weight_input")
    height = st.number_input("Height (cm)", min_value=1.0, key="height_input")
    age = st.number_input("Age", min_value=1 , key="age_input")
    
    # Normalizing intensity to match the dictionary keys
    intensity = st.selectbox("Activity Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extra Active"], key="intensity_input").lower().replace(' ', '_')
    
    objective = st.selectbox("Objective", ["Weight Loss", "Muscle Gain", "Health Maintenance"], key="objective_input").lower().replace(' ', '_')

    if st.button("Submit"):
        # حساب السعرات المطلوبة باليومية
        required_calories = compute_daily_caloric_intake(
            bmr=compute_bmr(gender, weight, height, age),
            activity_intensity=intensity,
            objective=objective
        )

        # عرض السعرات المطلوبة بطريقة واضحة
        st.subheader(f"Required Daily Calories: {required_calories} kcal")

        # عرض الوصفات المقترحة بشكل منسق
        suggested_recipes = suggest_recipes(gender, weight, height, age, intensity, objective)

        if suggested_recipes is not None and not suggested_recipes.empty:
            st.subheader("Top 5 Suggested Recipes:")
            for idx, recipe in suggested_recipes.iterrows():
                st.write(f"*{idx + 1}. {recipe['Name']}* - {recipe['Calories']} Calories")
                st.write('-' * 40)
        else:
            st.warning("Unfortunately, we couldn't find any recipes matching your preferences. Try adjusting your inputs or dietary preferences.")

# Disease Page
elif page == "Disease Page":
    st.header("Disease Page")
    st.image("Images/doctor.png", width=300)



    health_conditions = st.multiselect(
        "Select health conditions:",
        ['HeartDisease', 'Diabetes', 'HighBloodPressure', 'Obesity', 'Hyperglycemia', 'KidneyDisease'],
        key="health_conditions_input"
    )

    calorie_range = delayed_slider("Calorie Range", 0, 1500, (200, 800), key="calories_input")

    allergies = st.text_input("Enter allergies (e.g., nuts, milk):", key="allergies_input")

    if allergies.strip():
        allergy_list = [allergy.strip() for allergy in allergies.split(',')]
    else:
        allergy_list = []

    if st.button("Filter Recipes", key="filter_submit"):
        filtered_recipes = df[(df['Calories'] >= calorie_range[0]) & (df['Calories'] <= calorie_range[1])]

        if not filtered_recipes.empty:
            st.write("Recipes suitable for you:")
            st.dataframe(filtered_recipes[['Name', 'Calories', 'RecipeIngredientParts']])

            meal_counts = filtered_recipes['Name'].value_counts()
            st.bar_chart(meal_counts)

            st.download_button(
                label="Download Recipes as CSV",
                data=filtered_recipes.to_csv(index=False),
                file_name='filtered_recipes.csv',
                mime='text/csv',
            )
        else:
            st.warning("No recipes found for your preferences. Please adjust your filters.")

# Chatbot Page
elif page == "Chatbot":

    st.title("Chat with DietBot")
    st.image("Images/file.png",width=300)

    # Initialize the chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Capture user input through Streamlit's chat input
    if user_input := st.chat_input("Enter your question (e.g., 'What foods are good for heart disease?' or 'Suggest a low-calorie meal')"):

        # Display user's message and append to session state
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Simulate streaming by displaying response chunks
        with st.chat_message("assistant"):
            bot_placeholder = st.empty()
            bot_placeholder.markdown("_DietBot is thinking..._")

            # Get full response from Azure OpenAI
            full_response = get_response_from_openai(user_input)

            # Simulate streaming by revealing text in chunks
            displayed_response = ""
            for word in full_response.split():
                displayed_response += word + " "
                bot_placeholder.markdown(displayed_response)
                time.sleep(0.05)  # Adjust delay for preferred speed

        # Append the bot's final response to session state
        st.session_state.messages.append({"role": "assistant", "content": full_response})
