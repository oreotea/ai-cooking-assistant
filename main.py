from groq import Groq
import base64
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Groq client with API key
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def analyze_ingredient(image_bytes):
    # Convert the image to base64
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    # Send request to Groq API to identify ingredients
    response = client.chat.completions.create(
        model="llama-3.2-11b-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                     "type": "text",
                     "text": "Identify the ingredients in this image. 'Only the ingredients' comma separated and nothing else."
                  },
                    {
                     "type": "image_url",
                     "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                     }
                  }
                ]
            }
        ],
        stream=False,
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stop=None,
    )

    return response.choices[0].message.content

def suggest_recipe(ingredients):
    # Send identified ingredients to Llama3.2 to get recipe suggestions
    response = client.chat.completions.create(
        model="llama-3.2-11b-text-preview",
        messages=[
            {"role": "user", "content": f"Suggest a recipe using these ingredients: {ingredients}"}
        ]
    )
    return response.choices[0].message.content


import streamlit as st

# Streamlit UI
st.title("AI-Powered Cooking Assistant")

uploaded_files = st.file_uploader("Upload ingredient images", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

if uploaded_files:
    ingredients_list = []

    for uploaded_file in uploaded_files:
        # Analyze each uploaded image
        image_bytes = uploaded_file.read()
        ingredients = analyze_ingredient(image_bytes)
        ingredients_list.append(ingredients)
        st.write(f"Identified Ingredients in {uploaded_file.name}: {ingredients}")

    # Suggest recipes based on the identified ingredients
    if ingredients_list:
        all_ingredients = ", ".join(ingredients_list)
        st.write(f"All identified ingredients: {all_ingredients}")
        recipe = suggest_recipe(all_ingredients)
        st.write("Suggested Recipe:")
        st.write(recipe)