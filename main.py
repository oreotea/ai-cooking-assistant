from groq import Groq
import base64
import os
from dotenv import load_dotenv
import streamlit as st
from PIL import Image
import cv2
import numpy as np
import io

# Load environment variables from .env file
load_dotenv()

# Initialize Groq client with API key
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Image quality checks
def is_image_blurry(image, threshold=100):
    gray = np.array(image.convert("L"))
    variance_of_laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance_of_laplacian < threshold

def compress_image(image, quality=85):
    # Compress the image by saving it with reduced quality
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=quality)
    return buffered.getvalue()

def analyze_ingredient(image_bytes):
    try:
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

        # Check if the response contains the expected data
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content
        else:
            st.error("Failed to identify ingredients. Please try again.")
            return None

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

def suggest_recipe(ingredients):
    try:
        # Send identified ingredients to Llama3.2 to get recipe suggestions
        response = client.chat.completions.create(
            model="llama-3.2-11b-text-preview",
            messages=[
                {"role": "user", "content": f"Suggest a recipe using these ingredients: {ingredients}"}
            ]
        )

        # Check if the response contains the expected data
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content
        else:
            st.error("Failed to suggest a recipe. Please try again.")
            return None

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

# Streamlit UI
st.title("AI-Powered Cooking Assistant")

# File uploader
uploaded_files = st.file_uploader("Upload ingredient images", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

if uploaded_files:
    ingredients_list = []

    for uploaded_file in uploaded_files:
        # Open the image using PIL
        image = Image.open(uploaded_file)

        # Check the resolution of the image
        if image.size[0] < 300 or image.size[1] < 300:
            st.error(f"The resolution of {uploaded_file.name} is too low. Please upload a higher-resolution image.")
            continue

        # Check if the image is blurry
        if is_image_blurry(image):
            st.error(f"The image {uploaded_file.name} is too blurry. Please upload a clearer image.")
            continue

        # Compress the image before sending it to the API
        compressed_image_bytes = compress_image(image)

        # Analyze each compressed image with a loading spinner
        with st.spinner(f"Analyzing {uploaded_file.name}..."):
            ingredients = analyze_ingredient(compressed_image_bytes)
            if ingredients:
                ingredients_list.append(ingredients)
                st.success(f"Identified Ingredients in {uploaded_file.name}: {ingredients}")
            else:
                st.error(f"Could not identify ingredients in {uploaded_file.name}. Please try a different image.")

    # Suggest recipes based on the identified ingredients
    if ingredients_list:
        all_ingredients = ", ".join(ingredients_list)
        st.write(f"**All identified ingredients:** {all_ingredients}")
        with st.spinner("Suggesting a recipe..."):
            recipe = suggest_recipe(all_ingredients)
            if recipe:
                st.write("**Suggested Recipe:**")
                st.write(recipe)
            else:
                st.error("Failed to suggest a recipe based on the identified ingredients.")
else:
    st.warning("Please upload one or more images to continue.")
