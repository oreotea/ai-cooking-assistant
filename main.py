from groq import Groq
import base64
import os
from dotenv import load_dotenv
import streamlit as st
from PIL import Image, ImageFilter
import numpy as np
import io
import random

# Load environment variables from .env file
load_dotenv()

# Initialize Groq client with API key
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Image quality checks
def is_image_blurry(image, threshold=100):
    # Use Pillow to detect edges and calculate variance to estimate blurriness
    gray_image = image.convert("L")  # Convert to grayscale
    edges = gray_image.filter(ImageFilter.FIND_EDGES)
    variance = np.var(np.array(edges))
    return variance < threshold

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

def suggest_single_recipe(ingredients, servings=1):
    try:
        # Request a single recipe suggestion with customized servings
        prompt = (
            f"Suggest a recipe using these ingredients: {ingredients}. "
            f"Provide the recipe for {servings} servings, including the recipe name, ingredients list, "
            "and detailed instructions adjusted for the specified number of servings."
        )

        response = client.chat.completions.create(
            model="llama-3.2-11b-text-preview",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # Parse the response to get the recipe
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content
        else:
            st.error("Failed to suggest a recipe. Please try again.")
            return None

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

def get_additional_recipe_links():
    # Provide a few random recipe links from popular sites
    links = [
        "https://www.allrecipes.com/",
        "https://www.foodnetwork.com/",
        "https://www.bonappetit.com/",
        "https://www.epicurious.com/",
        "https://www.bbcgoodfood.com/"
    ]
    return random.sample(links, 3)

# Streamlit UI Layout
st.set_page_config(layout="wide")

# Title centered at the top
st.markdown("<h1 style='text-align: center;'>AI-Powered Cooking Assistant</h1>", unsafe_allow_html=True)

# File uploader in the center below the title
st.subheader("Upload Ingredients")
uploaded_files = st.file_uploader("Upload ingredient images", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

# Live camera section for capturing images in real-time
st.subheader("Or Use Your Camera")
camera_image = st.camera_input("Take a picture of the ingredients")

# Filters and servings below the file uploader
col1, col2 = st.columns([1, 1])

# Filters on the left (col1)
with col1:
    st.subheader("Recipe Filters")
    cuisine_filter = st.selectbox("Cuisine Type", options=["All", "Italian", "Mexican", "Indian", "Chinese", "American"])
    difficulty_filter = st.selectbox("Difficulty Level", options=["All", "Easy", "Medium", "Hard"])
    dietary_filter = st.selectbox("Dietary Restrictions", options=["All", "Vegetarian", "Vegan", "Gluten-Free", "None"])
    st.info("Recipe filtering is coming soon!")

# Servings on the right (col2)
with col2:
    st.subheader("Servings")
    servings = st.number_input("Number of Servings", min_value=1, max_value=20, value=1)

# Function to process and analyze images (from file upload or camera)
def process_image(image):
    # Check the resolution of the image
    if image.size[0] < 300 or image.size[1] < 300:
        st.error("The resolution is too low. Please upload a higher-resolution image.")
        return None

    # Check if the image is blurry using Pillow
    if is_image_blurry(image):
        st.error("The image is too blurry. Please upload a clearer image.")
        return None

    # Compress the image before sending it to the API
    compressed_image_bytes = compress_image(image)

    # Analyze the image with a loading spinner
    with st.spinner("Analyzing the image..."):
        ingredients = analyze_ingredient(compressed_image_bytes)
        return ingredients

# Analyze images from either file upload or live camera
ingredients_list = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        ingredients = process_image(image)
        if ingredients:
            ingredients_list.append(ingredients)
            st.success(f"Identified Ingredients in {uploaded_file.name}: {ingredients}")

elif camera_image:
    # Process the live camera image
    image = Image.open(camera_image)
    ingredients = process_image(image)
    if ingredients:
        ingredients_list.append(ingredients)
        st.success(f"Identified Ingredients: {ingredients}")

# Suggest a recipe based on the detected ingredients
if ingredients_list:
    all_ingredients = ", ".join(ingredients_list)
    st.write(f"**All identified ingredients:** {all_ingredients}")
    with st.spinner(f"Suggesting a recipe for {servings} servings..."):
        recipe = suggest_single_recipe(all_ingredients, servings)
        if recipe:
            st.write("### Suggested Recipe")
            st.write(recipe)

            # Display additional recipe links
            st.write("### You might also like these recipe sources:")
            for link in get_additional_recipe_links():
                if st.button(f"Open {link}"):
                    # Show the link content within a pop-up window
                    st.markdown(f"<iframe src='{link}' width='100%' height='600px'></iframe>", unsafe_allow_html=True)
        else:
            st.error("Failed to suggest a recipe based on the identified ingredients.")
else:
    st.warning("Please upload an image or take a picture to continue.")
