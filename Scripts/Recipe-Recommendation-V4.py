# Required libraries for making requests, handling data, and performing NLP/ML operations
from flask import Flask, request, jsonify
import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
import asyncio
import aiohttp
import nest_asyncio
import faiss
import pickle
import torch
import json
from transformers import DistilBertTokenizer, DistilBertModel
import codecs
import time
import os
from google.cloud import storage
from dotenv import load_dotenv
from functools import wraps  
from quart import Quart, request, jsonify, Response  # Ensure Quart is imported here
from functools import wraps
import json
from collections import OrderedDict
import random 

load_dotenv()
# Access the variables
STAGING_API_KEY = os.getenv("STAGING_API_KEY")
# Apply the nest_asyncio patch to enable asyncio to work in a nested context
nest_asyncio.apply()
USERNAME = os.getenv("STAGING_OPENSEARCH_USER")
PASSWORD = os.getenv("STAGING_OPENSEARCH_PWD")
OPENSEARCH_URL = os.getenv("STAGING_OPENSEARCH_HOST")
HEALTH_PROFILE_GROUP = os.getenv("STAGING_HEALTH_PROFILE_GROUP")

# Initialize Flask app
app = Quart(__name__)

# Function to normalize text (fixes encoding issues)
def normalize_text(text):
    return ftfy.fix_text(text)

def require_api_key(view_function):
    @wraps(view_function)
    async def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        print(f"Authorization Header: '{auth_header}'")  # Debugging output
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1].strip()
            print(f"Token extracted: '{token}'")  # Debugging output
            print(f"Token length: {len(token)}, API_KEY length: {len(STAGING_API_KEY)}")
            print(f"Token bytes: {token.encode('utf-8')}")
            print(f"API_KEY bytes: {STAGING_API_KEY.encode('utf-8')}")
            if token == STAGING_API_KEY:
                return await view_function(*args, **kwargs)  # Await the async view function
        return jsonify({"error": "Unauthorized API Key"}), 401
    return decorated_function


# Load fine-tuned FoodBERT model and tokenizer (for recipe ingredient embeddings)
model = DistilBertModel.from_pretrained('/home/rsa-key-20240306/Recipe-Recommendations/Staging/Weights/fine_tuned_distilbert_model')
tokenizer = DistilBertTokenizer.from_pretrained('/home/rsa-key-20240306/Recipe-Recommendations/Staging/Weights/fine_tuned_distilbert_tokenizer')

# Load precomputed FAISS index and recipe metadata
index = faiss.read_index('/home/rsa-key-20240306/Recipe-Recommendations/Staging/Data/recipe_faiss_ivfflat_index_42k.index')
with open('/home/rsa-key-20240306/Recipe-Recommendations/Staging/Data/recipe_metadata_42k.pkl', 'rb') as f_metadata:
    recipe_metadata = pickle.load(f_metadata)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():  # Disable gradient tracking
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)  # Averaging token embeddings
    return embedding.detach().numpy()

# Function to compute the user's ingredient vector using FoodBERT
def get_user_vector(user_ingredients):
    # Generate embeddings for each ingredient
    user_embeddings = [get_embedding(ingredient) for ingredient in user_ingredients]
    # Average the embeddings to create a single user vector
    user_vector = np.mean(user_embeddings, axis=0).astype('float32')
    
    # Ensure user_vector is 1-dimensional with the correct dimensionality
    user_vector = user_vector.flatten()  # This will make it a 1D array of shape (768,)
    
    # Check dimensionality consistency with the FAISS index
    if user_vector.shape[0] != index.d:
        raise ValueError(f"Dimension mismatch: user vector has shape {user_vector.shape}, but index expects dimension {index.d}")
    
    # Normalize the vector for cosine similarity
    faiss.normalize_L2(user_vector.reshape(1, -1))
    return user_vector


# Function to fetch batch scroll results from OpenSearch (for large queries)
async def fetch_scroll_batch(scroll_id, session):
    scroll_url = f"{OPENSEARCH_URL}/_search/scroll"
    async with session.post(
        scroll_url,
        auth=aiohttp.BasicAuth(USERNAME, PASSWORD),
        headers={"Content-Type": "application/json"},
        json={"scroll": "2m", "scroll_id": scroll_id}
    ) as response:
        if response.status == 200:
            return await response.json()
        else:
            print(f"Error during scroll: {response.status}, {await response.text()}")
            return None

# Function to retrieve meal recommendations for a user based on their health profile
def get_meal_recommendations_for_user(foodhak_user_id):
    url = f"{HEALTH_PROFILE_GROUP}/{foodhak_user_id}"
    headers = {
        "accept": "application/json",
        "Authorization": "Api-Key mS6WabEO.1Qj6ONyvNvHXkWdbWLFi9mLMgHFVV4m7",
        "X-CSRFToken": "K2JQAMgMyW91ofUU1nzGPKyGtMQu2F1K4tuJw6FdPuSf5Y2nBFussCcbFWAfhJi7"
    }
    
    try:
        # Send the GET request
        response = requests.get(url, headers=headers)
        
        # Check if the request was successful
        if response.status_code in [200, 201]:
            # Parse the JSON response and return the meal recommendations
            response_data = response.json()
            meal_recommendations = response_data.get("nutrition_values", [])
            return meal_recommendations
        else:
            # If the response code is not 200 or 201, return an empty list
            print(f"Failed to get data. Status code: {response.status_code}")
            print("Response text:", response.text)
            return []
    
    except requests.RequestException as e:
        # Handle exceptions like network errors, timeouts, etc.
        print(f"An error occurred: {e}")
        return []

# Function to extract meal information dynamically from meal recommendations
def extract_meal_info_dynamically(meal_recommendations):
    dynamic_info = {}
    for recommendation in meal_recommendations:
        item_name = recommendation.get('item')
        item_value = recommendation.get('value')
        item_unit = recommendation.get('unit')
        # Mapping certain key nutritional values to a standard format
        if item_name and item_value:
            if item_name.lower() == "energy":
                formatted_name = "Energy (KCAL)"
            elif item_name.lower() == "protein":
                formatted_name = "Protein (G)"
            elif item_name.lower() == "fats":
                formatted_name = "Total Fat (G)"
            elif item_name.lower() == "fatty acids - total saturated":
                formatted_name = "Saturated Fat (G)"
            elif item_name.lower() == "sodium":
                formatted_name = "Sodium Na (MG)"
            elif item_name.lower() == "carbohydrate":
                formatted_name = "Total Carbohydrate (G)"
            elif item_name.lower() == "dietary fibre":
                formatted_name = "Dietary Fiber (G)"
            elif item_name.lower() == "vitamin c":
                formatted_name = "Vitamin C (MG)"                
            elif item_name.lower() == "calcium":
                formatted_name = "Calcium (MG)"
            elif item_name.lower() == "iron":
                formatted_name = "Iron (MG)"
            elif item_name.lower() == "potassium":
                formatted_name = "Potassium K (MG)"                
            else:
                continue
            dynamic_info[formatted_name] = item_value
    return dynamic_info

# Function to get a user's profile from OpenSearch
def get_user_profile(foodhak_user_id):
    url = f"{OPENSEARCH_URL}/user-profiles/_search"
    query = {
        "query": {
            "match": {
                "foodhak_user_id": foodhak_user_id
            }
        }
    }
    # Send the request to OpenSearch
    response = requests.get(url, json=query, auth=HTTPBasicAuth(USERNAME, PASSWORD))
    if response.status_code == 200:
        results = response.json()
        if results['hits']['total']['value'] > 0:
            result = results['hits']['hits'][0]['_source']   
            # Get user goals and related information
            user_health_goals = result.get("user_health_goals", [])
            primary_goal = next((goal for goal in user_health_goals if goal['user_goal'].get('is_primary')), None)
            primary_goal_id = primary_goal['user_goal'].get('id') if primary_goal else user_health_goals[0]['user_goal'].get('id')
            primary_goal_title = primary_goal['user_goal'].get('title') if primary_goal else user_health_goals[0]['user_goal'].get('title')
            user_age = result.get("age")
            user_gender = result.get("sex")
            # Fetch meal recommendations
            meal_recommendations = get_meal_recommendations_for_user(foodhak_user_id)
            #print(meal_recommendations)
            dynamic_meal_info = extract_meal_info_dynamically(meal_recommendations) if meal_recommendations else {}
            #print(dynamic_meal_info)
            # Extract recommended ingredients
            if primary_goal and primary_goal.get("ingredients_to_recommend"):
                # Use ingredients from primary goal if available
                recommended_ingredients = [
                    ingredient.get("common_name")
                    for ingredient in primary_goal.get("ingredients_to_recommend", [])
                ]
            else:
                # If primary_goal does not have ingredients_to_recommend, fallback to first goal with ingredients
                first_available_goal = next(
                    (goal for goal in user_health_goals if goal.get("ingredients_to_recommend")),
                    None
                )
                recommended_ingredients = [
                    ingredient.get("common_name")
                    for ingredient in first_available_goal.get("ingredients_to_recommend", [])
                ] if first_available_goal else []

            #print(recommended_ingredients)
            # Create a profile info dictionary
            profile_info = {
                "User Name": result.get("name"),
                "User Age": user_age,
                "User Gender": user_gender,
                "Primary User Goal ID": primary_goal_id,
                "Primary User Goal Title": primary_goal_title,
                "Goal Titles": [goal['user_goal']['title'] for goal in user_health_goals if 'user_goal' in goal],
                "Dietary Restrictions": result.get("dietary_restrictions", {}).get("name"),
                "Allergens": [allergen.get("type") for allergen in result.get("allergens", [])],
                "Recommended Ingredients": recommended_ingredients,
                "Meal Recommendation": dynamic_meal_info  
            }
            return profile_info
        else:
            print("No matching user profile found.")
            return None
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Updated function to retrieve cosine similarity recipes
def retrieve_cosine_similarity_recipes(user_ingredients, top_k):
    # Set FAISS index nprobe for improved search accuracy
    index.nprobe = 150  # Ensures more clusters are considered during retrieval

    # Generate the user embedding vector
    user_vector = get_user_vector(user_ingredients)

    # Verify dimensionality consistency with FAISS index
    if user_vector.shape[0] != index.d:
        raise ValueError(f"Dimensionality mismatch: User vector has shape {user_vector.shape}, but index expects dimension {index.d}.")

    # Reshape for FAISS search if necessary
    user_vector = user_vector.reshape(1, -1)

    # Query the FAISS index for top `top_k` recipes
    D, I = index.search(user_vector, top_k)

    # Compute cosine similarity from squared L2 distances
    cosine_similarities = 1 - (D[0] / 2)  # Converting L2 distances to cosine similarity

    # Extract recipe URLs and scores
    results = []
    for i in range(top_k):
        recipe_index = I[0, i]
        similarity_score = cosine_similarities[i]  # Use computed cosine similarity
        if recipe_index != -1 and recipe_index < len(recipe_metadata):
            recipe_info = recipe_metadata[recipe_index]
            recipe_url = recipe_info.get('recipe_url', 'No URL Available')
            results.append((recipe_index, recipe_url, similarity_score))

    return results



# Function to fetch all recipes data from OpenSearch
async def fetch_recipes_from_opensearch():
    query = {
        "size": 5000,
        "query": {"match_all": {}},
        "sort": ["_doc"]
    }
    async with aiohttp.ClientSession() as session:
        url = f"{OPENSEARCH_URL}/recipes/_search"
        async with session.get(
            url,
            auth=aiohttp.BasicAuth(USERNAME, PASSWORD),
            headers={"Content-Type": "application/json"},
            json=query,
            params={"scroll": "10m"}
        ) as response:
            # Log the response status
            #print(f"OpenSearch response status: {response.status}")
            # Read the response content
            #response_text = await response.text()
            #print(f"OpenSearch response content: {response_text}")

            if response.status == 200:
                recipes_data = await response.json()
                total_hits = recipes_data.get("hits", {}).get("total", {}).get("value", 0)
                scroll_id = recipes_data.get("_scroll_id")
                hits = recipes_data.get("hits", {}).get("hits", [])
                recipes_list = [hit["_source"] for hit in hits]

                # Fetch the rest of the documents using scroll API
                while len(recipes_list) < total_hits:
                    scroll_data = await fetch_scroll_batch(scroll_id, session)
                    if scroll_data and "hits" in scroll_data:
                        hits = scroll_data.get("hits", {}).get("hits", [])
                        if not hits:
                            break
                        recipes_list.extend([hit["_source"] for hit in hits])
                        scroll_id = scroll_data.get("_scroll_id")
                    else:
                        break

                return recipes_list
            else:
                return []

def convert_to_float(value):
    """Convert a value to float, handling ranges like '1.6-2.2' by averaging them."""
    try:
        # Check if the value is a range (e.g., '1.6-2.2')
        if isinstance(value, str) and '-' in value:
            # Split the range and calculate the average
            low, high = map(float, value.split('-'))
            return (low + high) / 2
        # Convert single numeric strings to float
        return float(value)
    except ValueError:
        # Return a default value (e.g., 0) if conversion fails
        return 0.0

# Function to compute Euclidean similarity between the user and recipes
def compute_euclidean_similarity(df_recipes, user_info):
    # Identify common columns between user_info and recipe data
    common_columns = [col for col in user_info.keys() if col in df_recipes.columns]
    # Convert each item in user_info to a float (handling ranges)
    filtered_user_info = np.array([convert_to_float(user_info[k]) for k in common_columns])

    # Clean recipe data to be numeric and comparable
    df_recipes[common_columns] = df_recipes[common_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
    filtered_user_info = filtered_user_info.astype(np.float64)

    # Compute Euclidean distances and normalize them
    distances = cdist(df_recipes[common_columns], [filtered_user_info], metric='euclidean').flatten()
    scaler = MinMaxScaler()
    normalized_distances = 1 - scaler.fit_transform(distances.reshape(-1, 1))
    df_recipes['Normalized Distance'] = normalized_distances
    return df_recipes

# Function to replace Unicode sequences in text with proper symbols
def replace_unicode_sequences(text):
    replacements = {
        r"\u00bc": "¼",  # 1/4
        r"\u00bd": "½",  # 1/2
        r"\u00be": "¾",  # 3/4
        r"\u2153": "⅓",  # 1/3
        r"\u2154": "⅔",  # 2/3
        r"\u00a9": "©",  # Copyright symbol
        r"\u00ae": "®",  # Registered trademark symbol
        r"\u2122": "™",  # Trademark symbol
        r"\u00e9": "é",  # e with acute accent
        r"Â": "",  # Remove extra Â character
        r"\t": ", ",  # Replace tab with comma and space
        r"\u2019": "'",  # Curly apostrophe to straight apostrophe
        r"\u201c": '"',  # Left double quotation mark
        r"\u201d": '"',  # Right double quotation mark
        r"\u00f1": "ñ",  # n with tilde (Spanish ñ)
        r"â\u0085": "⅓", 
        r"â\u0085": "⅔",
    }
    # Replace each Unicode sequence in the text
    for unicode_seq, replacement in replacements.items():
        text = text.replace(unicode_seq, replacement)
    return text

# Function to display recipe details based on the similarity score type
def display_recipe_details(recipe, score_type, score_value, recipe_id=None, similarity_type="combined"):
    ingredients = recipe.get('detailed_ingredients', recipe.get('ingredients', 'No ingredients available'))
    # Clean up and structure the recipe data for better JSON output
    clean_recipe_data = OrderedDict([
        ("recipe_id", str(recipe_id) if recipe_id else 'No ID available'),
        ("recipe_description", recipe.get('recipe_description', 'No description available')),
        ("recipe_url", recipe.get('recipe_url', 'No URL available')),
        ("food_title", recipe.get('food_title', 'Unknown Title')),
        ("mealType", recipe.get('mealType', 'Unknown')),
        ("servings", str(recipe.get('servings', 'Unknown'))),
        ("ready_in_minutes", int(recipe.get('ready_in_minutes', 0))),
        ("image_url", recipe.get('image_url', 'No image available')),
        ("ingredients", ingredients),
        ("Energy (KCAL)", float(recipe.get('Energy (KCAL)', 0))),
        ("Total Fat (G)", float(recipe.get('Total Fat (G)', 0))),
        ("Saturated Fat (G)", float(recipe.get('Saturated Fat (G)', 0))),
        ("Cholesterol (MG)", float(recipe.get('Cholesterol (MG)', 0))),
        ("Sodium Na (MG)", float(recipe.get('Sodium Na (MG)', 0))),
        ("Total Carbohydrate (G)", float(recipe.get('Total Carbohydrate (G)', 0))),
        ("Dietary Fiber (G)", float(recipe.get('Dietary Fiber (G)', 0))),
        ("Protein (G)", float(recipe.get('Protein (G)', 0))),
        ("Vitamin C (MG)", float(recipe.get('Vitamin C (MG)', 0))),
        ("Calcium (MG)", float(recipe.get('Calcium (MG)', 0))),
        ("Iron (MG)", float(recipe.get('Iron (MG)', 0))),
        ("Potassium K (MG)", float(recipe.get('Potassium K (MG)', 0))),
        ("cautions", recipe.get('cautions', [])),
        ("diet_labels", recipe.get('diet_labels', [])),
        ("average_rating", str(recipe.get('average_rating', 'N/A'))),
        ("total_ratings", str(recipe.get('total_ratings', 'N/A'))),
        (score_type, round(float(score_value), 3))
    ])
    return clean_recipe_data

# Function to fetch recipe ID from OpenSearch based on recipe URL using aiohttp
async def fetch_recipe_id_by_url(recipe_url, session):
    search_url = f"{OPENSEARCH_URL}/recipes/_search"
    query = {
        "query": {
            "match": {
                "recipe_url": recipe_url
            }
        }
    }
    async with session.post(
        search_url,
        auth=aiohttp.BasicAuth(USERNAME, PASSWORD),
        headers={"Content-Type": "application/json"},
        json=query
    ) as response:
        # Check if the response is successful
        if response.status == 200:
            results = await response.json()
            # If the results contain any hits, return the recipe_id
            if results['hits']['total']['value'] > 0:
                return results['hits']['hits'][0]['_source']['recipe_id']  # Return the actual recipe_id
        # Return None if no match is found
        return None



# Function to display cosine similarity recipes with dynamic k
async def display_cosine_similarity_recipes(user_ingredients, k):
    # Get the user vector based on the ingredients
    user_vector = get_user_vector(user_ingredients)
    # Retrieve top recipes using cosine similarity
    cosine_results = retrieve_cosine_similarity_recipes(user_vector, k)
    print("Top recipes based on cosine similarity:")
    async with aiohttp.ClientSession() as session:
        # Display the top recipes in JSON format
        for idx, (recipe_id, score) in enumerate(cosine_results):
            recipe_data = recipe_metadata[recipe_id]
            if recipe_data:  # Handle missing recipe_id
                recipe_url = recipe_data.get('recipe_url', None)
                print(recipe_url)
                if recipe_url:
                    recipe_id = await fetch_recipe_id_by_url(recipe_url, session)
                    print(f"Fetched Recipe ID for {recipe_url}: {recipe_id}")
            # Build and print recipe details
            recipe_details = display_recipe_details(recipe_data, "Cosine Similarity", score, recipe_id, similarity_type="cosine")
            print(json.dumps(recipe_details, indent=4, ensure_ascii=False))  # Print recipe details in JSON

# Function to display Euclidean similarity recipes with dynamic k
def display_euclidean_similarity_recipes(df_recipes, user_info, k):
    # Compute Euclidean similarity for the recipes
    df_recipes = compute_euclidean_similarity(df_recipes, user_info)
    # Sort recipes by the normalized distance and get top k
    sorted_recipes = df_recipes.sort_values(by='Normalized Distance', ascending=False).head(k)
    print("\nTop recipes based on Euclidean similarity:")
    # Display the top recipes in JSON format
    for idx, (_, recipe) in enumerate(sorted_recipes.iterrows()):
        recipe_id = recipe['recipe_id']
        recipe_details = display_recipe_details(recipe, "Euclidean Similarity", recipe['Normalized Distance'], recipe_id, similarity_type="euclidean")
        print(json.dumps(recipe_details, indent=4, ensure_ascii=False))  # Print recipe details in JSON


async def save_cosine_results_to_json(cosine_results, json_filename='cosine_results.json'):
    # Convert to a list of dictionaries and ensure all values are JSON-serializable
    cosine_results_dict = [
        {"recipe_id": str(recipe_id), "cosine_score": float(cosine_score)} 
        for recipe_id, cosine_score in cosine_results
    ]

    # Save to JSON file
    with open(json_filename, 'w') as json_file:
        json.dump(cosine_results_dict, json_file, indent=4)
    
    print(f"Cosine results saved to {json_filename}")


# Function to fetch cosine similarity scores for the entire dataset
def fetch_cosine_similarity_for_all(user_ingredients, k=42000):
    """Compute cosine similarity for the user ingredients against the entire recipe dataset."""
    cosine_results = retrieve_cosine_similarity_recipes(user_ingredients, k)
    print(f"Cosine similarity: Retrieved {len(cosine_results)} recipes")

    # Convert to a dictionary for easy look-up by URL
    cosine_scores_by_url = {recipe_url: score for _, recipe_url, score in cosine_results}
    return cosine_scores_by_url


def convert_to_float(value):
    """
    Convert a value to float, handling strings and ranges (e.g., '1.6-2.2').
    """
    try:
        # Handle ranges like '1.6-2.2' by averaging
        if isinstance(value, str) and '-' in value:
            low, high = map(float, value.split('-'))
            return (low + high) / 2
        # Convert single numeric strings to float
        return float(value)
    except (ValueError, TypeError):
        # Return a default value (e.g., 0) if conversion fails
        return 0.0

def preprocess_meal_types(df):
    # Replace missing or invalid mealType values with an empty list
    df["mealType"] = df["mealType"].fillna("").str.lower().str.split(", ")
    return df

def filter_valid_meal_types(df, valid_meal_types={"breakfast", "lunch", "dinner"}):
    # Keep recipes where at least one valid meal type is present
    return df[df["mealType"].apply(lambda x: any(meal in valid_meal_types for meal in x))]


async def display_weighted_recipes(user_ingredients, df_recipes, user_info, meal_counts, cosine_weight=0.5, euclidean_weight=0.5):
    # Preprocess meal types in the dataset
    df_recipes = preprocess_meal_types(df_recipes)

    # Filter recipes to include only valid meal types
    df_recipes = filter_valid_meal_types(df_recipes)

    # Convert user_info values to float
    user_info = {k: convert_to_float(v) for k, v in user_info.items()}
    print("Converted User Info:", user_info)  # Debugging statement

    # Return immediately if no meal counts are specified
    if max(meal_counts.values()) == 0:
        return {meal_type: [] for meal_type in meal_counts}

    # Step 1: Retrieve cosine similarity scores for all recipes
    k = len(recipe_metadata)  # Set k to include all recipes
    if k > 0:
        cosine_results = retrieve_cosine_similarity_recipes(user_ingredients, k)
        print(f"Cosine similarity: Retrieved {len(cosine_results)} recipes")
    else:
        cosine_results = []  # Ensure `cosine_results` is initialized

    # Create a dictionary for cosine scores
    cosine_scores_by_url = {recipe_url: score for _, recipe_url, score in cosine_results}

    # Step 2: Retrieve top recipes using Euclidean similarity for each meal type
    selected_recipes = {meal_type: [] for meal_type in meal_counts}
    meal_type_percentages = {"breakfast": 0.2, "lunch": 0.4, "dinner": 0.4}

    for meal_type in meal_counts:
        if meal_counts[meal_type] == 0:
            continue  # Skip if no recipes are needed for this meal type
        # Initialize meal_user_info as an empty dictionary
        meal_user_info = {}  # Add this line to initialize the dictionary
        # Compute meal-specific user nutritional requirements
        for k, v in user_info.items():
            v = convert_to_float(v)
            meal_percentage = meal_type_percentages[meal_type]
            try:
                meal_user_info[k] = round(v * meal_percentage, 2)  # Round to 2 decimal places
            except TypeError as e:
                print(f"Error processing nutrient '{k}' with value '{v}': {e}")
                meal_user_info[k] = 0.0

        print(f"Meal User Info for {meal_type}:", meal_user_info)  # Debugging statement

        # Compute Euclidean similarity for the recipes
        df_recipes_meal = compute_euclidean_similarity(df_recipes.copy(), meal_user_info)

        # Filter recipes based on meal type
        meal_recipes = df_recipes_meal[df_recipes_meal["mealType"].apply(lambda x: meal_type in x)].copy()
        print(f"{meal_type.capitalize()}: Found {len(meal_recipes)} recipes")

        if meal_recipes.empty:
            print(f"Warning: No recipes available for {meal_type}")
            continue

        # Sort recipes by Euclidean similarity and select the top recipes
        meal_recipes = meal_recipes.sort_values(by="Normalized Distance", ascending=False).head(1000)

        # Retrieve cosine similarity scores for these recipes
        meal_recipes["cosine_similarity"] = meal_recipes["recipe_url"].map(cosine_scores_by_url).fillna(0)

        # Normalize the scores
        scaler = MinMaxScaler()
        meal_recipes[["euclidean_similarity_normalized", "cosine_similarity_normalized"]] = scaler.fit_transform(
            meal_recipes[["Normalized Distance", "cosine_similarity"]]
        )

        # Combine scores with weights
        meal_recipes["combined_score"] = (
            cosine_weight * meal_recipes["cosine_similarity_normalized"] +
            euclidean_weight * meal_recipes["euclidean_similarity_normalized"]
        )
                # Sort recipes by combined_score in descending order
        meal_recipes = meal_recipes.sort_values(by="combined_score", ascending=False)

        # Limit to the top 500 recipes
        top_recipes = meal_recipes.head(1000)

        # Select a random subset of recipes from the top 500
        count = meal_counts[meal_type]
        if len(top_recipes) >= count:
            sampled_recipes = top_recipes.sample(n=count)
            #sampled_recipes = top_recipes.head(count) picking the recipes from top 500 based on top scores
        else:
            sampled_recipes = top_recipes
        # Process selected recipes for the final output
        async with aiohttp.ClientSession() as session:
            for _, row in sampled_recipes.iterrows():
                recipe_url = row["recipe_url"]
                recipe_data = next((recipe for recipe in recipe_metadata if recipe.get("recipe_url") == recipe_url), None)
                if not recipe_data:
                    continue

                # Fetch the recipe ID if it's missing
                recipe_id = recipe_data.get("recipe_id", None)
                if not recipe_id or recipe_id == "No ID available":
                    recipe_id = await fetch_recipe_id_by_url(recipe_url, session)
                    recipe_data["recipe_id"] = recipe_id if recipe_id else "No ID available"

                # Append the recipe details
                selected_recipes[meal_type].append(
                    display_recipe_details(
                        recipe_data,
                        "Weighted Similarity",
                        row["combined_score"],
                        recipe_id,
                        similarity_type="combined"
                    )
                )

    # Return the randomly selected recipes for each meal type
    return selected_recipes






# Example of returning recommendations with all retrieved and random subset
@app.route('/weighted-recommendations', methods=['POST'])
@require_api_key
async def weighted_recommendations():
    request_data = await request.get_data()
    data = json.loads(request_data.decode('utf-8'), object_pairs_hook=OrderedDict)
    foodhak_user_id = data.get('foodhak_user_id')
    meal_counts = data.get('meal_counts', OrderedDict([('breakfast', 5), ('lunch', 5), ('dinner', 5)]))

    user_profile = get_user_profile(foodhak_user_id)
    if not user_profile:
        return jsonify({"error": "User profile not found"}), 404

    recipes_data = await fetch_recipes_from_opensearch()
    if recipes_data:
        df_recipes = pd.DataFrame(recipes_data)
        user_ingredients = user_profile["Recommended Ingredients"]
        print("User-Ingredients",user_ingredients)
        user_info = user_profile["Meal Recommendation"]

        # Get recommendations with both all retrieved and random subset
        recommendations = await display_weighted_recipes(user_ingredients, df_recipes, user_info, meal_counts)

        recommendations_json = json.dumps(recommendations, ensure_ascii=False, indent=4)
        return Response(recommendations_json, mimetype='application/json')

    else:
        return jsonify({"error": "No recipes found"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
