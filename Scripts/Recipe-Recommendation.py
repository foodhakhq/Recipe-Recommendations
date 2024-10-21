# Required libraries for making requests, handling data, and performing NLP/ML operations
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
import requests

# Function to normalize text (fixes encoding issues)
def normalize_text(text):
    return ftfy.fix_text(text)

# Apply the nest_asyncio patch to enable asyncio to work in a nested context
nest_asyncio.apply()

# OpenSearch and API configuration
OPENSEARCH_URL = "https://search-foodhak-staging-core-ffnbha54vi5fo2hm6vjcjkffpe.eu-west-2.es.amazonaws.com/recipes/_search"
USERNAME = "admin"
PASSWORD = "HealthyAsianKitchen1$3"

# Load fine-tuned FoodBERT model and tokenizer (for recipe ingredient embeddings)
model = DistilBertModel.from_pretrained('/home/ai/Recipe Recommendation/FoodBERT Embeddings/Weights2/fine_tuned_distilbert_model')
tokenizer = DistilBertTokenizer.from_pretrained('/home/ai/Recipe Recommendation/FoodBERT Embeddings/Weights2/fine_tuned_distilbert_tokenizer')

# Load precomputed FAISS index and recipe metadata
index = faiss.read_index('recipe_faiss_ivf_index.index')
with open('recipe_metadata.pkl', 'rb') as f_metadata:
    recipe_metadata = pickle.load(f_metadata)

# Function to generate embeddings for a given ingredient using FoodBERT
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    # Averaging token embeddings to get a single vector representation
    embedding = outputs.last_hidden_state.mean(dim=1)  
    return embedding.detach().numpy()

# Function to compute the user's ingredient vector using FoodBERT
def get_user_vector(user_ingredients):
    # Create an embedding for each ingredient
    user_embeddings = [get_embedding(ingredient) for ingredient in user_ingredients]
    # Average the embeddings to get a single user vector
    user_vector = np.mean(user_embeddings, axis=0).astype('float32')
    # Normalize the vector for cosine similarity
    faiss.normalize_L2(user_vector.reshape(1, -1))  
    return user_vector

# Function to fetch batch scroll results from OpenSearch (for large queries)
async def fetch_scroll_batch(scroll_id, session):
    scroll_url = "https://search-foodhak-staging-core-ffnbha54vi5fo2hm6vjcjkffpe.eu-west-2.es.amazonaws.com/_search/scroll"
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
    url = f"https://api-staging.foodhak.com/healthprofile-group-details/{foodhak_user_id}"
    headers = {
        "accept": "application/json",
        "Authorization": "Api-Key mS6WabEO.1Qj6ONyvNvHXkWdbWLFi9mLMgHFVV4m7",
        "X-CSRFToken": "K2JQAMgMyW91ofUU1nzGPKyGtMQu2F1K4tuJw6FdPuSf5Y2nBFussCcbFWAfhJi7"
    }

    # Send the GET request
    response = requests.get(url, headers=headers)
    if response.status_code in [200, 201]:
        # Parse the JSON response and return the meal recommendations
        response_data = response.json()
        meal_recommendations = response_data.get("nutrition_values", [])
        return meal_recommendations
    else:
        # Print error details in case the request fails
        print(f"Failed to get data. Status code: {response.status_code}")
        print("Response text:", response.text)
        return None

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
    url = "https://search-foodhak-staging-core-ffnbha54vi5fo2hm6vjcjkffpe.eu-west-2.es.amazonaws.com/user-profiles/_search"
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
            dynamic_meal_info = extract_meal_info_dynamically(meal_recommendations) if meal_recommendations else {}

            # Extract recommended ingredients
            recommended_ingredients = [
                ingredient.get("common_name")
                for ingredient in primary_goal.get("ingredients_to_recommend", []) if primary_goal
            ]

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

# Function to retrieve top recipes based on cosine similarity
def retrieve_cosine_similarity_recipes(user_vector, k=5):
    # Perform FAISS index search for top k results
    D, I = index.search(user_vector.reshape(1, -1), k)
    return [(I[0, i], D[0, i]) for i in range(k)]

# Function to fetch all recipes data from OpenSearch
async def fetch_recipes_from_opensearch():
    query = {
        "size": 5000,
        "query": {"match_all": {}},
        "sort": ["_doc"]
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(
            OPENSEARCH_URL,
            auth=aiohttp.BasicAuth(USERNAME, PASSWORD),
            headers={"Content-Type": "application/json"},
            json=query,
            params={"scroll": "10m"}
        ) as response:
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

# Function to compute Euclidean similarity between the user and recipes
def compute_euclidean_similarity(df_recipes, user_info):
    # Identify the common columns between the user's info and recipes
    common_columns = [col for col in user_info.keys() if col in df_recipes.columns]
    filtered_user_info = np.array([user_info[k] for k in common_columns])
    
    # Clean and process the recipe dataset for numeric comparison
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
    }
    
    # Replace each Unicode sequence in the text
    for unicode_seq, replacement in replacements.items():
        text = text.replace(unicode_seq, replacement)
    
    return text

# Function to display recipe details based on the similarity score type
def display_recipe_details(recipe, score_type, score_value, similarity_type="cosine"):
    # Choose which ingredient field to display based on the similarity type
    if similarity_type == "cosine":
        ingredients = recipe.get('detailed_ingredients', 'No ingredients listed')
    else:
        ingredients = recipe.get('ingredients', 'No ingredients listed')

    # Replace problematic Unicode sequences in the ingredients
    ingredients = replace_unicode_sequences(ingredients)
    
    # Clean up and structure the recipe data for better JSON output
    clean_recipe_data = {
        "recipe_description": recipe.get('recipe_description', 'No description available'),
        "recipe_url": recipe.get('recipe_url', 'No URL available'),
        "food_title": recipe.get('food_title', 'Unknown Title'),
        "mealType": recipe.get('mealType', 'Unknown'),
        "servings": recipe.get('servings', 'Unknown'),
        "ready_in_minutes": recipe.get('ready_in_minutes', 'N/A'),
        "image_url": recipe.get('image_url', 'No image available'),
        "ingredients": ingredients,
        "Energy (KCAL)": float(recipe.get('Energy (KCAL)', 0)),  # Convert numpy types to Python float
        "Total Fat (G)": float(recipe.get('Total Fat (G)', 0)),
        "Saturated Fat (G)": float(recipe.get('Saturated Fat (G)', 0)),
        "Cholesterol (MG)": float(recipe.get('Cholesterol (MG)', 0)),
        "Sodium Na (MG)": float(recipe.get('Sodium Na (MG)', 0)),
        "Total Carbohydrate (G)": float(recipe.get('Total Carbohydrate (G)', 0)),
        "Dietary Fiber (G)": float(recipe.get('Dietary Fiber (G)', 0)),
        "Protein (G)": float(recipe.get('Protein (G)', 0)),
        "Vitamin C (MG)": float(recipe.get('Vitamin C (MG)', 0)),
        "Calcium (MG)": float(recipe.get('Calcium (MG)', 0)),
        "Iron (MG)": float(recipe.get('Iron (MG)', 0)),
        "Potassium K (MG)": float(recipe.get('Potassium K (MG)', 0)),
        "cautions": recipe.get('cautions', '[]'),
        "diet_labels": recipe.get('diet_labels', '[]'),
        "average_rating": recipe.get('average_rating', 'N/A'),
        "total_ratings": recipe.get('total_ratings', 'N/A'),
        score_type: float(score_value)  # Ensure score is a Python float
    }

    return clean_recipe_data

# Function to display cosine similarity recipes
def display_cosine_similarity_recipes(user_ingredients, k=1000):
    # Get the user vector based on the ingredients
    user_vector = get_user_vector(user_ingredients)
    
    # Retrieve top recipes using cosine similarity
    cosine_results = retrieve_cosine_similarity_recipes(user_vector, k)
    print("Top recipes based on cosine similarity:")
    
    # Display the top recipes in JSON format
    for idx, (recipe_id, score) in enumerate(cosine_results):
        recipe_data = recipe_metadata[recipe_id]
        recipe_details = display_recipe_details(recipe_data, "Cosine Similarity", score, similarity_type="cosine")
        print(json.dumps(recipe_details, indent=4, ensure_ascii=False))  # Print the recipe details in JSON format
        
# Function to display Euclidean similarity recipes
def display_euclidean_similarity_recipes(df_recipes, user_info, k=1000):
    # Compute Euclidean similarity for the recipes
    df_recipes = compute_euclidean_similarity(df_recipes, user_info)
    
    # Sort recipes by the normalized distance and get top k
    sorted_recipes = df_recipes.sort_values(by='Normalized Distance', ascending=False).head(k)
    print("\nTop recipes based on Euclidean similarity:")
    
    # Display the top recipes in JSON format
    for idx, (_, recipe) in enumerate(sorted_recipes.iterrows()):
        recipe_details = display_recipe_details(recipe, "Euclidean Similarity", recipe['Normalized Distance'], similarity_type="euclidean")
        print(json.dumps(recipe_details, indent=4, ensure_ascii=False))  # Print the recipe details in JSON format

# Function to display weighted recipes based on both cosine and Euclidean scores
def display_weighted_recipes(user_ingredients, df_recipes, user_info, meal_counts, cosine_weight=0.5, euclidean_weight=0.5, k=1000):
    """
    :param user_ingredients: List of ingredients for cosine similarity
    :param df_recipes: Dataframe of all recipes
    :param user_info: User meal information for Euclidean similarity
    :param meal_counts: Dictionary with counts of meals { 'breakfast': 2, 'lunch': 0, 'dinner': 3 }
    :param cosine_weight: Weight for cosine similarity (default: 0.5)
    :param euclidean_weight: Weight for Euclidean similarity (default: 0.5)
    :param k: Total number of recipes to retrieve (default: 5)
    :return: JSON object with filtered recipes
    """

    # Get top k cosine similarity results
    user_vector = get_user_vector(user_ingredients)
    cosine_results = retrieve_cosine_similarity_recipes(user_vector, k)

    # Compute Euclidean similarity results and get top k
    df_recipes = compute_euclidean_similarity(df_recipes, user_info)
    euclidean_results = df_recipes.sort_values(by='Normalized Distance', ascending=False).head(k)

    # Create a dictionary to store combined scores
    combined_scores = {}

    # Add cosine similarity results to combined_scores
    for recipe_id, cosine_score in cosine_results:
        recipe_url = recipe_metadata[recipe_id]['recipe_url']
        combined_scores[recipe_url] = {"cosine_score": cosine_score, "euclidean_score": 0}  # Default Euclidean score is 0
    
    # Add Euclidean similarity results to combined_scores
    for index, row in euclidean_results.iterrows():
        recipe_url = row['recipe_url']
        euclidean_score = row['Normalized Distance']
        if recipe_url in combined_scores:
            combined_scores[recipe_url]["euclidean_score"] = euclidean_score
        else:
            combined_scores[recipe_url] = {"cosine_score": 0, "euclidean_score": euclidean_score}  # Default cosine score is 0

    # Calculate the combined weighted score for each recipe
    final_scores = []
    for recipe_url, scores in combined_scores.items():
        combined_score = (cosine_weight * scores['cosine_score']) + (euclidean_weight * scores['euclidean_score'])
        final_scores.append((recipe_url, combined_score))

    # Sort recipes by the combined weighted score
    final_scores.sort(key=lambda x: x[1], reverse=True)

    # Display recipes based on the requested meal types
    meal_type_counts = {'breakfast': meal_counts.get('breakfast', 0), 
                        'lunch': meal_counts.get('lunch', 0), 
                        'dinner': meal_counts.get('dinner', 0)}
    
    selected_recipes = {'breakfast': [], 'lunch': [], 'dinner': []}

    for idx, (recipe_url, score) in enumerate(final_scores):
        # Find the corresponding recipe in the list using the URL
        recipe_data = next(recipe for recipe in recipe_metadata if recipe.get('recipe_url') == recipe_url)
        
        # Get the meal type for the recipe and split it into a list if there are multiple values
        meal_types = recipe_data.get('mealType', '').lower().split(', ')

        # Assign recipe to the first available meal type category with open slots
        if 'breakfast' in meal_types and meal_type_counts['breakfast'] > 0:
            selected_recipes['breakfast'].append(display_recipe_details(recipe_data, "Weighted Similarity", score, similarity_type="cosine"))
            meal_type_counts['breakfast'] -= 1
        elif 'lunch' in meal_types and meal_type_counts['lunch'] > 0:
            selected_recipes['lunch'].append(display_recipe_details(recipe_data, "Weighted Similarity", score, similarity_type="euclidean"))
            meal_type_counts['lunch'] -= 1
        elif 'dinner' in meal_types and meal_type_counts['dinner'] > 0:
            selected_recipes['dinner'].append(display_recipe_details(recipe_data, "Weighted Similarity", score, similarity_type="euclidean"))
            meal_type_counts['dinner'] -= 1

        # Stop when all required meal types are satisfied
        if meal_type_counts['breakfast'] == 0 and meal_type_counts['lunch'] == 0 and meal_type_counts['dinner'] == 0:
            break

    # Return the selected recipes as a JSON object
    return json.dumps(selected_recipes, indent=4, ensure_ascii=False)

# Example configuration for meal type counts
meal_type_counts = {'breakfast': 30, 'lunch': 30, 'dinner': 30}

# Fetch the user's profile and display recommended recipes based on user information
foodhak_user_id = "20cee875-b696-4a04-9ed2-182c66641f9b"
profile = get_user_profile(foodhak_user_id)
if profile:
    user_ingredients = profile["Recommended Ingredients"]
    user_info = profile["Meal Recommendation"]
    
    # Fetch all recipes data from OpenSearch
    loop = asyncio.get_event_loop()
    recipes_data = loop.run_until_complete(fetch_recipes_from_opensearch())
    if recipes_data:
        df_recipes = pd.DataFrame(recipes_data)
        
        # Display weighted recipes based on both cosine and Euclidean scores
        recipes_json = display_weighted_recipes(user_ingredients, df_recipes, user_info, meal_type_counts)
        print(recipes_json)
