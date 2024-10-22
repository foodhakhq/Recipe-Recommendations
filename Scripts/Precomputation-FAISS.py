import numpy as np
import pandas as pd
import faiss
import pickle
import torch
from transformers import DistilBertTokenizer, DistilBertModel

# Load your fine-tuned FoodBERT model and tokenizer
model = DistilBertModel.from_pretrained('/home/ai/Recipe Recommendation/FoodBERT Embeddings/Weights2/fine_tuned_distilbert_model')
tokenizer = DistilBertTokenizer.from_pretrained('/home/ai/Recipe Recommendation/FoodBERT Embeddings/Weights2/fine_tuned_distilbert_tokenizer')

# Function to get the embedding for a given ingredient using FoodBERT
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)  # Averaging token embeddings to get a single vector representation
    return embedding.detach().numpy()

file_path = '/home/ai/Recipe Recommendation/Recipes.csv'
df = pd.read_csv(file_path, delimiter=',', encoding='ISO-8859-1')
df.columns = df.columns.str.strip()
df['ingredient'] = df['ingredient'].fillna('')

# Limit the DataFrame to the first 1000 rows
#df = df.head(1000)

recipe_vectors = []
recipe_metadata = []

for index, row in df.iterrows():
    ingredients = str(row['ingredient']) 
    metadata = {
        "recipe_description": row["recipe_description"],
        "recipe_url": row["recipe_url"],
        "food_title": row["food_title"],
        "mealType": row["mealType"],
        "servings": row["servings"],
        "ready_in_minutes": row["ready_in_minutes"],
        "image_url": row["image_url"],
        "detailed_ingredients": row["detailed_ingredients"],
        "ingredient": row["ingredient"],
        "Energy (KCAL)": row["Energy (KCAL)"],
        "Total Fat (G)": row["Total Fat (G)"],
        "Saturated Fat (G)": row["Saturated Fat (G)"],
        "Cholesterol (MG)": row["Cholesterol (MG)"],
        "Sodium Na (MG)": row["Sodium Na (MG)"],
        "Total Carbohydrate (G)": row["Total Carbohydrate (G)"],
        "Dietary Fiber (G)": row["Dietary Fiber (G)"],
        "Protein (G)": row["Protein (G)"],
        "Vitamin C (MG)": row["Vitamin C (MG)"],
        "Calcium (MG)": row["Calcium (MG)"],
        "Iron (MG)": row["Iron (MG)"],
        "Potassium K (MG)": row["Potassium K (MG)"],
        "cautions": row["cautions"],
        "diet_labels": row["diet_labels"],
        "average_rating": row["average_rating"],
        "total_ratings": row["total_ratings"],
    }

    # Get embedding for the ingredient list, split by comma
    if ingredients.strip():  # Ensure the ingredient string is not empty
        ingredient_embeddings = [get_embedding(ingredient.strip()) for ingredient in ingredients.split(',') if ingredient.strip()]
        ingredient_embeddings = [emb for emb in ingredient_embeddings if emb is not None]  # Filter out None values

        if ingredient_embeddings:  # Only compute if there are valid embeddings
            recipe_vector = np.mean(ingredient_embeddings, axis=0)  # Average ingredient vectors
            recipe_vectors.append(recipe_vector)
            recipe_metadata.append(metadata)

recipe_vectors = np.array(recipe_vectors).astype('float32')

# Ensure recipe_vectors is a 2D array with shape (n_samples, n_features)
if len(recipe_vectors.shape) != 2:
    recipe_vectors = recipe_vectors.reshape(-1, recipe_vectors.shape[-1])

# Normalize vectors
faiss.normalize_L2(recipe_vectors)

# Clustering and creating the inverted index
dimension = recipe_vectors.shape[1]  # Dimensionality of the vectors
nlist = 500  # Number of clusters (this can be tuned based on your dataset size)

# Create an IVF index
quantizer = faiss.IndexFlatIP(dimension)  # Quantizer for clustering
index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

# Train the index with your vectors
index.train(recipe_vectors)

# Add the vectors to the index
index.add(recipe_vectors)

# Save the FAISS index
faiss.write_index(index, 'recipe_faiss_ivf_index.index')

# Save metadata
with open('recipe_metadata.pkl', 'wb') as f_metadata:
    pickle.dump(recipe_metadata, f_metadata)

print("Recipe embeddings and metadata have been precomputed and saved.")
