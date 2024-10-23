# Recipe Recommendation Project

## Overview
This project provides a recipe recommendation system based on precomputed FAISS (Facebook AI Similarity Search) embeddings using cosine similarity and eucledian distance. The system uses a dataset of 40k recipes to recommend personalized meal options for users, which are categorized into different meal types (e.g., breakfast, lunch, dinner).

## Project Structure

The project is organized into two main folders:

### 1. Data Folder
The `Data` folder contains the following files:

- **`recipe_faiss_ivf_index.index`**: Precomputed FAISS index file generated from the recipe dataset for similarity-based recipe searching.
- **`recipe_metadata.pkl`**: Metadata file containing relevant details for each recipe, such as title, ingredients, and meal type.
- **`recipes.csv`**: CSV file containing the raw dataset of 40k recipes, with information such as recipe names, ingredients, and meal types.
- **Note: Install faiss-cpu or faiss-gpu on a conda environment
  
### 2. Scripts Folder
The `Scripts` folder includes the following Python scripts:

- **`Precomputation-FAISS.py`**: This script precomputes the FAISS index based on the recipes dataset (`recipes.csv`). It creates an efficient similarity-based index and stores it in the `Data/recipe_faiss_ivf_index.index` file.
  
- **`Recipe-Recommendation.py`**: This script takes inputs in the form of a `foodhak-userid` and meal type counts (e.g., number of breakfasts, lunches, and dinners) and outputs recipe recommendations in JSON format for each meal type.
- **`Recipe-Recommendation-V2.py`**: This script is the updated version of the `Recipe-Recommendation.py`, this script returns the recipe-id along with the recipes.
