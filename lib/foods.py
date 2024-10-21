import dotenv
dotenv.load_dotenv(override=True)


import os
import pprint
import sys
import datetime
import shutil
from typing import List
import pandas as pd
import numpy as np
import time
from pydantic import BaseModel, Field
import tiktoken
import asyncio
from tenacity import retry, stop_after_attempt, wait_random_exponential
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document, HumanMessage, SystemMessage
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.retrievers import BM25Retriever
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss  # Import faiss directly
from tqdm.asyncio import tqdm
import concurrent.futures
import threading
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from config.cache import diskcache

import json
from tabulate import tabulate

# Constants
DATA_FOLDER = "data"
INDEX_PATH = os.path.join(DATA_FOLDER, "faiss_index")
CSV_PATH = os.path.join(DATA_FOLDER, "food_data.csv")
ENRICHED_CSV_PATH = os.path.join(DATA_FOLDER, "food_data_enriched.csv")
CHAT_MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL_NAME = "text-embedding-3-large"
CACHE_DIR = "cache"
CACHE_FILE = os.path.join(CACHE_DIR, "enriched_descriptions_cache.json")

# Initialize models
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
chat_model = ChatOpenAI(model=CHAT_MODEL_NAME)

# Define evaluation prompt
evaluation_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a nutrition expert evaluating food search results."),
    HumanMessage(content="""
    Query: "{query}"
    Search Results:
    {results}

    Please evaluate the relevance of these search results to the query based on the following criteria:
    1. **Nutritional Content Match:** How well do the macronutrient profiles (protein, carbs, fats) align with the query?
    2. **Food Category Relevance:** Are the food categories appropriate for the query?
    3. **Ingredient Similarity:** Do the ingredients match or complement the query?
    4. **Use Case Alignment:** Are the foods suitable for the intended use case (e.g., breakfast, snack)?

    For each result, provide:
    - A brief comment on its relevance.
    - A relevance score from 1 to 10, where 10 is perfectly relevant.

    Finally, provide an overall evaluation and suggestions for improving the search results.
    """)
])

# Function to calculate the number of tokens in a text
def calculate_tokens(text):
    encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL_NAME)
    return len(encoding.encode(text))

# Function to normalize embeddings
def normalize_embeddings(embeddings_array):
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    return embeddings_array / norms

# Function to rename old index
def rename_old_index():
    if os.path.exists(INDEX_PATH):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        new_path = f"{INDEX_PATH}_{timestamp}"
        shutil.move(INDEX_PATH, new_path)
        print(f"Renamed old index to: {new_path}")

class EnrichedDescription(BaseModel):
    nice_name: str
    description: str
    ingredients: str
    nutritional_highlights: str
    use_cases: str
    servings: str = Field(default=[], description="Human readable serving sizes and units", examples=["1 cup:230 g", "1 tablespoon:15 g"])

# Function to load cache
def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

# Function to save cache
def save_cache(cache):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)

# Load cache at module level
enriched_descriptions_cache = load_cache()

def generate_enriched_description(food_item, category):
    cache_key = f"{food_item}_{category}"
    
    if cache_key in enriched_descriptions_cache:
        return EnrichedDescription(**enriched_descriptions_cache[cache_key])

    # If not in cache, generate the description
    prompt = f"""
    Given the food item "{food_item}" in the category "{category}", please provide:
    - A brief description of the food.
    - Common ingredients.
    - Nutritional highlights.
    - Typical use cases (e.g., ideal for breakfast, snack).
    - Human readable serving size and units.

    Each description should be 1 sentence.
    """

    parser = PydanticOutputParser(pydantic_object=EnrichedDescription)
    format_instructions = parser.get_format_instructions()

    prompt_template = ChatPromptTemplate.from_template(
        template="{prompt}\n\n{format_instructions}"
    )

    messages = prompt_template.format_messages(
        prompt=prompt,
        format_instructions=format_instructions
    )

    response = chat_model.invoke(messages)
    parsed_response = parser.parse(response.content)

    # Cache the result
    enriched_descriptions_cache[cache_key] = parsed_response.model_dump()
    save_cache(enriched_descriptions_cache)

    return parsed_response

# Create a thread-safe counter
class Counter:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()

    def increment(self):
        with self.lock:
            self.value += 1
        return self.value

def process_row(row, counter, pbar):
    food_item = row['FOOD ITEM']
    category = row['CATEGORY']
    enriched_description = generate_enriched_description(food_item, category)
    counter.increment()
    pbar.update(1)
    return {**row.to_dict(), **enriched_description.model_dump()}

def enrich_database(df, max_threads=20):
    # Filter out rows with empty 'FOOD ITEM' column
    df = df[df['FOOD ITEM'].notna() & df['FOOD ITEM'].str.strip().astype(bool)]
    
    total_tasks = len(df)
    enriched_descriptions = []
    counter = Counter()    
    with tqdm(total=total_tasks, desc="Enriching rows") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = [executor.submit(process_row, row, counter, pbar) for _, row in df.iterrows()]
            for future in concurrent.futures.as_completed(futures):
                enriched_descriptions.append(future.result())
    
    return pd.DataFrame(enriched_descriptions)

def enrich_and_save_database(df):    
    print("Enriching the database using LLM...")
    df_enriched = enrich_database(df)
    df_enriched.to_csv(ENRICHED_CSV_PATH, index=False)
    print(f"Enriched data saved to {ENRICHED_CSV_PATH}")
    return df_enriched

# Function to create sparse index
def create_sparse_index(df):    
    metadatas = df.to_dict(orient="records")
    texts = df['combined_text'].tolist()
    retriever = BM25Retriever.from_documents([
        Document(page_content=text, metadata=metadata) for text, metadata in zip(texts, metadatas)
    ])
    return retriever

# Function to create FAISS index and save embeddings
def create_food_embeddings(df):
    rename_old_index()
    df["description"] = (
        df["CATEGORY"].fillna("").astype(str) + ", " + 
        df["FOOD ITEM"].fillna("").astype(str) + ", " + 
        df["BRAND NAME"].fillna("").astype(str)
    )
    df['combined_text'] = df['description'] + ' ' + "\nNice Name: " + df['nice_name'].fillna('') + "\nServing Size: " + df['servings'].fillna('') + "\nIngredients: " + df['ingredients'].fillna('') + "\nNutritional Highlights: " + df['nutritional_highlights'].fillna('') + "\nUse Cases: " + df['use_cases'].fillna('')
    texts = df['combined_text'].tolist()
    metadatas = df.to_dict(orient="records")

    # Generate embeddings
    print("Creating embeddings...")
    embeddings_list = embeddings.embed_documents(texts)

    # Normalize embeddings
    embeddings_array = normalize_embeddings(np.array(embeddings_list).astype('float32'))

    # Create FAISS index
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_array)

    # Create Document instances
    from langchain.schema import Document
    docs = [Document(page_content=texts[i], metadata=metadatas[i]) for i in range(len(texts))]

    # Build vector store
    print("Building FAISS index...")
    vector_store = FAISS(
        index=index,
        embedding_function=embeddings,
        docstore=InMemoryDocstore({str(i): docs[i] for i in range(len(docs))}),
        index_to_docstore_id={i: str(i) for i in range(len(docs))}
    )

    # Save the index
    vector_store.save_local(INDEX_PATH)
    print(f"Embeddings created and saved for {len(df)} food items.")

# Function to load FAISS index
def load_food_embeddings():
    try:
        if not os.path.exists(INDEX_PATH):
            print(f"Index file not found at {INDEX_PATH}")
            return None

        print(f"Loading index from {INDEX_PATH}")
        vector_store = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        print("Index loaded successfully")
        return vector_store
    except Exception as e:
        print(f"Error loading index: {str(e)}")
        return None

def load_sparse_index():
    df_enriched = pd.read_csv(ENRICHED_CSV_PATH)
    df_enriched['combined_text'] = df_enriched['description'] + ' ' + df_enriched['nice_name'].fillna('') + "\nServing Size: " + df_enriched['servings'].fillna('') + "\nIngredients: " + df_enriched['ingredients'].fillna('') + "\nNutritional Highlights: " + df_enriched['nutritional_highlights'].fillna('') + "\nUse Cases: " + df_enriched['use_cases'].fillna('')
    return create_sparse_index(df_enriched)

food_vector_store = load_food_embeddings()
sparse_retriever = load_sparse_index()

# Function to perform hybrid search
@diskcache
def hybrid_search(query, top_k=10, alpha=0.8):    
    # Dense search
    dense_results = food_vector_store.similarity_search_with_score(query, k=top_k)
    dense_docs = [res[0] for res in dense_results]
    dense_scores = [res[1] for res in dense_results]

    # Sparse search using the 'retrieve' method
    sparse_results = sparse_retriever.invoke(query,config={"verbose":True})
    # pprint.pprint(sparse_results)
    # Limit to top_k results
    sparse_results = sparse_results[:top_k]
    # Assign default scores since BM25Retriever may not provide them
    sparse_scores = [1.0 for _ in sparse_results]

    # Normalize scores
    normalized_dense_scores = min_max_normalize(dense_scores)
    normalized_sparse_scores = min_max_normalize(sparse_scores)    
    # Combine results
    sparse_results = []
    combined = []
    for doc, score in zip(dense_docs, normalized_dense_scores):
        combined.append({'doc': doc, 'score': alpha * score, 'source': 'dense'})
    for doc, score in zip(sparse_results, normalized_sparse_scores):
        combined.append({'doc': doc, 'score': (1 - alpha) * score, 'source': 'sparse'})

    # Sort combined results
    combined_sorted = sorted(combined, key=lambda x: x['score'], reverse=True)
    return combined_sorted[:top_k]

# Function to normalize scores using min-max scaling
def min_max_normalize(scores):
    min_score = min(scores)
    max_score = max(scores)
    if max_score - min_score == 0:
        return [0.5] * len(scores)  # Avoid division by zero
    return [(score - min_score) / (max_score - min_score) for score in scores]

# Retry decorator for OpenAI API calls
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def evaluate_search_results(query, results):
    results_str = ''
    for res in results:
        doc = res['doc']
        score = res['score']
        food_item = doc.metadata.get('FOOD ITEM', 'Unknown')
        category = doc.metadata.get('CATEGORY', 'Unknown')
        result_str = f"- {food_item} ({category}) [Score: {score:.2f}]"
        results_str += result_str + "\n"

    formatted_prompt = evaluation_prompt.format_messages(query=query, results=results_str)
    response = chat_model(formatted_prompt)
    return response.content.strip()

# Function to run tests
def run_tests():
    print("Running search test cases with LLM evaluation...")

    test_keywords = [
        "apple fruit",
        "vegetarian protein",
        "spicy mexican",
        "low calorie snack",
        "gluten free breakfast"
    ]

    vector_store = load_food_embeddings()
    if vector_store is None:
        print("Failed to load vector store. Unable to perform search.")
        return

    df_enriched = pd.read_csv(ENRICHED_CSV_PATH)
    df_enriched['combined_text'] = df_enriched['description'] + ' ' + df_enriched['Enriched Description'].fillna('')
    sparse_retriever = create_sparse_index(df_enriched)

    for keywords in test_keywords:
        print(f"\nSearching for: '{keywords}'")
        try:
            results = hybrid_search(keywords, vector_store, sparse_retriever, top_k=10, alpha=0.5)
            if not results:
                print("No results found.")
                continue

            print("Top 10 relevant foods:")
            for res in results:
                doc = res['doc']
                score = res['score']
                source = res['source']
                food_item = doc.metadata.get('FOOD ITEM', 'Unknown')
                category = doc.metadata.get('CATEGORY', 'Unknown')
                print(f"- {food_item} ({category}) [Score: {score:.2f}, Source: {source}]")

            # LLM Evaluation
            print("\nLLM Evaluation:")
            try:
                evaluation = evaluate_search_results(keywords, results)
                print(evaluation)
            except Exception as e:
                print(f"Error during LLM evaluation: {str(e)}")

        except Exception as e:
            print(f"Error occurred during search: {str(e)}")

    print("\nSearch tests with LLM evaluation completed.")

# Function to calculate cost and time per embedding using a sample
def calculate_cost_and_time(df):
    df["description"] = df["CATEGORY"].fillna("") + ", " + df["FOOD ITEM"].fillna("") + ", " + df["BRAND NAME"].fillna("") 
    df['combined_text'] = df['description'] + "\nNice Name: " + df['nice_name'].fillna("") + "\nServing Size: " + df['servings'].fillna("") + "\nIngredients: " + df['ingredients'].fillna("") + "\nNutritional Highlights: " + df['nutritional_highlights'].fillna("") + "\nUse Cases: " + df['use_cases'].fillna("")

    total_tokens = df['combined_text'].apply(calculate_tokens).sum()
    cost_per_1k_tokens = 0.0004  # $0.0004 per 1K tokens for embeddings
    total_cost = (total_tokens / 1_000) * cost_per_1k_tokens

    print(f"Total tokens for embeddings: {total_tokens}")
    print(f"Estimated total cost for embeddings: ${total_cost:.4f}")

    return total_cost

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run_tests":
        run_tests()
    elif len(sys.argv) > 1 and sys.argv[1] == "search":        
        vector_store = load_food_embeddings()
        if vector_store is None:
            print("Failed to load vector store. Unable to perform search.")
            sys.exit(1)

        # skip first row
        df_enriched = pd.read_csv(ENRICHED_CSV_PATH)
        df_enriched = df_enriched.iloc[1:]
        df_enriched['combined_text'] = df_enriched['description'] + "\nNice Name: " + df_enriched['nice_name'].fillna("") + "\nServing Size: " + df_enriched['servings'].fillna("") + "\nIngredients: " + df_enriched['ingredients'].fillna("") + "\nNutritional Highlights: " + df_enriched['nutritional_highlights'].fillna("") + "\nUse Cases: " + df_enriched['use_cases'].fillna("")
        print("Loading sparse index...")
        sparse_retriever = create_sparse_index(df_enriched)
        print("Sparse index loaded successfully")
        
        while True:
            try:
                keywords = input("Enter search keywords (or type 'exit' to quit): ")
                if keywords.lower() == 'exit':
                    break
            # catch if ctrl+c is pressed
            except KeyboardInterrupt:
                print("Search interrupted by user.")
                break
            except Exception as e:
                print(f"Error occurred during search: {str(e)}")

            results = hybrid_search(keywords, vector_store, sparse_retriever, top_k=20, alpha=0.8)
            if not results:
                print("No results found.")
                continue

            table_data = []
            for res in results:
                doc = res['doc']
                # Use get with a default value of '-' to handle NaN or missing values
                food_item = doc.metadata.get('FOOD ITEM', '-') or '-'
                category = doc.metadata.get('CATEGORY', '-') if doc.metadata.get('BRAND NAME') or doc.metadata.get('BRAND NAME') == 'nan' else '-'
                quantity = doc.metadata.get('QUANTITY', '-') or '-'
                unit = doc.metadata.get('UNIT', '-') or '-'
                protein = doc.metadata.get('PROTEIN', '-') or '-'
                net_carbs = doc.metadata.get('NET CARBS', '-') or '-'
                dietary_fibre = doc.metadata.get('DIETARY FIBRE', '-') or '-'
                total_sugars = doc.metadata.get('TOTAL SUGARS', '-') or '-'
                fats = doc.metadata.get('FATS', '-') or '-'
                calories = doc.metadata.get('CALORIES', '-') or '-'

                table_data.append([category, food_item, quantity, unit, protein, net_carbs, dietary_fibre, total_sugars, fats, calories])

            headers = ["CATEGORY", "FOOD ITEM", "QUANTITY", "UNIT", "PROTEIN", "NET CARBS", "DIETARY FIBRE", "TOTAL SUGARS", "FATS", "CALORIES"]
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
    else:
        # Enrich and save the database
        df = pd.read_csv(CSV_PATH)
        # get first 1000 rows
        # df = df.head(1000)
        df_enriched = enrich_and_save_database(df)

        # Calculate cost per embedding
        total_cost = calculate_cost_and_time(df_enriched)

        num_descriptions = len(df_enriched)
        print(f"Warning: This process will create embeddings for {num_descriptions} food items.")
        print(f"Estimated cost: ${total_cost:.4f}")
        user_input = input("Do you want to proceed? (yes/no): ")
        if user_input.lower() == "yes":
            # Create and save FAISS index
            start_time = time.time()
            create_food_embeddings(df_enriched)
            end_time = time.time()
            print(f"Embeddings created and saved successfully in {end_time - start_time:.2f} seconds.")
        else:
            print("Operation aborted by the user.")
