import json
from fasthtml.common import *
import pandas as pd
from shad4fast import *

from lib.foods import ENRICHED_CSV_PATH, create_sparse_index, hybrid_search, load_food_embeddings
from lib.functions import calculate_calories_for_user
from lib.llm_functions import generate_foods_list_by_meal_plan, generate_meal_plan_using_food_data
from screens.home import HomeScreen
from screens.layout import MainLayout
from screens.calculator.profile import BuildMealPlan, CaloriesResult, MealPreferences, ProfileForm, UserProfile
from openinference.instrumentation.langchain import LangChainInstrumentor
from lib.models import meal_types
from phoenix.otel import register
from pprint import pprint

# Script(src='https://cdn.tailwindcss.com'),
    
app, rt = fast_app(
    live=True, 
    pico=False,
    hdrs=(ShadHead(tw_cdn=True))
)
rt = app.route


vector_store = load_food_embeddings()
if vector_store is None:
    print("Failed to load vector store. Unable to perform search.")
    sys.exit(1)

# skip first row
# df_enriched = pd.read_csv(ENRICHED_CSV_PATH)
# df_enriched = df_enriched.iloc[1:]
# df_enriched['combined_text'] = df_enriched['description'] + "\nNice Name: " + df_enriched['nice_name'].fillna("") + "\nServing Size: " + df_enriched['servings'].fillna("") + "\nIngredients: " + df_enriched['ingredients'].fillna("") + "\nNutritional Highlights: " + df_enriched['nutritional_highlights'].fillna("") + "\nUse Cases: " + df_enriched['use_cases'].fillna("")
# print("Loading sparse index...")
# sparse_retriever = create_sparse_index(df_enriched)

tracer_provider = register(project_name="zenith-method")
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

@rt("/")
def landing_page():
    return MainLayout(None,
        Div(cls="flex flex-col items-center h-screen mt-10")(
            HomeScreen()
        )    
    )

foods_per_meal = [
    {
        "meal_name": "Breakfast",
        "foods": [
            
        ]
    },
    {
        "meal_name": "Lunch",
        "foods": [
            
        ]
    },
    {
        "meal_name": "Dinner",
        "foods": [
            
        ]
    }
]

@rt("/calculate", methods=["get", "post"])
async def profile(session, request: Request = None):    
    if request and request.method == "GET" and request.query_params.get("reset"):
        session.pop('user', None)    

    user = None
    meal_preferences = None
    user_meal_preferences = None
    calories = None
    meal_plan = None
    has_user_profile = False
    if request and request.method == "POST" and not request.query_params.get("mealplan"):
        form_data = await request.form()
        session['user'] = dict(form_data)
        user = UserProfile(**form_data)
        has_user_profile = True
    elif 'user' in session:
        user = UserProfile(**session['user'])
        has_user_profile = True
    
    if has_user_profile:
        calories = calculate_calories_for_user(user)        
    
    zenith_instructions = "Begin your day with a nutrient-dense breakfast, focusing on protein and fiber-rich carbs; for lunch, include a variety of vegetables, lean protein, and whole grains; at dinner, opt for lighter portions of lean protein and vegetables, minimizing heavy carbs; hydrate well, and snack on nuts, fruits, or yogurt when necessary"    
    serving_options = None
    if request and request.method == "POST" and request.query_params.get("mealplan"):        
        form_data = await request.form()
        user_meal_preferences = form_data.get("user_meal_preferences")
        # Initialize the meal preferences dictionary
        meal_preferences = {}

        # Loop over the form data to capture meal_preferences fields
        for key, value in form_data.items():
            if key.startswith("meal_preferences["):
                # Extract the field name inside the brackets
                meal_type = key[key.index("[") + 1: key.index("]")]
                if value:
                    meal_preferences[meal_type] = value
        
        print("Meal Preferences:", meal_preferences)  # Debug print
        foods_list_by_meal_plan = generate_foods_list_by_meal_plan(user, caloric_profile=calories, zenith_instructions=zenith_instructions, user_meal_preferences=user_meal_preferences, meal_preferences=meal_preferences)
        # pprint(foods_list_by_meal_plan.dict(), sort_dicts=False, indent=2)
        meal_plan = generate_meal_plan_using_food_data(foods_list_by_meal_plan, user, calories)

    return MainLayout("Calculate Calories",                        
        Div(cls="flex flex-col md:flex-row")(            
            Div(cls="w-full pl-0 md:pl-4")(
                CaloriesResult(calories, user),
                MealPreferences(meal_preferences),
                BuildMealPlan(meal_plan) if meal_plan else None
            ) if has_user_profile 
            else Div(cls="w-full pr-0 md:pr-4")(
                ProfileForm(user)
            )            
        )
    )

@app.get("/search_foods")
async def search_foods(request: Request, food_search: str):    
    results = hybrid_search(food_search, top_k=20, alpha=0.8)
    
    table_data = []
    for res in results:
        doc = res['doc']
        # Use get with a default value of '-' to handle NaN or missing values
        food_item = doc.metadata.get('FOOD ITEM', '-') or '-'
        brand = doc.metadata.get('BRAND NAME', '-') if doc.metadata.get('BRAND NAME') or doc.metadata.get('BRAND NAME') == 'nan' else '-'
        quantity = doc.metadata.get('QUANTITY', '-') or '-'
        unit = doc.metadata.get('UNIT', '-') or '-'
        quantity_unit = f"{quantity} {unit}".strip()
        calories = doc.metadata.get('CALORIES', '-') or '-'
        protein = doc.metadata.get('PROTEIN', '-') or '-'
        net_carbs = doc.metadata.get('NET CARBS', '-') or '-'
        dietary_fibre = doc.metadata.get('DIETARY FIBRE', '-') or '-'
        total_sugars = doc.metadata.get('TOTAL SUGARS', '-') or '-'
        fats = doc.metadata.get('FATS', '-') or '-'

        table_data.append([brand, food_item, quantity_unit, calories, protein, net_carbs, dietary_fibre, total_sugars, fats])

    headers = ["CATEGORY", "FOOD ITEM", "QUANTITY", "CALORIES", "PROTEIN", "NET CARBS", "DIETARY FIBRE", "TOTAL SUGARS", "FATS"]
    
    return Table(
        TableHeader(
            TableRow(
                *[TableHead(header, cls="bg-gray-100") for header in headers]
            )
        ),
        TableBody(
            *[
                TableRow(
                    *[TableCell(data) for data in row]
                ) for row in table_data
            ]
        )
    )

serve()
