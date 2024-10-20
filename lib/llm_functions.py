
from concurrent.futures import ThreadPoolExecutor
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from .models import Meal, MealPlan, FoodListByMealPlan
from .foods import hybrid_search
from config.cache import diskcache


# set_llm_cache(SQLiteCache(database_path=".langchain.db"))

def get_llm(model_name):
    return ChatOpenAI(temperature=0.1, model=model_name)
# @diskcache
def _generate_response(final_prompt, response_model: BaseModel, model_name: str = "gpt-4o"):
    structured_llm = get_llm(model_name).with_structured_output(response_model, method="json_schema")     
    return structured_llm.invoke(final_prompt)

def generate_foods_list_by_meal_plan(user_profile: dict, caloric_profile: dict, zenith_instructions: str, user_meal_preferences: str, meal_preferences: dict):
    user_meal_preferences = user_meal_preferences if user_meal_preferences else "There are no user preferences."
    prompt_template = """You are a nutritionist (TheZenithMethod). Your task is to generate a list of foods for each meal, given the following:
    
[[# The user's profile#]]
{user_profile}

[[# The caloric profile goals are #]]    
{caloric_profile}

[[# The zenith instructions are #]]
{zenith_instructions}    

[[# The user's preferences are #]]    
{user_meal_preferences}

[[# The meal preferences are #]]
{meal_preferences}

**IMPORTANT GUIDELINES:**
1. Make sure you provide list of actual foods (NOT recipes) names which can be searched in the database.
2. Use Zenith instructions to come up with the list of foods, if user has not provided any preferences.
4. In order to create coherent meal plan, you may add/remove foods, even if not on the list.
5. *Calories Distribution* Make sure distribution of calories from protein, fats, and carbs match user caloric profile goals. This step is very important.
6. Before you generate the serving options, think step by step about how you would create a meal plan for the user given their goals and profile.
3. DO not create more meals than the user has provided in meal preferences.
7. Use reasoning format defined below, to come up with correct calories distribution for each meal.
<reasoning_format>
First, I will come up with calories distrution for each meal:
**Meal Name 1**: X Kcal
...
**Meal Name N**: X Kcal

Now, I will generate list of foods for each meal which match the calories distribution for that meal and will adjust serving sizes if needed.
**Meal Name 1**: 
1. Food 1: <serving size>
2. Food 2: <serving size>
...
**Meal Name N**: 
1. Food 1: <serving size>
2. Food 2: <serving size>
...
</reasoning_format>
Respond in JSON format.
"""
    prompt_template = PromptTemplate(input_variables=["user_profile", "caloric_profile", "zenith_instructions", "user_meal_preferences", "meal_preferences"], template=prompt_template)
    final_prompt = prompt_template.format(user_profile=user_profile, caloric_profile=caloric_profile, zenith_instructions=zenith_instructions, user_meal_preferences=user_meal_preferences, meal_preferences=meal_preferences)        
    food_list_by_meal_plan = _generate_response(final_prompt, FoodListByMealPlan)
    return food_list_by_meal_plan
    

def plan_meal(meal_template: dict):
    prompt_template = """You are an expert nutritionist. Your task is to generate meal from list of options from each food.

[[# Meal Template #]]
{meal_template}

Follow these guidelines:
1. Always match the target nutrients for the meal.
    - Select appropriate food from the list of provided foods
    - Review how close the target nutrients are to the provided foods
    - If target nutrients are not met, your can add/remove foods, change serving sizes, etc.        
    - IMPORTANT: It is not acceptable to have a meal that does not meet target nutrients for the meal, so update the serving size to match the target nutrients for the meal

2. use `serving_conversion_formula` to convert target serving to human consumable serving size.
    - Never use 100g as a serving size, always use human consumable portion size, and round it nearest whole number.
    - Always use `serving_conversion_formula` to convert target serving size to human consumable portion size

3. Generate a concise paragraph explaining how to consume the meal and which order. 
    - DO NOT give prepare instructions.
    - Use nicer short names for foods, instead of generic names.

4. Use the same meal name provided in the meal template.

5. Writ your reasoning in valid markdown format, for each step you take explaining why you selected the food and serving size.

Respond in JSON format.
    """
    prompt_template = PromptTemplate(input_variables=["meal_template"], template=prompt_template)    
    final_prompt = prompt_template.format(meal_template=meal_template)
    meal_plan = _generate_response(final_prompt, Meal)
    return meal_plan

def generate_meal_plan_using_food_data(foods_list_by_meal_plan: FoodListByMealPlan, user_profile: dict, caloric_profile: dict):
    meal_plan = []
    for meal in foods_list_by_meal_plan.meals:
        meal_foods = []
        for food in meal.foods:
            results = hybrid_search(food.name, top_k=3)
            meal_foods.append([
                {
                    'name': r['doc'].metadata['FOOD ITEM'],
                    'target_calories': food.serving,
                    'serving_conversion_formula': r['doc'].metadata['servings'].replace(':', '='),
                    'nutrients_per_serving': {
                        'serving': str(r['doc'].metadata['QUANTITY']) + ' ' + str(r['doc'].metadata['UNIT']), 
                        'calories': r['doc'].metadata['CALORIES'], 
                        'protein': r['doc'].metadata['PROTEIN'], 
                        'carbs': r['doc'].metadata['NET CARBS'], 
                        'fats': r['doc'].metadata['FATS']
                    }
                } 
                for r in results
            ])
        meal_plan.append({
            "meal_name": meal.meal_name,
            "foods": meal_foods,            
            'target_nutrients':{
                'calories': meal.target_calories,
                'protein': meal.target_protein,
                'fats': meal.target_fats,
                'carbs': meal.target_carbs
            }
        })

    full_meal_plan = []
    # use threads to process each meal
    with ThreadPoolExecutor(max_workers=len(meal_plan)) as executor:
        full_meal_plan = list(executor.map(plan_meal, meal_plan))
    
    return full_meal_plan
