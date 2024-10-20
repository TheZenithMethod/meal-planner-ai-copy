
from typing import List
from pydantic import BaseModel, Field

meal_types = {
        'Breakfast': 'brown bread, omelette, tea or coffee',
        'Morning Snack': 'fruits salad, yogurt, nuts',
        'Lunch': 'beef steak, salad',
        'Afternoon Snack': 'fruits salad, nuts',
        'Dinner': 'chicken breast, salad, brown rice',
        'Evening Snack': 'apple'    
    }

class Meal(BaseModel):
    class Food(BaseModel):
        name: str
        serving: str
        calories: float
        protein: float
        carbs: float
        fats: float

    step_by_step_rationale: str
    meal_name: str    
    foods: list[Food]
    instructions: str

class MealPlan(BaseModel):
    step_by_step_rationale: str
    meals: list[Meal]


class MealServing(BaseModel):
    meal_name: str = Field(description="The name of the meal this serving belongs to e.g. 'Breakfast', 'Lunch', 'Dinner'")
    zenith_reasoning: str = Field(description="The reasoning for why this serving is optimal for the user based on the zenith method")
    food: List[str] = Field(description="Name of the food - a few words which can be searched for in a food database")
    target_calories: float = Field(description="The target calories for this serving")

class FoodListByMealPlan(BaseModel):
    
    class Meal(BaseModel):
        class Food(BaseModel):
            name: str = Field(description="Name of the food - a few words which can be searched for in a food database")
            serving: str = Field(description="Serving size of the food. e.g. 1 cup, 2 units, 0.5 piece")        

        meal_name: str
        foods: list[Food] = Field(description="List of foods for this meal")
        target_calories: str = Field(description="Target calories for this meal"),
        target_fats: str
        target_carbs: str
        target_protein: str        

    step_by_step_rationale: str = Field(description="Here is my step by step reasoning to create list of foods for each meal...")
    meals: list[Meal]
