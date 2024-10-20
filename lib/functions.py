# Function to adjust food portions to meet meal's calorie target
from screens.calculator.profile import UserProfile


def adjust_portions(meal_data):
    # Get the meal target based on its distribution percentage
    total_calories = meal_data['target']['calories']
    distribution_percent = float(meal_data['meals'][0]['distribution'].strip('%')) / 100
    meal_calories_target = total_calories * distribution_percent
    
    # Macronutrient targets for the meal
    target_protein = meal_data['target']['protein'] * distribution_percent
    target_carbs = meal_data['target']['carbs'] * distribution_percent
    target_fats = meal_data['target']['fats'] * distribution_percent
    
    # Initialize totals for current meal macros and calories
    current_protein = 0
    current_carbs = 0
    current_fats = 0
    current_calories = 0
    
    # Iterate through the foods and calculate current totals
    for food in meal_data['meals'][0]['foods']:
        protein = food['protein']
        carbs = food['carbs']
        fats = food['fats']
        
        # Calculate the calories for each food
        calories = 4 * protein + 4 * carbs + 9 * fats
        
        # Add to the total
        current_protein += protein
        current_carbs += carbs
        current_fats += fats
        current_calories += calories
    
    # Calculate the scaling factor
    scaling_factor = meal_calories_target / current_calories
    
    # Adjust portions
    adjusted_foods = []
    for food in meal_data['meals'][0]['foods']:
        adjusted_food = food.copy()
        adjusted_food['quantity'] = round(float(food['quantity']) * scaling_factor, 2)
        adjusted_food['protein'] = round(food['protein'] * scaling_factor, 2)
        adjusted_food['carbs'] = round(food['carbs'] * scaling_factor, 2)
        adjusted_food['fats'] = round(food['fats'] * scaling_factor, 2)
        adjusted_foods.append(adjusted_food)
    
    return {
        'meal_name': meal_data['meals'][0]['name'],
        'target_calories': meal_calories_target,
        'scaled_foods': adjusted_foods,
        'scaled_total_protein': round(current_protein * scaling_factor, 2),
        'scaled_total_carbs': round(current_carbs * scaling_factor, 2),
        'scaled_total_fats': round(current_fats * scaling_factor, 2),
        'scaled_total_calories': round(current_calories * scaling_factor, 2)
    }


def calculate_calories_for_user(user: UserProfile) -> dict:
    # Implementation of the Zenith formula
    # Step 1 - BMR Calculation
    if user.gender.lower() == "male":
        bmr = 10 * user.weight + 6.25 * user.height - 5 * user.age + 5
    else:
        bmr = 10 * user.weight + 6.25 * user.height - 5 * user.age - 161

    # Step 2 - Total Calories
    activity_factors = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very active": 1.9
    }
    calories = bmr * activity_factors[user.activity_level.lower()]

    # Step 3 - Adjust Calorie As per Goal
    goal_adjustments = {
        "maintain weight": 0,
        "weight loss": -500,
        "fast weight loss": -1000,
        "weight gain": 500,
        "fast weight gain": 1000
    }
    adjusted_calories = calories + goal_adjustments[user.goal.lower()]

    # Step 4 - Calculate Macronutrients
    protein = user.weight * 2  # 2g per kg of total body weight
    protein_calories = protein * 4

    if user.activity_level.lower() in ["active", "very active"] and adjusted_calories > 2500:
        fats_calories = 0.2 * adjusted_calories
    else:
        fats_calories = 0.15 * adjusted_calories
    fats = fats_calories / 9

    carbs_calories = adjusted_calories - protein_calories - fats_calories
    carbs = carbs_calories / 4

    return {
        "total_calories": round(adjusted_calories),
        "protein": round(protein),
        "fats": round(fats),
        "carbs": round(carbs)
    }