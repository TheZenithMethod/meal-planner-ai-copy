from fasthtml.common import *
from pydantic import BaseModel
from shad4fast import *
from lib.models import FoodListByMealPlan, MealPlan, meal_types
import markdown
from fasthtml.components import Raw  # Add this import

class UserProfile(BaseModel):
    age: int
    gender: str
    height: float
    weight: float
    activity_level: str
    goal: str

def ProfileForm(user: UserProfile):
    gender = ['male', 'female']
    activity_level = ['Sedentary', 'Light', 'Moderate', 'Active', 'Very Active']
    weight_goal = ['Maintain Weight', 'Weight Loss', 'Fast Weight Loss', 'Weight Gain', 'Fast Weight Gain']
    # print(user)
    return Form(
        Div(cls="max-w-xl shadow-md rounded p-8")(
            P("Please fill in the following information to calculate your calories.", cls="mb-4"),
            # Age and Gender row
            Div(cls="flex flex-wrap -mx-2 mb-4")(
                Div(cls="w-1/2 px-2")(
                    Label("Age", For="age", cls="block text-sm font-bold mb-2"),
                    Input(id="age", name="age", type="number", placeholder="Enter your age", required=True, cls="shadow appearance-none border rounded w-full py-2 px-3 leading-tight focus:outline-none focus:shadow-outline", value=(user.age if user else ''))
                ),
                Div(cls="w-1/2 px-2")(
                    Label("Gender", For="gender", cls="block text-sm font-bold mb-2"),
                    Div(cls="flex space-x-4")(
                        *[Label(
                            Input(id=g, name="gender", type="radio", value=g, required=True, checked=(g == user.gender if user else False), cls="mr-2"),
                            g.capitalize(), cls="inline-flex items-center"
                        ) for g in gender]
                    )
                )
            ),
            # Height and Weight row
            Div(cls="flex flex-wrap -mx-2 mb-4")(
                Div(cls="w-1/2 px-2")(
                    Label("Height (cm)", For="height", cls="block text-sm font-bold mb-2"),
                    Input(id="height", name="height", type="number", step="0.1", placeholder="Enter your height in cm", required=True, cls="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline", value=(user.height if user else ''))
                ),
                Div(cls="w-1/2 px-2")(
                    Label("Weight (kg)", For="weight", cls="block text-sm font-bold mb-2"),
                    Input(id="weight", name="weight", type="number", step="0.1", placeholder="Enter your weight in kg", required=True, cls="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline", value=(user.weight if user else ''))
                )
            ),
            Div(cls="mb-4")(
                Label("Activity Level", For="activity_level", cls="block text-sm font-bold mb-2"),
                Select(
                    SelectTrigger(
                        SelectValue(placeholder="Choose your activity level"),
                        cls="w-full"
                    ),
                    SelectContent(
                        SelectGroup(
                            *[SelectItem(al, value=al.lower(), checked=str(al.lower() == user.activity_level.lower() if user else "").lower()) for al in activity_level]
                        ),
                        id='activity_level',
                    ),
                    standard=True,
                    id='activity_level',
                    name='activity_level',
                )
            ),
            Div(cls="mb-6")(
                Label("Goal", For="goal", cls="block text-sm font-bold mb-2"),
                Select(                            
                    SelectTrigger(
                        SelectValue(placeholder="Choose your goal"),
                        cls="w-full"
                    ),
                    SelectContent(
                        SelectGroup(
                            *[SelectItem(g, value=g.lower(), checked=str(g.lower() == user.goal.lower() if user else "").lower() ) for g in weight_goal]
                        ),
                        id='goal',
                    ),
                    state="opened",
                    standard=True,
                    id='goal',
                    name='goal',
                    required=True,
                )
            ),

            Div(cls="flex items-center justify-start")(
                Button("Calculate", type="submit"),
                A("Clear", href="/calculate?reset=1", cls="ml-4") if user else None
            ),

        ),
        method="post", action="/calculate"
    )

def CaloriesResult(calories: dict, user: UserProfile):
    if not calories:
        return Div(cls="mt-4 w-full")(
            P("Please fill in the form to calculate your calories.")
        )
    return Div(cls="w-full")(    
        Div(cls="flex flex-wrap -mx-2")(
            # Left column: Caloric needs and nutrients
            Div(cls="w-full md:w-1/2 px-2 mb-4")(
                Div(cls="border-l-4 border-green-500 bg-green-100 p-4 h-full shadow-md")(
                    P(f"{calories['total_calories']} KCal - Daily intake", cls="text-3xl font-bold mb-4"),
                    Div(cls="grid grid-cols-3 gap-4")(
                        Div(cls="text-center p-3 border-l-4 border-red-500 bg-red-100")(
                            P("Protein", cls="text-sm font-bold"),
                            P(f"{calories['protein']}g", cls="text-xl font-bold")
                        ),                        
                        Div(cls="text-center p-3 border-l-4 border-blue-500 bg-blue-100")(
                            P("Carbs", cls="text-sm font-medium"),
                            P(f"{calories['carbs']}g", cls="text-xl font-bold")
                        ),
                        Div(cls="text-center p-3 border-l-4 border-orange-500 bg-orange-100")(
                            P("Fats", cls="text-sm font-medium"),
                            P(f"{calories['fats']}g", cls="text-xl font-bold")
                        ),
                    )
                )
            ),
            # Right column: User profile
            Div(cls="w-full md:w-1/2 px-2 mb-4")(
                Div(cls="border-l-4 border-purple-500 bg-purple-100 p-4 h-full shadow-md")(
                    Div(cls="flex justify-between items-center")(
                        H3("Your Profile", cls="text-lg font-semibold mb-4"),
                        A("Edit", href="/calculate?reset=1", cls="text-blue-500 underline")
                    ),
                    Div(cls="grid grid-cols-3 gap-4")(
                        Div(cls="flex flex-col")(
                            Span("Age", cls="text-gray-700 font-bold"),
                            Span(f"{user.age} years", cls="text-gray-900")
                        ),
                        Div(cls="flex flex-col")(
                            Span("Gender", cls="text-gray-700 font-bold"),
                            Span(user.gender.capitalize(), cls="text-gray-900")
                        ),
                        Div(cls="flex flex-col")(
                            Span("Height", cls="text-gray-700 font-bold"),
                            Span(f"{user.height} cm", cls="text-gray-900")
                        ),
                        Div(cls="flex flex-col")(
                            Span("Weight", cls="text-gray-700 font-bold"),
                            Span(f"{user.weight} kg", cls="text-gray-900")
                        ),
                        Div(cls="flex flex-col")(
                            Span("Activity Level", cls="text-gray-700 font-bold"),
                            Span(user.activity_level.capitalize(), cls="text-gray-900")
                        ),
                        Div(cls="flex flex-col")(
                            Span("Goal", cls="text-gray-700 font-bold"),
                            Span(user.goal.capitalize(), cls="text-gray-900")
                        )
                    )
                )
            )
        )
    )

def MealPreferences(meal_preferences: dict):
    if not meal_preferences:
        meal_preferences = meal_types

    return Div(cls="mt-4 w-full")(
        H5("Please provide your preferences for the meal plan:", cls="text-lg font-bold mb-4"),
        Form(cls="", method="post", action="/calculate?mealplan=1")(
            Div(cls="grid grid-cols-3 gap-4")(
                *[Div(cls="")(
                    Label(mt.replace('_', ' ').title(), For=f"{mt.lower().replace(' ', '_')}_preferences", cls="block text-md font-bold mb-2"),
                    Div(cls="relative")(
                        Input(
                            id=f"{mt.lower().replace(' ', '_')}_preferences",
                            name=f"meal_preferences[{mt.lower().replace(' ', '_')}]",
                            cls="shadow appearance-none border rounded w-full py-2 px-3 leading-tight focus:outline-none focus:shadow-outline",
                            value=meal_preferences[mt],
                            oninput="this.nextElementSibling.style.display = this.value ? 'flex' : 'none';"
                        ),
                        Span(
                            "✖",
                            cls="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-500 cursor-pointer",
                            onclick="this.previousElementSibling.value=''; this.style.display='none';"
                        )                    
                    )
                ) for mt in meal_preferences.keys()]
            ),
            Div(cls="mt-4 text-right")(
                A("Reset Preferences", href="/calculate?mealplan=0", cls="mr-4"),
                Button("Generate Meal Plan", type="submit", cls="font-bold py-2 px-4 rounded", id="generate-meal-plan-btn", onclick="this.disabled=true; this.textContent='Please wait...'; this.form.submit();")
            ),
        )
    )

def BuildMealPlan(meal_plan: MealPlan):
    return Div(id="mealplan_blueprint", cls="mt-4 w-full")(        
        *[Div(cls="mb-6 shadow-md rounded-lg overflow-hidden")(
            Div(cls="bg-primary p-3 flex justify-between items-center border-b")(
                P(meal.meal_name, cls="text-lg font-bold text-secondary"),
                Div(cls="text-sm text-secondary flex")(
                    Span("Serving", cls="w-24 text-right"),
                    Span("Protein", cls="w-24 text-right"),
                    Span("Carbs", cls="w-24 text-right"),
                    Span("Fats", cls="w-24 text-right"),
                    Span("Calories", cls="w-24 text-right"),
                )
            ),

            # foods per meal
            *[Div(cls="flex justify-between items-center p-3 border-b last:border-b-0")(
                Div(f"{food.name}", cls="flex-grow"),
                Div(cls="flex")(
                    Span(f"{food.serving}", cls="w-50 text-right"),
                    Span(f"{round(food.protein)}", cls="w-24 text-right"),
                    Span(f"{round(food.carbs)}", cls="w-24 text-right"),
                    Span(f"{round(food.fats)}", cls="w-24 text-right"),
                    Span(f"{round(food.calories)}", cls="w-24 text-right"),
                )
            ) for food in meal.foods],
            Div(cls="text-sm p-3")(
                NotStr(markdown.markdown(meal.instructions)),
                # NotStr(markdown.markdown(meal.step_by_step_rationale))
            ),
            # total per meal
            Div(cls="flex justify-between items-center p-2 font-bold")(
                Div(cls="w-2/6")(
                    Input(type="text", placeholder=f"Search foods for {meal.meal_name}...", cls="w-full border rounded")
                ),
                Div(cls="flex text-sm")(
                    Span(f"{round(sum(food.protein for food in meal.foods))}", cls="w-24 text-right"),
                    Span(f"{round(sum(food.carbs for food in meal.foods))}", cls="w-24 text-right"),
                    Span(f"{round(sum(food.fats for food in meal.foods))}", cls="w-24 text-right"),
                    Span(f"{round(sum(food.calories for food in meal.foods))}", cls="w-24 text-right"),
                )
            )
        ) for meal in meal_plan],

        # show total for macronutrients from all meals
        Div(cls="flex justify-between items-center p-2 font-bold")(
            Div(cls="w-2/6")(
                P("Total", cls="text-lg font-bold")
            ),
            Div(cls="flex text-sm")(
                Span(f"{round(sum(sum(food.protein for food in meal.foods) for meal in meal_plan))}g", cls="w-24 text-right"),
                Span(f"{round(sum(sum(food.carbs for food in meal.foods) for meal in meal_plan))}g", cls="w-24 text-right"),
                Span(f"{round(sum(sum(food.fats for food in meal.foods) for meal in meal_plan))}g", cls="w-24 text-right"),
                Span(f"{round(sum(sum(food.calories for food in meal.foods) for meal in meal_plan))} KCal", cls="w-24 text-right"),
            )
        )
    )

