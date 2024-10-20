from fasthtml.common import *
from shad4fast import *

def SearchFoods():
    return Div(cls="w-full mx-auto")(
        Div(cls="max-w-md mx-auto")(  # New wrapper for search input            
            Div(cls="mb-4")(            
                Div(cls="relative")(
                    Input(
                        id="food_search",
                        name="food_search",
                        placeholder="Start typing to search foods...",
                        cls="shadow appearance-none border rounded w-full py-2 px-3 leading-tight focus:outline-none focus:shadow-outline",
                        hx_get="/search_foods",
                        hx_trigger="keyup changed delay:500ms",
                        hx_target="#search_results",
                        hx_indicator="#search_indicator"
                    ),
                    # Add custom spinner
                    Div(
                        id="search_indicator",
                        cls="htmx-indicator absolute inset-y-0 right-0 flex items-center pr-3"
                    )(
                        Div(
                            cls="w-6 h-6 border-4 border-t-4 border-blue-500 rounded-full animate-spin"
                        )
                    )
                )
            )
        ),
        Div(id="search_results", cls="mt-4 w-full")  # Updated to take full width
    )

def HomeScreen():
    return Div(cls="flex flex-col w-full min-h-screen p-4")(
        Div(cls="text-center mb-8 w-full")(
            H1("Welcome to the Calorie Calculator", cls="text-4xl font-bold mb-4"),
            P("This is a simple calorie calculator that allows you to calculate the calories in your meals.", cls="text-lg mb-4"),
            Button("Try Now...", onclick="window.location.href='/calculate'", cls="mb-8")
        ),
        SearchFoods()
    )
