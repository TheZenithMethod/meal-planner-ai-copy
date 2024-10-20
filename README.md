# FastHTML Calorie Calculator

This is a simple web application built with FastHTML that calculates daily caloric intake based on user input.

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install fasthtml
   ```

## Running the application

To run the application, use the following command:

```
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

## API Documentation

FastAPI automatically generates API documentation. You can access it at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
