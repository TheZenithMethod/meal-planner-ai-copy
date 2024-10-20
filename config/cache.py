import diskcache as dc
from functools import wraps

# Initialize a disk cache in the 'cache_dir' folder
cache = dc.Cache('.cache')

def diskcache(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create a cache key based on function arguments
        cache_key = (func.__name__, args, frozenset(kwargs.items()))
        
        # Check if the result is in the cache
        if cache_key in cache:
            print("Cache hit")
            return cache[cache_key]
        
        # Call the actual function and store the result in cache
        print("Cache miss")
        result = func(*args, **kwargs)
        cache[cache_key] = result
        return result
    
    return wrapper