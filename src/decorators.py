import time

def measure_time(base_function):

    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = base_function(*args, **kwargs)
        t2 = time.time()
        print(f"Function {base_function.__name__} run in {t2-t1}s")
        return result
    return wrapper
