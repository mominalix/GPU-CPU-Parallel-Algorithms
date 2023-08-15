import time
import cProfile

def slow_function():
    total = 0
    for i in range(1000000):
        total += i
    return total

def main():
    print("Starting performance profiling...")
    
    start_time = time.time()
    slow_function()
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"Time taken by slow_function: {elapsed_time:.6f} seconds")
    
    print("\nProfiling with cProfile:")
    profiler = cProfile.Profile()
    profiler.enable()
    
    slow_function()
    
    profiler.disable()
    profiler.print_stats(sort='cumulative')

if __name__ == "__main__":
    main()
