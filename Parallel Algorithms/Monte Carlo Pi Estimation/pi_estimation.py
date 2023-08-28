import random
import threading

num_threads = 4
total_points = 1000000
points_inside_circle = 0
lock = threading.Lock()

def monte_carlo_pi(num_points):
    global points_inside_circle

    inside_circle = 0

    for _ in range(num_points):
        x = random.random()
        y = random.random()

        if x * x + y * y <= 1.0:
            inside_circle += 1

    with lock:
        points_inside_circle += inside_circle

def main():
    threads = []
    points_per_thread = total_points // num_threads

    for _ in range(num_threads):
        thread = threading.Thread(target=monte_carlo_pi, args=(points_per_thread,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    pi_estimate = 4.0 * points_inside_circle / total_points
    print("Estimated value of π:", pi_estimate)

if __name__ == "__main__":
    main()
