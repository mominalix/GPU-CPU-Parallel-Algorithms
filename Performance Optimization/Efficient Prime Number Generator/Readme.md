# Efficient Prime Number Generator

This Python script implements the Sieve of Eratosthenes algorithm to efficiently generate prime numbers within a given range. It offers a user-friendly interface to input the upper limit for prime number generation.

## How to Run

1. **Prerequisites**: Make sure you have Python (3.x) installed on your system.

2. **Navigate to the Directory**:
   ```bash
   cd efficient-prime-generator
   ```

3. **Run the Script**:
   ```bash
   python prime_generator.py
   ```

4. **Input Upper Limit**: Enter a positive integer greater than 1 as the upper limit for prime number generation.

5. **Output**: The script will display a list of prime numbers within the specified range.

## Functionality

The script `prime_generator.py` performs the following steps:

1. Accepts user input for the upper limit of prime number generation.
2. Implements the Sieve of Eratosthenes algorithm to efficiently compute prime numbers up to the specified limit.
3. Displays the list of prime numbers within the given range.

The algorithm's efficiency comes from marking multiples of prime numbers as non-prime, significantly reducing the number of divisions required.
