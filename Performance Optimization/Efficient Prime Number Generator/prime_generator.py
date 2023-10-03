#Sieve of Eratosthenes Function
def sieve_of_eratosthenes(limit):
    primes = []
    is_prime = [True] * (limit + 1)
    
    for num in range(2, int(limit**0.5) + 1):
        if is_prime[num]:
            primes.append(num)
            for multiple in range(num*num, limit + 1, num):
                is_prime[multiple] = False
    
    for num in range(int(limit**0.5) + 1, limit + 1):
        if is_prime[num]:
            primes.append(num)
    
    return primes

def main():
    print("Welcome to Prime Number Generator!")
    try:
        upper_limit = int(input("Enter the upper limit for prime numbers: "))
        if upper_limit < 2:
            print("Please enter a valid upper limit greater than 1.")
            return
        
        prime_list = sieve_of_eratosthenes(upper_limit)
        print(f"Prime numbers up to {upper_limit}:")
        print(prime_list)
        
    except ValueError:
        print("Invalid input. Please enter a valid integer.")
    
if __name__ == "__main__":
    main()
