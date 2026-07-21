import logging

def is_prime(n: int) -> bool:
    """
    Check if a number is prime.

    Args:
        n: The integer to check for primality

    Returns:
        True if n is prime, False otherwise

    Raises:
        TypeError: If n is not an integer
    """
    logging.debug(f"Checking if {n} is prime")

    if not isinstance(n, int):
        error_msg = f"Expected integer, got {type(n).__name__}: {n}"
        logging.error(error_msg)
        raise TypeError(error_msg)

    if n < 2:
        logging.debug(f"{n} is less than 2, not prime")
        return False

    if n == 2:
        return True

    if n % 2 == 0:
        return False

    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            logging.debug(f"{n} is divisible by {i}, not prime")
            return False

    logging.info(f"{n} is prime")
    return True