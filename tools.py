def calculator(a: float, b: float, operation: str):
    """
    Perform a basic mathematical calculation.
    """

    if operation == "add":
        return a + b

    elif operation == "subtract":
        return a - b

    elif operation == "multiply":
        return a * b

    elif operation == "divide":
        if b == 0:
            return "Cannot divide by zero."

        return a / b

    else:
        return "Invalid operation."