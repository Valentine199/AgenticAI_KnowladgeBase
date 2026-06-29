from fastmcp import FastMCP

mcp = FastMCP(name = "Calculator")

@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers
    
    args: a (float) The first number.
          b (float) The second number.

    returns: float: The product of the two numbers
    """
    return a * b

@mcp.tool(
        name = "add",
        description = "add two numbers",
        tags = {"math", "arithmetic"})
def add(a: float, b: float) -> float:
    """add two numbers
    
    args: a (float) The first number.
          b (float) The second number.

    returns: float: The addition results of the two numbers
    """
    return a + b

@mcp.tool()
def subtract(a: float, b: float) -> float:
    """subtract two numbers
    
    args: a (float) The first number.
          b (float) The second number.

    returns: float: The differnece of a and b
    """
    return a - b

@mcp.tool()
def division(a: float, b: float) -> float:
    """Divide two numbers
    
    args: a (float) The first number.
          b (float) The second number.

    returns: float: a divided by b
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    

    return a / b

if __name__ == "__main__":
    mcp.run(transport="http", host="localhost", port =8003)  # HTTP  
    # sas 
    