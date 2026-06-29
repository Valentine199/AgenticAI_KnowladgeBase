from fastapi import FastAPI
from fastapi_mcp import FastApiMCP

app = FastAPI(title="Calculator API")

@app.post("/multiply")
def multiply(a: float, b: float):
    """Multiply two numbers
    
    args: a (float) The first number.
          b (float) The second number.

    returns: float: The product of the two numbers
    """

    result = a * b
    return {"result": result}

@app.post("/add")
def add(a: float, b: float):
    """add two numbers
    
    args: a (float) The first number.
          b (float) The second number.

    returns: float: The addition results of the two numbers
    """
    result = a + b
    return {"result": result}

@app.post("/subtract")
def subtract(a: float, b: float):
    """subtract two numbers
    
    args: a (float) The first number.
          b (float) The second number.

    returns: float: The differnece of a and b
    """
    result = a - b
    return {"result": result}

@app.post("/divide")
def divide(a: float, b: float):
    """Divide two numbers
    
    args: a (float) The first number.
          b (float) The second number.

    returns: float: the quotient of a and b
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")

    result = a / b
    return {"result": result}

# Convert into MCP
mcp = FastApiMCP(app, name="Calculator MCP")
mcp.mount_http()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8002)