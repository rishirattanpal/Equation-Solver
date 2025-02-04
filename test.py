import numpy as np
from sympy import symbols, Eq, Matrix, parse_expr, Add, S

def rearrange_to_ax_b(eq):
    """
    Rearranges a linear equation into the form Ax = b.
    
    Args:
        eq (sympy.Eq): The equation to rearrange.
    
    Returns:
        sympy.Eq: The equation in the form Ax = b.
    """
    lhs = eq.lhs
    rhs = eq.rhs
    
    # Move all terms to the LHS
    all_terms = lhs - rhs
    
    # Separate variable terms and constant terms
    variables = all_terms.as_coefficients_dict()
    constant_term = variables.pop(S.One, S.Zero)  # Extract the constant term
    
    # Move the constant term to the RHS
    lhs_rearranged = Add(*[coeff * var for var, coeff in variables.items()])
    rhs_rearranged = -constant_term
    
    return Eq(lhs_rearranged, rhs_rearranged)

def system_to_ax_b(equations):
    """
    Converts a system of equations into the form Ax = b.
    
    Args:
        equations (list of str): The system of equations as strings.
    
    Returns:
        A (numpy.ndarray): Coefficient matrix.
        b (numpy.ndarray): Constant vector.
    """
    # Parse the strings into sympy equations
    sympy_eqs = []
    for eq_str in equations:
        lhs, rhs = eq_str.split('=')
        lhs_expr = parse_expr(lhs.strip())
        rhs_expr = parse_expr(rhs.strip())
        sympy_eqs.append(Eq(lhs_expr, rhs_expr))
    
    # Rearrange each equation
    rearranged_eqs = [rearrange_to_ax_b(eq) for eq in sympy_eqs]
    
    # Extract coefficients and constants
    variables = list(sympy_eqs[0].free_symbols)  # Get all variables
    A = []
    b = []
    
    for eq in rearranged_eqs:
        row = [float(eq.lhs.coeff(var)) for var in variables]  # Convert to float for NumPy
        A.append(row)
        b.append(float(eq.rhs))  # Convert to float for NumPy
    
    # Convert to NumPy arrays
    A_np = np.array(A, dtype=float)
    b_np = np.array(b, dtype=float)
    
    return A_np, b_np

# Example input
equations = ['x = 0.5 * y', 'y = 0.5 * x + 3']

# Convert to Ax = b
A, b = system_to_ax_b(equations)

print("Coefficient Matrix A:")
print(A)
print("Constant Vector b:")
print(b)