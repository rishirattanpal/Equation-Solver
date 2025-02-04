import re
import numpy as np
from sympy import symbols, Eq, solve, sympify, parse_expr, S, Add

def get_user_input():
    systemOfEquations = False

    equations = []
    print("Enter each equation (press Enter on an empty line to stop):")

    while True:
        equation = input("Enter equation: ").strip()
        if not equation:
            break
        equations.append(equation)

    if len(equations) == 1:
        return equations, systemOfEquations
    elif len(equations) > 1:
        systemOfEquations = True
        return equations, systemOfEquations
    else:
        return equations, "no equations entered"

def classify_equation(equation):
    if any(op in equation for op in ['<', '>', '<=', '>=']):
        return "inequality"

    elif '=' in equation:
        lhs, rhs = equation.split('=')

        lhs_sympy = sympify(lhs)
        #print(f"sympify = {lhs_sympy}")
        rhs_sympy = sympify(rhs)

        lhs_poly = lhs_sympy.as_poly()
        #print(f"poly = {lhs_poly}")
        rhs_poly = rhs_sympy.as_poly()

        lhs_degree = lhs_poly.degree() if lhs_poly is not None else 0
        rhs_degree = rhs_poly.degree() if rhs_poly is not None else 0

        degree = max(lhs_degree, rhs_degree)

        if degree == 1:
            return "linear"

        elif degree == 2:
            return "quadratic"

        else:
            return "polynomial"

def preprocess_equation(equation):

    equation = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', equation)
    return equation

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
    print(all_terms)
    
    # Separate variable terms and constant terms
    variables = all_terms.as_coefficients_dict()
    print(variables)
    constant_term = variables.pop(S.One, S.Zero)  # Extract the constant term
    print(constant_term)
    # Move the constant term to the RHS
    lhs_rearranged = Add(*[coeff * var for var, coeff in variables.items()])
    print(lhs_rearranged)
    rhs_rearranged = -constant_term
    print(rhs_rearranged)
    
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
    # Preprocess equations to handle implicit multiplication
    equations = [preprocess_equation(eq) for eq in equations]
    
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
    variables = sorted(sympy_eqs[0].free_symbols, key=lambda x: str(x))  # Sort variables for consistent order
    A = []
    b = []
    
    for eq in rearranged_eqs:
        row = [float(eq.lhs.coeff(var)) for var in variables]  # Extract coefficients in order
        A.append(row)
        b.append(float(eq.rhs))  # Extract constant term
    
    # Convert to NumPy arrays
    A_np = np.array(A, dtype=float)
    b_np = np.array(b, dtype=float)
    
    return A_np, b_np


# main
equations, systemOfequations = get_user_input()
equations = [preprocess_equation(equation) for equation in equations]


print(equations)
print(systemOfequations)



if systemOfequations == True:
    print("Implement systems method")
    A, b = system_to_ax_b(equations)
    print(f"A {A}")
    print(f"b {b}")

    

else:
    equationType = classify_equation(equations[0])
    print(equationType)




