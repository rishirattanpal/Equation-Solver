import re
import numpy as np
from sympy import symbols, Eq, sympify, parse_expr, S, Add, expand
import math

# user input functions
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

    equation = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', equation) # convert 2x to x**2
    equation = re.sub(r'\^', r'**', equation)

    equation = re.sub(r'\)\s*\(', r')*(', equation)

    lhs, rhs = equation.split('=')

    lhs_expr = sympify(lhs.strip())
    rhs_expr = sympify(rhs.strip())

    # Step 6: Expand LHS and RHS
    lhs_expr = expand(lhs_expr)
    rhs_expr = expand(rhs_expr)

    # Step 7: Reconstruct the equation as a string
    expanded_equation = f"{lhs_expr} = {rhs_expr}"

    return expanded_equation

# solving sim/linear equations functions
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
    for eq_str in equations:
        lhs, rhs = eq_str.split('=')
        lhs_expr = parse_expr(lhs.strip())
        rhs_expr = parse_expr(rhs.strip())
        sympy_eqs.append(Eq(lhs_expr, rhs_expr))
    print(f"sympy_eqs: {sympy_eqs}")
    
    # Rearrange each equation
    rearranged_eqs = [rearrange_to_ax_b(eq) for eq in sympy_eqs]
    print(f"rearranged_eqs {rearranged_eqs}")
    
    # collect vars and keep in a consistent order
    all_symbols = set()
    for eq in sympy_eqs:
        all_symbols |= eq.free_symbols
    variables = sorted(all_symbols, key=lambda x: str(x))  # Sort variables for consistent order
    print(f"variables {variables}")

    # Extract coefficients and constants
    A = []
    b = []
    
    for eq in rearranged_eqs:
        row = [float(eq.lhs.coeff(var)) for var in variables]  # Extract coefficients in order
        A.append(row)
        b.append(float(eq.rhs))  # Extract constant term
    
    # Convert to NumPy arrays
    A_np = np.array(A, dtype=float)
    b_np = np.array(b, dtype=float).reshape(-1, 1)
    
    return A_np, b_np

def row_reduction(A):
# =============================================================================
# A is a NumPy array that represents an augmented matrix of dimension n x (n+1)
# associated with a linear system.  RowReduction returns B, a NumPy array that
# represents the row echelon form of A.  RowReduction may not return correct
# results if the the matrix A does not have a pivot in each column.
# =============================================================================
   
    m = A.shape[0]  # A has m rows 
    n = A.shape[1]  # It is assumed that A has m+1 columns
    
    B = np.copy(A).astype('float64')

    # For each step of elimination, we find a suitable pivot, move it into
    # position and create zeros for all entries below.
    
    for k in range(m):
        # Set pivot as (k,k) entry
        pivot = B[k][k]
        pivot_row = k
        
        # Find a suitable pivot if the (k,k) entry is zero
        while(pivot == 0 and pivot_row < m-1):
            pivot_row += 1
            pivot = B[pivot_row][k]
            
        # Swap row if needed
        if (pivot_row != k):
            B = row_swap(B,k,pivot_row)
            
        # If pivot is nonzero, carry on with elimination in column k
        if (pivot != 0):
            B = row_scale(B,k,1./B[k][k])
            for i in range(k+1,m):    
                B = row_add(B,k,i,-B[i][k])
        else:
            print("Pivot could not be found in column",k,".")

    return B

def back_substitution(U,B):
# =============================================================================
#     U is a NumPy array that represents an upper triangular square mxm matrix.  
#     B is a NumPy array that represents an mx1 vector     
#     BackSubstitution will return an mx1 vector that is the solution of the
#     system UX=B.
# =============================================================================
    m = U.shape[0]  # m is number of rows and columns in U
    X = np.zeros((m,1))
    
    for i in range(m-1,-1,-1):  # Calculate entries of X backward from m-1 to 0
        X[i] = B[i]
        for j in range(i+1,m):
            X[i] -= U[i][j]*X[j]
        if (U[i][i] != 0):
            X[i] /= U[i][i]
        else:
            print("Zero entry found in U pivot position",i,".")
    return X

def solve_system(A,B):
    # =============================================================================
    # A is a NumPy array that represents a matrix of dimension n x n.
    # B is a NumPy array that represents a matrix of dimension n x 1.
    # SolveSystem returns a NumPy array of dimension n x 1 such that AX = B.
    # If the system AX = B does not have a unique solution, SolveSystem may not
    # generate correct results.
    # =============================================================================

    infiniteSolutions = False

    # Check shape of A
    if (A.shape[0] != A.shape[1]):
        print("SolveSystem accepts only square arrays.")
        return
    n = A.shape[0]  # n is number of rows and columns in A
    
    # 1. Join A and B to make the augmented matrix
    A_augmented = np.hstack((A,B))

    # 2. Carry out elimination    
    R = row_reduction(A_augmented)

    print(f"reduced matrix {R}")

    #feasibility = check_system_feasibility(R)

    # check feasibility
    for row in R:
        # no solutions
        if all(c == 0 for c in row[:-1]) and row[-1] != 0:
            return "System has no solution (inconsistent system).", infiniteSolutions
        # infinite solutions
        if all(c == 0 for c in row[:-1]) and row[-1] == 0:
            infiniteSolutions = True


    # 3. Split R back to nxn piece and nx1 piece
    B_reduced = R[:,n:n+1]
    A_reduced = R[:,0:n]

    # 4. Do back substitution
    X = back_substitution(A_reduced,B_reduced)
    return X, infiniteSolutions

def solve_systems_and_linear(equations):
    A, b = system_to_ax_b(equations) # get augmented matrix
    
    variables = set().union(*[eq.free_symbols for eq in sympy_eqs])
    variables = sorted(variables, key=lambda x: str(x))

    answer, infiniteSols = solve_system(A, b)
    print(infiniteSols)

    if isinstance(answer, str):
        print(answer)
        return answer
    elif infiniteSols == True:
        print("This is a system with infinite solutions")
        for i, var in enumerate(variables):
            print(f"{var} = {answer[i].item():.2f}")
        
        return answer

    else:
        for i, var in enumerate(variables):
            print(f"{var} = {answer[i].item():.2f}")

        return answer


# matrix manipulation functions
def row_swap(A,k,l):
# =============================================================================
#     A is a NumPy array.  RowSwap will return duplicate array with rows
#     k and l swapped.
# =============================================================================
    m = A.shape[0]  # m is number of rows in A
    n = A.shape[1]  # n is number of columns in A
    
    B = np.copy(A).astype('float64')
        
    for j in range(n):
        temp = B[k][j]
        B[k][j] = B[l][j]
        B[l][j] = temp
        
    return B

def row_scale(A,k,scale):
# =============================================================================
#     A is a NumPy array.  RowScale will return duplicate array with the
#     entries of row k multiplied by scale.
# =============================================================================
    m = A.shape[0]  # m is number of rows in A
    n = A.shape[1]  # n is number of columns in A
    
    B = np.copy(A).astype('float64')

    for j in range(n):
        B[k][j] *= scale
        
    return B

def row_add(A,k,l,scale):
# =============================================================================
#     A is a numpy array.  RowAdd will return duplicate array with row
#     l modifed.  The new values will be the old values of row l added to 
#     the values of row k, multiplied by scale.
# =============================================================================
    m = A.shape[0]  # m is number of rows in A
    n = A.shape[1]  # n is number of columns in A
    
    B = np.copy(A).astype('float64')
        
    for j in range(n):
        B[l][j] += B[k][j]*scale
        
    return B

# quadratic functions
def lhs_subtract_rhs(eq):
    lhs, rhs = eq.split('=')

    lhs_expr = parse_expr(lhs.strip())
    rhs_expr = parse_expr(rhs.strip())

    standard_lhs = lhs_expr - rhs_expr


    return standard_lhs

def solve_quadratic(equation):
        standard_quadratic = lhs_subtract_rhs(equation)
        print(standard_quadratic)

        variables_dict = standard_quadratic.as_coefficients_dict()

        # quadratic formula = (-b±√(b²-4ac))/(2a)
        a = variables_dict[x**2]
        b = variables_dict[x]
        c = variables_dict[1]
        print(a)
        print(b)
        print(c)

        # check feasibility via discriminant
        discriminant = pow(b,2) - (4*a*c)

        if discriminant > 0:

            # give the exact form
            x1_1 = f"(-{b} + {math.sqrt(pow(b, 2) - (4 * a * c))} / {(2*a)})"
            x2_1 = f"(-{b} - {math.sqrt(pow(b, 2) - (4 * a * c))} / {(2*a)})"

            print(f"x = {x1_1}")
            print(f"x = {x2_1}")

            # decimal form
            x1_2 = (-b + math.sqrt(pow(b, 2) - (4 * a * c))) / (2 * a)
            x2_2 = (-b - math.sqrt(pow(b, 2) - (4 * a * c))) / (2 * a)

            print(f"x = {x1_2}")
            print(f"x = {x2_2}")

            return x1_2, x2_2
        
        elif discriminant == 0:
            x1_1 = f"({-b} + {math.sqrt(pow(b, 2) - (4 * a * c))} / {(2*a)})"
            print(f"x = {x1_1}")

            x1_2 = (-b + math.sqrt(pow(b, 2) - (4 * a * c))) / (2 * a)
            print(f"x = {x1_2}")

            return x1_2

        else:
            print('infeasible quadratic - no real roots')
            return None


#------------------------------------
# main

# get user input
x, y, z = symbols('x y z')

# test = "(x+1)(x+2)"
# print(test)

# test_expr = sympify(test)
# expand_expr = expand(test_expr)
# print(expand_expr)


equations, systemOfequations = get_user_input()
equations = [preprocess_equation(equation) for equation in equations]

print(equations)
print(systemOfequations)



if systemOfequations == True:
    sympy_eqs = []
    solve_systems_and_linear(equations)

else:
    equationType = classify_equation(equations[0])
    print(equationType)

    if equationType == 'linear':
        sympy_eqs = []
        solve_systems_and_linear(equations)

    elif equationType == 'quadratic':
        equation = equations[0]
        solve_quadratic(equation)




 








