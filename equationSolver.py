import re
import numpy as np
from sympy import symbols, Eq, sympify, parse_expr, S, Add, expand, Pow, Symbol, diff, lambdify
import math
from collections import defaultdict
import timeit 

# user input functions
def get_user_input():

    equations = []
    print("Enter each equation (press Enter on an empty line to stop):")

    while True:
        equation = input("Enter equation: ").strip()
        if not equation:
            break
        equations.append(equation)

    if len(equations) >= 1:
        return equations
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

# basic arithmetic
def if_basic_arithmetic(expression):

    pattern = r'^[\d\s\.\+\-\*\/\^\(\)]+$' # checks for any arithmetic expression
    return re.match(pattern, expression)

def solve_basic_arithmetic(expression):
    try:
        result = parse_expr(expression).evalf()
        result = round(float(result), 3)
    
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": f"Failed to evaluate the expression: {e}"}


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
    sympy_eqs = []

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
    
    return A_np, b_np, sympy_eqs

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
    A, b, sympy_eqs = system_to_ax_b(equations)  # Get augmented matrix
    
    # Collect variables from the equations
    variables = set().union(*[eq.free_symbols for eq in sympy_eqs])
    variables = sorted(variables, key=lambda x: str(x))

    print(f"A = {A}")
    print(f"B = {b}")

    answer, infiniteSols = solve_system(A, b)
    
    solution_dict = {
        "equations" : equations
    }


    if isinstance(answer, str):
        return {"error": answer}
    

    for i, var in enumerate(variables):
        solution_dict[str(var)] = round(answer[i].item(), 2)

    if infiniteSols:
        solution_dict["note"] = "This system has infinitely many solutions"

    return solution_dict
    

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
        x = symbols('x')

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

        result = {
            "equation" : str(equation)
        }

        # check feasibility via discriminant
        discriminant = pow(b,2) - (4*a*c)

        if discriminant > 0:

            x1 = (-b + math.sqrt(discriminant) / (2*a))
            x2 = (-b - math.sqrt(discriminant) / (2*a))

            result['solutions'] = {
                'x1' : round(x1, 2),
                'x2' : round(x2, 2)
            }

        
        elif discriminant == 0:
            x1 = (-b + math.sqrt(discriminant) / (2*a))

            result['solutions'] = {
                'x1' : round(x1, 2),
            }


        else:
            result['solutions'] = None
            result['note'] = "No real roots"

        return result



# newton rhapson stuff
def cauchy_bound(coefficients):
    # finds the upper bound a root can have

    if len(coefficients) == 0:
        return 1.0
    
    maxCoeff = max(abs(coeff) for coeff in coefficients [:-1])

    return maxCoeff + 1


def newton_raphson(f, df, guess, delta = 1e-10, maxIter = 1000):

    for _ in range(maxIter):
        fVal = f(guess)
        dfVal = df(guess)

        if dfVal == 0:
            raise ValueError("Deriviative is zero, no sol found")
        
        nextGuess = guess - fVal / dfVal

        if abs(np.complex128(nextGuess) - np.complex128(guess)) < delta:
            return nextGuess
        
        guess = nextGuess

    raise ValueError("Max num of iterations reached, no sol found")
        

def find_roots(coefficients, f, df, numPoints = 100, delta = 1e-10, maxIter = 100):
    
    bound = cauchy_bound(coefficients)
    realRange = np.linspace(-bound, bound, numPoints)
    imagRange = np.linspace(-bound, bound, numPoints)

    roots = []
    counter = 0

    for r in realRange:
        for i in imagRange:
            guess = complex(r, i)

            try:
                root = newton_raphson(f, df, guess, delta, maxIter)

                if root is not None:
                    if not any(abs(root - existingRoot) < delta for existingRoot in roots):
                        roots.append(root) # add root if unique 
            except Exception:
                continue

            counter += 1
            print(f"progress: {counter} / {counter*counter}")


    # check without this
    cleaned_roots = []
    for root in roots:
        # If imaginary part is very small, convert to real
        if abs(root.imag) < delta:
            cleaned_roots.append(round(root.real, 3))
        else:
            cleaned_roots.append(root)
    
    return cleaned_roots
    

def solve_polynomial(equation):
    x = symbols('x')

    print(equation)
    print(type(equation))


    # Assuming lhs_subtract_rhs is a function that rearranges the equation to standard form
    standardPolynomial = lhs_subtract_rhs(equation)

    # Get the coefficients dictionary
    variables_dict = standardPolynomial.as_coefficients_dict()
    print("Variables dict:", variables_dict)


    print(variables_dict.keys())

    # get highest order 
    highestPower = max(
        variables_dict.keys(),
        key=lambda k: k.as_base_exp()[1] if isinstance(k, Pow) else (1 if isinstance(k, Symbol) else 0)
    )



    _, exponent = highestPower.as_base_exp()

    # Length of the coefficients array
    length = exponent + 1
    print("Length of coefficients array:", length)

    # Initialize the coefficients array
    coefficients = [0] * length



    # fill in coefficients
    for term, coeff in variables_dict.items():
        print(f"Key: {term} (Type: {type(term).__name__}), Value: {coeff}")

        # extract components
        if isinstance(term, Pow):
            order = term.as_base_exp()[1]  
        elif isinstance(term, Symbol):
            order = 1  
        else:
            # handles the constant
            order = 0

        # Place the coefficient in the correct position
        coefficients[length - 1 - order] = coeff

    print("Coefficients array:", coefficients)

    f = lambdify(x, standardPolynomial, 'numpy')
    df = lambdify(x, diff(standardPolynomial), 'numpy')

    roots = find_roots(coefficients, f, df)


    return roots


#------------------------------------
# main

# get user input

# test = "(x+1)(x+2)"
# print(test)

# test_expr = sympify(test)
# expand_expr = expand(test_expr)
# print(expand_expr)



def solve_equation(equations):
    x, y, z = symbols('x y z')
    
    if len(equations) > 1:
        systemOfequations = True
    else:
        systemOfequations = False

        # basic sums
    if not systemOfequations and if_basic_arithmetic(equations[0]):
        return solve_basic_arithmetic(equations[0])

    equations = [preprocess_equation(equation) for equation in equations]




    if systemOfequations:
        return solve_systems_and_linear(equations)

    else:
        equationType = classify_equation(equations[0])
        print(equationType)

        if equationType == 'linear':
            return solve_systems_and_linear(equations)

        elif equationType == 'quadratic':
            equation = equations[0]
            return solve_quadratic(equation)
        
        elif equationType == 'polynomial':
            equation = equations[0]
            return solve_polynomial(equation)

    


if __name__ == "__main__":
    
    equations = get_user_input()

    start = timeit.timeit()
    result = solve_equation(equations)

    end = start = timeit.timeit()
    timeTaken = end - start

    print(result)
    print(timeTaken)