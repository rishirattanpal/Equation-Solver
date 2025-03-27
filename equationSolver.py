import re
import numpy as np
from sympy import symbols, Eq, sympify, parse_expr, S, Add, expand, Pow, Symbol, diff, lambdify
import math
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

# operator stuff
OPERATORS = ["<=", ">=", ">", "<", "="]

def get_operator(equation):
    for op in OPERATORS:
        if op in equation:
            return op
    else:
        return None
    

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

    # check operator
    op = get_operator(equation)
    if op is None:
        raise ValueError("No equation entered")
    


    # preprocess symbols
    equation = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', equation) 
    equation = re.sub(r'\^', r'**', equation)
    equation = re.sub(r'\)\s*\(', r')*(', equation)

    lhs, rhs = [part.strip() for part in equation.split(op, 1)]

    lhsExpr = sympify(lhs)
    rhsExpr = sympify(rhs)

    # Step 6: Expand LHS and RHS
    lhsExpr = expand(lhsExpr)
    rhsExpr = expand(rhsExpr)

    # Step 7: Reconstruct the equation as a string
    expanded_equation = f"{lhsExpr} {op} {rhsExpr}"

    return expanded_equation

# basic arithmetic
def if_basic_arithmetic(expression):

    pattern = r'^[\d\s\.\+\-\*\/\^\(\)]+$' # checks for any arithmetic expression
    return re.match(pattern, expression)

def solve_basic_arithmetic(expression):
    try:
        result = sympify(expression).evalf()
        result = round(float(result), 3)
    
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": f"Failed to evaluate the expression: {e}"}


# solving sim/linear equations functions
def rearrange_to_ax_b(eq):
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
 
    m = A.shape[0]  
    n = A.shape[1]  
    
    B = np.copy(A).astype('float64')

    #find a suitable pivot, move it into position and create zeros for all entries below.
    
    for k in range(m):
        # Set pivot 
        pivot = B[k][k]
        pivot_row = k
        
        # Find a suitable pivot if the (k,k) entry is zero
        while(pivot == 0 and pivot_row < m-1):
            pivot_row += 1
            pivot = B[pivot_row][k]
            
        
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
            return "System has no solution", infiniteSolutions
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
    
    result = {
        "equation" : equations
    }
    print(f"result: {result}")


    if isinstance(answer, str):
        result["solutions"] = answer
        return result
    

    result["solutions"] = {str(var): round(answer[i].item(), 2) for i, var in enumerate(variables)}
    
    if infiniteSols:
        result["note"] = "System has infinite solutions"
    
    operators = [get_operator(eq) for eq in equations]
    result["operators"] = operators

    return result
    


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

            x1 = round(float(((-b + math.sqrt(discriminant)) / (2*a))), 3)
            print(x1)
            x2 = round(float(((-b - math.sqrt(discriminant)) / (2*a))), 3)
            print(x2)

            result['solutions'] = [x1, x2]

        
        elif discriminant == 0:
            x=[]
            x1 = round(((-b) / (2*a)), 3)
            x.append(x1)
            
            result['solutions'] = x

        else:
            result['solutions'] = "No real roots"
            result['note'] = "No real roots"

        return result



# newton rhapson stuff
def clean_roots(roots, tol = 1e-3):
    print("cleaning roots")
    cleaned = []
    seen = set()

    for root in roots:
        if root == -0.0:
            root = 0
        
        if abs(root.imag) < tol:
            root = root.real
    
        if isinstance(root, complex):
            rounded = complex(round(root.real, 3), round(root.imag,3))
        else:
            rounded = round(root.real, 3)

        root_str = str(rounded)

        if root_str not in seen:
            seen.add(root_str)
            cleaned.append(rounded)
        
        print(cleaned)

    return cleaned

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
        

def find_roots(coefficients, f, df, numPoints = 50, delta = 1e-10, maxIter = 100):
    
    bound = cauchy_bound(coefficients)
    realRange = np.linspace(float(-bound), float(bound), numPoints)
    imagRange = np.linspace(-float(bound), float(bound), numPoints)

    roots = []
    counter = 0

    for r in realRange:
        for i in imagRange:
            guess = complex(r, i)

            try:
                root = newton_raphson(f, df, guess, delta, maxIter)

                if root is not None:
                    if not any(abs(root - existingRoot) < delta for existingRoot in roots):
                        print(root)
                        roots.append(root) # add root if unique 
            except Exception:
                continue

            counter += 1
            #print(f"progress: {counter} / {counter*counter}")


    # # check without this
    # cleaned_roots = []
    # for root in roots:
    #     # If imaginary part is very small, convert to real
    #     if abs(root.imag) < delta:
    #         cleaned_roots.append(round(root.real, 3))
    #     else:
    #         cleaned_roots.append(root)

    cleanedRoots = clean_roots(roots)
    
    return cleanedRoots
    

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

    result = {
        "equation" : str(equation),
        "solutions" : roots,
        "note" : f"Polynomial of degree {exponent}"
    }


    return result



# graph stuff
def show_graph(equations):
    plt.figure(figsize=(6, 3))
    
    # Setup axes and grid
    plt.axhline(y=0, color='w', linewidth=0.5)
    plt.axvline(x=0, color='w', linewidth=0.5)
    plt.grid(True)
    
    # Color cycle
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    
    # Plot range
    x_vals = np.linspace(-10, 10, 1000)
    
    for i, equation in enumerate(equations):
        color = colors[i % len(colors)]
        
        try:
            # Preprocess equation
            eq = preprocess_equation(equation)
            
            # Handle special cases first
            if handle_constant_equation(eq, color):
                continue
                
            # Handle y = f(x) equations
            if handle_y_equation(eq, x_vals, color, equation):
                continue
                
            # Handle general f(x) = g(x) case
            handle_general_equation(eq, x_vals, color)
            
        except Exception as e:
            print(f"Could not plot {equation}: {e}")
            continue
    
    # Finalize graph
    plt.title("Graph of Equations")
    plt.xlabel("x")
    plt.ylabel("y")
    adjust_axes_limits()
    
    plt.savefig("static/graph.png", dpi=111, bbox_inches='tight')
    plt.close()
    
    return True


def handle_constant_equation(eq, color):
    """Handle equations of form x/y/z = constant"""
    lhs, rhs = eq.split('=')
    lhs, rhs = lhs.strip(), rhs.strip()
    
    # Check if right side is a number
    if not rhs.lstrip('-').replace('.', '').isdigit():
        return False
    
    constant = float(rhs)
    
    # x = constant (vertical line)
    if lhs == 'x':
        plt.axvline(x=constant, color=color, label=f'x = {constant}')
        return True
        
    # y/z = constant (horizontal line)
    if lhs in ('y', 'z'):
        plt.axhline(y=constant, color=color, label=f'{lhs} = {constant}')
        return True
        
    # Also check if constant is on left side
    if rhs == 'x' and lhs.lstrip('-').replace('.', '').isdigit():
        plt.axvline(x=float(lhs), color=color, label=f'x = {lhs}')
        return True
        
    if rhs in ('y', 'z') and lhs.lstrip('-').replace('.', '').isdigit():
        plt.axhline(y=float(lhs), color=color, label=f'{rhs} = {lhs}')
        return True
        
    return False

def handle_y_equation(eq, x_vals, color, original_eq):
    """Handle equations explicitly solving for y"""
    lhs, rhs = eq.split('=')
    lhs, rhs = lhs.strip(), rhs.strip()
    
    if 'y' not in eq:
        return False
    
    # Determine which side has y
    y_expr = rhs if lhs == 'y' else lhs
    
    # Evaluate for all x values
    y_points = []
    for x_val in x_vals:
        try:
            y_val = eval(y_expr.replace('x', f'({x_val})'), {'math': math})
            y_points.append(y_val)
        except:
            y_points.append(np.nan)
    
    plt.plot(x_vals, y_points, color=color, label=original_eq)
    return True

def handle_general_equation(eq, x_vals, color):
    """Handle general f(x) = g(x) case"""
    lhs, rhs = eq.split('=')
    lhs, rhs = lhs.strip(), rhs.strip()
    
    lhs_y, rhs_y = [], []
    
    for x_val in x_vals:
        try:
            lhs_val = eval(lhs.replace('x', f'({x_val})'), {'math': math})
            rhs_val = eval(rhs.replace('x', f'({x_val})'), {'math': math})
            lhs_y.append(lhs_val)
            rhs_y.append(rhs_val)
        except:
            lhs_y.append(np.nan)
            rhs_y.append(np.nan)
    
    # Plot LHS
    plt.plot(x_vals, lhs_y, color=color, linestyle='-', label=f'{lhs} (LHS)')
    
    # Find and plot intersections
    diffs = np.array(lhs_y) - np.array(rhs_y)
    sign_changes = np.where(np.diff(np.sign(diffs)))[0]
    
    for idx in sign_changes:
        if idx < len(x_vals) - 1:
            x0, x1 = x_vals[idx], x_vals[idx+1]
            y0, y1 = diffs[idx], diffs[idx+1]
            
            if y0 != y1:
                x_root = x0 - y0 * (x1 - x0) / (y1 - y0)
                y_root = eval(lhs.replace('x', f'({x_root})'), {'math': math})
                plt.plot(x_root, y_root, 'ro', markersize=5)

def adjust_axes_limits():
    y_min, y_max = plt.ylim()
    if abs(y_min) > 50 or abs(y_max) > 50:
        plt.ylim(max(y_min, -20), min(y_max, 20))
    
    x_min, x_max = plt.xlim()
    if abs(x_min) > 50 or abs(x_max) > 50:
        plt.xlim(max(x_min, -20), min(x_max, 20))

# fix for y = c e.g. y = 2

#------------------------------------
# main

# get user input

# test = "(x+1)(x+2)"
# print(test)

# test_expr = sympify(test)
# expand_expr = expand(test_expr)
# print(expand_expr)



def solve_equation(equations):
    
    if len(equations) > 1:
        systemOfequations = True
    else:
        systemOfequations = False

    # basic sums
    if not systemOfequations and if_basic_arithmetic(equations[0]):
        return solve_basic_arithmetic(equations[0])

    equations = [preprocess_equation(equation) for equation in equations]

    # pre process equations
    processedEqs = []
    operators = []
    for eq in equations:
        op = get_operator(eq)
        print(op)
        if op is None:
            raise ValueError(f"Invalid equation")
        operators.append(op)
        processedEqs.append(preprocess_equation(eq))

    tempEqs = [eq.replace(op, "=") for eq, op in zip(processedEqs, operators)]

    if systemOfequations:
        result = solve_systems_and_linear(tempEqs)

    else:
        equationType = classify_equation(tempEqs[0])
        print(equationType)

        if equationType == 'linear':
            result = solve_systems_and_linear(tempEqs)

        elif equationType == 'quadratic':
            result = solve_quadratic(tempEqs[0])
        
        elif equationType == 'polynomial':
            result = solve_polynomial(tempEqs[0])
        else:
            raise ValueError(f"Unsupported equation type")
        
    result["operators"] = operators
    return result
        


if __name__ == "__main__":
    
    equations = get_user_input()
    result = solve_equation(equations)
    print(equations)


    print(result)
    #show_graph(equations)
    print("Test")