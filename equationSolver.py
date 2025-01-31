import re
import numpy as np
from sympy import symbols, Eq, solve, sympify



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



# main
equations, systemOfequations = get_user_input()
equations = [preprocess_equation(equation) for equation in equations]


print(equations)
print(systemOfequations)



if systemOfequations == True:
    print("no way")

else:
    equationType = classify_equation(equations[0])
    print(equationType)



