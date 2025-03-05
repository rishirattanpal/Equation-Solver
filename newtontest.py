import numpy as np
import matplotlib.pyplot as plt

def f(x):
    # return 4*x**4 + 2*x - 1
    return x**3 - 6*x**2 + 11*x - 6 


def df(x):
    # return 16*x**3 + 2
    return 3*x**2 - 12*x + 11

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


        if abs(nextGuess - guess) < delta:
            return nextGuess
        
        guess = nextGuess

    raise ValueError("Max num of iterations reached, no sol found")
        

def find_roots(coefficients, numPoints = 100, delta = 1e-10, maxIter = 1000):
    
    bound = cauchy_bound(coefficients)
    realRange = np.linspace(-bound, bound, numPoints)
    imagRange = np.linspace(-bound, bound, numPoints)

    roots = []

    for r in realRange:
        for i in imagRange:
            guess = r + i*1j

            try:
                root = newton_raphson(f, df, guess, delta, maxIter)

                if not any(abs(root - existingRoot) < delta for existingRoot in roots):
                    roots.append(root) # add root if unique 
            except ValueError:
                continue
        
    return roots

# 4*x**4 + 2*x - 1 
#coefficients = [4, 0, 0, 2, -1]

# x**3 - 6*x**2 + 11x - 6
coefficients = [1, -6, 11, -6]

roots = find_roots(coefficients)
roundedRoots = [np.round(root, 10) for root in roots]
realRoots = [float(r.real) for r in roundedRoots]
print(realRoots)
