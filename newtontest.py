import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 4*x**4 + 2*x - 1


def df(x):
    return 16*x**3 + 2

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
coefficients = [4, 0, 0, 2, -1]

roots = find_roots(coefficients)
print(roots)



# xvalues = np.linspace(-1, 1, 100)
# yvalues = f(xvalues)

# plt.axhline(0)
# plt.plot(xvalues, yvalues)
# # plt.show()

# delta = 1e-10


# rvalues = np.linspace(-5, 5, 100)
# ivalues = np.linspace(-5, 5, 100)

# roots = []

# for r in rvalues:
#     for i in ivalues:
#         guess = r + i*1j

#         for n in range(1000):
#             nextGuess = guess - f(guess)/df(guess)
#             if abs(nextGuess - guess) < delta:
#                 alreadyIn = False
#                 for root in roots:
#                     if abs(nextGuess - root) < delta:
#                         alreadyIn = True
#                         break
#                 if not alreadyIn:
#                     roots.append(nextGuess)
#                 break
                
#             guess = nextGuess
     
    
# print(roots)