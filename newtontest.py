import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 4*x**4 + 2*x - 1


def df(x):
    return 16*x**3 + 2


xvalues = np.linspace(-1, 1, 100)
yvalues = f(xvalues)

plt.axhline(0)
plt.plot(xvalues, yvalues)
# plt.show()

delta = 1e-10


rvalues = np.linspace(-5, 5, 100)
ivalues = np.linspace(-5, 5, 100)

roots = []

for r in rvalues:
    for i in ivalues:
        guess = r + i*1j

        for n in range(1000):
            nextGuess = guess - f(guess)/df(guess)
            if abs(nextGuess - guess) < delta:
                alreadyIn = False
                for root in roots:
                    if abs(nextGuess - root) < delta:
                        alreadyIn = True
                        break
                if not alreadyIn:
                    roots.append(nextGuess)
                break
                
            guess = nextGuess
     
    
print(roots)