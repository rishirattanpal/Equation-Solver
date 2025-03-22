import matplotlib.pyplot as plt
import numpy as np



x = np.linspace(-5, 5)
y = "x**3 + 12*x - 16"
y = eval(y)

plt.plot(x,y)

show = plt.show()

# x**3 + 12*x - 16 = 0