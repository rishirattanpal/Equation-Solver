import matplotlib.pyplot as plt

plt.axvline(x=10, color='r', label='2x + 23 = 43')  # Vertical line at x = 10
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.legend()
plt.grid()
plt.show()