import matplotlib.pyplot as plt
import numpy as np

# Rearrange the equation to y = mx + b form
# 2x + 23 = 43
# 2x = 43 - 23
# 2x = 20
# x = 10
# So the plot will show the line y = 2x + 23

def plot_equation():
    # Create x values
    x = np.linspace(-5, 15, 100)
    
    # Calculate y values
    y = 2*x + 23
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='2x + 23 = 43', color='blue')
    
    # Add horizontal and vertical lines at x and y axes
    plt.axhline(y=0, color='k', linestyle='--')
    plt.axvline(x=0, color='k', linestyle='--')
    
    # Highlight the point where the equation is true
    plt.plot(10, 43, 'ro', label='Solution Point (10, 43)')
    
    # Customize the plot
    plt.title('Plot of Equation: 2x + 23 = 43')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, linestyle=':')
    plt.legend()
    
    # Show the plot
    plt.show()

# Call the function to create the plot
plot_equation()