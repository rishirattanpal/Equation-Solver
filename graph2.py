import matplotlib.pyplot as plt
import numpy as np
from equationSolver import preprocess_equation

# Rearrange the equation to y = mx + b form
# 2x + 23 = 43
# 2x = 43 - 23
# 2x = 20
# x = 10
# So the plot will show the line y = 2x + 23
# graph stuff
def show_graph(equations):
    plt.figure(figsize=(6, 3))
    #plt.tight_layout()
    
    # Make axis
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.grid(True)
    
    # Colors for each line
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    
    x_vals = np.linspace(-10, 10, 1000)
    
    for i, equation in enumerate(equations):
        color = colors[i % len(colors)]
        
        try:
            # Preprocess the equation
            eq = preprocess_equation(equation)
            lhs, rhs = eq.split('=')
            lhs = lhs.strip()
            rhs = rhs.strip()
            
            # x = c where c can be decimal, neg or pos
            if lhs == 'x' and rhs.lstrip('-').replace('.', '').isdigit(): 
                x_const = float(rhs)
                plt.axvline(x=x_const, color=color, label=f'x = {x_const}')
                continue
            elif rhs == 'x' and lhs.lstrip('-').replace('.', '').isdigit():
                x_const = float(lhs)
                plt.axvline(x=x_const, color=color, label=f'x = {x_const}')
                continue
                
            # y = c
            if lhs == 'y' and rhs.lstrip('-').replace('.', '').isdigit():
                y_const = float(rhs)
                plt.axhline(y=y_const, color=color, label=f'y = {y_const}')
                continue
            elif rhs == 'y' and lhs.lstrip('-').replace('.', '').isdigit():
                y_const = float(lhs)
                plt.axhline(y=y_const, color=color, label=f'y = {y_const}')
                continue
                
            # y = or = y 
            if 'y' in eq:
                if lhs == 'y':
                    y_expr = rhs
                else:
                    y_expr = lhs
                
                # Evaluate the expression
                y_points = []
                for x_val in x_vals:
                    try:
                        y_val = eval(y_expr.replace('x', f'({x_val})'), {'math': math})
                        y_points.append(y_val)
                    except:
                        y_points.append(np.nan)
                
                plt.plot(x_vals, y_points, color=color, label=equation)
            
            else:
                # Equation without y (f(x) = g(x))
                lhs_y = []
                rhs_y = []
                for x_val in x_vals:
                    try:
                        lhs_val = eval(lhs.replace('x', f'({x_val})'), {'math': math})
                        rhs_val = eval(rhs.replace('x', f'({x_val})'), {'math': math})
                        lhs_y.append(lhs_val)
                        rhs_y.append(rhs_val)
                    except:
                        lhs_y.append(np.nan)
                        rhs_y.append(np.nan)
                
                # Plot both sides
                plt.plot(x_vals, lhs_y, color=color, linestyle='-', label=f'{lhs} (LHS)')
                plt.plot(x_vals, rhs_y, color=color, linestyle='--', label=f'{rhs} (RHS)')
                
                # Find and plot intersection points
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
        
        except Exception as e:
            print(f"Could not plot {equation}: {e}")
            continue
    
    plt.title("Graph of Equations")
    plt.xlabel("x")
    plt.ylabel("y")
    
    # Adjust legend position and size
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # Set reasonable limits
    current_y_lim = plt.ylim()
    if abs(current_y_lim[0]) > 50 or abs(current_y_lim[1]) > 50:
        plt.ylim(-10, 10)
    
    current_x_lim = plt.xlim()
    if abs(current_x_lim[0]) > 50 or abs(current_x_lim[1]) > 50:
        plt.xlim(-10, 10)
    
    plt.savefig("static/graph.png")
    plt.show()
    plt.close()
    
    return True

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
#plot_equation()

equations = ['y=x']
show_graph(equations)