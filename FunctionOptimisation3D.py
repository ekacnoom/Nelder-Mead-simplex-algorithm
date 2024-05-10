import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to calculate F(x, y)
def function_F(x):
    return 10 * (x[1] - np.sin(x[0]))**2 + 0.1 * x[0]**2

# Function to create the initial simplex
def create_initial_simplex(centre, bound):
    side_length = min(centre[0] - bound[0], bound[1] - centre[0], 
                      centre[1] - bound[0], bound[1] - centre[1])
    return np.array([centre, 
                     [centre[0] + side_length, centre[1]], 
                     [centre[0], centre[1] + side_length]])

# Optimising the function
def optimise_function(centre, tolerance, bound):
    initial_simplex = create_initial_simplex(centre, bound)
    result = minimize(function_F, centre, method='Nelder-Mead', 
                      tol=tolerance, options={'initial_simplex': initial_simplex})
    return result

# Visualisation
def visualise_function():
    x = np.linspace(-3, 3, 30)
    y = np.linspace(-3, 3, 30)
    x, y = np.meshgrid(x, y)
    z = function_F([x, y])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('F(x, y)')
    plt.show()

# Execute optimisation and visualisation
centre = [1, 3]  # Example centre coordinates
tolerance = 0.001  # Example specified precision
bound = [-3, 3]  # Limits of the area

result = optimise_function(centre, tolerance, bound)
print("Optimised coordinates:", result.x)
print("Minimum value of the function:", result.fun)

visualise_function()
