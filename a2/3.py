import numpy as np
import matplotlib.pyplot as plt

# Create array of x values from -10 to 10
x = np.linspace(-10, 10, 200)  # Using 200 points for smooth curve

# Calculate y values (x squared)
y = x**2

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'b-', label='y = x²')

# Add labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title('Quadratic Function: y = x²')
plt.grid(True)
plt.legend()

# Add x and y axis lines
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

# Show the plot
plt.show()