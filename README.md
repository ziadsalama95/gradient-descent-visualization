# Gradient Descent Visualization

Welcome! In this [notebook](https://github.com/ziadsalama95/gradient-descent-visualization/blob/main/gradient_descent_visualization.ipynb), we’ll explore **Gradient Descent (GD)**, one of the most important algorithms in machine learning and optimization. By the end of this notebook, you’ll understand how GD works, why it’s useful, and how to apply it to find the minimum of a function. Let’s dive in!

---

## What is Gradient Descent?

Gradient Descent is an iterative optimization algorithm used to minimize a function. Imagine you’re standing on a hill, and your goal is to reach the bottom. Gradient Descent works by taking small steps in the direction of the steepest slope (the gradient) until you reach the lowest point.

Mathematically, Gradient Descent updates a parameter $x$ using the following rule:

$$
x_{\text{new}} = x_{\text{old}} - \alpha \cdot \nabla f(x_{\text{old}})
$$

Where:
- $x_{\text{old}}$ is the current value of the parameter.
- $\alpha$ is the **learning rate**, which controls the size of the steps.
- $\nabla f(x_{\text{old}})$ is the **gradient** (derivative) of the function at $x_{\text{old}}$.

---

## Step 1: Choosing a Function

To see Gradient Descent in action, we first need a function to minimize. You can choose from the following options:

1. $f(x) = x^2$  
2. $f(x) = x^3 - 3x + 1$  
3. $f(x) = e^x$  
4. $f(x) = \ln(x)$ (only valid for $x > 0$)  
5. $f(x) = \sin(x)$  
6. $f(x) = x^2 + \sin(5x)$  

Here’s how the code works:

```python
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Math

x = sp.Symbol('x') # Symbolic variable

functions = {
    1: x**2,
    2: x**3 - 3*x + 1,
    3: sp.exp(x),
    4: sp.ln(x), # Only valid for x > 0
    5: sp.sin(x),
    6: x**2 + sp.sin(5*x)
}

print('Choose a function for gradient descent:')
print('1: f(x) = x^2')
print('2: f(x) = x^3 - 3x + 1')
print('3: f(x) = e^x')
print('4: f(x) = ln(x) (only for x > 0)')
print('5: f(x) = sin(x)')
print('6: f(x) = x^2 + sin(5x)')

choice = int(input('Enter the number of your chosen function (1-6): '))

if choice not in functions:
    raise ValueError('Invalid choice. Please restart and select a valid number.')

f_expr = functions[choice]

df_expr = sp.diff(f_expr, x) # Compute derivative

# Convert symbolic expressions to numerical functions
f = sp.lambdify(x, f_expr, 'numpy')
df = sp.lambdify(x, df_expr, 'numpy')
print('----------------')
display(Math(f"f(x) = {sp.latex(f_expr)}"))
display(Math(f"f'(x) = {sp.latex(df_expr)}"))
print('----------------')
```

### Example
If you choose $f(x) = x^2$, the output will be:

$$
f(x) = x^2
$$

$$
f'(x) = 2x
$$

---

## Step 2: Applying Gradient Descent

Now that we’ve chosen a function, let’s apply Gradient Descent to find its minimum. Here’s how it works:

1. Start with an initial guess for $x$.
2. Compute the gradient (derivative) at the current $x$.
3. Update $x$ using the gradient descent rule:
   $$
   x_{\text{new}} = x_{\text{old}} - \alpha \cdot f'(x_{\text{old}})
   $$
4. Repeat until the gradient is close to zero or the change in $f(x)$ is very small.

Here’s the code:

```python
import numpy as np
import matplotlib.pyplot as plt

# Gradient descent parameters
learning_rate = 0.1
iterations = 30
x_init = 5
tolerance = 1e-6  # Tolerance for stopping when the gradient is small or change is negligible

# Initialize values
x_values = [x_init]
x = x_init
prev_f_x = f(x)

# Gradient descent process
for i in range(iterations):
    grad = df(x)  # Gradient (derivative)

    x = x - learning_rate * grad # Update x using gradient descent
    x_values.append(x)

    f_x = f(x) # Compute new function value

    # Check stopping conditions
    if np.abs(grad) < tolerance:  # Gradient is too small
        print("Stopping early due to small gradient.")
        break
    if np.abs(f_x - prev_f_x) < tolerance:  # Function value change is too small
        print("Stopping early due to small function value change.")
        break

    prev_f_x = f_x  # Update previous function value

final_x = x_values[-1]
final_f_x = f(final_x)
print(f"Final point after {len(x_values) - 1} iterations: x = {final_x}, f(x) = {final_f_x}")

# Generate function values for visualization
x_plot = np.linspace(-6, 6, 100)
y_plot = f(x_plot)

# Plot function and gradient descent steps
plt.figure(figsize=(8, 5))
plt.plot(x_plot, y_plot, label='Function f(x)')
plt.scatter(x_values, [f(x) for x in x_values], color='red', label='Gradient Descent Steps')
plt.plot(x_values, [f(x) for x in x_values], linestyle='dashed', color='red')

# Highlight the final point in green
plt.scatter(final_x, final_f_x, color='green')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.title('Gradient Descent Visualization')
plt.show()
```

### Example Output
For $f(x) = x^2$, the output might look like:
```
Final point after 30 iterations: x = 0.006189700196426903, f(x) = 3.8312388521647236e-05
```

---

## Visualizing Gradient Descent

The plot below shows the function $f(x)$ and the steps taken by Gradient Descent to find the minimum. The red dots represent the steps, and the green dot marks the final point.

![Gradient Descent Visualization](https://raw.githubusercontent.com/ziadsalama95/gradient-descent-visualization/refs/heads/main/Gradient%20Descent%20Visualization.png)

---

## Key Takeaways

1. **Learning Rate ($\alpha$):** Controls the step size. If it’s too large, the algorithm may overshoot the minimum; if it’s too small, convergence will be slow.
2. **Gradient ($\nabla f(x)$):** Tells us the direction of the steepest ascent. We move in the opposite direction to minimize the function.
3. **Stopping Conditions:** The algorithm stops when the gradient is close to zero or the change in the function value is negligible.

---

## Experiment and Explore!

Feel free to experiment with different functions and learning rates. Observe how the algorithm behaves and try to answer these questions:
- What happens if the learning rate is too large or too small?
- How does the choice of function affect the convergence of Gradient Descent?
- Can you find a function where Gradient Descent gets stuck in a local minimum?

---

Happy learning! 🚀
