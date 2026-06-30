# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="Fkd-th4I_y4R"
# # Model Fitting and Selection
#
# **Course Title:** ENM 3800: Learning from Data
#
# **Instructor:** Eva Dyer
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nerdslab/learningfromdata-course/blob/main/notebooks/Notebook_2/Notebook_2b_Model_Selection.ipynb)
#
# In the previous notebook, we built our first models. Now we ask a harder question:
#
# > How do we choose a model that learns the signal without memorizing the noise?
#
# This notebook focuses on:
#
# - loss functions,
# - optimization and gradient descent,
# - train/validation/test splits,
# - overfitting and underfitting,
# - model complexity,
# - interpolation vs. extrapolation.
#
# Big theme:
#
# > Training error tells us how well a model fit the data it saw. Generalization tells us whether it learned something useful.

# %% [markdown] id="FMhHc4yy4RCW"
# ## Model Optimization = Searching for Good Parameters
#

# %% [markdown] id="KlSd1cco-4hC"
# ### Defining the loss function
#
# Every model we will build has **parameters**
# (e.g., slope & intercept, neural network weights, polynomial coefficients).
#
# Training means:
# $$
# \textbf{find parameters that make good predictions}
# $$
#
# To measure “good”, we use a **loss function**
# $$
# L(\beta) = \text{how wrong the model is}
# $$
#
# Examples:
#
# - Mean Squared Error (regression):
# $$
# L(\beta)=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2
# $$
#
# - Classification Cross-Entropy (later in class!)
#
# Training = **Optimization problem**
# $$
# \beta^* = \arg\min_{\beta} L(\beta)
# $$
#
# But how do we *actually* find \($\beta^*$\)?
# Not magic ➜ **gradient descent**
#

# %% [markdown] id="0yWyQcWi-yvI"
# ### Searching for the Best Parameters (Before Gradient Descent)
#
# Before using calculus and gradients, there is a simpler idea:
#
# Try many possible parameter values
# Compute the loss for each
# Pick the one with the smallest loss
#
# This is called **exhaustive search** or **grid search**.
#
# Good:
#
# - very intuitive
# - guaranteed to find best value (if grid is dense enough)
#
# Bad:
#
# - very slow in high dimensions
# - becomes impossible when models have many parameters
#
# Let's see it in 1D first.
#
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 524} executionInfo={"elapsed": 656, "status": "ok", "timestamp": 1766254717719, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="FoTOc-BA_B9-" outputId="69185929-1ac6-4f25-e5c3-649ee74cafea"
import matplotlib.pyplot as plt
import numpy as np


# Define a simple loss function
def loss(beta):
    return (beta - 2) ** 2 + 1  # minimum at beta = 2


# Search over candidate values
candidates = np.linspace(-6, 6, 200)
values = loss(candidates)

best_idx = np.argmin(values)
best_beta = candidates[best_idx]
best_loss = values[best_idx]

plt.figure(figsize=(7, 5))
plt.plot(candidates, values, label="loss function")
plt.scatter(best_beta, best_loss, color="red", s=80, label="Best found")
plt.title("Searching for Best Parameter by Trying Many Values")
plt.xlabel("beta")
plt.ylabel("Loss")
plt.legend()
plt.show()

print("Best beta found:", best_beta)
print("Loss at best beta:", best_loss)


# %% [markdown] id="BELSqeSs_KIy"
# ### What did we just do?
#
# We tried many possible parameter values.
# For each value we asked:
# \[
# \text{How bad is this choice?}
# \]
#
# Then we picked the one with the smallest loss.
#
# This is conceptually what many ML systems do for hyperparameters
# (e.g., choosing number of neighbors in kNN, polynomial degree, regularization strength).
#
# But there is a problem…
#

# %% [markdown] id="2mZ_GMe4_Qwk"
# ### Why brute-force search is not enough
#
# A real ML model may have:
#
# - 10 parameters (already bad)
# - 100 parameters (impossible)
# - millions of parameters (neural networks!)
#
# If each parameter needs:
#
# - 100 candidate values
# - 100^10 combinations = \(10^{20}\)
# - cannot search them all
#
# So we need a smarter way:
# instead of trying everything,
# we **follow the slope of the loss function**.
#
# This leads to…
# 👉 Gradient Descent
#

# %% [markdown] id="vs3Yvhn04l6R"
# ### Gradient Descent: The Key Learning Algorithm
#
# Gradient descent is like hiking downhill in fog:
#
# - you cannot see the whole mountain
# - but you CAN feel the local slope
# - so you step in the steepest downward direction
# - repeat until you reach a valley
#
# Mathematically:
# $$
# \beta \leftarrow \beta - \eta \,\nabla_\beta L(\beta)
# $$
#
# where
#
# - \($\nabla_\beta L$\) = gradient (direction of steepest increase)
# - \($\eta$\) = learning rate (step size)
#
# Small \($\eta$\) → slow but safe
# Large \($\eta$\) → fast but risky (may overshoot or diverge)
#
# We stop when steps become very small or after some iterations.
#
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 487} executionInfo={"elapsed": 297, "status": "ok", "timestamp": 1766253098167, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="SIxmsCnF40yJ" outputId="c119ebbd-9838-4db6-89bd-64faab8a2a8e"
import matplotlib.pyplot as plt
import numpy as np


def loss(beta):
    return (beta - 2) ** 2 + 1  # minimum at beta = 2


def grad(beta):
    return 2 * (beta - 2)


beta = -4
eta = 0.2
trajectory = [beta]

for _ in range(20):
    beta = beta - eta * grad(beta)
    trajectory.append(beta)

# Plot
b = np.linspace(-6, 6, 200)
plt.figure(figsize=(7, 5))
plt.plot(b, loss(b))
plt.scatter(trajectory, [loss(x) for x in trajectory], color="red")
plt.title("Gradient Descent in 1D")
plt.xlabel("beta")
plt.ylabel("Loss")
plt.show()


# %% [markdown]
# ### Interactive Gradient Descent: Learning Rate Matters
#
# The learning rate controls step size.
#
# Use the slider below to see three regimes:
#
# - too small: slow progress,
# - just right: steady convergence,
# - too large: bouncing or divergence.
#


# %%
def run_gradient_descent_1d(eta=0.2, start=-4.0, steps=20):
    def loss(beta):
        return (beta - 2) ** 2 + 1

    def grad(beta):
        return 2 * (beta - 2)

    beta = start
    trajectory = [beta]
    for _ in range(steps):
        beta = beta - eta * grad(beta)
        trajectory.append(beta)

    b = np.linspace(-6, 6, 400)
    losses = [loss(x) for x in trajectory]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(b, loss(b), color="steelblue")
    axes[0].scatter(trajectory, losses, color="crimson", s=35)
    axes[0].plot(trajectory, losses, color="crimson", alpha=0.5)
    axes[0].set_title(f"Gradient descent path, eta={eta:.2f}")
    axes[0].set_xlabel("parameter beta")
    axes[0].set_ylabel("loss")

    axes[1].plot(losses, marker="o", color="crimson")
    axes[1].set_title("Loss over iterations")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("loss")

    plt.tight_layout()
    plt.show()

    print("final beta:", trajectory[-1])
    print("final loss:", losses[-1])


try:
    import ipywidgets as widgets
    from IPython.display import display

    ui = widgets.interactive(
        run_gradient_descent_1d,
        eta=widgets.FloatSlider(
            value=0.2, min=0.01, max=1.2, step=0.01, description="eta"
        ),
        start=widgets.FloatSlider(
            value=-4.0, min=-6.0, max=6.0, step=0.25, description="start"
        ),
        steps=widgets.IntSlider(value=20, min=1, max=60, step=1, description="steps"),
    )
    display(ui)
except Exception as e:
    print("Widgets are not available here. Running one default example instead.")
    run_gradient_descent_1d(eta=0.2, start=-4.0, steps=20)
    print(e)


# %% [markdown] id="_ZwFVhW847f_"
# Try changing the learning rate!
#
# - Too small → slow convergence
# - Too large → bouncing / divergence
# - Just right → smooth descent
#
# This is the same behavior we will see in neural networks later.
#
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 564} executionInfo={"elapsed": 1124, "status": "ok", "timestamp": 1766253136480, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="aBSwyANN4_vp" outputId="23b33c98-76b3-4116-f021-615d19201593"
import matplotlib.pyplot as plt
import numpy as np


def loss(beta):
    x, y = beta
    return x**2 + 0.5 * y**2 + 1


def grad(beta):
    x, y = beta
    return np.array([2 * x, y])


eta = 0.2
beta = np.array([3.0, 3.0])
trajectory = [beta.copy()]

for _ in range(25):
    beta -= eta * grad(beta)
    trajectory.append(beta.copy())

trajectory = np.array(trajectory)

# Plot contours
x = np.linspace(-4, 4, 200)
y = np.linspace(-4, 4, 200)
X, Y = np.meshgrid(x, y)
Z = X**2 + 0.5 * Y**2 + 1

plt.figure(figsize=(6, 6))
plt.contour(X, Y, Z, levels=20)
plt.plot(trajectory[:, 0], trajectory[:, 1], marker="o", color="red")
plt.title("Gradient Descent Path on 2D Loss Landscape")
plt.xlabel("x")
plt.ylabel("y")
plt.gca().set_aspect("equal")
plt.show()


# %% [markdown] id="VpGIH0Oy5H92"
# ### Gradient Descent for Linear Regression
#
# Consider a simple regression model:
# $$
# \hat{y}_i = w x_i + b
# $$
#
# Loss:
# $$
# L(w,b)=\frac{1}{N}\sum_i (y_i-\hat{y}_i)^2
# $$
#
# We can compute gradients (you do not need to memorize):
# $$
# \frac{\partial L}{\partial w} = -\frac{2}{N}\sum_i x_i(y_i-\hat{y}_i)
# $$
#
# $$
# \frac{\partial L}{\partial b} = -\frac{2}{N}\sum_i (y_i-\hat{y}_i)
# $$
#
# Update rules:
# $$
# w \leftarrow w - \eta \frac{\partial L}{\partial w}
# $$
#
# $$
# b \leftarrow b - \eta \frac{\partial L}{\partial b}
# $$
#
# This is what libraries like sklearn and PyTorch do for you.
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 509} executionInfo={"elapsed": 335, "status": "ok", "timestamp": 1766253273010, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="iNAqWgjw5hcR" outputId="fb914bdf-7108-48e2-fd36-482a67d39571"
np.random.seed(0)

X = np.linspace(0, 5, 40)
true_w, true_b = 2.5, 1.0
y = true_w * X + true_b + np.random.randn(len(X)) * 1

w, b = -3, 0
eta = 0.01


def predict(X):
    return w * X + b


def mse(y_pred, y):
    return np.mean((y_pred - y) ** 2)


losses = []

for _ in range(200):
    y_pred = predict(X)
    dw = -2 * np.mean(X * (y - y_pred))
    db = -2 * np.mean(y - y_pred)
    w -= eta * dw
    b -= eta * db
    losses.append(mse(y_pred, y))

plt.plot(losses)
plt.title("Training Loss Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.show()

print("Learned w:", w)
print("Learned b:", b)


# %% [markdown] id="M8kFwxmz6foE"
# ### Additional examples
# Optimization (finding minima or maxima of a function) is a large field in mathematics, and optimization of complicated functions can be rather involved. Here we will only look at a few very simple cases.
#
# For a more detailed introduction to optimization with SciPy see: https://scipy-lectures.org/advanced/mathematical_optimization/
#
#

# %% [markdown] id="CyhwUyy4_40_"
# #### Example 1: Finding a function's minimum


# %% colab={"base_uri": "https://localhost:8080/", "height": 452} executionInfo={"elapsed": 340, "status": "ok", "timestamp": 1766253577472, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="uWIfTpB36rsX" outputId="0ccd2ce7-5a61-422c-bd5d-e2eda03d856a"
#| code-fold: true
def f(x):
    return 4 * x**3 + (x - 2) ** 2 + x**4


x = np.linspace(-5, 3, 100)
plt.plot(x, f(x))
plt.title("Function f(x)")
plt.show()

# %% [markdown] id="ZdPFa4nL6yYw"
# The goal is to find the minimum of f(x).
#
# There are many types of optimizers available. If you are interested, you can read more in the [documentation](https://docs.scipy.org/doc/scipy/reference/optimize.html).

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 20, "status": "ok", "timestamp": 1766253620183, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="myZb_3fP62Q_" outputId="f642520b-3ad1-412a-a2a7-07121e6f40a1"
from scipy.optimize import minimize

x_min = minimize(f, -4)
print("The minimum is:", x_min.x)


# %% [markdown] id="IUa68BHd667J"
# What happens if we start at a different initial point?

# %% [markdown] id="L7DB8dG22qPc"
# #### Example 2: Solving an equation
# A related problem is solving an equation, which can be achieved with the `fsolve` function.
#
# Here we want to find `x` such that `g(x)=0`


# %% colab={"base_uri": "https://localhost:8080/", "height": 452} executionInfo={"elapsed": 563, "status": "ok", "timestamp": 1724873267442, "user": {"displayName": "Eva Dyer", "userId": "05212169819659068372"}, "user_tz": 240} id="XxosIF6j2jgz" outputId="d13e9f51-bc12-4642-c971-1a3b1443a7e3"
#| code-fold: true
def g(x):
    return np.sin(3 * x) * (1 / x) - 0.2


x = np.linspace(1, 10, 100)
plt.plot(x, g(x))
plt.plot([0, 10], [0, 0])
plt.title("Equation g(x)=0")
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1724873267443, "user": {"displayName": "Eva Dyer", "userId": "05212169819659068372"}, "user_tz": 240} id="wc8Xx2w-2pr0" outputId="6d72f155-96c0-4193-e0f0-25ec407e9967"
from scipy.optimize import fsolve

ans = fsolve(g, 3)
print("The solution is:", ans)

# %% [markdown] id="zKBfcMG68ZP5"
# ## Overfitting and Generalization
#
# So far, we’ve focused on *fitting* a model to data.
# But the real goal is bigger:
#
# We want a model that performs well on **new, unseen data**.
#
# This property is called **generalization**.

# %% [markdown] id="Ny67vut79Ns2"
# ### Overfitting
#
# A model can:
#
# - **Underfit** → too simple, misses structure
# - **Just right** → captures true pattern + tolerates noise
# - **Overfit** → memorizes training data, learns noise

# %% [markdown] id="ugDMDROZ8lGM"
# ![mlconcepts_image5.png](data:image/png;base64,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)

# %% [markdown] id="dvbtbiWD8uYD"
# Image credit: https://docs.aws.amazon.com/machine-learning/latest/dg/model-fit-underfitting-vs-overfitting.html

# %% [markdown] id="fEJYAVBl87NK"
#
# Training error alone is **not enough** to judge a model.
# We need to look at how a model behaves on data it did *not* see during training.
#

# %% [markdown] id="f2aVPWl9z9JO"
# ### Cross-validation
#
# One way to achieve this goal is through "cross-validation", where some examples ("validation" examples) are hidden when the model is fit to "training" examples, and the loss function is assessed on the hidden validation samples.
#

# %% [markdown] id="8jz9Eea0SFK8"
# There are many strategies for cross-validation:
#
# * *hold-out*: randomly leave out a percentage (usually ~30%) of the data during training.
# * *k-fold*: select `k` (usually 3-5) randomly-assigned sub-groups of data, and train `k` times holding each group out.
# * *leave p-out*: leave `p` (usually 1) samples out of the training and assess the error for the `p` that were left out. Repeat for all possible `p` subsets of the sample.
# * *stratified splitting (or stratified k-fold)*: maintains class proportions across the train/test splits.
#
# Cross-validation is used to determine hyperparameters. In this case, even the "test" sets are used to optimize the model. It is common to select an additional "validation" or "holdout" subset for a final validation of the model.
#
# Important (and often violated) assumption: **The collected data is representative of future data.**

# %% [markdown]
# ### Visualizing the Roles of Train, Validation, and Test Sets
#
# A useful workflow is:
#
# 1. **Training set:** fit model parameters.
# 2. **Validation set:** choose hyperparameters and model complexity.
# 3. **Test set:** estimate final performance once, after model selection.
#
# The test set should not guide choices during model development.
#

# %%
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fig, ax = plt.subplots(figsize=(10, 2.5))
ax.axis("off")

segments = [
    (0.05, 0.35, 0.55, "Train", "fit parameters", "#88c0d0"),
    (0.62, 0.35, 0.20, "Validation", "choose hyperparameters", "#ebcb8b"),
    (0.84, 0.35, 0.11, "Test", "final estimate", "#bf616a"),
]

for x, y, w, label, subtitle, color in segments:
    ax.add_patch(
        Rectangle((x, y), w, 0.35, facecolor=color, edgecolor="black", linewidth=1.2)
    )
    ax.text(
        x + w / 2, y + 0.23, label, ha="center", va="center", fontsize=13, weight="bold"
    )
    ax.text(x + w / 2, y + 0.09, subtitle, ha="center", va="center", fontsize=9)

ax.text(
    0.5,
    0.88,
    "One dataset, different responsibilities",
    ha="center",
    fontsize=14,
    weight="bold",
)
ax.text(
    0.5, 0.12, "Do not use the test set to choose the model.", ha="center", fontsize=12
)
plt.show()


# %% [markdown] id="o5RvJY_9RVuK"
# ![Train-test-cross-validation-split-methodology-used-in-this-paper-The-first-operation.jpg](data:image/jpeg;base64,/9j/4AAQSkZJRgABAQADAAMAAAD/4QEqRXhpZgAATU0AKgAAAAgABgESAAMAAAABAAEAAAEaAAUAAAABAAAAVgEbAAUAAAABAAAAXgExAAIAAAAmAAAAZgEyAAIAAAAUAAAAjIdpAAQAAAABAAAAoAAAAAAAAAMAAAAAAQAAAwAAAAABV2luZG93cyBQaG90byBFZGl0b3IgMTAuMC4xMDAxMS4xNjM4NAAyMDIwOjAzOjI2IDEzOjM0OjMzAAAHkAMAAgAAABQAAAD6kAQAAgAAABQAAAEOkpEAAgAAAAMwMAAAkpIAAgAAAAMwMAAAoAEAAwAAAAEAAQAAoAIABAAAAAEAAAHNoAMABAAAAAEAAAE3AAAAADIwMjA6MDM6MjYgMTE6NDI6MjgAMjAyMDowMzoyNiAxMTo0MjoyOAD/4QqeaHR0cDovL25zLmFkb2JlLmNvbS94YXAvMS4wLwA8P3hwYWNrZXQgYmVnaW49Iu+7vyIgaWQ9Ilc1TTBNcENlaGlIenJlU3pOVGN6a2M5ZCI/PiA8eDp4bXBtZXRhIHhtbG5zOng9ImFkb2JlOm5zOm1ldGEvIiB4OnhtcHRrPSJYTVAgQ29yZSA2LjAuMCI+IDxyZGY6UkRGIHhtbG5zOnJkZj0iaHR0cDovL3d3dy53My5vcmcvMTk5OS8wMi8yMi1yZGYtc3ludGF4LW5zIyI+IDxyZGY6RGVzY3JpcHRpb24gcmRmOmFib3V0PSIiIHhtbG5zOnhtcD0iaHR0cDovL25zLmFkb2JlLmNvbS94YXAvMS4wLyIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnBob3Rvc2hvcD0iaHR0cDovL25zLmFkb2JlLmNvbS9waG90b3Nob3AvMS4wLyIgeG1wOkNyZWF0ZURhdGU9IjIwMjAtMDMtMjZUMTE6NDI6MjguMDAiIHhtcDpDcmVhdG9yVG9vbD0iV2luZG93cyBQaG90byBFZGl0b3IgMTAuMC4xMDAxMS4xNjM4NCIgeG1wOk1vZGlmeURhdGU9IjIwMjAtMDMtMjZUMTM6MzQ6MzMiIHhtcE1NOkluc3RhbmNlSUQ9InV1aWQ6ZmFmNWJkZDUtYmEzZC0xMWRhLWFkMzEtZDMzZDc1MTgyZjFiIiBwaG90b3Nob3A6RGF0ZUNyZWF0ZWQ9IjIwMjAtMDMtMjZUMTE6NDI6MjguMDAiLz4gPC9yZGY6UkRGPiA8L3g6eG1wbWV0YT4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICA8P3hwYWNrZXQgZW5kPSJ3Ij8+AP/tAHhQaG90b3Nob3AgMy4wADhCSU0EBAAAAAAAPxwBWgADGyVHHAIAAAIAAhwCPwAGMTE0MjI4HAI+AAgyMDIwMDMyNhwCNwAIMjAyMDAzMjYcAjwABjExNDIyOAA4QklNBCUAAAAAABBY7JOICBE7DtMirFhKQQPS/8AAEQgBNwHNAwEiAAIRAQMRAf/EAB8AAAEFAQEBAQEBAAAAAAAAAAABAgMEBQYHCAkKC//EALUQAAIBAwMCBAMFBQQEAAABfQECAwAEEQUSITFBBhNRYQcicRQygZGhCCNCscEVUtHwJDNicoIJChYXGBkaJSYnKCkqNDU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6g4SFhoeIiYqSk5SVlpeYmZqio6Slpqeoqaqys7S1tre4ubrCw8TFxsfIycrS09TV1tfY2drh4uPk5ebn6Onq8fLz9PX29/j5+v/EAB8BAAMBAQEBAQEBAQEAAAAAAAABAgMEBQYHCAkKC//EALURAAIBAgQEAwQHBQQEAAECdwABAgMRBAUhMQYSQVEHYXETIjKBCBRCkaGxwQkjM1LwFWJy0QoWJDThJfEXGBkaJicoKSo1Njc4OTpDREVGR0hJSlNUVVZXWFlaY2RlZmdoaWpzdHV2d3h5eoKDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uLj5OXm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)

# %% [markdown] id="ZfGykXk4Dpwx"
# See this [sklearn tutorial](https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html#sphx-glr-auto-examples-model-selection-plot-cv-indices-py) with many examples of splitting strategies for defining training, test, and validation sets!
#

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1900, "status": "ok", "timestamp": 1766253445551, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="pZUiE9f76J4H" outputId="3ce09002-0def-4adc-999a-6eb95094af01"
import numpy as np
from sklearn.model_selection import KFold

X = np.arange(10)

kf = KFold(n_splits=5, shuffle=True, random_state=0)

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}")
    print(" Train indices:", train_idx)
    print(" Test indices:", test_idx)
    print()


# %% [markdown] id="8Aw29DiegwGm"
# ## Example
#
# ### Generate Noisy Dataset
#
# Let's consider the following function
# $$f(x) = \cos(\frac{3}{2} \pi x) + \varepsilon, \forall x \in [0,1]$$
# where $\varepsilon \sim \mathcal{N}(0,\sigma^2)$ is drawn from a Gaussian distribution with mean $0$ and variance $\sigma^2$.
#
# Given $n$ samples $(x_i, y_i)$ with $y_i=f(x_i)$, the goal is to find an approximation $\hat f$ of $f$.


# %% colab={"base_uri": "https://localhost:8080/", "height": 449} executionInfo={"elapsed": 874, "status": "ok", "timestamp": 1724873268315, "user": {"displayName": "Eva Dyer", "userId": "05212169819659068372"}, "user_tz": 240} id="WvL3pxwkbwqe" outputId="5ec11c44-832e-4036-f246-2d78217074d9"
def f(x):
    return np.cos(1.5 * np.pi * x)


def generate_data(n_samples, seed=12):
    np.random.seed(seed=seed)
    # sample points
    X = np.random.rand(n_samples)

    # apply function
    y = f(X)

    # add noise
    y = y + np.random.randn(n_samples) * 0.2  # multiply by standard deviation (sigma)
    return X, y


X, y = generate_data(50)

plt.scatter(X, y)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# %% [markdown] id="3WlTK3Fd46jq"
# Note that there is only one feature so X will be one dimensional.

# %% [markdown] id="k0sSdkIzff_l"
# ### Define your train/test split
#
# First step is splitting the data into a train set and test set. The test set will be use to evaluate the generalization performance of the model and should at no point be seen by the model during learning.
#
# `sklearn` provides a utility, [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn-model-selection-train-test-split), that splits arrays into random train and test subsets. We will hold out 30% of the data for testing.

# %% colab={"base_uri": "https://localhost:8080/", "height": 501} executionInfo={"elapsed": 3273, "status": "ok", "timestamp": 1724873271587, "user": {"displayName": "Eva Dyer", "userId": "05212169819659068372"}, "user_tz": 240} id="RtQSHZUKfptN" outputId="7d6244c6-96ab-43ad-8e12-9d82c14cda5f"
#| code-fold: true
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

print("Total number of samples:", len(X))
print("Number of training samples:", len(X_train))
print("Number of testing samples:", len(X_test))

plt.scatter(X_train, y_train)
plt.scatter(X_test, y_test)
plt.xlabel("x")
plt.ylabel("y")
plt.legend(["Train samples", "Test samples"])
plt.show()

# %% [markdown] id="1E6chzVcc9q0"
# ### Fit a model on the training data
#
# We consider a simple model, we can perform linear regression to approximate function `f`. The model will look like this:
#
# $$\hat f(x) = w_0 + w_1 x$$

# %% colab={"base_uri": "https://localhost:8080/", "height": 75} executionInfo={"elapsed": 173, "status": "ok", "timestamp": 1724873271759, "user": {"displayName": "Eva Dyer", "userId": "05212169819659068372"}, "user_tz": 240} id="cZlv_qSFQw3D" outputId="a872afc9-66f9-4ca8-85b8-0b171de4b397"
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(
    X_train.reshape(-1, 1), y_train
)  # X_train is reshape because reg expects a matrix of size (num_samples, num_feats)

# %% [markdown] id="EFhOZq8FeG7z"
# Let's visualize the fitted model

# %% colab={"base_uri": "https://localhost:8080/", "height": 430} executionInfo={"elapsed": 1061, "status": "ok", "timestamp": 1724873272817, "user": {"displayName": "Eva Dyer", "userId": "05212169819659068372"}, "user_tz": 240} id="hV8RxrYxeJYb" outputId="04203b12-6399-4d2d-90f5-e69f56b56b90"
# plot data
plt.scatter(X_train, y_train)

# plot true function
x_grid = np.linspace(X.min(), X.max(), 300)
plt.plot(x_grid, f(x_grid), c="red")

# plot approximate function
plt.plot(x_grid, reg.predict(x_grid.reshape(-1, 1)), c="green")

plt.legend(["True f", "Approximated f", "Samples"])
plt.show()


# %% [markdown] id="4m9EYdjFg3ra"
# We can clearly see how the model (in green) is **underfitting** the true function (in red). This will be further reflected when we evaluate the train and test errors.

# %% [markdown] id="vzExUSL8VDPJ"
# Compute the error (loss) on our training and test set.


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1724873272818, "user": {"displayName": "Eva Dyer", "userId": "05212169819659068372"}, "user_tz": 240} id="mzgDELLxfQ4d" outputId="46add436-bba0-470c-a00a-c37b3ba05701"
def compute_mean_squared_error(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


print(
    "Train error:",
    compute_mean_squared_error(reg.predict(X_train.reshape(-1, 1)), y_train),
)
print(
    "Test error:",
    compute_mean_squared_error(reg.predict(X_test.reshape(-1, 1)), y_test),
)


# %% [markdown] id="c_Cti25bhhXM"
# The model performs poorly on both sets.
#
# ### Increase the complexity of the features in the model
#
# The next step in this case is to use a more complex model, with potentially more parameters. We can perform **polynomial regression**!
#
# A third-order polynomial, for example, will have the following formula:
#
# $$\hat f(x) = w_0 + w_1 x + w_2 x^2 + w_3 x^3$$
#
# `sklearn` does not provide a PolynomialRegression class unfortunately, but, by computing polynomial features, we can use `LinearRegression`. This is done as follows: X contains one feature of value $x_i$ for each of its $n$ samples. We add polynomial features (up to the desired degree, here 3) as new features of X:
#
# $$
# X = \begin{bmatrix}
# x_1 \\
# x_2 \\
# \vdots \\
# x_n \\
# \end{bmatrix}
# \Longrightarrow
# \begin{bmatrix}
# x_1 & x_1^2 & x_1^3 \\
# x_2 & x_2^2 & x_2^3\\
# \vdots \\
# x_n & x_n^2 & x_n^3\\
# \end{bmatrix}
# $$


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 5, "status": "ok", "timestamp": 1724873272818, "user": {"displayName": "Eva Dyer", "userId": "05212169819659068372"}, "user_tz": 240} id="zbKzkj1Rz9rB" outputId="f9def954-d777-4aed-df6d-48798b3aadd5"
def add_polynomial_features(x, deg):
    feats = []
    for i in range(1, deg + 1):
        feats.append(x**i)
    return np.vstack(feats).T


X_train_poly = add_polynomial_features(X_train, deg=3)
X_test_poly = add_polynomial_features(X_test, deg=3)

print(
    "The data is now of shape",
    X_train_poly.shape,
    "and has",
    X_train_poly.shape[1],
    "features.",
)

# %% [markdown] id="BEkEe5julnrB"
# We can now perform polynomial regression using `LinearRegression`.

# %% colab={"base_uri": "https://localhost:8080/", "height": 75} executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1724873272818, "user": {"displayName": "Eva Dyer", "userId": "05212169819659068372"}, "user_tz": 240} id="U_KdbL8olmUN" outputId="5860679d-9bc3-4db5-95f9-71ea6137b4b4"
polyreg = LinearRegression()
polyreg.fit(X_train_poly, y_train)

# %% colab={"base_uri": "https://localhost:8080/", "height": 430} executionInfo={"elapsed": 1025, "status": "ok", "timestamp": 1724873273840, "user": {"displayName": "Eva Dyer", "userId": "05212169819659068372"}, "user_tz": 240} id="LkfRF0hilx36" outputId="dc035145-acee-4ad7-827c-feb840810d16"
# plot data
plt.scatter(X_train, y_train)

# plot true function
x_grid = np.linspace(X.min(), X.max(), 300)
plt.plot(x_grid, f(x_grid), c="red")

# plot approximate function
x_grid_poly = add_polynomial_features(x_grid, deg=3)
plt.plot(x_grid, polyreg.predict(x_grid_poly), c="green")

plt.legend(["True f", "Approximated f", "Samples"])
plt.show()

# %% [markdown] id="BDHg6Jl1nDwx"
# The model already looks better, let's look at the errors.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2, "status": "ok", "timestamp": 1724873273840, "user": {"displayName": "Eva Dyer", "userId": "05212169819659068372"}, "user_tz": 240} id="Z4zpRrSEmF8v" outputId="295b82a5-f56f-4133-c38a-40b8369eae06"
print(
    "Train error:", compute_mean_squared_error(polyreg.predict(X_train_poly), y_train)
)
print("Test error:", compute_mean_squared_error(polyreg.predict(X_test_poly), y_test))


# %% [markdown] id="-bSkaTl7oMvI"
# The performance indeed improved a lot!
# But can we do better? Let's give our model more parameters, let's use a polynomial of degree 8.


# %% colab={"base_uri": "https://localhost:8080/", "height": 487} executionInfo={"elapsed": 1384, "status": "ok", "timestamp": 1724873275222, "user": {"displayName": "Eva Dyer", "userId": "05212169819659068372"}, "user_tz": 240} id="3Fvp2dG_oVNK" outputId="b6cfe3c7-73d7-496c-9f08-a10e6b1f9b28"
def fit_polynomial(X_train, y_train, X_test, y_test, deg, plot=False):
    X_train_poly = add_polynomial_features(X_train, deg=deg)
    X_test_poly = add_polynomial_features(X_test, deg=deg)

    polyreg = LinearRegression()
    polyreg.fit(X_train_poly, y_train)

    train_error = compute_mean_squared_error(polyreg.predict(X_train_poly), y_train)
    test_error = compute_mean_squared_error(polyreg.predict(X_test_poly), y_test)

    if plot:
        # plot data
        plt.scatter(X_train, y_train)

        # plot true function
        x_grid = np.linspace(X.min(), X.max(), 300)
        plt.plot(x_grid, f(x_grid), c="red")

        # plot approximate function
        x_grid_poly = add_polynomial_features(x_grid, deg=deg)
        plt.plot(x_grid, polyreg.predict(x_grid_poly), c="green")

        plt.legend(["True f", "Approximated f", "Samples"])
        plt.title("Degree %d polynomial" % deg)
        plt.show()
    return train_error, test_error


train_error, test_error = fit_polynomial(
    X_train, y_train, X_test, y_test, deg=10, plot=True
)
print("Train error:", train_error)
print("Test error:", test_error)


# %% [markdown] id="ESmPCd8BxNFc"
# The model now looks more complex, the red curve passes through most training samples, but no longer matches the true function well. While the train error decreases, the test error starts increasing significatly: the model is **overfitting**.

# %% [markdown]
# ### Interactive Model Complexity Demo
#
# Polynomial degree is a hyperparameter that controls model complexity.
#
# Use the slider to watch the fitted curve change.
#
# Look for:
#
# - underfitting: curve too simple, high train and test error,
# - good fit: captures the main pattern,
# - overfitting: curve bends to chase noisy training points.
#


# %%
def plot_polynomial_degree(deg=3):
    X_train_poly = add_polynomial_features(X_train, deg=deg)
    X_test_poly = add_polynomial_features(X_test, deg=deg)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    train_pred = model.predict(X_train_poly)
    test_pred = model.predict(X_test_poly)
    train_error = compute_mean_squared_error(train_pred, y_train)
    test_error = compute_mean_squared_error(test_pred, y_test)

    x_grid = np.linspace(-0.05, 1.05, 400)
    x_grid_poly = add_polynomial_features(x_grid, deg=deg)

    plt.figure(figsize=(8, 5))
    plt.scatter(X_train, y_train, label="train", alpha=0.8)
    plt.scatter(X_test, y_test, label="test", alpha=0.8)
    plt.plot(x_grid, f(x_grid), color="red", linewidth=2, label="true function")
    plt.plot(
        x_grid, model.predict(x_grid_poly), color="green", linewidth=2, label="model"
    )
    plt.ylim(-2, 2)
    plt.title(
        f"Polynomial degree {deg}: train MSE={train_error:.3f}, test MSE={test_error:.3f}"
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


try:
    import ipywidgets as widgets
    from IPython.display import display

    ui = widgets.interactive(
        plot_polynomial_degree,
        deg=widgets.IntSlider(value=3, min=1, max=20, step=1, description="degree"),
    )
    display(ui)
except Exception as e:
    print("Widgets are not available here. Running selected examples instead.")
    for degree in [1, 3, 10, 20]:
        plot_polynomial_degree(degree)
    print(e)


# %% [markdown] id="AgbYe7Swx0jf"
# ## Model selection: Validation set
#
# Now that we have seen three different models, how do we proceed to find the appropriate number of parameters or complexity of our model?
#
# One commonly used approach for model selection is defining a set for validation to tune hyperparameters over. We can simulate a "test" set by holding out a few samples from the train set, to create what is called a **validation** set. It is used for hyperparameter tuning (in this case, we need to tune the degree of the polynomial) and acts as a surrogate for the test set.
#

# %% [markdown] id="gdlm6FOHWoJh"
# Let's transform 20% of the training samples to validation samples. The rest of the 80% of the training set will be used to train the model.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1724873275223, "user": {"displayName": "Eva Dyer", "userId": "05212169819659068372"}, "user_tz": 240} id="cZRwlPOjwG-L" outputId="8ee3a971-67b4-4b6e-a1b1-4ca0ce008cee"
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=3
)

print("Number of remaining training samples:", len(X_train))
print("Number of validation samples:", len(X_val))

# %% [markdown] id="IB_5vxbv6-FJ"
# We compute errors for different degrees, this allows us to draw the **training curve** and the **validation curve**.

# %% colab={"base_uri": "https://localhost:8080/", "height": 449} executionInfo={"elapsed": 1389, "status": "ok", "timestamp": 1724873276610, "user": {"displayName": "Eva Dyer", "userId": "05212169819659068372"}, "user_tz": 240} id="VQh0byTQzi3y" outputId="d888e72a-406e-4acd-8f40-bd68ddada15a"
#| code-fold: true
train_errors, val_errors = [], []
deg_range = range(1, 10)
for deg in deg_range:
    train_error, val_error = fit_polynomial(X_train, y_train, X_val, y_val, deg=deg)
    train_errors.append(train_error)
    val_errors.append(val_error)

plt.plot(deg_range, train_errors)
plt.plot(deg_range, val_errors)
plt.legend(["Training curve", "Validation curve"])
plt.xlabel("Degree of polynomial")
plt.ylabel("Error")
plt.show()


# %% [markdown]
# ### Interactive Validation Curve
#
# A validation curve plots error as a function of model complexity.
#
# The best model is usually not the one with the lowest training error. It is the one with low validation error.
#


# %%
def validation_curve(max_degree=15):
    degrees = range(1, max_degree + 1)
    train_errors, val_errors = [], []

    for deg in degrees:
        train_error, val_error = fit_polynomial(X_train, y_train, X_val, y_val, deg=deg)
        train_errors.append(train_error)
        val_errors.append(val_error)

    best_idx = int(np.argmin(val_errors))
    best_degree = list(degrees)[best_idx]

    plt.figure(figsize=(8, 5))
    plt.plot(degrees, train_errors, marker="o", label="training error")
    plt.plot(degrees, val_errors, marker="o", label="validation error")
    plt.axvline(
        best_degree, color="gray", linestyle="--", label=f"best degree = {best_degree}"
    )
    plt.xlabel("Polynomial degree")
    plt.ylabel("MSE")
    plt.title("Training vs Validation Error")
    plt.legend()
    plt.show()

    print("Best validation degree:", best_degree)
    print("Validation error:", val_errors[best_idx])


try:
    import ipywidgets as widgets
    from IPython.display import display

    ui = widgets.interactive(
        validation_curve,
        max_degree=widgets.IntSlider(
            value=15, min=3, max=25, step=1, description="max degree"
        ),
    )
    display(ui)
except Exception as e:
    print("Widgets are not available here. Running default validation curve instead.")
    validation_curve(max_degree=15)
    print(e)


# %% [markdown] id="Yy5OhMMf1qFA"
# We see that after increasing the degree of the polynomial beyond 5, the model starts overfitting the data. Thus, a 5-degree polynomial would be appropriate.
#
#

# %% [markdown] id="zv166Lf_XBEv"
# Now we can specify our final model.

# %% colab={"base_uri": "https://localhost:8080/", "height": 487} executionInfo={"elapsed": 1196, "status": "ok", "timestamp": 1724873277803, "user": {"displayName": "Eva Dyer", "userId": "05212169819659068372"}, "user_tz": 240} id="uEAg33LU1zlk" outputId="94eb6548-841b-4d84-9ff4-15d16e51dbb2"
deg = 5

X_train_poly = add_polynomial_features(X_train, deg=deg)
X_test_poly = add_polynomial_features(X_test, deg=deg)

polyreg = LinearRegression()
polyreg.fit(X_train_poly, y_train)

print(
    "Train error:", compute_mean_squared_error(polyreg.predict(X_train_poly), y_train)
)
print("Test error:", compute_mean_squared_error(polyreg.predict(X_test_poly), y_test))

# plot data
plt.scatter(X_train, y_train)

# plot true function
x_grid = np.linspace(X.min(), X.max(), 300)
plt.plot(x_grid, f(x_grid), c="red")

# plot approximate function
x_grid_poly = add_polynomial_features(x_grid, deg=deg)
plt.plot(x_grid, polyreg.predict(x_grid_poly), c="green")

plt.legend(["True f", "Approximated f", "Samples"])
plt.title("Degree %d polynomial" % deg)
plt.show()

# %% [markdown] id="CE6aDVkE0Lju"
# ## Interpolation vs. Extrapolation
#
# In general, machine learning models can **only** interpolate. There are possible exceptions, but this requires some specialized model development and/or prior knowledge of the nature of the model.
#
# If, for example, we extend the domain definition of our previous function $f$ to from $x\in[0, 1]$ to  $x\in[-1, 2]$, we will see how the model fails to extrapolate outside of the range it was trained on.

# %% colab={"base_uri": "https://localhost:8080/", "height": 452} executionInfo={"elapsed": 1270, "status": "ok", "timestamp": 1724873279072, "user": {"displayName": "Eva Dyer", "userId": "05212169819659068372"}, "user_tz": 240} id="3gpXT5QQ7lwO" outputId="014b379b-4728-444f-8cd6-78ceda385d27"
# plot true function
x_grid = np.linspace(-1, 2, 400)
plt.plot(x_grid, f(x_grid), c="red")

# plot approximate function
x_grid_poly = add_polynomial_features(x_grid, deg=deg)
plt.plot(x_grid, polyreg.predict(x_grid_poly), c="green")

plt.legend(["True f", "Approximated f"])
plt.title("Degree %d polynomial" % deg)
plt.ylim([-5, 2])
plt.vlines([0, 1], -5, 2, linestyles="dashed")
plt.show()

# %% [markdown] id="H2Y525Pg7ZLS"
#
# "Extrapolation" with machine learning models is typically achieved through search/exploration algorithms or "adaptive learning". These algorithms utilize machine-learning models to produce an iterative experimental design scheme that involves collection of new data. This effectively turns extrapolation problems into interpolation problems.

# %% [markdown] id="R6fZd9QyAYZ9"
# #### Summary
#
#
#
# **Additional resources:**
#
# - Distill article on momentum: https://distill.pub/2017/momentum/
# - Overview on cross-validation here: https://scikit-learn.org/stable/modules/cross_validation.html
#
# **Tutorials and related concepts in sklearn:**
#
# - Visualizing cross-validation behavior: https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html#sphx-glr-auto-examples-model-selection-plot-cv-indices-py
# - Comparing various optimizers: https://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_comparison.html#sphx-glr-auto-examples-linear-model-plot-sgd-comparison-py

# %% [markdown]
# ### Model Selection Checklist
#
# Before trusting a model, ask:
#
# 1. What data did the model train on?
# 2. What data were used to choose hyperparameters?
# 3. What data were held out for final testing?
# 4. Does the model perform similarly on training and validation/test data?
# 5. Are errors concentrated in a particular part of the input space?
# 6. Are we interpolating within the training range or extrapolating beyond it?
# 7. Could a simpler model perform almost as well?
#
# The goal is not to find the most complicated model. The goal is to find a model that generalizes.
#

# %% [markdown] id="fz4kIYpnASx4"
# ### ✏️ Additional Questions

# %% [markdown] id="mPeGKp8BXdm7"
# **Challenge 1.**
#
# 1. Randomly assigning samples into a training and test split is the most common approach for cross-validation. Describe at least three of the other evaluation approaches and why they may be useful for different biomedical applications.
#
# >
#
# 2. In which scenarios is random sampling no longer sufficient?
#
# >

# %% [markdown] id="79-Rqt9v9wfA"
# **Challenge 2.**
#
# 1️⃣ Why is a model that perfectly fits training data *not always good*?
#
# 2️⃣ Where do you see “overfitting” in real life?
# Examples:
#
# - memorizing exam answers vs understanding
# - overtraining in sports
# - overly tuned rules or policies
#
# 3️⃣ Why can’t we judge a model using only training error?
#

# %% [markdown] id="VP9xE6MYAxeS"
# ```

# %% [markdown] id="jGrZiQtZAePh"
# Contributors: Mehdi Azabou, Eva Dyer, AJ Medford. Some examples adapted from materials from COE 3803 taught by EL Dyer and AJ Medford.
