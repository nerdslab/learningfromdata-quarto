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

# %% [markdown] id="nb1a-title"
# # Overview & Data Representation
#
# **Course Title:** ENM 3800: Learning from Data
#
# **Instructor:** Eva Dyer
#
# **Lecture:** 2
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nerdslab/learningfromdata-course/blob/main/notebooks/Notebook_1/Notebook_1a_Overview_and_Data_Representation.ipynb)
#
# ## Module Theme
#
# This module asks a deceptively simple question:
#
# > What does it mean to learn from data?
#
# Before we train models, we need to understand what data are, how data become numbers, what kinds of questions we can ask, and where uncertainty enters the process.
#
# This first notebook covers the **overview** of learning from data and the **computational tools** we use to represent it: Python, NumPy, linear algebra, images, and tables. The companion notebook (Lecture 3) turns to **noise, variability, and how we model data**.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 61, "status": "ok", "timestamp": 1766276534145, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="dl44GoRCR4kN" outputId="62ca8d0e-bcc4-417d-e9a9-82afa300d02f"
print(r"""
 __        __   _                            _
 \ \      / /__| | ___ ___  _ __ ___   ___  | |_ ___
  \ \ /\ / / _ \ |/ __/ _ \| '_ ` _ \ / _ \ | __/ _ \
   \ V  V /  __/ | (_| (_) | | | | | |  __/ | || (_) |
    \_/\_/ \___|_|\___\___/|_| |_| |_|\___|  \__\___/

           Welcome to Learning From Data!
        Let's explore, discover, and build models together :)
""")


# %% [markdown]
# ## What Does It Mean to Learn From Data?
#
# Machine learning and AI are now used across science, medicine, engineering, finance, policy, entertainment, and everyday life. These tools can automate tasks, support human decision-making, reveal patterns, and make predictions.
#
# But learning from data does not start with an algorithm.
#
# It starts with a question.
#
# ---
#
# **Examples:**
#
# - Can we predict how long a food delivery will take?
# - Can we detect whether a manufactured product is defective?
# - Can we recommend a song someone might enjoy?
# - Can we estimate the risk of wildfire in different regions?
# - Can we predict how much a house will sell for?
#
# Each of these questions has to be translated into a formal data problem before a model can do anything useful.

# %% [markdown]
# ### From a Human Question to a Learning Problem
#
# A human question is usually messy:
#
# > Can we predict how much a house will sell for?
#
# To turn it into a learning problem, we need to define the building blocks.
#
# | Building Block | Meaning | House Price Example |
# | --- | --- | --- |
# | **Examples** | What is one item or row in the dataset? | One house |
# | **Features** | What information do we use as input? | Square footage, bedrooms, location, age |
# | **Target / output** | What are we trying to predict or estimate? | Sale price |
# | **Task type** | What kind of learning problem is this? | Regression |
# | **Evaluation** | How do we know if the model did well? | Average prediction error |
#
# The machine learning problem does not exist naturally in the world. We create it by deciding what counts as data, what counts as an answer, and how we will judge success.

# %% [markdown]
# ### Prediction, Explanation, and Discovery
#
# When we build models, we usually have one of several goals.

# %% [markdown]
# #### Prediction
#
# Can we make a good estimate for a new or future example?
#
# - Will this product fail inspection?
# - How long will this delivery take?
# - What will tomorrow's temperature be?
#
# Prediction cares about **future performance**.

# %% [markdown]
# #### Explanation
#
# Can we understand why something happens or which variables are associated with an outcome?
#
# - Which features are most related to house price?
# - Which conditions are associated with higher wildfire risk?
# - How does changing a process affect product quality?
#
# Explanation cares about **interpretation**, and sometimes causality.

# %% [markdown]
# #### Discovery
#
# Can we find structure without a known target?
#
# - Are there natural groups of songs or listeners?
# - Are there clusters of neighborhoods with similar transportation patterns?
# - Can we summarize high-dimensional data in two dimensions?
#
# Discovery cares about **patterns and representation**.

# %% [markdown]
# ### The Modeling Loop
#
# Learning from data is not "push button -> magic." It is an iterative process of turning a real-world question into something we can test with data.
#
# A useful version of the loop is:
#
# $$
# \text{Question}
# \rightarrow \text{Data}
# \rightarrow \text{Representation}
# \rightarrow \text{Model}
# \rightarrow \text{Evaluation}
# \rightarrow \text{Revision}
# $$
#
# Each step involves choices.
#
# | Step | Guiding Question |
# | --- | --- |
# | **Question** | What do we want to know, predict, explain, or discover? |
# | **Data** | What observations do we have, and how were they collected? |
# | **Representation** | How do we turn the real-world object into numbers? |
# | **Model** | What assumptions are we making about the pattern we want to learn? |
# | **Evaluation** | How will we decide whether the model is useful? |
# | **Revision** | What did we learn that changes our question, data, representation, or model? |
#
# The loop is important because a model can only learn from the information we give it. If the question is vague, the data are incomplete, the representation hides important structure, or the evaluation metric rewards the wrong behavior, then even a technically impressive model can give a misleading answer.
#
# A major theme of this course:
#
# > We will not just fit models. We will learn how to ask what a model learned, what it missed, and when it should be trusted.
#

# %% [markdown]
# ### What Is Data?
#
# To a computer, there is no house, patient, song, image, or delivery route. There are numbers.
#
# Different data types require different representations.
#
# | Data Type | Natural Object | Numerical Representation |
# | --- | --- | --- |
# | Tabular data | houses, patients, products | rows and columns |
# | Time series | temperature, motion, sensor traces | ordered measurements over time |
# | Images | photos, medical scans | grids of pixel values |
# | Text | reviews, documents, messages | counts, embeddings, or token sequences |
# | Networks | friendships, roads, molecules | nodes and edges |
#
# Representation shapes what a model can see.
#
# If important information is missing from the representation, the model cannot recover it by magic.
#
# This brings us to a second important question:
#
# > Who or what might be missing from the data?

# %% [markdown] id="9D4gLiC3Q7DR"
# # From Ideas to Code: Why Python?
#
# Once we define a learning problem, we need to work with the data directly.
#
# This course uses **Python** because it is widely used for data science, machine learning, and scientific computing.
#
# We will use:
#
# - `numpy` for vectors, matrices, and numerical computation,
# - `pandas` for tables,
# - `matplotlib` and `seaborn` for visualization,
# - `scikit-learn` for machine learning models.
#
# You do not need to be a Python expert. We will use Python as a tool for thinking clearly with data.
#

# %% [markdown] id="MxGwpgOsQ_4Y"
# ## What We'll Do Next
#
# In the rest of this notebook, we will build the computational foundation for the course:
#
# 1. Python basics for working in notebooks.
# 2. Arrays, vectors, and matrices as data representations.
# 3. Distances and dot products as ways to compare and combine features.
# 4. Matrix transformations as a preview of models.
# 5. Images and tables as concrete data types.
#
# Probability, variance, and covariance — the tools for describing uncertainty and the shape of data — come in the next notebook.
#
# We will not rush into complex models yet. First we build comfort with data itself.
#

# %% [markdown] id="a0XK9PHUv6w0"
# ## Python Basics

# %% [markdown] id="MRLKV_a9CnlT"
# A **cell** is a container for code to be executed by the python **kernel**. When you run the cell, its output will be displayed below. You can click the run button or press `Shift+Enter`.

# %% [markdown] id="JFZka4z6FDpz"
# ### Basic variable manipulation
#

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1766276534472, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="uOAj9XCOE2dd" outputId="938d4ec4-c35c-4bb1-cfe2-040eae832f72"
print("Hello world!")

# %% [markdown] id="g0MG1vS7ObeW"
# The equal sign `=` is used to assign a value to a variable.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1766276534475, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="blkHcrJeCDAC" outputId="03e854e7-b471-464c-cdc7-bce51db9efde"
a = 5**2  # 5 squared
print(a)

a = (a - 10) * 2
print(a)

# %% [markdown] id="VUStMrAvEs0H"
# ### `for` Statements
#

# %% [markdown] id="HKAuZyVhOe1k"
# The `for` statement is used to iterate over the indented code.
#
# Use `range()` to iterate over a sequence of numbers for example.
# Note that `range(n)` goes from `0` to `n-1`.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1, "status": "ok", "timestamp": 1766276534476, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="5ErSGqqDEluc" outputId="acc146d0-03af-4821-fb41-9afc2acc35d2"
cumsum = 0  # holds cumulative sum

for i in range(3):
    print(i)
    cumsum = cumsum + i
    print("Cumulative Sum:", cumsum)

# %% [markdown] id="TqvWmPgdML4w"
# ### Lists
#

# %% [markdown] id="pAbEVJPiOhd3"
# Lists are used to group together multiple values. They might contain items of different types, but usually the items all have the same type.
#
# `len()` is used to access the length of a list.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1, "status": "ok", "timestamp": 1766276534478, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="yJ9C68BHMe1W" outputId="e09004cd-3418-4730-b2e9-c23fde4abf10"
squares = [1, 4, 9, 16, 25]
print("This list has", len(squares), "elements.")

# %% [markdown] id="d84mmB9tMrIO"
# To access an element in the list, use indexing. Note that the first element in a list has index `0`.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2, "status": "ok", "timestamp": 1766276534482, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="J7Zwc2BbNExU" outputId="70255fd8-60e7-4a30-c1f1-a7cbd32b7f7a"
print("first element:", squares[0])  # indexing returns the item
print("third element:", squares[2])
print("last element:", squares[4])
print("also last element:", squares[-1])

# %% [markdown] id="dI7TviyyNocL"
# Lists can be sliced. `squares[a:b]` will return a new list with the elements between indices `a` and `b-1`.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1766276534488, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="lFdQSKwhNIWN" outputId="43595f8a-69d4-4c96-df25-acdbff0e9cc7"
print(squares[1:3])


# %% [markdown] id="YMC5sdUSORLK"
# ### Defining Functions

# %% [markdown] id="ZJSJIkoGPLpH"
# When certain blocks of code are to be used multiple times, defining functions can be useful. In general, it is helpful to break down the code into small an modular components.
#
# Below is an example of a function that takes in arguments and returns a value.


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1766276534492, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="LJjKaa1sOjdC" outputId="899f759e-fcc0-4672-f559-7d0e305df82c"
def compute_sum(l):
    sum = 0
    for i in range(len(l)):
        sum += l[i]
    return sum


print("The sum of", squares, "is", compute_sum(squares))

# %% [markdown] id="D_LeedYIQSOD"
# ### Python Packages

# %% [markdown] id="iBymDBKXQYB-"
# Python comes with a library of standard modules, like `math` for example. The module itself, or a specific function can be loaded in.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 5, "status": "ok", "timestamp": 1766276534501, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="43kB_mTOQ4xh" outputId="fee183f9-1189-44f7-9131-70d9dfeba44e"
import math

print(math.cos(0))

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 5, "status": "ok", "timestamp": 1766276534508, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="KRQ45LcXRAVH" outputId="e53c794a-d5ef-47bf-803f-ef69cb603f36"
from math import cos, pi

print(cos(pi))

# %% [markdown] id="tT9Ou3dxRJAH"
# More modules can be installed through `pip`, the package installer for Python. We can install the `art` for example and then load it in to make cool ASCII art.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 12961, "status": "ok", "timestamp": 1766276547471, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="jNwbJOu4RnQr" outputId="713e4853-2b8a-4fed-ca22-8ef10c9147c4"
# !pip install art #<- note that ! lets us run a bash command

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 65, "status": "ok", "timestamp": 1766276547539, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="BGmusH44RyQQ" outputId="a5bb1288-b07e-48a2-f411-8e8d41f3b7a4"
from art import art, tprint

# print pretty text
tprint("\(^-^)/")

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 12, "status": "ok", "timestamp": 1766276547557, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="P483kA-JSoOX" outputId="17fa9231-19a3-4181-9a0b-b0d824dbe32b"
# draw 3 random Strings. (Hint: try running the cell multiple times.)
for i in range(3):
    print(art("random"))

# %% [markdown] id="G7vfp0uQSq6U"
# ### Example: Creating a Matrix
#
# A list can contain any type of object, that includes lists themselves. These are called nested lists. Let's create a 3x4 matrix.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 11, "status": "ok", "timestamp": 1766276547574, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="-9PO8JaWS9lA" outputId="a70961c7-2268-442a-f915-e1d68de2df2f"
matrix = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]  # concatenates rows

print("first row:", matrix[0])
print("element at row 0 and column 2:", matrix[0][2])

# %% [markdown] id="0TBHDQprTu_6"
# Let's add 1s to the main diagonal of the matrix.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 30, "status": "ok", "timestamp": 1766276547607, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="ZzV9XnTnT_hx" outputId="1aee5a1f-7501-4078-af3d-7ced50ab60f6"
for i in range(3):
    for j in range(4):
        if i == j:
            matrix[i][j] = 1

print(matrix)

# %% [markdown] id="CEBm7bbXS2cP"
# ### More Resources:
# - https://docs.python.org/3/tutorial/introduction.html
# - https://docs.python.org/3/tutorial/

# %% [markdown] id="LbtzesVlv9u-"
# ## Linear Algebra and NumPy
#
# Linear algebra gives us the basic objects of data science.
#
# | Object | Data Science Meaning |
# | --- | --- |
# | Scalar | one number, such as a price or error |
# | Vector | one example represented by features |
# | Matrix | many examples arranged as rows and columns |
# | Dot product | a weighted sum of features |
# | Matrix multiplication | many weighted sums or a transformation of data |
#
# The key representation in many machine learning problems is:
#
# $$
# X \in \mathbb{R}^{n \times d}
# $$
#
# where:
#
# - $n$ = number of examples,
# - $d$ = number of features,
# - each row is one example,
# - each column is one feature.
#

# %% [markdown] id="1lhGMjEbYd-K"
# ### Matrices as Datasets
#
# A NumPy array is the main Python object we will use for vectors and matrices.
#
# For example, suppose each house is represented by four features:
#
# 1. square footage,
# 2. number of bedrooms,
# 3. number of bathrooms,
# 4. age of the house.
#
# Then one house is a vector:
#
# $$
# x = [1800, 3, 2, 25]
# $$
#
# and five houses form a matrix:
#
# $$
# X =
# \begin{bmatrix}
# 1800 & 3 & 2 & 25 \\
# 2400 & 4 & 3 & 10 \\
# 950 & 2 & 1 & 60 \\
# 1600 & 3 & 2 & 35 \\
# 3000 & 5 & 4 & 5
# \end{bmatrix}
# $$
#
# This row-by-column structure will appear throughout the course.
#

# %% executionInfo={"elapsed": 30, "status": "ok", "timestamp": 1766276547637, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="d6uEVt7cCD98"
import numpy as np  # <- very common shorthand for numpy

# %% [markdown] id="67FSf4ApCAF5"
# ### Creating `numpy` Arrays
#
# There are a number of ways to initialize new numpy arrays, for example from
#
# * a Python list or tuples
# * using functions that are dedicated to generating numpy arrays, such as `arange`, `linspace`, etc.
# * reading data from files

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 13, "status": "ok", "timestamp": 1766276547648, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="eipvy-g8CLSX" outputId="9715f433-1371-4b31-cc0d-0ea9b4e14bd8"
# a vector: the argument to the array function is a Python list
v = np.array([1, 2, 3, 4])

print(v)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 42, "status": "ok", "timestamp": 1766276547689, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="95sRG26JXtqo" outputId="af668cb1-b4e1-4f46-bfb7-4dbee0573ed2"
# a matrix: the argument to the array function is a nested Python list
M = np.array([[1, 2], [3, 4]])  # stacks rows

print(M)

# %% [markdown] id="FW19JfKUCKgG"
# The difference between the `v` and `M` arrays is only their shapes. We can get information about the shape and size of an array by using the `shape` and `size` properties.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1766276547693, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="HdzCjPUWCNfN" outputId="6928809e-3dcc-44f1-b0c3-7d0e6dccdb3a"
print("v: number of dimensions=", v.ndim, ", shape=", v.shape)
print("M: number of dimensions=", M.ndim, ", shape=", M.shape)

# %% [markdown] id="S7RoMKY0CTHv"
# Arrays are similar to lists, but they must contain a single type:

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1766276547696, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="byH2-pSoCVyp" outputId="3eab0b1e-d7c8-40a7-a479-4e6e989ac614"
M[0, 0] = 10
print(M)

# %% [markdown] id="NFBadfaxCY6z"
# If we want, we can explicitly define the type of the array data when we create it, using the `dtype` keyword argument:

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1, "status": "ok", "timestamp": 1766276547698, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="Mf797lLlCfEB" outputId="7c57d206-f0a8-4b51-fb4d-32a4708cad98"
v = np.array([1, 2, 3, 4], dtype=np.uint8)
print(v, v.dtype)

v = np.array([1, 2, 3, 4], dtype=np.float64)
print(v, v.dtype)

# %% [markdown] id="t-oLHLcHCbLU"
# ### Creating Arrays with Functions

# %% [markdown] id="_WGa7XYyYtPI"
# It is often more efficient to generate large arrays instead of creating them from lists. There are a few useful functions for this in numpy.
#
# `np.arange` creates a range with a specified step size (endpoints not included)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1766276547705, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="8bx2dcwoYqNQ" outputId="bb810604-0e36-49e0-f008-b396ce47fcb1"
x = np.arange(0, 4, 0.5)  # arguments: start, stop, step
print(x)

# %% [markdown] id="yanJlVjkY3ZM"
# `np.linspace` creates a range with a specified number of points (endpoints are included)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2, "status": "ok", "timestamp": 1766276547707, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="tARhIIReY7aK" outputId="366aba54-71d5-45ff-b72d-1abbc089bd8f"
x = np.linspace(0, 10, 5)
print(x)

# %% [markdown] id="rw76th5ZZIY6"
# `np.zeros` creates a matrix of zeros.
#
# `np.ones` creates a matrix of ones.
#
# `np.eye` creates an identity matrix.
#

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 52, "status": "ok", "timestamp": 1766276547759, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="KEfVrLp2ZAK_" outputId="cd0cca00-374a-453e-b211-33e81f58ab13"
print("\n 2d-Matrix of shape (2,3) filled with zeros\n", np.zeros((2, 3)))
print("\n 3d-Matrix of shape (2,2,2) filled with ones\n", np.ones((2, 3, 4)))
print("\n Identity matrix of shape (3,3)\n", np.eye(3))

# %% [markdown] id="k5TEllRMcEEm"
# ### Manipulating Arrays
#
# Once we generate `numpy` arrays, we need to interact with them. This involves a few operations:
#
# * indexing - accessing certain elements
# * index "slicing" - accessing certain subsets of elements
# * fancy indexing - combinations of indexing and slicing
#
# This is not very different from Matlab.
#
# We can index elements in an array using square brackets and indices:

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1766276547763, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="bIH4My01c2qD" outputId="1362caca-fd35-40c2-90d6-5abf0298b1a4"
# v is a vector, and has only one dimension, taking one index
print(v[0])
# M is a matrix, or a 2 dimensional array, taking two indices
print(M[1, 1])
# If an index is ommitted then the whole row is returned
print(M[1])

# %% [markdown] id="SiUdlcF_c8Rn"
# We can assign new values to elements or rows in an array using **indexing**:

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 5, "status": "ok", "timestamp": 1766276547768, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="D9MPgFNBc6Ab" outputId="bb2a4acc-4d31-4558-9259-3a31bdac0e5a"
M[:, 1] = -1
print(M)

# %% [markdown] id="qdsjmAIgdY4v"
# **Index slicing** is the name for the syntax `M[lower:upper]` to extract a subset of an array:

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2, "status": "ok", "timestamp": 1766276547771, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="s_c_z_6gdayC" outputId="4e8b0e71-0b22-4ac1-be9b-2c0c0a80b0e1"
A = np.arange(1, 20)
print(A)
print(A[1:8])

# %% [markdown] id="-ttoyhvKdtv8"
# **Fancy indexing** is the name for when an array or list is used in-place of an index:

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1766276547775, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="1WSE1Pk3dyE6" outputId="88cc444c-da01-4535-8184-6f896d0e71cb"
R = np.eye(4)
print(R, "\n")

row_indices = np.array([1, 3])
print(R[row_indices])

# %% [markdown] id="B1b7bo31eAfF"
# ### Transposing Arrays
#
# Arrays can easily be transposed with `.T`.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2, "status": "ok", "timestamp": 1766276547776, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="pnvgY0ZDeCab" outputId="eca02670-7137-497a-c439-c046bb4e15c1"
M = np.array([[1, 2, 3], [2, 1, 4]])
print(M)
print("shape", M.shape)

print("\n")

print(M.T)
print("shape", M.T.shape)

# %% [markdown] id="fWHVvhA7v8Ej"
# ### Exponentiating

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 33, "status": "ok", "timestamp": 1766276547809, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="fz74HVv6wDR0" outputId="93672d32-ae1d-4427-b850-ceaa0b45c8a5"
pow_two = 2 ** np.arange(4, 12)
print(pow_two)

# %% [markdown] id="jaNt1da2u0y_"
# ## Measuring Distance: Lᵖ Norms
#
# Once we represent data as vectors, we need a way to measure **distance** between them.
# Distance defines:
#
# - what it means for two points to be “similar”
# - how nearest-neighbor models behave
# - how optimization moves in parameter space
# - how geometry shapes learning
#
# A powerful general way to define distance is through **Lᵖ norms**.
#
# For a vector:
#
# $$
# \mathbf{x} = (x_1, x_2, ..., x_n)
# $$
#
# the **Lᵖ norm** is:
#
# $$
# \|\mathbf{x}\|_p =
# \left( \sum_{i=1}^n |x_i|^p \right)^{1/p}
# $$
#
# Two important special cases:
#
# ### L2 norm (Euclidean distance)
# $$
# \|\mathbf{x}\|_2 = \sqrt{x_1^2 + x_2^2 + ... + x_n^2}
# $$
#
# - “straight-line distance”
# - rotationally symmetric
# - what we naturally think of as distance
#
# ### L1 norm (Manhattan distance)
# $$
# \|\mathbf{x}\|_1 = |x_1| + |x_2| + ... + |x_n|
# $$
#
# - distance if you can only move horizontally & vertically
# - less sensitive to outliers
# - promotes sparsity in modeling (later: LASSO!)
#
# Different norms = different geometry = different behavior.
# Let’s visualize.
#

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1766276547810, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="DlduroaEvoLp" outputId="ab927fd4-6596-4077-f481-14601da03206"
## Simple example, computing L2 distance

v = np.array([1, 2, 0, 4, 10, 8])
w = np.array([2, 1, 2, 7, 8, 9])

dist = np.sqrt(np.sum((v - w) ** 2))
print(dist)


# %% colab={"base_uri": "https://localhost:8080/", "height": 638} executionInfo={"elapsed": 371, "status": "ok", "timestamp": 1766276548179, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="MLfpv8bmu0RF" outputId="012eef62-4041-4485-af06-f1b4de3307da"
import matplotlib.pyplot as plt
import numpy as np

# Create grid
grid = np.linspace(-1.5, 1.5, 500)
X, Y = np.meshgrid(grid, grid)

points = np.stack([X, Y], axis=-1)  # shape (N,N,2)

# Compute norms
L2 = np.linalg.norm(points, ord=2, axis=-1)
L1 = np.linalg.norm(points, ord=1, axis=-1)

# %%
#| code-fold: true
plt.figure(figsize=(6, 6))

# Draw contour where norm = 1
plt.contour(X, Y, L2, levels=[1], colors="blue", linewidths=3, label="L2")
plt.contour(X, Y, L1, levels=[1], colors="red", linewidths=3, label="L1")

plt.axhline(0, color="gray", linewidth=1)
plt.axvline(0, color="gray", linewidth=1)
plt.gca().set_aspect("equal", adjustable="box")

plt.title("Unit Contours of L1 vs L2 Norms\nComputed Directly Using np.linalg.norm")
plt.legend(
    handles=[
        plt.Line2D([0], [0], color="blue", lw=3, label="L2 norm"),
        plt.Line2D([0], [0], color="red", lw=3, label="L1 norm"),
    ]
)

plt.show()


# %% [markdown] id="rVMDTZJutCt9"
# ### More resources
# - https://numpy.org/doc/stable/user/absolute_beginners.html

# %% [markdown] id="QGvN-41qwb6v"
# ## Matrix-Vector Operations
#
#
# ### Matrix-vector multiplication as a transformation
# We now understand data as **vectors** of numbers.
# But most models transform one vector into another. How?
#
# A matrix is simply a **table of numbers**:
#
# $$
# A =
# \begin{bmatrix}
# 2 & 1 \\
# -1 & 3
# \end{bmatrix}
# $$
#
# A vector is a **list of numbers**:
#
# $$
# \mathbf{x} =
# \begin{bmatrix}
# 4 \\
# 1
# \end{bmatrix}
# $$
#
# When we multiply them:
#
# $$
# A\mathbf{x} =
# \begin{bmatrix}
# 2 & 1 \\
# -1 & 3
# \end{bmatrix}
# \begin{bmatrix}
# 4 \\
# 1
# \end{bmatrix}
# =
# \begin{bmatrix}
# 2\cdot4 + 1\cdot1 \\
# -1\cdot4 + 3\cdot1
# \end{bmatrix}
# =
# \begin{bmatrix}
# 9 \\
# -1
# \end{bmatrix}
# $$
#
# Matrix–vector multiplication = **weighted combinations of features**.
#
# Why this matters:
#
# - linear regression uses matrix–vector multiplication
# - neural networks are stacks of matrix–vector multiplications + nonlinearities
# - PCA transforms data with matrices
# - distances, projections, rotations → matrices!
#
# Let's compute our first example.
#

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 10, "status": "ok", "timestamp": 1766276548190, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="M7jPxFQew6Pb" outputId="79357dbe-2d5d-4409-b3aa-3e8299a75353"
import numpy as np

A = np.array([[2, 1], [-1, 3]])

x = np.array([4, 1])

Ax = A @ x  # matrix–vector multiplication

print("Matrix A:\n", A)
print("\nVector x:\n", x)
print("\nResult A x:\n", Ax)


# %% [markdown] id="1bxogGVXxFlg"
# ### What did the matrix *do*?
#
# You can think of a matrix as a **machine**:
#
# Input  →  process →  output
#
# $$
# \mathbf{x} \rightarrow A \rightarrow A\mathbf{x}
# $$
#
# Matrices can:
#
# - stretch vectors
# - squash vectors
# - rotate vectors
# - shear (skew) vectors
# - mix features together
#
# They reshape space.
# This is why matrices are the language of machine learning.
#

# %% [markdown]
# ### Visualizing Matrices as Transformations
#
# A matrix does not only transform one vector. It transforms every vector in the space.
#
# We will visualize this with a **flow field**: a grid of arrows showing how each point moves after applying a matrix.
#
# For a point $x$, the transformed point is:
#
# $$
# Ax
# $$
#
# The arrow shows:
#
# $$
# Ax - x
# $$
#
# So each arrow points from the original location toward the transformed location. Long arrows mean the matrix moves that part of space a lot. Short arrows mean that part of space barely moves.
#

# %%
#| code-fold: true
import matplotlib.pyplot as plt
import numpy as np


def plot_matrix_flow(
    A, title="Matrix Flow Field", lim=2.6, grid_size=11, color="purple"
):
    """
    Visualize a 2x2 matrix as a flow field.

    Each arrow starts at a point x and points in the direction A x - x.
    In other words, the arrow shows how the matrix moves that point.
    """
    grid_x, grid_y = np.meshgrid(
        np.linspace(-2, 2, grid_size), np.linspace(-2, 2, grid_size)
    )

    points = np.stack([grid_x.flatten(), grid_y.flatten()], axis=1)
    transformed = (A @ points.T).T
    displacement = transformed - points

    arrow_length = np.linalg.norm(displacement, axis=1)

    plt.figure(figsize=(6, 6))
    plt.quiver(
        points[:, 0],
        points[:, 1],
        displacement[:, 0],
        displacement[:, 1],
        arrow_length,
        angles="xy",
        scale_units="xy",
        scale=1,
        cmap="plasma",
        alpha=0.9,
    )

    plt.scatter(points[:, 0], points[:, 1], s=10, color="black", alpha=0.35)
    plt.axhline(0, color="gray", linewidth=1)
    plt.axvline(0, color="gray", linewidth=1)
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.colorbar(label="amount moved")
    plt.title(title)
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.show()

    print("Matrix A:")
    print(A)


def plot_matrix_before_after(A, title="Before and After", lim=3, grid_size=11):
    """
    Show original grid points and their transformed locations.
    This complements the flow field by showing where the points land.
    """
    grid_x, grid_y = np.meshgrid(
        np.linspace(-2, 2, grid_size), np.linspace(-2, 2, grid_size)
    )

    points = np.stack([grid_x.flatten(), grid_y.flatten()], axis=1)
    transformed = (A @ points.T).T

    plt.figure(figsize=(6, 6))
    plt.scatter(
        points[:, 0], points[:, 1], s=16, color="black", alpha=0.35, label="original"
    )
    plt.scatter(
        transformed[:, 0],
        transformed[:, 1],
        s=18,
        color="teal",
        alpha=0.75,
        label="transformed",
    )

    for start, end in zip(points, transformed):
        plt.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color="purple",
            alpha=0.15,
            linewidth=0.8,
        )

    plt.axhline(0, color="gray", linewidth=1)
    plt.axvline(0, color="gray", linewidth=1)
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend(loc="upper left")
    plt.title(title)
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.show()


# %% [markdown]
# ### Example 1: The Identity Matrix
#
# The identity matrix leaves every vector unchanged.
#
# $$
# I =
# \begin{bmatrix}
# 1 & 0 \\
# 0 & 1
# \end{bmatrix}
# $$
#
# This is the "do nothing" transformation.

# %%
I = np.array([[1, 0], [0, 1]])

plot_matrix_flow(I, title="Identity Matrix: No Flow Because Nothing Moves")

# %% [markdown]
# ### Example 2: Stretching Space
#
# This matrix stretches the first coordinate and slightly shrinks the second coordinate.
#
# $$
# A =
# \begin{bmatrix}
# 1.8 & 0 \\
# 0 & 0.6
# \end{bmatrix}
# $$
#
# The axes stay lined up, but distances change.

# %%
A_stretch = np.array([[1.8, 0], [0, 0.6]])

plot_matrix_flow(
    A_stretch, title="Stretch/Shrink Matrix: Flow Away from and Toward Axes"
)

# %% [markdown]
# ### Example 3: Rotating Space
#
# A rotation matrix turns every point by the same angle.
#
# $$
# R =
# \begin{bmatrix}
# \cos \theta & -\sin \theta \\
# \sin \theta & \cos \theta
# \end{bmatrix}
# $$
#
# Rotation changes direction, but preserves distances from the origin.

# %%
theta = np.radians(35)

A_rotate = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

plot_matrix_flow(A_rotate, title="Rotation Matrix: Circular Flow Around the Origin")

# %% [markdown]
# ### Example 4: Mixing Features with a Shear
#
# This matrix mixes the two coordinates.
#
# $$
# A =
# \begin{bmatrix}
# 1 & 0.8 \\
# 0 & 1
# \end{bmatrix}
# $$
#
# The first transformed coordinate depends partly on the second original coordinate. This is a simple example of **feature mixing**.

# %%
A_shear = np.array([[1, 0.8], [0, 1]])

plot_matrix_flow(A_shear, title="Shear Matrix: Flow Depends on the Other Feature")

# %% [markdown]
# ### Example 5: A General Transformation
#
# Now try a matrix that stretches, rotates, and mixes coordinates all at once.
#
# This is close to the kind of transformation that appears inside models: inputs are combined and moved into a new representation.

# %%
A = np.array([[1.2, 0.6], [-0.3, 1.1]])

plot_matrix_flow(A, title="General Matrix: Stretching, Rotating, and Mixing")

# Optional: show where the same points land after the transformation.
plot_matrix_before_after(A, title="Before and After the General Matrix Transformation")

# %% [markdown] id="ex-matrix-diagnose"
# ### Exercise: Diagnose a Matrix Transformation
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nerdslab/learningfromdata-course/blob/main/notebooks/Notebook_1/Notebook_1a_Overview_and_Data_Representation.ipynb#scrollTo=ex-matrix-diagnose)
#
# Try changing the matrix below.
#
# For each matrix you test, answer:
#
# 1. Does the matrix stretch, shrink, rotate, flip, or shear the space?
# 2. Do the transformed points stay on the same axes, or do the axes get mixed?
# 3. Are some regions of space moved more than others?
# 4. What would this mean if the two coordinates were real features in a dataset?
#
# Try at least three matrices:
#
# - one diagonal matrix,
# - one matrix with off-diagonal entries,
# - one matrix with a negative entry.

# %%
# Exercise: edit this matrix and rerun the cell.
A_exercise = np.array([[1.0, 0.0], [0.0, 1.0]])

plot_matrix_flow(A_exercise, title="My Matrix Flow Field")

# %% [markdown] id="9c9zSnYMxbve"
# ### Dot Product: Weighted Sums of Features
#
# Given two vectors
#
# $$
# \mathbf{a} = [a_1, a_2], \quad \mathbf{b} = [b_1, b_2]
# $$
#
# The dot product is:
#
# $$
# \mathbf{a} \cdot \mathbf{b} = a_1 b_1 + a_2 b_2
# $$
#
# Geometric meaning:
#
# - measures **similarity**
# - large if pointing in same direction
# - zero if perpendicular
# - negative if opposite
#
# Machine learning uses dot products everywhere:
#
# - linear regression predictions
# - logistic regression decisions
# - neural network activations
# - cosine similarity
#

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 8, "status": "ok", "timestamp": 1766276548420, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="9O0QkDbSyGlL" outputId="cd8675ee-6983-4099-e79e-d9ecd33088b1"
a = np.array([2, 1])
b = np.array([3, 0])
c = np.array([-1, -2])

print("a·b =", np.dot(a, b))
print("a·c =", np.dot(a, c))


# %% [markdown] id="7h0cnBLI1iUs"
# ### Quick Questions
#
# 1. What does multiplying A x do conceptually?
# - combine features?
# - stretch space?
# - just magic?
#
# 2. What might go wrong if A is badly chosen?
#
# 3. Where might you already have seen matrices?
# - image filters?
# - rotations in graphics?
# - spreadsheet tables?
#

# %% [markdown] id="SG5PDPYQ147A"
# ### Additional Operations

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1766276548425, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="Qd90pq9crmyb" outputId="32ca60f1-6e6c-4655-bf25-6c779ca3ff61"
v = np.arange(0, 5)
print("v:", v)

print("v*2:", v * 2)

print("v+2:", v + 2)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2, "status": "ok", "timestamp": 1766276548428, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="Np-zKKFNr6dD" outputId="aa19db6a-8d25-4593-cdd8-f88b888b5a32"
M = np.ones((2, 2))
print("M:\n", M)
print("M*2:\n", M * 2)
print("M+2:\n", M + 2)

# %% [markdown] id="FEoIdrjBr56r"
# When we add, subtract, multiply and divide arrays with each other, the default behaviour is **element-wise** operations. This is different from Matlab!

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 15, "status": "ok", "timestamp": 1766276548443, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="yngd8fgXr9jN" outputId="0c13b2e1-a559-474f-86fe-2e1460697154"
v = np.arange(2, 6)
print("v:", v)
print("v.v:", v * v)
print("v/v:", v / v)

M = np.array([[1, 2], [3, 4]])
print("M:\n", M)
print("M.M:\n", M * M)

# %% [markdown] id="UV-bUnFL2Lav"
# Matrix inversion

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 13, "status": "ok", "timestamp": 1766276548457, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="LQA_JfDNsHr7" outputId="6fcdb522-8dfd-488f-ec8f-7851022e38fd"
A = np.array([[-1, 2], [3, -1]])
print("A:\n", A)
print("inv(A):\n", np.linalg.inv(A))

# %% [markdown] id="ezSK4sTI2Ydz"
# ### Why its important
#
# Matrix–vector operations are the computational backbone of modern machine learning.
# Everything that follows builds on this idea:
#
# - Regression: combine features linearly
# - Logistic regression: linear combination + nonlinear squashing
# - Neural networks: many matrix multiplications stacked
# - PCA: special matrices that rotate & re-express data
#

# %% [markdown] id="ex-matrix-practice"
# ### Practice
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nerdslab/learningfromdata-course/blob/main/notebooks/Notebook_1/Notebook_1a_Overview_and_Data_Representation.ipynb#scrollTo=ex-matrix-practice)
#
# 1. Create your own 2×2 matrix and 2D vector.
# Compute A x.
#
# 2. Change one number in the matrix.
# How does the output change?
#
# 3. Try a matrix that *flips* x and y:
# $$
# \begin{bmatrix}
# 0 & 1 \\
# 1 & 0
# \end{bmatrix}
# $$
#
# 4. Try a scaling matrix:
# $$
# \begin{bmatrix}
# 2 & 0 \\
# 0 & 0.5
# \end{bmatrix}
# $$
#
# Which stretches?
# Which squashes?
#

# %% [markdown] id="ayBIw7uGmqF8"
# ## Images
#

# %% [markdown] id="1JscUW8hpnUt"
# ### Images as Data
#
# Images may *look* like pictures to us, but to a computer an image is just:
#
# > A grid of numbers.
#
# Each pixel is a measurement of light intensity.
# Different types of images simply store different kinds of numbers:
#
# - **Grayscale image**
#   - each pixel = 1 number (brightness)
# - **Color image**
#   - each pixel = 3 numbers (Red, Green, Blue)
#
# So an image is not a special object…
# it is simply a **matrix** (or sometimes a stack of matrices).
#
# This means images fit naturally into everything we’ve discussed:
#
# - vectors
# - matrices
# - arrays
# - features
#

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2, "status": "ok", "timestamp": 1766276555207, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="VITcwrHwl1VR" outputId="a4947b74-f0df-4d21-ad58-d21546f0c6dd"
# Loading a sample image
import matplotlib.cbook as cbook

w, h = 256, 256
with cbook.get_sample_data("s1045.ima.gz") as datafile:
    s = datafile.read()
img = np.frombuffer(s, np.uint16).astype(float).reshape((w, h))

print(img)
print("type:", img.dtype)
print("shape:", img.shape)

# %% colab={"base_uri": "https://localhost:8080/", "height": 452} executionInfo={"elapsed": 358, "status": "ok", "timestamp": 1766276555565, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="dqJSv6qwmGiD" outputId="01bedc28-c515-4c82-b3c4-b46c97ac1d7b"
plt.imshow(img, cmap=plt.cm.gray)
plt.title("MRI")
plt.show()

# %% [markdown] id="oq0QHn4lpv6P"
# ### What does the grayscale matrix represent?
#
# - Each entry is a **pixel**
# - The number tells us how bright the pixel is
# - Bigger number = more light
# - Smaller number = darker pixel
#
# Common conventions:
#
# - Floating point images → numbers in `[0, 1]`
# - Integer images → numbers in `[0, 255]`
#
# This structure is powerful because it means:
# > We can manipulate images using pure linear algebra.
#

# %% [markdown] id="CUk3OG4wp_B_"
# ### Color Images: 3 Numbers Per Pixel
#
# A **color image** adds another dimension.
#
# Instead of a 2D matrix:
#
# $$
# H \times W
# $$
#
# it becomes a **3D array**:
#
# $$
# H \times W \times 3
# $$
#
# because each pixel has 3 channels:
#
# | Channel | Meaning           |
# |--------|-------------------|
# | R      | Red intensity     |
# | G      | Green intensity   |
# | B      | Blue intensity    |
#
# So a color image is literally:
#
# > Three grayscale images stacked together.
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 446} executionInfo={"elapsed": 505, "status": "ok", "timestamp": 1766276556073, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="ZgROVOlHqG03" outputId="271cf981-6abb-4035-816b-bb5a9e994928"
from skimage import data

rgb = data.astronaut()

print("Color Image Shape:", rgb.shape)  # (height, width, 3)

plt.imshow(rgb)
plt.title("Color Image")
plt.axis("off")
plt.show()


# %% colab={"base_uri": "https://localhost:8080/", "height": 295} executionInfo={"elapsed": 1086, "status": "ok", "timestamp": 1766276557161, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="5Xn0gUI4qVrJ" outputId="8a14dd55-8ab0-473b-c216-f41dcee3e04b"
R = rgb[:, :, 0]
G = rgb[:, :, 1]
B = rgb[:, :, 2]

# %%
#| code-fold: true
fig, ax = plt.subplots(1, 4, figsize=(14, 4))

ax[0].imshow(rgb)
ax[0].set_title("Original")
ax[0].axis("off")

ax[1].imshow(R, cmap="Reds")
ax[1].set_title("Red Channel")
ax[1].axis("off")

ax[2].imshow(G, cmap="Greens")
ax[2].set_title("Green Channel")
ax[2].axis("off")

ax[3].imshow(B, cmap="Blues")
ax[3].set_title("Blue Channel")
ax[3].axis("off")

plt.show()


# %% [markdown] id="ex-image-representation"
# ### Quick Image Representation Exercise
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nerdslab/learningfromdata-course/blob/main/notebooks/Notebook_1/Notebook_1a_Overview_and_Data_Representation.ipynb#scrollTo=ex-image-representation)
#
# 1. Inspect the `shape` of different images:
#    - grayscale
#    - color
# 2. Answer:
#    - What does each dimension represent?
#    - Why does color need 3 channels?
# 3. Create a diagonal bright line and add Gaussian noise to it. Plot the image before and after noise.
#
# Optional challenge:
# > Zero out one color channel in the RGB image.
# What happens visually? Why?
#

# %% [markdown] id="dZujryRFnwom"
# ## Tables
#

# %% [markdown] id="CyTfxZtwoHQR"
# ### Visualizing DataFrames Using Seaborn
#
# Seaborn provides a high-level interface for drawing attractive and informative statistical graphics. It is based on matplotlib.
#
# Let's use it to visualize a widely used built in dataset consisting of different features and attributes collected from three different types of penguins (Adelie, Chinstrap, Gentoo).

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 9476, "status": "ok", "timestamp": 1766276566640, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="77Px0fRVn6XJ" outputId="7459d5f0-981a-4033-e134-6b5181024f56"
import seaborn as sns

sns.set_theme(style="ticks")

df = sns.load_dataset("penguins")
sns.pairplot(df, hue="species")
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 363} executionInfo={"elapsed": 5, "status": "ok", "timestamp": 1766276566647, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="rK6j_IxB3nj8" outputId="4ccf3ac0-1bbb-4ad8-a008-044d6deb89c0"
df.head(10)

# %% [markdown] id="0umwS5_Loamr"
# **More resources**
#
# - matplotlib cheatsheets: https://github.com/matplotlib/cheatsheets#cheatsheets
# - matplotlib gallery: https://matplotlib.org/stable/gallery/index.html
# - seaborn gallery: https://seaborn.pydata.org/examples/index.html

# %% [markdown] id="bRFEetBD2ezm"
# ### Pandas for Table Manipulation

# %% [markdown] id="wKA3dc3to_3i"
# pandas is a library for manipulating numerical tables and time series, and is a powerful tool for data analysis.
#
# pandas dataframes are a very convenient way to interact with low-dimensional structured data. The basic dataframe object acts very similarly to an Excel file, but data can be manipulated with Python rather than clumsy Excel functions.

# %% executionInfo={"elapsed": 1, "status": "ok", "timestamp": 1766276566656, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="Ba-qIxs2p1V1"
import pandas as pd

# %% [markdown] id="qQPpbAdmrTNw"
# ### Loading the Iris Dataset
#
# Load in the iris dataset!
#
# **3 classes**:  Three different types of iris flowers, Iris Setosa, Iris Versicolor, and Iris Virginica.
#
# **4 features**: Petal length and width, Sepal length and width

# %% [markdown] id="Ph3VmEdY4Zh3"
# ![index.jpg](data:image/jpeg;base64,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)

# %% executionInfo={"elapsed": 51, "status": "ok", "timestamp": 1766276566709, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="_V_4OlLAqg6K"
df = pd.read_csv(
    "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 677} executionInfo={"elapsed": 127, "status": "ok", "timestamp": 1766276566886, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="J1BhvgxAqx-k" outputId="1a377007-f916-4c22-d35b-568368396c1b"
df.head(20)

# %% [markdown] id="uS5c_2hureJ2"
# ### Computing Statistics

# %% colab={"base_uri": "https://localhost:8080/", "height": 210} executionInfo={"elapsed": 34, "status": "ok", "timestamp": 1766276566918, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="O5jMTl84sBjt" outputId="24748c2f-bf73-41ed-c155-23c94b2ecc52"
df["variety"].value_counts()

# %% colab={"base_uri": "https://localhost:8080/", "height": 300} executionInfo={"elapsed": 111, "status": "ok", "timestamp": 1766276567029, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="xFBd0VJfrgM-" outputId="4b0c5987-a023-4a6e-fe28-d54ae8fbcbc6"
df.describe()

# %% [markdown] id="TyUtlGfbtcf8"
# ### More Visualization with Jointplot

# %% colab={"base_uri": "https://localhost:8080/", "height": 601} executionInfo={"elapsed": 785, "status": "ok", "timestamp": 1766276567812, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="nYvokUYgrsMG" outputId="a2224182-ca0c-496f-c98b-5652a04d3ff5"
sns.jointplot(x="sepal.length", y="sepal.width", data=df, height=6)
plt.show()

# %% [markdown] id="E5FuJbzzthef"
# ### Computing New Features

# %% [markdown] id="GS6W-mg-t1gt"
# Because Pandas is designed to work with NumPy, most NumPy functions will work on DataFrame objects.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 14, "status": "ok", "timestamp": 1766276567827, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="ior26vA_thDv" outputId="46d1bd42-0bc3-44a7-de88-5e1acd46b8e7"
df["length_diff"] = df["sepal.length"] - df["petal.length"]
print(df.head())

# %% [markdown] id="1OswYeSDH7hv"
# **Challenge:**
#
# - Find three different tabular datasets in spreadsheet (CSV) format and load them into Colab.
#
# - Generate a **pairplot** across at most 5 features in your dataset, and provide a discussion about what you can take away about the dataset from your pairplot visualization.
#
# - Describe each dataset using the analyses in the notebook and compare both datasets in terms of their signal and noise characteristics.
#
# - Load three images into Colab that represent your interests or hobbies. Rescale and crop them to all be of equal size, and visualize them as subplots using matplotlib.

# %% executionInfo={"elapsed": 2, "status": "ok", "timestamp": 1766276567830, "user": {"displayName": "Eva Dyer", "userId": "13751912255938119410"}, "user_tz": 300} id="R28jBMzP_bJG"
# Please add code here


# %% [markdown] id="V_MB2smWtL86"
# ### More resources
# - pandas cheatsheet: https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf

# %% [markdown] id="x-27VROmIJg0"
# *Contributors*: Mehdi Azabou, Eva Dyer
