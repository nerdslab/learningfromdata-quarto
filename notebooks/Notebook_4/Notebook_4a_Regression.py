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

# %% [markdown] id="ZR5YPpPICaYP"
# # Regression and Regularization
#
# **Course Title:** ENM 3800: Learning from Data
#
# **Instructor:** Eva Dyer
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nerdslab/learningfromdata-course/blob/main/notebooks/Notebook_4/Notebook_4a_Regression.ipynb)
#
# Regression models predict continuous outcomes. In this notebook, we use the diabetes dataset to study:
#
# - linear regression with one feature and many features,
# - residuals and error metrics,
# - model coefficients as learned feature weights,
# - ridge, lasso, and elastic-net regularization,
# - robust regression with RANSAC.
#
# Big theme:
#
# > Regression is not only about fitting a line. It is about understanding prediction error, feature contributions, and how regularization changes what a model is allowed to learn.

# %% [markdown] id="_4hABC4Ds6DZ"
#
# In this lecture, we will study:
#
# * least-squares regression,
# * ridge regression,
# * sparse regression (LASSO)

# %% [markdown] id="Wt2FIl-wmaog"
# #### Dataset
#
# <p><strong>Diabetes Data Set Characteristics:</strong></p>
#
# <p>Ten baseline variables, age, body mass index, average blood
# pressure, and six blood serum measurements were obtained for each of n =
# 442 diabetes patients, as well as the response of interest, a
# *quantitative measure of disease progression one year after baseline*.</p>
#
#
# <div><dl class="field-list simple">
# <dt class="field-odd">Number of Instances</dt>
# <dd class="field-odd"><p>442</p>
# </dd>
# <dt class="field-even">Number of Attributes</dt>
# <dd class="field-even"><p>First 10 columns are numeric predictive values</p>
# </dd>
# <dt class="field-odd">Target</dt>
# <dd class="field-odd"><p>Column 11 is a quantitative measure of disease progression one year after baseline</p>
# </dd>
# <dt class="field-even">Attribute Information</dt>
#
#
# *   **age** - age in years
# *   **sex** - discrete variable
# * **bmi** - body mass index
# *  **bp** - average blood pressure
# * **tc** - (s1) total serum cholesterol
# * **ldl** - (s2) low-density lipoproteins
# * **hdl** - (s3) high-density lipoproteins
# * **tch** - (s4) total cholesterol / HDL
# * **ltg** - (s5) possibly log of serum triglycerides level
# * **glu** - (s6) blood sugar level
#
# <p>Note: Each of these 10 feature variables have been mean centered and scaled by the standard deviation times <code class="docutils literal notranslate"><span class="pre">n_samples</span></code> (i.e. the sum of squares of each column totals 1).</p>
# <p>Source URL:
# <a class="reference external" href="https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html">https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html</a></p>
# <p>For more information see:
# Bradley Efron, Trevor Hastie, Iain Johnstone and Robert Tibshirani (2004) “Least Angle Regression,” Annals of Statistics (with discussion), 407-499.
# (<a class="reference external" href="https://web.stanford.edu/~hastie/Papers/LARS/LeastAngle_2002.pdf">https://web.stanford.edu/~hastie/Papers/LARS/LeastAngle_2002.pdf</a>)</p>

# %% id="l_F49-y6QQty"
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

diabetes = datasets.load_diabetes()


# %% [markdown]
# ### Visualization Helpers for Regression
#
# These helper functions will let us inspect predictions, residuals, and coefficient behavior across models.
#


# %%
#| code-fold: true
def plot_predicted_vs_actual(y_true, y_pred, title="Predicted vs Actual"):
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(y_true, y_pred, alpha=0.7, edgecolor="black")
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    plt.plot(
        [lo, hi], [lo, hi], color="crimson", linestyle="--", label="perfect prediction"
    )
    plt.xlabel("actual target")
    plt.ylabel("predicted target")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_residuals(y_true, y_pred, title="Residual Diagnostics"):
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(y_pred, residuals, alpha=0.7, edgecolor="black")
    axes[0].axhline(0, color="crimson", linestyle="--")
    axes[0].set_xlabel("predicted target")
    axes[0].set_ylabel("residual = actual - predicted")
    axes[0].set_title("Residuals vs Predictions")

    sns.histplot(residuals, kde=True, ax=axes[1])
    axes[1].axvline(0, color="crimson", linestyle="--")
    axes[1].set_title("Residual Distribution")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_coefficients(coefs, names, title="Model Coefficients"):
    order = np.argsort(np.abs(coefs))
    plt.figure(figsize=(8, 5))
    plt.barh(np.array(names)[order], np.array(coefs)[order])
    plt.axvline(0, color="black", linewidth=1)
    plt.title(title)
    plt.xlabel("coefficient value")
    plt.show()


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1725910138408, "user": {"displayName": "Eva Dyer", "userId": "05212169819659068372"}, "user_tz": 240} id="_B6C6-Cu4KPW" outputId="d9c87f34-9034-436c-f486-36a8ca22b51e"
feature_names = diabetes["feature_names"]
X = diabetes["data"]
y = diabetes["target"]

print("Feature names:", feature_names)
print("Number of samples:", X.shape[0])

# %% [markdown] id="Dvn3FUeZ6Vb9"
# **Splitting the dataset into a train and test set**

# %% id="ETQ5p0om6UAs"
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# %% [markdown] id="e2xK9-XN6Fm5"
# #### Least-squares regression
#
# Linear Regression fits a linear model with coefficients
#  to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation. Mathematically it solves a problem of the form:
#
#  $$\min_{\beta} || X \beta - y||_2^2$$
#
#  In a previous lecture we saw how to derive the LS estimator:
#
# $$\widehat{\beta} = (X^T X)^{+}X^T y$$
#

# %% [markdown] id="79QiQLTq7Z_M"
# ##### 2.1. Fit the model to a single feature
#
# We're already familiar with `sklearn` API. We can use the `LinearRegression` class to define parameters, fit training data and predict a measure of the disease progress. To begin, we can *fit the model to a single feature*, the BMI.
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 449} executionInfo={"elapsed": 1625, "status": "ok", "timestamp": 1725910140030, "user": {"displayName": "Eva Dyer", "userId": "05212169819659068372"}, "user_tz": 240} id="URfYZIpa6GDZ" outputId="fcd4090b-8414-48ec-d0b4-87a5712694df"
bmi_train = X_train[:, 2:3]
bmi_test = X_test[:, 2:3]

# create linear regression object
regr = linear_model.LinearRegression()

# fit training samples
regr.fit(bmi_train, y_train)

# predict on test samples
y_test_pred = regr.predict(bmi_test)

plt.scatter(bmi_train, y_train, color="black")
plt.scatter(bmi_test, y_test, color="red")
plt.plot(bmi_test, y_test_pred, color="blue", linewidth=3)
plt.xlabel("BMI")
plt.ylabel("disease progression")
plt.legend(["linear fit", "train samples", "test samples"])
plt.show()


# %% [markdown] id="al8buqQw70Kk"
# **Model details:**
#
# This linear model deals with one feature only, it has the following equation:
# $$\hat y(x) = a x + b$$
#
# When `.fit` is called, the model will find the slope $a$ and the intercept $b$ that minimize the squared error over the training samples.
#
# The parameters are saved in the `regr` object as attributes `coef_` and `intercept_` for $a$ and $b$ respectively.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1725910140030, "user": {"displayName": "Eva Dyer", "userId": "05212169819659068372"}, "user_tz": 240} id="xu3MInGf8wtF" outputId="d265a80a-d0be-40a1-96d5-e454e6affa37"
print("Slope: %.2f" % regr.coef_[0])
print("Intercept: %.2f" % regr.intercept_)

# %% [markdown] id="NXe10PPF7Yeb"
# ##### 2.2. Accuracy measures
#
# The linear regression model is now fit to the data. It can be used on any sample to predict the corresponding $y$, using `.predict()`
#
# We can compute the **mean squared error** or the **coefficient of determination** on the training set and the testing set.
#
# Recall that the coefficient of determination or the $R^2$ is defined as:
# $$ R^2 = 1 - \frac{\sum_{i} (y_i - \widehat{y}_i)^2}{\sum_{i} (y_i - \bar{y})^2},$$
# where $\bar{y} = \frac{1}{n} \sum_{i=1}^{n} y_i$ and $\widehat{y}_i$ is the prediction for the $i^{\rm th}$ sample.
#

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1725910140030, "user": {"displayName": "Eva Dyer", "userId": "05212169819659068372"}, "user_tz": 240} id="O4EHEddeC8IH" outputId="5221b2e7-c7d0-474e-d264-a22cc5ea83a0"
from sklearn.metrics import mean_squared_error, r2_score

# Metrics on the training set
y_train_pred = regr.predict(bmi_train)
print("Train set: mean squared error: %.2f" % mean_squared_error(y_train, y_train_pred))
print("Train set: coefficient of determination: %.2f" % r2_score(y_train, y_train_pred))

# Metrics on the test set
y_test_pred = regr.predict(bmi_test)
print("Test set: mean squared error: %.2f" % mean_squared_error(y_test, y_test_pred))
print("Test set: coefficient of determination: %.2f" % r2_score(y_test, y_test_pred))

# %% [markdown] id="-RlJcF9f-mbf"
# #### Least-squares (Multi-features)
#
# We find that the linear model doesn't perform well on the test set. While this indicates that the model does not generalize well yet. To improve the performance of our predictive model, we can use all of the features we have at our disposal.
#
#

# %% [markdown] id="cdbNwdfcV-Z6"
# ##### 3.1 Take a closer look at the features
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000, "output_embedded_package_id": "118-OCMCKMZsSBKtd69U1Hv-LJrGliQNe"} executionInfo={"elapsed": 77376, "status": "ok", "timestamp": 1725910217403, "user": {"displayName": "Eva Dyer", "userId": "05212169819659068372"}, "user_tz": 240} id="fZfxQRpuAAZT" outputId="54917344-b214-4cfe-a895-d57e4482e3b6"
import pandas as pd
import seaborn as sns

df = pd.DataFrame(
    np.hstack([X_train, y_train[:, np.newaxis]]), columns=feature_names + ["y"]
)
sns.pairplot(df, hue="y", diag_kind=None)

# %% [markdown] id="HpBl9IGW_-uf"
# ##### 3.2. Fit a linear regression model to all features

# %% id="nDN7JPQ9_Oh1"
# create linear regression object
regr = linear_model.LinearRegression()

# fit training samples
regr.fit(X_train, y_train)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 37, "status": "ok", "timestamp": 1725910217403, "user": {"displayName": "Eva Dyer", "userId": "05212169819659068372"}, "user_tz": 240} id="FFiDCbC8_WIQ" outputId="98bfde07-331f-45f6-cbd8-a6e4c768f05f"
y_train_pred = regr.predict(X_train)
print("Train set: mean squared error: %.2f" % mean_squared_error(y_train, y_train_pred))
print("Train set: coefficient of determination: %.2f" % r2_score(y_train, y_train_pred))

y_test_pred = regr.predict(X_test)
print("Test set: mean squared error: %.2f" % mean_squared_error(y_test, y_test_pred))
print("Test set: coefficient of determination: %.2f" % r2_score(y_test, y_test_pred))

# %% [markdown]
# ### Visual Diagnostic: Residuals
#
# A single error number is useful, but it hides *where* the model is wrong.
#
# Residual plots help us ask:
#
# - Are errors centered around zero?
# - Are errors larger for high or low predictions?
# - Are there outliers?
# - Is the model missing nonlinear structure?
#

# %%
plot_predicted_vs_actual(
    y_test, y_test_pred, title="Linear Regression: Predicted vs Actual"
)
plot_residuals(y_test, y_test_pred, title="Linear Regression Residual Diagnostics")


# %% [markdown] id="ex-read-residual-plot"
# #### Exercise: Read the Residual Plot
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nerdslab/learningfromdata-course/blob/main/notebooks/Notebook_4/Notebook_4a_Regression.ipynb#scrollTo=ex-read-residual-plot)
#
# 1. Are residuals mostly centered around zero?
# 2. Are large target values predicted well or poorly?
# 3. Do you see any outliers?
# 4. What might a pattern in the residuals suggest about the model?
#

# %% [markdown] id="gHsk-RAxCpO1"
# ##### 3.3. The performance of the model increased. Let's take a look at the weights assigned to each feature:

# %% colab={"base_uri": "https://localhost:8080/", "height": 385} executionInfo={"elapsed": 36, "status": "ok", "timestamp": 1725910217403, "user": {"displayName": "Eva Dyer", "userId": "05212169819659068372"}, "user_tz": 240} id="658leqb0_ZOe" outputId="9ae3b1cd-25bc-4e77-ce9d-c1865451114a"
#| code-fold: true
plt.figure(figsize=(10, 4))
plt.bar(feature_names, regr.coef_)
plt.ylabel("weight")
print("Intercept: %.2f" % regr.intercept_)

# %% [markdown] id="BTlHiWvJDm3F"
# #### Ridge Regression

# %% [markdown] id="82YpL9Q1DvZN"
# $$\min_{\beta} || X \beta - y||_2^2 + \alpha ||\beta||_2^2$$
#
# The hyperparameter  $\alpha$ controls the amount of shrinkage: the larger the value of $\alpha$, the greater the amount of shrinkage and thus the coefficients become more robust to collinearity.
#

# %% [markdown] id="-7jZNXbXyHpZ"
# ##### 4.1 Apply ridge to diabetes dataset
#
# Now let's apply the ridge estimator to our data from before.

# %% id="s5aSU_QlDOvx"
regr = linear_model.Ridge(alpha=0.5)

# fit training samples
regr.fit(X_train, y_train)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 33, "status": "ok", "timestamp": 1725910217404, "user": {"displayName": "Eva Dyer", "userId": "05212169819659068372"}, "user_tz": 240} id="aYmvymeUD3uT" outputId="bffb435d-72c0-4e14-85a6-70988039bc49"
y_train_pred = regr.predict(X_train)
print("Train set: mean squared error: %.2f" % mean_squared_error(y_train, y_train_pred))
print("Train set: coefficient of determination: %.2f" % r2_score(y_train, y_train_pred))

y_test_pred = regr.predict(X_test)
print("Test set: mean squared error: %.2f" % mean_squared_error(y_test, y_test_pred))
print("Test set: coefficient of determination: %.2f" % r2_score(y_test, y_test_pred))

# %% [markdown] id="mobxbJTQEtjn"
# ##### 4.2. What happens as we increase the amount of regularization?

# %% colab={"base_uri": "https://localhost:8080/", "height": 340} executionInfo={"elapsed": 33, "status": "ok", "timestamp": 1725910217405, "user": {"displayName": "Eva Dyer", "userId": "05212169819659068372"}, "user_tz": 240} id="nLwtyOSOE1tW" outputId="9cf7f273-90e0-4ef7-a576-eb9689c64367"
fig, axs = plt.subplots(1, 4, figsize=(25, 4))

for i, alpha in enumerate([0.1, 0.5, 1.0, 10.0]):
    regr = linear_model.Ridge(alpha=alpha)
    regr.fit(X_train, y_train)

    axs[i].bar(feature_names, regr.coef_)
    axs[i].set_ylim([-260, 530])
    axs[i].set_title("alpha=%.1f" % alpha)
axs[0].set_ylabel("weight")
plt.show()

# %% [markdown] id="QP5BPkxeFL7Y"
# ### Lasso

# %% [markdown] id="Tsyg42hNF6Xi"
# Instead of regularizing with the L2-norm, we can use other p-norms. When we use the L1-norm, this regularized estimator is called the LASSO (or Basis Pursuit Denoising, BPDN).
#
# $$\min_{\beta} \frac{1}{2}||X \beta - y||_2 ^ 2 + \lambda ||\beta ||_1$$

# %% [markdown] id="4Cg8g4zhy31H"
# ##### 5.1 Apply LASSO to diabetes data

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 32, "status": "ok", "timestamp": 1725910217405, "user": {"displayName": "Eva Dyer", "userId": "05212169819659068372"}, "user_tz": 240} id="SMRG-XW2FIDH" outputId="bbbcaaa7-a1a9-4115-91fb-9d2a8ddaa65a"
regr = linear_model.Lasso(alpha=0.5)

# fit training samples
regr.fit(X_train, y_train)

y_train_pred = regr.predict(X_train)
print("Train set: mean squared error: %.2f" % mean_squared_error(y_train, y_train_pred))
print("Train set: coefficient of determination: %.2f" % r2_score(y_train, y_train_pred))

y_test_pred = regr.predict(X_test)
print("Test set: mean squared error: %.2f" % mean_squared_error(y_test, y_test_pred))
print("Test set: coefficient of determination: %.2f" % r2_score(y_test, y_test_pred))

# %% [markdown] id="J5S3b07KyxkM"
# ##### 5.2. Varying regularization strength
#
# Let's examine the impact of regularization on the coefficients obtained via the LASSO.

# %% colab={"base_uri": "https://localhost:8080/", "height": 340} executionInfo={"elapsed": 30, "status": "ok", "timestamp": 1725910217405, "user": {"displayName": "Eva Dyer", "userId": "05212169819659068372"}, "user_tz": 240} id="XOqk3OO7GEk3" outputId="d9aca8aa-36c3-45c9-92e6-824bb7c8ce5e"
fig, axs = plt.subplots(1, 4, figsize=(25, 4))

for i, alpha in enumerate([0.01, 0.1, 0.5, 1.0]):
    regr = linear_model.Lasso(alpha=alpha)
    regr.fit(X_train, y_train)

    axs[i].bar(feature_names, regr.coef_)
    axs[i].set_ylim([-260, 530])
    axs[i].set_title("alpha=%.1f" % alpha)
axs[0].set_ylabel("weight")
plt.show()

# %% [markdown] id="QmUzLfagmqqa"
# **Challenge:** How does this picture look different from what we saw in ridge regression?
#
# `
# Please provide your reply here.
# `
#

# %% [markdown]
# ### Visualizing Regularization Paths
#
# Regularization changes the learned coefficients.
#
# - Ridge shrinks coefficients smoothly toward zero.
# - Lasso can shrink some coefficients exactly to zero, performing feature selection.
#
# The next visualization shows how coefficients change as the regularization strength `alpha` increases.
#

# %%
from sklearn.linear_model import Lasso, Ridge

alphas = np.logspace(-3, 2, 80)
ridge_coefs = []
lasso_coefs = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha, max_iter=10000)
    ridge.fit(X_train, y_train)
    ridge_coefs.append(ridge.coef_)

    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train, y_train)
    lasso_coefs.append(lasso.coef_)

ridge_coefs = np.array(ridge_coefs)
lasso_coefs = np.array(lasso_coefs)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for j, name in enumerate(feature_names):
    axes[0].plot(alphas, ridge_coefs[:, j], label=name)
    axes[1].plot(alphas, lasso_coefs[:, j], label=name)

for ax, title in zip(axes, ["Ridge coefficient path", "Lasso coefficient path"]):
    ax.set_xscale("log")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("alpha regularization strength")
    ax.set_ylabel("coefficient")
    ax.set_title(title)

axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()


# %% [markdown] id="ex-compare-ridge-lasso"
# #### Exercise: Compare Ridge and Lasso
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nerdslab/learningfromdata-course/blob/main/notebooks/Notebook_4/Notebook_4a_Regression.ipynb#scrollTo=ex-compare-ridge-lasso)
#
# 1. Which method sets coefficients exactly to zero?
# 2. Which coefficients survive longest as `alpha` increases?
# 3. Why might sparse coefficients be easier to interpret?
# 4. Why might too much regularization underfit?
#

# %% [markdown] id="G82CIh90OONC"
# ### Elastic-Net
#
# In elastic net regularization, the penalty term is a linear combination of the L1 and L2 penalties:

# %% [markdown] id="Oyc9L3IbOk5q"
# $$ ||X \beta - y||_2 ^ 2 + \alpha \rho ||\beta||_1 +
# \frac{\alpha(1-\rho)}{2} ||\beta||_2 ^ 2 $$

# %% [markdown] id="jzrJoP9cnUyq"
# 5.1 Implementing elastic net
#
# In `scikit-learn`, $\rho$ is controlled by the 'l1_ratio' parameter: An 'l1_ratio' of 1 corresponds to an L1 penalty, and anything lower is a combination of L1 and L2. When you set the ratio to zero, you get ridge regression.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 29, "status": "ok", "timestamp": 1725910217405, "user": {"displayName": "Eva Dyer", "userId": "05212169819659068372"}, "user_tz": 240} id="nYfyFiRjICCS" outputId="687cf320-2b95-4cb9-8212-b4fe1bbf93db"
regr = linear_model.ElasticNet(alpha=0.05, l1_ratio=0.8)

# fit training samples
regr.fit(X_train, y_train)

y_train_pred = regr.predict(X_train)
print("Train set: mean squared error: %.2f" % mean_squared_error(y_train, y_train_pred))
print("Train set: coefficient of determination: %.2f" % r2_score(y_train, y_train_pred))

y_test_pred = regr.predict(X_test)
print("Test set: mean squared error: %.2f" % mean_squared_error(y_test, y_test_pred))
print("Test set: coefficient of determination: %.2f" % r2_score(y_test, y_test_pred))

# %% [markdown] id="ShXr0WmdIYXy"
# 5.2 Cross-validation for model selection
#
# Tuning these hyperparameters can get tricky, we can use the cross-validation strategy to identify the best set of parameters
#
# The class `ElasticNetCV` can be used to set the parameters alpha and l1_ratio by cross-validation.

# %% id="Mosd2CkFOaL8"
# cross-validation for hyperparameter selection
regr = linear_model.ElasticNetCV(
    cv=5, random_state=1, l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1]
)

# %% [markdown] id="ytl129W5nGJi"
# 3. Fit model with selected hyperparameters

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 28, "status": "ok", "timestamp": 1725910217406, "user": {"displayName": "Eva Dyer", "userId": "05212169819659068372"}, "user_tz": 240} id="h8hXUOOynE6-" outputId="3cd9233e-4efb-4494-cedf-af510270579c"
# fit training samples
regr.fit(X_train, y_train)

y_train_pred = regr.predict(X_train)
print("Train set: mean squared error: %.2f" % mean_squared_error(y_train, y_train_pred))
print("Train set: coefficient of determination: %.2f" % r2_score(y_train, y_train_pred))

y_test_pred = regr.predict(X_test)
print("Test set: mean squared error: %.2f" % mean_squared_error(y_test, y_test_pred))
print("Test set: coefficient of determination: %.2f" % r2_score(y_test, y_test_pred))

# %% [markdown] id="kyV07HPtI5Hw"
# The optimal hyperparameters found for this model were:

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 27, "status": "ok", "timestamp": 1725910217406, "user": {"displayName": "Eva Dyer", "userId": "05212169819659068372"}, "user_tz": 240} id="RmAVMZsbI4Th" outputId="cfde613a-5325-443c-9e93-b28b7744a46b"
print("alpha:", regr.alpha_)
print("l1_ratio:", regr.l1_ratio_)

# %% [markdown] id="M_YM8mAHnraH"
# Which values did we search over?

# %% colab={"base_uri": "https://localhost:8080/", "height": 75} executionInfo={"elapsed": 26, "status": "ok", "timestamp": 1725910217406, "user": {"displayName": "Eva Dyer", "userId": "05212169819659068372"}, "user_tz": 240} id="ViB5Xlodnt_2" outputId="57967454-9de0-4504-fe07-3676c00d7a0b"
regr

# %% [markdown]
# ### Why Robust Regression?
#
# Least-squares regression can be strongly affected by outliers because squared error gives very large penalties to large residuals.
#
# RANSAC tries to fit a model using a subset of points that appear consistent, while treating unusual points as possible outliers.
#

# %%
from sklearn.linear_model import LinearRegression, RANSACRegressor

rng = np.random.default_rng(3)
x_demo = np.linspace(0, 10, 80)
y_demo = 2.0 * x_demo + 1.0 + rng.normal(0, 1.0, size=len(x_demo))

# Add outliers
outlier_idx = rng.choice(len(x_demo), size=10, replace=False)
y_demo[outlier_idx] += rng.normal(18, 4, size=len(outlier_idx))

X_demo = x_demo.reshape(-1, 1)
ols = LinearRegression().fit(X_demo, y_demo)
ransac = RANSACRegressor(LinearRegression(), random_state=0).fit(X_demo, y_demo)

x_grid = np.linspace(0, 10, 200).reshape(-1, 1)

plt.figure(figsize=(8, 5))
plt.scatter(x_demo, y_demo, alpha=0.75, label="data")
plt.scatter(
    x_demo[outlier_idx],
    y_demo[outlier_idx],
    facecolor="none",
    edgecolor="crimson",
    s=110,
    linewidth=2,
    label="injected outliers",
)
plt.plot(x_grid, ols.predict(x_grid), label="ordinary least squares", linewidth=2)
plt.plot(x_grid, ransac.predict(x_grid), label="RANSAC", linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Outliers Can Pull Least-Squares Regression")
plt.legend()
plt.show()


# %% [markdown] id="k9ee9CZ-QY1_"
# ### RANSAC: RANdom SAmple Consensus

# %% [markdown] id="1EB12P37Qj-6"
# RANSAC consists of the following steps:
#
# 1. Fit a model to **min_samples random samples** from the original data (base_estimator.fit).
#
# 2. Classify all data as inliers or outliers by calculating the residuals to the estimated model (base_estimator.predict(X) - y) - **all data samples with absolute residuals smaller than the residual_threshold are considered as inliers**.
#
# 3. Save fitted model as **best model if number of inlier samples is maximal**. If the current estimated model has the same number of inliers, it is only considered as the best model if it has better score.
#
# These steps are performed either a maximum number of times (max_trials) or until one of the special stop criteria are met (see stop_n_inliers and stop_score). **The final model is estimated using all inlier samples (consensus set) of the previously determined best model.**
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 75} executionInfo={"elapsed": 25, "status": "ok", "timestamp": 1725910217406, "user": {"displayName": "Eva Dyer", "userId": "05212169819659068372"}, "user_tz": 240} id="HpNSof_4PeYY" outputId="d48808fb-ab41-4879-ac5e-4edcd517c138"
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets, linear_model

n_samples = 1000
n_outliers = 50

X, y, coef = datasets.make_regression(
    n_samples=n_samples,
    n_features=1,
    n_informative=1,
    noise=10,
    coef=True,
    random_state=0,
)

# Add outlier data
np.random.seed(0)
X[:n_outliers] = 3 + 0.5 * np.random.normal(size=(n_outliers, 1))
y[:n_outliers] = -3 + 10 * np.random.normal(size=n_outliers)

# Fit line using all data
lr = linear_model.LinearRegression()
lr.fit(X, y)

# %% id="-kpyL80ulmaN"
# Robustly fit linear model with RANSAC algorithm
ransac = linear_model.RANSACRegressor()
ransac.fit(X, y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

# Predict data of estimated models
line_X = np.arange(X.min(), X.max())[:, np.newaxis]
line_y = lr.predict(line_X)
line_y_ransac = ransac.predict(line_X)

# %% colab={"base_uri": "https://localhost:8080/", "height": 484} executionInfo={"elapsed": 171, "status": "ok", "timestamp": 1725910217553, "user": {"displayName": "Eva Dyer", "userId": "05212169819659068372"}, "user_tz": 240} id="wSuhhGn2le0A" outputId="1a69e773-4fcf-4e64-b948-2704241d09f2"
# Compare estimated coefficients
print("Estimated coefficients (true, linear regression, RANSAC):")
print(coef, lr.coef_, ransac.estimator_.coef_)

lw = 2
plt.scatter(
    X[inlier_mask], y[inlier_mask], color="yellowgreen", marker=".", label="Inliers"
)
plt.scatter(
    X[outlier_mask], y[outlier_mask], color="gold", marker=".", label="Outliers"
)
plt.plot(line_X, line_y, color="navy", linewidth=lw, label="Linear regressor")
plt.plot(
    line_X,
    line_y_ransac,
    color="cornflowerblue",
    linewidth=lw,
    label="RANSAC regressor",
)
plt.legend(loc="lower right")
plt.xlabel("Input")
plt.ylabel("Response")
plt.show()

# %% [markdown] id="bc4zVeD1NXhh"
# **Challenge:**
#
# - Build at least 4 different regression models for the diabetes dataset using sklearn. You can use any of the [linear models in sklearn](https://https://scikit-learn.org/stable/modules/linear_model.html), polynomial regression, or other regression models you find in sklearn. Report their accuracy on the same training, validation, and test set.
#
# - Then, visualize the accuracy of your models as a barplot, with 3 different *subfigures* for the accuracy on your (a) training, (b) validation, and (c) test data.
#
# - Add a text block discussing your results, and whether the models are under- or over-fitting. Go back to the pairplot or other visualizations to make sense of your results.
#

# %% id="rO9nqj1fJkGo"
# add code here
#

# %% [markdown] id="YPhFMIbvBLKq"
# **Tutorials and related concepts in sklearn:**
#
#
# * Orthogonal matching pursuit (OMP) - https://scikit-learn.org/stable/modules/linear_model.html#orthogonal-matching-pursuit-omp
# * Kernel ridge regression - https://scikit-learn.org/stable/modules/kernel_ridge.html
# *   Linear regressors - https://scikit-learn.org/stable/modules/classes.html#classical-linear-regressors
# *   Regressors with variable selection - https://scikit-learn.org/stable/modules/classes.html#regressors-with-variable-selection
# * Multitask regressors - https://scikit-learn.org/stable/modules/classes.html#multi-task-linear-regressors-with-variable-selection
# * Outlier-robust regressors - https://scikit-learn.org/stable/modules/classes.html#multi-task-linear-regressors-with-variable-selection
#
#

# %% [markdown] id="1CjtMVDls-j0"
# Contributors: Mehdi Azabou, Eva Dyer
