# Import necessary libraries
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.outliers_influence import OLSInfluence
from scipy.optimize import fsolve
from scipy.stats import t
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Load the dataset
auto_data = pd.read_csv('/Users/virensasalu/Documents/coding/UOA Coding Files/Course 3/download.csv')

# Preprocess the data
PreProcessedData = auto_data.copy()
PreProcessedData.replace('?', np.nan, inplace=True)
PreProcessedData = PreProcessedData.dropna()
PreProcessedData.drop('name', axis=1, inplace=True)
PreProcessedData['horsepower'] = PreProcessedData['horsepower'].astype(int)

# Simple Linear Regression with "mpg" as the response and "horsepower" as the predictor
X = PreProcessedData['horsepower']
y = PreProcessedData['mpg']
X = sm.add_constant(X)  # Add a constant term for the intercept

# Fit the model
linear_model = sm.OLS(y, X).fit()

# Calculate Residual Standard Error (RSE)
YPredicted = linear_model.predict(X)
residuals = y - YPredicted
mse = np.mean(residuals**2)
rse = np.sqrt(mse)

# Print model summary and RSE
print(linear_model.summary())
print(f"Residual Standard Error (RSE): {rse}")

# Values from the model summary for predictions
const_coef = linear_model.params['const']
horsepower_coef = linear_model.params['horsepower']
std_err_horsepower = linear_model.bse['horsepower']
n = len(PreProcessedData)
std_err_regression = linear_model.bse['const']

# Predicted MPG for horsepower = 98
horsepower = 98
predicted_mpg = const_coef + horsepower_coef * horsepower

# Calculate the critical t-value for 95% confidence interval
alpha = 0.05
dof = n - 2
t_critical = t.ppf(1 - alpha / 2, dof)

# Calculate confidence and prediction intervals
ConfidenceInterval = (predicted_mpg - t_critical * std_err_horsepower, predicted_mpg + t_critical * std_err_horsepower)
PredictionInterval = (predicted_mpg - t_critical * std_err_regression, predicted_mpg + t_critical * std_err_regression)

# Print results of predictions and intervals
print(f"Predicted MPG for horsepower = 98: {predicted_mpg:.2f}")
print(f"95% Confidence Interval: {ConfidenceInterval[0]:.2f} - {ConfidenceInterval[1]:.2f}")
print(f"95% Prediction Interval: {PredictionInterval[0]:.2f} - {PredictionInterval[1]:.2f}")

# Inverse prediction using fsolve to find horsepower from predicted mpg
def equation(horsepower):
    return const_coef + horsepower_coef * horsepower - predicted_mpg

InferredHorsepower = fsolve(equation, horsepower)[0]
print(f"Inferred 'horsepower' from the predicted MPG: {InferredHorsepower:.2f}")

# Visualization using seaborn's lmplot to display the least squares regression line
sns.lmplot(x='horsepower', y='mpg', data=PreProcessedData, height=6, aspect=2)
plt.title("Least Squares Regression Line")
plt.xlabel("Horsepower ---> ")
plt.ylabel("Miles Per Gallon ---> ")
plt.show()

# Diagnostic plots using plot_fit from statsmodels library 
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sm.graphics.influence_plot(linear_model, ax=axes[0, 0], criterion="cooks")
axes[0, 0].set_title("Influence Plot")
axes[0, 1].scatter(linear_model.fittedvalues, residuals)
axes[0, 1].axhline(0, color='red', linestyle='--')
axes[0, 1].set_title("Residuals vs Fitted")
axes[0, 1].set_xlabel("Fitted values")
axes[0, 1].set_ylabel("Residuals")
axes[1, 0].hist(residuals)
axes[1, 0].set_title("Histogram of Residuals")
sm.qqplot(residuals, line='s', ax=axes[1, 1])
axes[1, 1].set_title("QQ Plot of Residuals")
plt.tight_layout()
plt.show()

# Polynomial Regression to capture non-linear relationships
degree = 3  # Degree of the polynomial
poly_features = PolynomialFeatures(degree)
X_poly = poly_features.fit_transform(PreProcessedData[['horsepower']])

# Fit the polynomial regression model
poly_model = sm.OLS(y, X_poly).fit()
print(poly_model.summary())

# Cross-validation for linear regression model performance assessment
scores = cross_val_score(LinearRegression(), PreProcessedData[['horsepower']], PreProcessedData['mpg'], cv=5)
print(f"Cross-validated scores: {scores}")
print(f"Mean CV score: {scores.mean()}")

# Adding interaction terms and fitting a new model with these features
PreProcessedData['horsepower_squared'] = PreProcessedData['horsepower'] ** 2

X_new = PreProcessedData[['horsepower', 'horsepower_squared']]
X_new = sm.add_constant(X_new)
new_model = sm.OLS(y, X_new).fit()
print(new_model.summary())

# Ridge Regression to prevent overfitting in more complex models
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(PreProcessedData[['horsepower']], PreProcessedData['mpg'])
print(f"Ridge coefficient: {ridge_model.coef_}")

# Calculate AIC and BIC for model comparison between linear and polynomial models 
aic_linear = linear_model.aic
bic_linear = linear_model.bic

aic_poly = poly_model.aic
bic_poly = poly_model.bic

print(f"AIC (Linear Model): {aic_linear}, BIC (Linear Model): {bic_linear}")
print(f"AIC (Polynomial Model): {aic_poly}, BIC (Polynomial Model): {bic_poly}")