import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

np.random.seed(0)
house_sizes = np.random.randint(1000, 5000, 100)
house_prices = 100000 + 200 * house_sizes + np.random.normal(0, 50000, 100)

X = house_sizes.reshape(-1, 1)
y = house_prices

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual data')
plt.plot(X_test, y_pred, color='red', label='Regression line')
plt.xlabel('House Size (sq ft)')
plt.ylabel('House Price ($)')
plt.title('Linear Regression: House Price vs Size')
plt.legend()
plt.grid(True)

print(f"Intercept: {model.intercept_:.2f}")
print(f"Coefficient: {model.coef_[0]:.2f}")

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean squared error: {mse:.2f}")
print(f"R-squared score: {r2:.2f}")

plt.show()
