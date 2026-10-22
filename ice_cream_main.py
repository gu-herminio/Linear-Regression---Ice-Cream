from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# X for temperature

X  = [[15],[18],[20],[25],[30],[35],[40]]

# Y for ice cream sales

Y = [5, 10, 14, 25, 45, 60, 70]

# Create a logistic regression model

model = LogisticRegression()

# Train the model

model.fit(X, Y)

# Predict the ice cream sales for a new temperature

new_temperature = 22

predicted_sales = model.predict([[new_temperature]])

# Print the predicted sales

print(f"Predicted sales for {new_temperature} degrees: {predicted_sales[0]}")

# Inclination of the model

print(f"Inclination of the model: {model.coef_[0][0]}")

# Intercept of the model

print(f"Intercept of the model: {model.intercept_[0]}")

# R-squared of the model

print(f"R-squared of the model: {model.score(X, Y)}")

# Plot the model with better visualization, the title should be "Ice Cream Sales vs Temperature", the x label should be "Temperature", the y label should be "Sales", the text should be "Inclination: {model.coef_[0][0]}", "Intercept: {model.intercept_[0]}", the text should be in the top right corner of the plot.

plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='blue', label='Data')
plt.plot(X, model.predict(X), color='red', label='Model')
plt.title('Ice Cream Sales vs Temperature')
plt.xlabel('Temperature')
plt.ylabel('Sales')
plt.legend()
plt.show()

