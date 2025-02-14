import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Title
st.title("Linear Regression: Predict Weight Based on Height")

# Step 1: Create lists of height and weight data
heights = [150, 160, 165, 170, 175, 180, 185, 190, 195, 200]
weights = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

# Step 2: Create a DataFrame
data = pd.DataFrame({"Height": heights, "Weight": weights})

# Display dataset
st.header("Dataset")
st.write(data)

# Step 3: Prepare the data for regression
X = data["Height"].values.reshape(-1, 1)  # Feature (height)
y = data["Weight"].values  # Target (weight)

# Step 4: Train the model
model = LinearRegression()
model.fit(X, y)

# Step 5: User Input for height prediction
st.header("Predict Weight Based on Your Height")

# User input for height (in cm)
user_height = st.number_input("Enter Height (in cm):", min_value=100, max_value=250, value=170)

# Step 6: Predict the weight based on the input height
predicted_weight = model.predict([[user_height]])

# Display the predicted weight
st.write(f"Predicted Weight for height {user_height} cm is: {predicted_weight[0]:.2f} kg")

# Step 7: Display the regression line
st.header("Regression Line")
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.title("Height vs Weight Linear Regression")
plt.legend()
st.pyplot(plt)
