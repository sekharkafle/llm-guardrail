import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Streamlit app
st.title("Iris Classification")

# Input features
st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("Sepal Length", 4.3, 7.9, 5.8)
sepal_width = st.sidebar.slider("Sepal Width", 2.0, 4.4, 3.1)
petal_length = st.sidebar.slider("Petal Length", 1.0, 6.9, 3.8)
petal_width = st.sidebar.slider("Petal Width", 0.1, 2.5, 1.2)

# Make predictions
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(input_data)
predicted_class = iris.target_names[prediction[0]]

# Display results
st.write("Predicted Class:", predicted_class)