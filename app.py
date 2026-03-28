import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Title
st.title("📚 Library Seat Occupancy Modelling")

st.write("This model predicts library seat usage during exam season using Logistic Growth Model.")

# Inputs
st.sidebar.header("Input Parameters")

K = st.sidebar.slider("Total Seats (K)", 50, 500, 200)
O0 = st.sidebar.slider("Initial Occupancy (O₀)", 1, 100, 20)
r = st.sidebar.slider("Growth Rate (r)", 0.1, 1.0, 0.6)
days = st.sidebar.slider("Number of Days", 5, 30, 15)

# Time array
t = np.arange(0, days)

# Logistic formula
occupancy = K / (1 + ((K - O0) / O0) * np.exp(-r * t))

# Display results
st.subheader("📊 Predicted Seat Occupancy")

for i in range(days):
    st.write(f"Day {i}: {occupancy[i]:.2f} seats occupied")

# Plot graph
st.subheader("📈 Occupancy Graph")

fig, ax = plt.subplots()
ax.plot(t, occupancy)
ax.set_xlabel("Days")
ax.set_ylabel("Occupied Seats")
ax.set_title("Library Seat Occupancy Over Time")

st.pyplot(fig)

# Peak occupancy
st.subheader("📌 Peak Occupancy")

peak_day = np.argmax(occupancy)
st.write(f"Peak occupancy occurs around Day {peak_day}")
st.write(f"Maximum seats occupied ≈ {occupancy[peak_day]:.2f}")