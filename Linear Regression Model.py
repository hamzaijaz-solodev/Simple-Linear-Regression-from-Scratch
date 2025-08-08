import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")

#Mean Squared Error
def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].study_time
        y = points.iloc[i].score
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(points))

#Gradient Descent
def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    n = len(points)

    for i in range(n):
        x = points.iloc[i].study_time
        y = points.iloc[i].score

        m_gradient += -(2/n) * x * (y - ((m_now * x + b_now)))
        b_gradient += -(2/n) * (y - ((m_now * x + b_now)))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L

    return m, b


m = 0
b = data.score.min()
#learning Rate & Epochs
L = 0.00005
epochs = 10000

for i in range(epochs):
    if i % 50 == 0:
        error = loss_function(m, b, data)
        print(f"Epochs: {i}, Error: {error}")
    m, b = gradient_descent(m, b, data, L)

print(f"m= {m} , b= {b}\n")

study_time = float(input("Enter Student's Total Study Hours: "))
predicted_score = m * study_time + b
print(f"Predicted Score: {predicted_score}\n")

x = data.study_time
plt.scatter(data.study_time, data.score, color="black", s=7)
plt.plot(data.study_time, m * x + b, color="red", linewidth=4)
plt.show()