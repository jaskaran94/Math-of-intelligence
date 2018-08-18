import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
# import seaborn as sns


# Import csv dataset with pandas
data = pd.read_csv('kc_house_data.csv')

data = data[:50000] # Using first 10000 rows for training

# Using sqft_living as X_Values and price as Y_Values
points = data.as_matrix(['sqft_living', 'price'])

f, ax = plt.subplots(figsize=(14, 8))

ax.set_xlabel('Size of living area (sqft)')
ax.set_ylabel('House Price ($)')
plt.scatter(points[:,0], points[:,1])
#plt.show()

def compute_error_for_line_given_points(b, m, points):
  totalError = 0
  for i in range(0, len(points)):
    x = points[i, 0]
    y = points[i, 1]
    totalError += (y - (m * x + b)) ** 2
  return totalError/float(len(points))

def step_gradient(b_current, m_current, points, learningRate):
  b_gradient = 0
  m_gradient = 0
  N = float(len(points))
  for i in range(0, len(points)):
    x = points[i, 0]
    y = points[i, 1]
    b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
    m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))

  new_b = b_current - (learningRate * b_gradient)
  new_m = m_current - (learningRate * m_gradient)
  return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learningRate, num_iterations):
  b = starting_b
  m = starting_m
  for i in range(num_iterations):
    b, m = step_gradient(b, m, np.array(points), learningRate)
    # Every 100th iterations, print b, m and error
    if i % 100 == 0:
      print('Iteration {0}, b:{1}, m:{2}, error:{3}'.format(i, b, m, compute_error_for_line_given_points(b, m, points)))
  return [b, m]

def run(learningRate, num_iterations):
  initial_b = 0
  initial_m = 0
  print('Starting gradient descent at b = {0}, m={1}, error={2}'.format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
  print('Running')
  [b, m] = gradient_descent_runner(points, initial_b, initial_m, learningRate, num_iterations)
  print('After {0} iterations b = {1}, m = {2}, error = {3}'.format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))

  f, ax = plt.subplots(figsize=(14, 8))
  ax.set_xlabel('Size of Living Area (sqft)')
  ax.set_ylabel('House Price ($)')
  plt.plot(points[:,0], predict(b, m, points[:,0]))
  plt.scatter(points[:,0], points[:,1])
  plt.show()
  predict_demo(b, m, 6000)

def predict_demo(b, m, x):
  predict = m * x + b
  print('Price of house with size {0} is ${1}'.format(x, predict))


def predict(b, m, x_values):
    predicted_y = list()
    for x in x_values:
        y = m * x + b
        predicted_y.append(y)
    return predicted_y

if __name__ == '__main__':
  run(0.0000001, 1000)