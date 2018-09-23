import numpy as np
from matplotlib import pyplot as plt
# %matplotlib inline

# Define our data
# Input data - Of the form [x, y, Bias term]
X = np.array([
  [-2, 4, -1],
  [4, 1, -1],
  [1, 6, -1],
  [2, 4, -1],
  [6, 2, -1]
])

# Associated output labels - First 2 examples are labeled '-1' and last 3 are '+1'
y = np.array([-1, -1, 1, 1, 1])

# Plot examples on 2D graph
for d, sample in enumerate(X):
  print(sample)
  # Plot the -ve samples (first 2)
  if d < 2: 
    plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
  else:
    plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)

# Print a possible hyperplane, that is seperating two classes.
plt.plot([-2, 6], [6, 0.5])
plt.show()

# Perform stochastic gradient descent to learn the seperating hyperplane between both classes
def svm_sgd_plot(X, Y):
  # Initialize SVMs weight vector with zeroes (3 values)
  w = np.zeros(len(X[0]))
  # The learning rate
  eta = 1
  # Iterations to train for
  epochs = 100000
  # store misclassifications so we can plot how they change over time
  errors = []

  # training part, gradient descent part
  for epoch in range(1, epochs):
    error = 0
    for i, x in enumerate(X):
      # misclassfication
      if (Y[i]*np.dot(X[i], w)) < 1:
        # misclassified update for weights
        w = w + eta * ( (X[i] * Y[i]) + (-2 * (1/epoch) * w))
        error = 1
      else:
        # correct classification, update our weights
        w = w + eta * (-2 * (1/epoch)*w)
      errors.append(error)
  
  # plot the rate of classification errors during training for SVM
  plt.plot(errors, '|')
  plt.ylim(0.5, 1.5)
  plt.axes().set_yticklabels([])
  plt.xlabel('Epoch')
  plt.ylabel('Misclassified')

  return w

for d, sample in enumerate(X):
  if d < 2:
    plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
  else: 
    plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)

# Add our test samples
plt.scatter(2,2, s=120, marker='_', linewidths=2, color='yellow')
plt.scatter(4, 3, s=120, marker='+', linewidths=2, color='blue')

w = svm_sgd_plot(X, y)

# Print the hyperplane calculate by svm_sgd()
x2=[w[0],w[1],-w[1],w[0]]
x3=[w[0],w[1],w[1],-w[0]]

x2x3 =np.array([x2,x3])
X,Y,U,V = zip(*x2x3)
ax = plt.gca()
ax.quiver(X, Y, U, V, scale=1, color='blue')
plt.show()
