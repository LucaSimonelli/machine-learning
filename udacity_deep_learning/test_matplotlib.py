import numpy as np
import matplotlib.pyplot as plt

X_data = np.arange(100, step=.1)
y_data = X_data + 20 * np.sin(X_data/10)

plt.scatter(X_data, y_data)
plt.show()
