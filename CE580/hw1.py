import matplotlib.pyplot as plt
import numpy as np

# example plot
x = np.linspace(0, 10, 100)
y = np.sin(x) + 1

plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('x')
plt.ylabel('sin(x) + 1')
plt.grid()
plt.show()
