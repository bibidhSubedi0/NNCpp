import matplotlib.pyplot as plt
import pandas as pd

# Load the data from CSV
data = pd.read_csv('error_vs_epoch.csv')

# Plot the data
plt.plot(data['Epoch'], data['Error'], marker='.')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Error vs. Epoch')
plt.grid(True)
plt.show()
