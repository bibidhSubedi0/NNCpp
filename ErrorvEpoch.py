'''import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the data from CSV
data = pd.read_csv('error_vs_epoch.csv')

# Apply log10 transformation to the 'Epoch' column
data['Log_Epoch'] = np.log10(data['Epoch'])

# Plot the data
plt.plot(data['Log_Epoch'], data['Error'], marker='.')
plt.xlabel('Log10(Epoch)')
plt.ylabel('Error')
plt.title('Error vs. Log10 of Epoch')
plt.grid(True)
plt.show()
'''
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
