import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.ticker import MaxNLocator

# Data from the LaTeX table
data = {
    "Study": [
        "Sansone & Martens (1978)", "Ward et al. (1989)", "de Angelis et al. (1993)", 
        "Mau et al. (2012)", "Gentz et al. (2014)", "Steinle et al. (2016)", 
        "Weinstein et al. (2016)", "Leonte et al. (2017)", "Mau et al. (2017)", 
        "Uhlig et al. (2018)", "Pack et al. (2015)", "Steinle et al. (2017)", 
        "Gründger et al. (2021)", "De Groot et al. (2024)"
    ],
    "k_ox [s^{-1}]": [
        0.0347*10**-6, 0.0926*10**-6, 0.0695*10**-6, 0.0579*10**-6, 0.255*10**-6, 0.0162*10**-6, 0.926*10**-6, 1.16*10**-6, 
        0.405*10**-6, 0.104*10**-6, 0.694*10**-6, 0.0347*10**-6, 0.00174*10**-6, 0.139*10**-6
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Calculate the average k_ox value
average_k_ox = df["k_ox [s^{-1}]"].mean()

# Set up the matplotlib figure with a narrower width
plt.figure(figsize=(8, 6))  # Adjusted width from 10 to 8
sns.set(style="whitegrid")

# Plot the histogram of k_ox values with a logarithmic scale on the x-axis
sns.histplot(df["k_ox [s^{-1}]"], bins=10, color="blue", log_scale=(True, False))

# Add a vertical line at the average value
plt.axvline(average_k_ox, color='red', linestyle='dashed', linewidth=2, label=f'Average: {average_k_ox:.2e}')

# Customize the plot
plt.title("Rate Coefficients (Logarithmic Scale)", fontsize=16)
plt.xlabel(r"$k_{ox} \, s^{-1}$", fontsize=14)  # Changed fontsize to 12
plt.ylabel("Frequency", fontsize=14)  # Changed fontsize to 12
plt.legend()
plt.grid(True)

# Ensure y-axis has only integer values
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

# Show the plot
plt.tight_layout()
plt.show()