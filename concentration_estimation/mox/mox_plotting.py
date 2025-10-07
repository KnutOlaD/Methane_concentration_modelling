import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.ticker import MaxNLocator
import numpy as np

color_1 = '#7e1e9c'
color_2 = '#014d4e'

# Data from the LaTeX table
data = {
    "Location": [
        "Cariaco Trench, Caribbean Sea", "Saanich Inlet, British Columbia", "Eastern Tropical North Pacific",
        "Juan De Fuca v.", "Deepwater Horizon, Gulf of Mexico", "North Sea gas blowout",
        "Cape Lookout Bight, North Carolina", "Santa Barbara Channel, California", "Boknis Eck, Baltic Sea",
        "South China Sea", "Hudson Canyon, US Atlantic", "Elson Lagoon, Alaska",
        "Norskebanken", "Hinlopen Trough", "Prins Karl Forland (2015)", "Prins Karl Forland (2016)",
        "Prins Karl Forland (2017)", "Prins Karl Forland", "Hornsundbanken", "Isfjordenbanken",
        "Storfjordrenna", "Storfjorden"
    ],
        "Category": [
        "Oxic/anoxic interface", "Oxic/anoxic interface", "Oxic/anoxic interface",
        "Hydrothermal plume", "Man-made accidents", "Man-made accidents",
        "Seep environment", "Seep environment", "Seep environment",
        "Seep environment", "Seep environment", "Seep environment",
        "Cold seeps - Svalbard Continental margin", "Cold seeps - Svalbard Continental margin", 
        "Cold seeps - Svalbard Continental margin", "Cold seeps - Svalbard Continental margin", 
        "Cold seeps - Svalbard Continental margin", "Cold seeps - Svalbard Continental margin", 
        "Cold seeps - Svalbard Continental margin", "Cold seeps - Svalbard Continental margin", 
        "Cold seeps - Svalbard Continental margin", "Cold seeps - Svalbard Continental margin"
    ],
    "Temp (℃)": [
        None, 9, None, None, None, 10, "23-27", "5-16", "1-3", "2-5", None, -1.8, 4.7, 3.5, "~3", "~1.5", "~3", "1.6-4.8", ">3", ">3", -0.5, -1.5
    ],
    "φ (nM)": [
        "<12100", "<1580", 19, "<390", "<183000", "<42097", "<740", "<1900", "300-466", "<1000", "<335", "<53.8", "<83.1", "<874", "<334", "<437", "<262", "<524", "<878", "<100", "<82", "<72.3"
    ],
    "k_ox (10^{-6} s^{-1})": [
        0.03, 0.02, 0.10, 1.74, 0.73, 0.41, 0.08, 0.09, 0.50, 0.04, 0.93, 0.12, 0.06, 0.23, 0.98, 0.02, 0.02, 0.21, 0.41, 0.62, 0.22, 0.35
    ],
    "t_{0.5} (days)": [
        277, 535, 77, 5, 11, 20, 107, 93, 16, 229, 9, 69, 125, 35, 8, 433, 385, 38, 20, 13, 36, 23
    ],
    "Reference": [
        "Ward et al. (1987)", "Ward et al. (1989)", "Pack et al. (2015)", "de Angelis et al. (1993)", "Valentine et al. (2010)",
        "Steinle et al. (2016)", "Sansone & Martens (1978)", "Mau et al. (2012)", "Steinle et al. (2017)", "Mau et al. (2020)",
        "Weinstein et al. (2016)", "Uhlig et al. (2018)", "Sert et al. (2023)", "De Groot et al. (2024)", "Gründger et al. (2021)",
        "Gründger et al. (2021)", "Gründger et al. (2021)", "Gentz et al. (2014)", "Mau et al. (2017)", "Mau et al. (2017)",
        "Sert et al. (2020)", "Mau et al. (2013)"
    ]
}


# Multiply all the data with 10^-6 since we're doing a log plot we dont need to worry about this
data["k_ox (10^{-6} s^{-1})"] = [x * 1e-6 for x in data["k_ox (10^{-6} s^{-1})"]]
data["φ (nM)"] = [x for x in data["φ (nM)"]]
data["t_{0.5} (days)"] = [x for x in data["t_{0.5} (days)"]]
data["Temp (℃)"] = [x for x in data["Temp (℃)"]]

# Create a DataFrame
df = pd.DataFrame(data)

# Calculate the average k_ox value
average_k_ox = df["k_ox (10^{-6} s^{-1})"].mean()
# Calculate the median k_ox value
median_k_ox = df["k_ox (10^{-6} s^{-1})"].median()

#take the average of only the cold seep values
average_k_ox_coldseeps = np.mean(df[df["Category"] == "Cold seeps - Svalbard Continental margin"]["k_ox (10^{-6} s^{-1})"])
print(f"Average k_ox for cold seeps: {average_k_ox_coldseeps:.2e}")
#average_k_ox_coldseeps = 2.15e-7

# Set up the matplotlib figure with a narrower width
plt.figure(figsize=(7, 6))

sns.set(style="whitegrid")

sns.set_style('ticks')

# Plot the histogram of k_ox values with a logarithmic scale on the x-axis and KDE line
sns.histplot(df["k_ox (10^{-6} s^{-1})"], bins=8, color=color_1, kde=True, log_scale=(True, False), kde_kws={'bw_method': 'silverman'},label='Datasets in range')

# Add a vertical line at the average value of all locations
plt.axvline(average_k_ox, color=color_2, linestyle='dotted', linewidth=2,label=f'Average: {average_k_ox:.2e}')

# Add a vertical line at the average value of cold seeps
plt.axvline(median_k_ox, color=color_2, linestyle='dashed', linewidth=2,label=f'Median: {median_k_ox:.2e}')

# make a new color
color_3 = '#e37222'
# Add a vertical line at the median and average value of cold seeps
#plt.axvline(average_k_ox_coldseeps, color=color_3, linestyle='dotted', linewidth=2,label=f'Average cold seeps: {average_k_ox_coldseeps:.2e}')  
# calculate median of cold seeps
median_k_ox_coldseeps = df[df["Location"].str.contains("Prins Karl Forland|Hornsundbanken|Isfjordenbanken|Storfjordrenna|Storfjorden")]["k_ox (10^{-6} s^{-1})"].median()   
#plt.axvline(median_k_ox_coldseeps, color=color_3, linestyle='dashed', linewidth=2,label=f'Median cold seeps: {median_k_ox_coldseeps:.2e}')   

# take the standard deviation of all values
std_k_ox = df["k_ox (10^{-6} s^{-1})"].std()

# Add a dummy line for the KDE label
plt.plot([], [], color=color_1, label='Kernel density fit', linewidth=2.5)

#Make dummy line for average label
#plt.plot([], [], color=color_2, linewidth=2.5, linestyle='dashed')

# Customize the plot
#plt.title("Methane Oxidation Rate Coefficients ($k_{ox}$)", fontsize=14)
plt.xlabel(r"$k_{ox} \, [\text{s}^{-1}]$, logarithmic scale, max values", fontsize=16)
plt.ylabel("Datasets", fontsize=16)
plt.legend(fontsize=12,loc=[0.14,0.77])
plt.grid(False)

#Increase size of ticks
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Ensure y-axis has only integer values
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
#Create a list of xticks. Should go from min to max of k_ox in logspace with
#ticks at 0.01,0.011,0.012,0.013,0.014,0.015,0.016,0.017,0.018,0.019,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10
xticks = np.array([0.01,0.011,0.012,0.013,0.014,0.015,0.016,0.017,0.018,0.019,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10])*10**-6#force xticks
plt.xticks(xticks)
#and labels at each order of magnitude
plt.xticks(np.array([0.01,0.1,1,10])*10**-6,['$10^{-8}$','$10^{-7}$','$10^{-6}$','$10^{-5}$'])

#adjust x-axis limits
plt.xlim(np.min(df["k_ox (10^{-6} s^{-1})"]), np.max(df["k_ox (10^{-6} s^{-1})"]))
   
# Show the plot
plt.tight_layout()
plt.show()

#################################
####### WITH COLORED BARS #######
#################################

# Create a DataFrame
df = pd.DataFrame(data)

# Calculate the average k_ox value
average_k_ox = df["k_ox (10^{-6} s^{-1})"].mean()

# Define colors for each category
category_colors = {
    "Oxic/anoxic interface": "blue",
    "Hydrothermal plume": "green",
    "Man-made accidents": "red",
    "Seep environment": "purple",
    "Cold seeps - Svalbard Continental margin": "orange"
}

# Set up the matplotlib figure
plt.figure(figsize=(12, 8))  
sns.set_style('whitegrid')

# Define bins for the histogram
bins = np.logspace(np.log10(df["k_ox (10^{-6} s^{-1})"].min()), np.log10(df["k_ox (10^{-6} s^{-1})"].max()), 9)

# Prepare data for stacked histogram
stacked_data = []
stacked_labels = []
stacked_colors = []

for category, color in category_colors.items():
    subset = df[df["Category"] == category]["k_ox (10^{-6} s^{-1})"]
    stacked_data.append(subset)
    stacked_labels.append(category)
    stacked_colors.append(color)

# Plot the stacked histogram
plt.hist(stacked_data, bins=bins, stacked=True, color=stacked_colors, label=stacked_labels, alpha=0.7)

# Add a KDE line for the total sum
sns.kdeplot(df["k_ox (10^{-6} s^{-1})"], bw_method='silverman', color='black', label='Gaussian KDE (Silverman)', log_scale=True)

# Add a vertical line at the average value
plt.axvline(average_k_ox, color='black', linestyle='dashed', linewidth=2, label=f'Average: {average_k_ox:.2e}')

# Customize the plot
plt.xscale('log')
plt.xlim(bins[0], bins[-1])  # Set x-axis limits based on the bins
plt.title("Methane Oxidation Rate Coefficients ($k_{ox}$) by Category", fontsize=16)
plt.xlabel(r"$k_{ox} \, (10^{-6} \, \text{s}^{-1})$", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.legend()
plt.grid(True)

# Ensure y-axis has only integer values
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

# Show the plot
plt.tight_layout()
plt.show()

# Bubble rising speed averages.. 

#Calculate average of all seep location mox rates

#make a vector of all the seep mox rates
seep_mox_rates = [0.08,0.09,0.5,0.04,0.93,0.12,0.06,0.23,0.98,0.02,0.02,0.21,0.41,0.62,0.22,0.35]

seep_mox_rates_avg = np.mean(seep_mox_rates)
seep_mox_rates_median = np.median(seep_mox_rates)
seep_mox_rates_std = np.std(seep_mox_rates)

print(f"Average: {seep_mox_rates_avg:.2e}")
print(f"Median: {seep_mox_rates_median:.2e}")
print(f"Standard Deviation: {seep_mox_rates_std:.2e}")

#create a histogram similar to the first histogram in this script
plt.figure(figsize=(7, 6))
sns.set(style="whitegrid")

# Plot the histogram of k_ox values with a logarithmic scale on the x-axis and KDE line
sns.histplot(seep_mox_rates, bins=8, color=color_1, kde=True, log_scale=(True, False), kde_kws={'bw_method': 'silverman'})

# Add a vertical line at the average value
#plt.axvline(seep_mox_rates, color=color_2, linestyle='dashed', linewidth=2, label=f'Average: {seep_mox_rates:.2e}',)

# Add a dummy line for the KDE label
plt.plot([], [], color=color_1, label='Gaussian KDE', linewidth=2.5)

# Customize the plot
#plt.title("Methane Oxidation Rate Coefficients ($k_{ox}$)", fontsize=14)
plt.xlabel(r"$k_{ox} \, (10^{-6} \, \text{s}^{-1})$, logarithmic scale", fontsize=16)
plt.ylabel("Frequency", fontsize=16)
plt.legend(fontsize=12,loc=[0.2,0.85])
plt.grid(True)

#adjust x-axis limits
plt.xlim(np.min(seep_mox_rates), np.max(seep_mox_rates))

#Increase size of ticks
plt.xticks(fontsize=14)

# Ensure y-axis has only integer values
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

# Show the plot
plt.tight_layout()
plt.show()
