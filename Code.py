import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, t, sem

#read data
data = pd.read_csv("data.csv")
df = pd.DataFrame(data)

# Perform one-way ANOVA
fvalue, pvalue = f_oneway(*df.values.T)

# Calculate the LSD (Least Significant Difference)
num_treatments = len(df.columns)
num_replicates = len(df)
df_error = num_replicates * (num_treatments - 1)
critical_value = t.ppf(1 - 0.05 / 2, df_error)
standard_error = sem(df.values.flatten())
lsd = critical_value * standard_error

# Perform post hoc test using LSD
means = df.mean()
comparisons = []
significant = []
for i in range(num_treatments):
    for j in range(i + 1, num_treatments):
        diff = np.abs(means[i] - means[j])
        significant_diff = diff > lsd
        comparisons.append((df.columns[i], df.columns[j], diff, significant_diff))
        if significant_diff:
            significant.append((df.columns[i], df.columns[j]))

# Print ANOVA results and LSD test summary
print("ANOVA p-value:", pvalue)
print("LSD:", lsd)
print("All Pairwise Comparisons (LSD):")
for comp in comparisons:
    print(f"{comp[0]} vs {comp[1]}: Difference={comp[2]}, Significant={comp[3]}")
print()

# Calculate means and standard deviations
stds = df.std()

# Generate bar chart with error bars
plt.bar(means.index, means.values, yerr=stds.values, capsize=5)
plt.xlabel('Treatment')
plt.ylabel('Seed Yield')
plt.title('Seed Yield for Different Treatments')
plt.show()

# Print standard deviations
print("Standard Deviations:")
print(stds)

# Export the ANOVA and LSD results as a CSV file
output_data = {
    'Comparison': [f"{comp[0]} vs {comp[1]}" for comp in comparisons],
    'Difference': [comp[2] for comp in comparisons],
    'Significant': [comp[3] for comp in comparisons],
    'ANOVA p-value': pvalue,
    'LSD': lsd,
}
output_df = pd.DataFrame(output_data)
output_df.to_csv('ANOVA_LSD_Results.csv', index=False)

# Export the standard deviations as a separate CSV file
stds_df = pd.DataFrame(stds, columns=['Standard Deviations'])
stds_df.to_csv('StD.csv', index=True)

