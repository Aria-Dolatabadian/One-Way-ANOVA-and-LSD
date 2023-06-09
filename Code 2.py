import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, t, sem

# Define the observation names and treatments
observations = ['Seed Yield', 'Seed Weight', 'Seed Number', 'Plant Height', 'Leaf Number']
treatments = ['Control', '50 ppm', '100 ppm', '200 ppm', '400 ppm']

# Create an empty dictionary to store the data
data = {}

# Read the CSV files for each observation and store the data in the dictionary
for observation in observations:
    filename = f'{observation}_Data.csv'
    df = pd.read_csv(filename)
    data[observation] = df

# Perform the analysis for each observation
for observation in observations:
    print(f"Analysis for {observation}:")

    # Get the DataFrame for the observation
    df = data[observation]

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

    # Export ANOVA table as a separate CSV file
    anova_data = {'Source of Variation': ['Between Groups', 'Within Groups', 'Total'],
                  'Sum of Squares': [fvalue * (num_replicates - 1),
                                     fvalue * (num_replicates - num_treatments),
                                     fvalue * num_replicates * num_treatments],
                  'Degree of Freedom': [num_treatments - 1,
                                        num_replicates - num_treatments,
                                        num_replicates * num_treatments - 1],
                  'Mean Square': [fvalue,
                                  fvalue * (num_replicates - num_treatments),
                                  ''],
                  'F-value': [fvalue,
                              '',
                              ''],
                  'p-value': [pvalue,
                              '',
                              '']}
    anova_df = pd.DataFrame(anova_data)
    anova_df.to_csv(f'{observation}_ANOVA.csv', index=False)

    # Calculate coefficient of variation (CV)
    cv = (stds / means) * 100

    # Export coefficient of variation as a separate CSV file
    cv_df = pd.DataFrame(cv, columns=['Coefficient of Variation'])
    cv_df.to_csv(f'{observation}_CV.csv', index=True)

    # Export standard deviations as a separate CSV file
    stds_df = pd.DataFrame(stds, columns=['Standard Deviations'])
    stds_df.to_csv(f'{observation}_StdDev.csv', index=True)

    # Export All Pairwise Comparisons (LSD) as a separate CSV file
    lsd_data = {'Comparison': [f"{comp[0]} vs {comp[1]}" for comp in comparisons],
                'Difference': [comp[2] for comp in comparisons],
                'Significant': [comp[3] for comp in comparisons]}
    lsd_df = pd.DataFrame(lsd_data)
    lsd_df.to_csv(f'{observation}_LSD.csv', index=False)


# Generate bar chart with error bars
    plt.bar(means.index, means.values, yerr=stds.values, capsize=5)
    plt.xlabel('Treatment')
    plt.ylabel(observation)
    plt.title(f'{observation} for Different Treatments')

    # Save the plot as JPEG
    plt.savefig(f'{observation}_Plot.jpg', format='jpg')

    plt.show()


