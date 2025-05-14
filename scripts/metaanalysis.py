import statsmodels.api as sm
from scipy.stats import permutation_test, ttest_1samp, bootstrap
from statsmodels.regression.linear_model import OLS
import pandas as pd
import numpy as np
df = pd.read_excel('../data/processed/df_metanalysis.xlsx')

paper_id_to_citation = {
    20: "Wiem & Lachiri, 2017",
    23: "Ayata et al., 2017",
    32: "Siddharth et al., 2018",
    38: "Ayata et al., 2017",
    63: "Sharma et al., 2019",
    66: "Ganapathy et al., 2020",
    74: "Chang et al., 2019",
    82: "Santamaria-Granados et al., 2018",
    86: "Ganapathy & Swaminathan, 2020",
    91: "Susanto et al., 2020",
    94: "Yin et al., 2019",
    97: "Ganapathy & Swaminathan, 2019",
    109: "Bota et al., 2023",
    113: "Selvi & Vijayakumaran, 2023",
    116: "Jung & Sejnowski, 2019",
    117: "Saffaryazdi et al., 2024",
    #129: "Ganapathy et al., 2021",
    131: "Pidgeon et al., 2022",
    133: "Dessai & Virani, 2023",
    135: "Gahlan & Sethia, 2024",
    138: "Chen et al., 2022",
    139: "Mu et al., 2024",
    142: "Perry Fordson et al., 2022",
    145: "Zhu et al., 2023",
    150: "Yin et al., 2022",
    154: "Joo et al., 2024",
    156: "Raheel et al., 2021",
    157: "Veeranki et al., 2024",
    161: "Shukla et al., 2019",
    162: "Chatterjee et al., 2022",
    163: "Tabbaa et al., 2021",
    166: "Gohumpu et al., 2023",
    171: "Singh et al., 2023",
    173: "Kumar & Fredo, 2025",
    174: "Liu et al., 2023",
    182: "Elalamy et al., 2021"
}
df

#%%

df['diff_acc'] = df['accuracy_arousal'] - df['accuracy_valence']

df['mean_acc'] = np.mean([df['accuracy_arousal'], df['accuracy_valence']], axis=0)

#%%

from scipy.stats import linregress
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Unique paper IDs
unique_paper_ids = df['paper_id'].unique()

# Create a color map for paper IDs
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_paper_ids)))

plt.figure(figsize=(22, 9.5))  # Significantly increased width to accommodate the legend

# Calculate the mean difference in accuracy
mean_diff_acc = np.mean(df['diff_acc'])

# Add a horizontal line to represent the mean difference in accuracy
plt.axhline(mean_diff_acc, color='black', linestyle='-', linewidth=2)

# Add text to indicate the mean value
plt.text(max(df['accuracy_valence']) * 1.04, mean_diff_acc, f'Mean difference = {mean_diff_acc:.2f}%', verticalalignment='bottom', horizontalalignment='right', fontsize=16)

# Calculate the correlation coefficient and p-value
slope, intercept, r_value, p_value, std_err = linregress(df['accuracy_valence'], df['diff_acc'])

for i, paper_id in enumerate(unique_paper_ids):
    subset = df[df['paper_id'] == paper_id]
    
    # Check if database is DEAP for the subset
    is_DEAP = subset['database'] == 'DEAP'
    
    # Get the citation for the paper ID
    citation = paper_id_to_citation.get(paper_id, f"Paper ID {paper_id}")
    
    # Plot points with database == DEAP using triangles
    if any(is_DEAP):
        plt.scatter(subset['accuracy_valence'][is_DEAP], subset['diff_acc'][is_DEAP], 
                    color=colors[i], marker='^', label=f"{citation}")
    
    # Plot other points using circles
    if any(~is_DEAP):
        plt.scatter(subset['accuracy_valence'][~is_DEAP], subset['diff_acc'][~is_DEAP], 
                    color=colors[i], label=citation)

# Adding horizontal lines
plt.axhline(np.mean(df['diff_acc']) + 1.96 * np.std(df['diff_acc']), linestyle='--')
plt.axhline(np.mean(df['diff_acc']) - 1.96 * np.std(df['diff_acc']), linestyle='--')

# Adding labels and title
plt.xlabel('Accuracy of Valence Models', fontsize=16)
plt.ylabel('Difference in Accuracy (Arousal - Valence)', fontsize=16)
plt.title('Valence Accuracy vs Difference in Arousal and Valence Accuracy', fontsize=17, fontweight='bold')

# Adjust the layout to maintain the plot size
plt.tight_layout(rect=[0, 0, 0.45, 1])  # Adjusted to leave about 1/3 of figure width for legends

# Get handles and labels for the references
handles, labels = plt.gca().get_legend_handles_labels()

# Single legend for all references with increased font size for better readability
legend1 = plt.legend(handles, labels, title='References', 
                    bbox_to_anchor=(1.01, 1.03), loc='upper left', ncol=1, 
                    fontsize=9, title_fontsize=10)
plt.gca().add_artist(legend1)
plt.setp(legend1.get_title(), weight='bold')

# Legend for marker types
legend3 = plt.legend([plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markersize=10),
                      plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10)],
                     ['DEAP Database                       ', 'Other Data'], 
                     title='Database', 
                     bbox_to_anchor=(1.01, 0.09), loc='upper left')
plt.setp(legend3.get_title(), weight='bold')

# Remove spines
sns.despine()

#plt.savefig(r'./figures/FIG10.png', dpi=300, bbox_inches='tight')
# Show the plot
plt.show()

#%%
x = df['diff_acc']
np.mean(x)

#%%

def my_stat(x):
    return ttest_1samp(x, popmean=0).statistic

#%%

permutation_test((x.values,), my_stat, permutation_type='samples')

#%%

bootstrap((x.values,), my_stat)

#%%

# Create dummy variables for model_family (categorical predictor)
X = pd.get_dummies(df['model_family'], prefix='family', drop_first=True)

# Create dummy variables for database
X_db = pd.get_dummies(df['database'], prefix='db', drop_first=True)

# Add the continuous predictors
X = pd.concat([X, X_db, df[['N', 'year', 'mean_acc']]], axis=1)

# Make sure all columns are numeric and handle NaN values
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(0)  # Fill NaN values with 0 to avoid object dtype issues

# Convert to numpy arrays explicitly to avoid pandas object dtype issues
X_np = np.asarray(X, dtype=float)
y_np = np.asarray(df['diff_acc'], dtype=float)

# Add constant
X_np = sm.add_constant(X_np)

# Define the dependent variable
y = df['diff_acc']

# Create the model with numpy arrays to avoid the pandas object dtype error
model = OLS(y_np, X_np)
res = model.fit()

# Print the summary with variable names
print("\nVariables in the model:")
for i, var_name in enumerate(['const'] + list(X.columns)):
    print(f"x{i} = {var_name}")

print("\nRegression Results:")
print(res.summary())
# %%

# This code performs a robust linear regression (RLM) analysis which is less sensitive to outliers
# than traditional OLS regression. It uses Huber's norm (HuberT) to reduce the influence of outliers.
# It examines how the independent variables (N, year, mean_acc) affect the difference in accuracy between
# arousal and valence (diff_acc), providing more robust estimates when there are extreme data points.
rlm_mod = sm.RLM(y_np, X_np, M=sm.robust.norms.HuberT())
rlm_res = rlm_mod.fit()
rlm_res.summary()
# %%
