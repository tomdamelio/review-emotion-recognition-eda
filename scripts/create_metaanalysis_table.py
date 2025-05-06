#%%
import pandas as pd
import numpy as np
import scipy.stats as stats
import os

#%%
# Reseteo del directorio principal
os.getcwd()
os.chdir("../")

#%%
# Read the data
df_statistical_learning_models = pd.read_excel('.\\data\\processed\\Processed - Normalized Table - Statistical learning model - Performances.xlsx')
df_participants = pd.read_excel('.\\data\\cleaned\\Normalized Table - Participants.xlsx')
df_metadata = pd.read_excel('.\\data\\cleaned\\Normalized Table - Metadata.xlsx')

#%%
# Filter rows where add_column has value 'x'
df_filtered = df_statistical_learning_models[df_statistical_learning_models['add_column'] == 'x']

# Get all columns that start with 'class_'
class_columns = [col for col in df_filtered.columns if col.startswith('class_') and col != 'class_model_output_categories']

# Create ML_model column based on which class_* column has 'x' or 'X'
def get_ml_model(row):
    # First check if there's a value in class_other
    if pd.notna(row['class_other']) and str(row['class_other']).strip():
        return str(row['class_other']).strip()
    
    # If no value in class_other, check other class columns
    for col in class_columns:
        if col != 'class_other' and pd.notna(row[col]) and str(row[col]).strip().upper() in ['X', 'x']:
            return col.replace('class_', '')
    return None

df_filtered['ML_model'] = df_filtered.apply(get_ml_model, axis=1)

# Create a dictionary to map output categories
output_categories = {
    'HA; LA': 'arousal',
    'HV; LV': 'valence'
}

# Add a column to identify if the model is for arousal or valence
df_filtered['output_type'] = df_filtered['class_model_output_categories'].map(output_categories)

# Create separate dataframes for arousal and valence
df_arousal = df_filtered[df_filtered['output_type'] == 'arousal'].copy()
df_valence = df_filtered[df_filtered['output_type'] == 'valence'].copy()

# Rename columns for arousal
df_arousal = df_arousal.rename(columns={
    'accuracy': 'accuracy_arousal',
    'class_model_output_categories': 'output_arousal'
})

# Rename columns for valence
df_valence = df_valence.rename(columns={
    'accuracy': 'accuracy_valence',
    'class_model_output_categories': 'output_valence'
})

# Merge the dataframes on paper_id and paired_id
df_meta_analysis = pd.merge(
    df_arousal[['paper_id', 'paired_id', 'apa_citation', 'year', 'output_arousal', 'accuracy_arousal', 'ML_model']],
    df_valence[['paper_id', 'paired_id', 'output_valence', 'accuracy_valence']],
    on=['paper_id', 'paired_id'],
    how='inner'
)

# Select and reorder columns, using paired_id as model_id
df_meta_analysis = df_meta_analysis[[
    'paper_id', 'paired_id', 'apa_citation', 'year',
    'output_arousal', 'accuracy_arousal',
    'output_valence', 'accuracy_valence',
    'ML_model'
]]

# Rename paired_id to model_id
df_meta_analysis = df_meta_analysis.rename(columns={'paired_id': 'model_id'})

# Convert accuracy values from 0-1 to 0-100 range with 2 decimals
df_meta_analysis['accuracy_arousal'] = (df_meta_analysis['accuracy_arousal'] * 100).round(2)
df_meta_analysis['accuracy_valence'] = (df_meta_analysis['accuracy_valence'] * 100).round(2)

# Merge with participants table to add 'n' column
df_meta_analysis = pd.merge(
    df_meta_analysis,
    df_participants[['paper_id', 'n']],
    on='paper_id',
    how='left'
)

# Merge with metadata table to get the year
df_meta_analysis = pd.merge(
    df_meta_analysis,
    df_metadata[['paper_id', 'year']],
    on='paper_id',
    how='left',
    suffixes=('', '_metadata')
)

# Drop the original year column and rename the new one
df_meta_analysis = df_meta_analysis.drop('year', axis=1)
df_meta_analysis = df_meta_analysis.rename(columns={'year_metadata': 'year'})

# Rename 'n' to 'N'
df_meta_analysis = df_meta_analysis.rename(columns={'n': 'N'})

# Reorder columns in the specified order
df_meta_analysis = df_meta_analysis[[
    'paper_id', 'model_id', 'apa_citation', 'year', 'N',
    'output_arousal', 'accuracy_arousal',
    'output_valence', 'accuracy_valence',
    'ML_model'
]]

# Print the head of the resulting dataframe
print("\nHead of the meta-analysis table:")
print(df_meta_analysis.head())

#%%
# Save the meta-analysis table
df_meta_analysis.to_excel('.\\data\\processed\\Processed - Normalized Table - Meta-analysis.xlsx', index=False)

#%%
# Find papers with duplicate model_ids
duplicate_papers = df_meta_analysis[df_meta_analysis.duplicated(['paper_id', 'model_id'], keep=False)].sort_values(['paper_id', 'model_id'])

# Create a text file with the duplicate papers information
with open('.\\data\\processed\\duplicate_model_ids.txt', 'w', encoding='utf-8') as f:
    f.write("Papers with duplicate model_ids:\n\n")
    for paper_id in duplicate_papers['paper_id'].unique():
        paper_data = duplicate_papers[duplicate_papers['paper_id'] == paper_id]
        f.write(f"Paper ID: {paper_id}\n")
        f.write(f"Citation: {paper_data['apa_citation'].iloc[0]}\n")
        f.write("Entries:\n")
        for _, row in paper_data.iterrows():
            f.write(f"  Model ID: {row['model_id']}\n")
            f.write(f"  N: {row['N']}\n")
            f.write(f"  ML Model: {row['ML_model']}\n")
            f.write("\n")
        f.write("-" * 80 + "\n\n")

#%%
# Remove duplicate rows based on paper_id and model_id
df_meta_analysis_no_duplicates = df_meta_analysis.drop_duplicates(subset=['paper_id', 'model_id'], keep='first')

# Save the meta-analysis table without duplicates
df_meta_analysis_no_duplicates.to_excel('.\\data\\processed\\Processed - Normalized Table - Meta-analysis - No Duplicates.xlsx', index=False)

#%%
# Read the first Excel file
df_first = pd.read_excel('.\\data\\processed\\new_melted_df_excel_paired_ALL.xlsx')

# Read the second Excel file
df_second = pd.read_excel('.\\data\\processed\\Processed - Normalized Table - Meta-analysis - No Duplicates.xlsx')

# Define the desired column order
column_order = [
    'paper_id', 'model_id', 'apa_citation', 'year', 'N',
    'output_arousal', 'accuracy_arousal',
    'output_valence', 'accuracy_valence',
    'ML_model'
]

# Ensure both dataframes have the same columns and order
df_first = df_first[column_order]
df_second = df_second[column_order]

# Perform vstack
df_final = pd.concat([df_first, df_second], axis=0, ignore_index=True)

# Ensure final dataframe has the correct column order
df_final = df_final[column_order]

# Save the final combined dataframe
df_final.to_excel('.\\data\\processed\\final_melted_df_excel_paired_ALL.xlsx', index=False)

#%%
# Decime cuantos model_id unicos y cuantos paper_id unicos hay en el dataframe final
print(f"Number of unique model_ids: {df_final['model_id'].nunique()}")
print(f"Number of unique paper_ids: {df_final['paper_id'].nunique()}")

#%%
duplicados = df_final.duplicated(subset=['paper_id', 'model_id'], keep=False)
print(df_final[duplicados])
# %%
