# Data Analysis Script for Neurocomputing Journal Paper (2nd Revision)
# This script performs comprehensive analysis on research data, 
# including statistical tests, data visualization, and network analysis
# for the second revision of our submission to Neurocomputing journal.

#%% 
# Imports principales
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px
import zepid
from zepid.graphics import EffectMeasurePlot
import networkx as nx
from numpy import genfromtxt
from scipy import stats
from IPython.display import Image
from thefuzz import fuzz
from matplotlib.patches import Patch
import re
import matplotlib.gridspec as gridspec
import geopandas as gpd

# Importar funciones personalizadas (ajusta el path si es necesario)
import functions as fn

#%% 
# Configuración del directorio de trabajo
# Get the parent directory
current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)

# Change to the parent directory
os.chdir(parent_directory)

# Get the new current working directory to confirm
new_directory = os.getcwd()
print(f"New Directory: {new_directory}")

#%% 
# Cargar el dataset principal
df_metadata = pd.read_excel(r'.\data\cleaned\Normalized Table - Metadata.xlsx')

# Limpieza inicial de datos
df_metadata = df_metadata.fillna('-')
df_metadata_without_duplicates = df_metadata.drop_duplicates(subset='paper_id')

#%% 
# Conteo de países de afiliación del primer autor
print("\nConteo de países de afiliación del primer autor:")
country_counts = df_metadata_without_duplicates['first_author_country_affiliation'].value_counts()
print(country_counts)

# Mostrar el total de países únicos
print(f"\nTotal de países únicos: {len(country_counts)}")

print(f"\n Lista de países únicos: {country_counts.index.tolist()}")
#%% 
# Clasificación de países por continente
asia_countries = ['China', 'India', 'Japan', 'South Korea', 'Malaysia', 'Pakistan', 'Taiwan', 
                 'Indonesia', 'Saudi Arabia', 'Iran', 'Turkey', 'United Arab Emirates']
europe_countries = ['Germany', 'UK', 'Spain', 'Italy', 'Switzerland', 'Romania', 'Greece',
                   'Austria', 'Finland', 'Slovenia', 'Portugal', 'France', 'Lithuania',
                   'Macedonian', 'Netherlands', 'Ireland', 'Sweden', 'Norway', 'Poland']
america_countries = ['USA', 'Canada', 'Colombia']
oceania_countries = ['Australia', 'New Zealand']
africa_countries = ['Tunisia', 'Egypt']

# Crear diccionario para mapear países a continentes
continent_mapping = {}
for country in asia_countries:
    continent_mapping[country] = 'Asia'
for country in europe_countries:
    continent_mapping[country] = 'Europe'
for country in america_countries:
    continent_mapping[country] = 'America'
for country in oceania_countries:
    continent_mapping[country] = 'Oceania'
for country in africa_countries:
    continent_mapping[country] = 'Africa'

# Agregar el continente a cada país en el conteo
continent_counts = {}
uncategorized_countries = {}
for country, count in country_counts.items():
    if country in continent_mapping:
        continent = continent_mapping[country]
        continent_counts[continent] = continent_counts.get(continent, 0) + count
    else:
        uncategorized_countries[country] = count

# Mostrar resultados por continente
print("\nConteo de papers por continente:")
for continent, count in sorted(continent_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{continent}: {count}")

# Mostrar países sin categorizar
print("\nPaíses sin categorizar:")
for country, count in uncategorized_countries.items():
    print(f"{country}: {count}")

# Mostrar totales
total_categorized = sum(continent_counts.values())
total_uncategorized = sum(uncategorized_countries.values())
print(f"\nTotal de papers categorizados: {total_categorized}")
print(f"Total de papers sin categorizar: {total_uncategorized}")
print(f"Total general: {total_categorized + total_uncategorized}")

# Crear DataFrame para visualización
continent_df = pd.DataFrame({
    'Continent': list(continent_counts.keys()),
    'Count': list(continent_counts.values())
})

# Visualizar resultados
plt.figure(figsize=(10, 6))
sns.barplot(data=continent_df, x='Continent', y='Count')
plt.title('Number of Papers by Continent')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%%
# Crear tabla pivote con conteo de países
countries = df_metadata_without_duplicates.pivot_table(index=['first_author_country_affiliation'], aggfunc='size')
df_countries = pd.DataFrame(countries)

# Calcular la suma total de papers por país
# Usar el nombre de la columna en lugar del índice numérico para evitar KeyError
total_papers = df_countries['count'].sum() if 'count' in df_countries.columns else df_countries.iloc[:, 0].sum()
print(f"Suma total de papers por país: {total_papers}")

df_countries.columns = ['count']


# Cargamos el GeoDataFrame con geometría de países
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Renombrar "USA" a "United States of America" en df_countries
df_countries.rename(index={'USA':'United States of America'}, inplace=True)

# Coordenadas de París
paris_coords = [2.3522, 48.8566]

# Coordenadas para ajustar Hungría
slovenia_adjust = [19.5033, 47.1625]

# Filtrar el GeoDataFrame para excluir la Antártida
world = world[world['name'] != 'Antarctica']

# Fusionamos el GeoDataFrame con tu DataFrame
world_merged = world.merge(df_countries, how="left", left_on="name", right_index=True)

# Crear el gráfico
fig, ax = plt.subplots(1, figsize=(20, 12), dpi=300)
world_merged.boundary.plot(ax=ax, linewidth=0.8, color='grey')
world_merged.plot(column='count', ax=ax, cmap='Blues',
                  missing_kwds={"color": "lightgrey"}, linewidth=0.5, edgecolor='black')

for idx, row in world_merged.iterrows():
    if not pd.isna(row['count']):
        x, y = row.geometry.centroid.x, row.geometry.centroid.y
        label = str(int(row['count']))
        
        # Ajustes específicos para Francia
        if row['name'] == 'France':
            x, y = paris_coords
        
        # Tamaño del círculo (20% más grande para China)
        circle_radius = 3 if row['name'] != 'China' else 3.6
        
        circle = plt.Circle((x, y), circle_radius, color='white', ec='black', fill=True, alpha=0.5, linewidth=1.5)
        ax.add_patch(circle)
        
        # Ajuste de la posición de la etiqueta para Slovenia
        if row['name'] == 'Slovenia':
            ax.annotate(label, xy=(x, y), xytext=(x + 3, y - 1),
                        fontsize=14, ha='center', va='center', fontweight='bold')
        else:
            ax.text(x, y, label, fontsize=14, ha='center', va='center', fontweight='bold')


# Título y barra de color
plt.title('Number of Articles by Country', fontsize=20, pad=20, fontweight='bold')
ax.set_axis_off()

sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=df_countries['count'].min(), vmax=df_countries['count'].max()))
sm.set_array([])

cbar = fig.colorbar(sm, orientation="horizontal", pad=0.01, shrink=0.5, ax=ax, fraction=0.046)
cbar.set_label('Number of Articles by Country', fontsize=16, fontweight='bold')
cbar.ax.tick_params(labelsize=14)
cbar.ax.set_position([0.1, 0.1, 0.4, 0.02])

plt.show()

#%% 
# Analysis of Source Titles (Journals/Conferences)
print("\nSource Title Analysis:")
source_counts = df_metadata_without_duplicates['source_title'].value_counts()

# Display source counts
print("\nFrecuencia absoluta de fuentes de publicación:")
for source, count in source_counts.items():
    print(f"{source}: {count}")

# Display total number of unique sources
print(f"\nTotal de fuentes únicas: {len(source_counts)}")

#%% 
# Count papers by source type
print("\nSource Type Analysis:")

# Count papers for each source type
journal_count = len(df_metadata_without_duplicates[df_metadata_without_duplicates['source_type_journal'] == 'x'])
conference_count = len(df_metadata_without_duplicates[df_metadata_without_duplicates['source_type_conference'] == 'x'])
preprint_count = len(df_metadata_without_duplicates[df_metadata_without_duplicates['source_type_preprint'] == 'x'])

# Display results
print(f"Number of journal papers: {journal_count}")
print(f"Number of conference papers: {conference_count}")
print(f"Number of preprint papers: {preprint_count}")

# Calculate percentages
total_papers = len(df_metadata_without_duplicates)
print(f"\nPercentages:")
print(f"Journal papers: {(journal_count/total_papers)*100:.2f}%")
print(f"Conference papers: {(conference_count/total_papers)*100:.2f}%")
print(f"Preprint papers: {(preprint_count/total_papers)*100:.2f}%")

# Create visualization
source_types = ['Journal', 'Conference', 'Preprint']
counts = [journal_count, conference_count, preprint_count]

plt.figure(figsize=(10, 6))
sns.barplot(x=source_types, y=counts)
plt.title('Distribution of Papers by Source Type')
plt.xlabel('Source Type')
plt.ylabel('Number of Papers')
plt.tight_layout()
plt.show()
#%%%
print(f"Number of journal papers: {journal_count}")

# Analyze journal papers specifically
print("\nJournal Analysis:")
journal_papers = df_metadata_without_duplicates[df_metadata_without_duplicates['source_type_journal'] == 'x']
journal_counts = journal_papers['source_title'].value_counts()

print("\nNumber of papers by journal (in descending order):")
for journal, count in journal_counts.items():
    print(f"{journal}: {count}")

# Count papers not in major journals
major_journals = [
    'Sensors',
    'IEEE Transactions on Affective Computing',
    'IEEE Access', 
    'Biomedical Signal Processing and Control',
    'IEEE Journal of Biomedical and Health Informatics',
    'Frontiers in Neuroscience',
    'Electronics (Switzerland)',
    'IEEE Sensors Journal',
    'Frontiers in ICT',
    'Studies in health technology and informatics',
    'Scientific Reports',
    'Expert Systems with Applications',
    'IEEE Transactions on Instrumentation and Measurement',
    'Frontiers in Psychology',
    'Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies'
]

other_journals = journal_papers[~journal_papers['source_title'].isin(major_journals)]
print(f"\nNumber of papers not in {', '.join(major_journals)}: {len(other_journals)}")

#%% 

# Database Analysis
print("\nDatabase Analysis:")

# Load database file
df_databases = pd.read_excel(r'.\data\cleaned\Normalized Table - Data type.xlsx')
df_databases = df_databases.drop_duplicates(subset='paper_id')

# 1. Database Access Analysis
print("\nDatabase Access Analysis:")
db_access_counts = df_databases['db_access'].value_counts()
print("Database access distribution:")
print(db_access_counts)

# Calculate percentages
total_papers_db = len(df_databases)
print("\nPercentages:")
for access_type, count in db_access_counts.items():
    print(f"{access_type}: {(count/total_papers_db)*100:.2f}%")

# 2. Temporal Analysis of Database Access
print("\nTemporal Analysis of Database Access:")
# Create a pivot table of database access by year
db_access_by_year = pd.pivot_table(
    df_databases,
    values='paper_id',
    index='year',
    columns='db_access',
    aggfunc='count',
    fill_value=0
)

# Calculate percentages by year
db_access_by_year_pct = db_access_by_year.div(db_access_by_year.sum(axis=1), axis=0) * 100

# 3. Analysis of Open Databases
#print("\nAnalysis of Open Databases:")
# Filter for papers with open databases
open_db_papers = df_databases[df_databases['db_access'] == 'open']

# Count database types
db_types = {
    'pre-established': len(open_db_papers[open_db_papers['db_public'] == 'x']),
    'new_public': len(open_db_papers[open_db_papers['db_private_and_public'] == 'x']),
    'upon_request': len(open_db_papers[open_db_papers['db_uppon_request'] == 'x'])
}

print("\nOpen Database Types Distribution:")
for db_type, count in db_types.items():
    print(f"{db_type}: {count} ({(count/sum(db_types.values())*100):.2f}%)")

# 4. Most Used Public Databases
print("\nMost Used Public Databases:")
# Get all database columns
db_columns = ['deap', 'amigos', 'mahnob', 'case', 'ascertain', 'cog_load', 
              'multimodal_dyadic_behavior', 'recola', 'decaf', 'driving_workload',
              'liris', 'sense_emotion', 'pmemo', 'hazumi1911', 'bio_vid_emo_db',
              'non_eeg_biosignals_data_set_for_assessment_and_visualization_of_neurological_status',
              'stress_recognition_in_automobile_drivers_data_set', 'pspm_hra1',
              'a multimodal dataset for mixed  emotion recognition', 'affective road',
              'bm-swu', 'bvhp', 'clas', 'cleas', 'ds-3', 'feng et al., 2024',
              'k-emocon', 'lab_to_daily', 'pafew', 'poca', 's-test', 'ubfc-phys',
              'usi_laughs', 'utd', 'verbio', 'vreed', 'wesad']

# Count usage of each database in open access papers
db_usage = {}
for db in db_columns:
    count = len(open_db_papers[open_db_papers[db] == 'x'])
    if count > 0:  # Only show databases that were used
        db_usage[db] = count

# Sort by usage count
db_usage = dict(sorted(db_usage.items(), key=lambda x: x[1], reverse=True))

print("\nPublic Database Usage Count:")
for db, count in db_usage.items():
    print(f"{db}: {count} ({(count/len(open_db_papers)*100):.2f}%)")


# 5. Create summary table for the paper
print("\nSummary Table for Paper:")
summary_data = {
    'Metric': [
        'Total database usages',
        'Restricted databases',
        'Open databases',
        'Pre-established databases',
        'New public databases',
        'Available upon request',
        'DEAP usage',
        'AMIGOS usage',
        'MAHNOB usage'
    ],
    'Count': [
        len(df_databases[df_databases['is_database'] == 'x']),
        len(df_databases[df_databases['db_access'] == 'restricted']),
        len(df_databases[df_databases['db_access'] == 'open']),
        db_types['pre-established'],
        db_types['new_public'],
        db_types['upon_request'],
        db_usage.get('deap', 0),
        db_usage.get('amigos', 0),
        db_usage.get('mahnob', 0)
    ]
}

summary_df = pd.DataFrame(summary_data)
summary_df['Percentage'] = summary_df['Count'] / summary_df['Count'].iloc[0] * 100
print("\nSummary Table:")
print(summary_df.to_string(index=False))

# Análisis adicional para sustentar claims sobre bases de datos
print("\nAnálisis para sustentar claims sobre bases de datos:")

# Tendencia temporal de uso de bases de datos
db_usage_by_year = df_databases[df_databases['is_database'] == 'x'].groupby('year').size()
print(f"\nTendencia de uso de bases de datos por año:")
print(db_usage_by_year)

# Proporción de bases de datos abiertas vs. restringidas
total_dbs = len(df_databases[df_databases['is_database'] == 'x'])
open_dbs = len(df_databases[df_databases['db_access'] == 'open'])
restricted_dbs = len(df_databases[df_databases['db_access'] == 'restricted'])
print(f"\nProporción de bases de datos:")
print(f"Abiertas: {open_dbs} ({open_dbs/total_dbs*100:.2f}%)")
print(f"Restringidas: {restricted_dbs} ({restricted_dbs/total_dbs*100:.2f}%)")

# Análisis de citaciones para las bases de datos más populares
top_dbs = ['deap', 'amigos', 'mahnob']
for db in top_dbs:
    papers_using_db = df_databases[df_databases[db] == 'x']['paper_id'].unique()
    print(f"\nPapers que utilizan {db.upper()}: {len(papers_using_db)}")
    if len(papers_using_db) > 0:
        years_using_db = df_databases[df_databases[db] == 'x']['year'].value_counts().sort_index()
        print(f"Uso por año de {db.upper()}:")
        print(years_using_db)

# Análisis de disponibilidad de datos para reproducibilidad
print("\nDisponibilidad de datos para reproducibilidad:")
reproducible_dbs = len(df_databases[(df_databases['db_access'] == 'open') & 
                                   ((df_databases['db_public'] == 'x') | 
                                    (df_databases['db_private_and_public'] == 'x') |
                                    (df_databases['db_uppon_request'] == 'x'))])
print(f"Bases de datos reproducibles: {reproducible_dbs} ({reproducible_dbs/total_dbs*100:.2f}%)")

#%%
# Figure: Database Access Over Time and Database Usage Frequency
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

sns.set_context("talk")

# Prepare data for panel A (stacked bar by year for access type)
df_access = df_databases[["paper_id", "db_access", "year"]].drop_duplicates()
df_access_crosstab = pd.crosstab(index=df_access['year'], columns=df_access['db_access'], normalize='index')

# Ensure all expected columns are present and in the right order
for col in ['open', 'restricted', 'both']:
    if col not in df_access_crosstab.columns:
        df_access_crosstab[col] = 0
# Reorder columns
panel_a_order = ['open', 'restricted', 'both']
df_access_crosstab = df_access_crosstab[panel_a_order]

# Custom colors for panel A
panel_a_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # blue, orange, green

# Prepare data for panel B (database usage frequency)
db_columns = ['deap', 'amigos', 'mahnob', 'case', 'ascertain', 'cog_load',
              'multimodal_dyadic_behavior', 'recola', 'decaf', 'driving_workload',
              'liris', 'sense_emotion', 'pmemo', 'hazumi1911', 'bio_vid_emo_db',
              'non_eeg_biosignals_data_set_for_assessment_and_visualization_of_neurological_status',
              'stress_recognition_in_automobile_drivers_data_set', 'pspm_hra1',
              'a multimodal dataset for mixed  emotion recognition', 'affective road',
              'bm-swu', 'bvhp', 'clas', 'cleas', 'ds-3', 'feng et al., 2024',
              'k-emocon', 'lab_to_daily', 'pafew', 'poca', 's-test', 'ubfc-phys',
              'usi_laughs', 'utd', 'verbio', 'vreed', 'wesad']

# Melt the database columns for countplot
melted_db = df_databases.melt(id_vars=['paper_id'], value_vars=db_columns, var_name='Database', value_name='Used')
db_freq = melted_db[melted_db['Used'] == 'x']['Database'].value_counts().reset_index()
db_freq.columns = ['Database', 'Count']

# Filter to only databases used by more than one paper
db_freq = db_freq[db_freq['Count'] > 1]

# Custom labels for the most frequent databases (top 11 or all if fewer)
def custom_db_label(db):
    if db.lower() == 'case':
        return 'CASE'
    elif db.lower() == 'vreed':
        return 'VREED'
    elif db.lower() == 'clas':
        return 'CLAS'
    elif db.lower() == 'k-emocon':
        return 'K-EmoCon'
    elif db.lower() == 'amigos':
        return 'AMIGOS'
    elif db.lower() == 'deap':
        return 'DEAP'
    elif db.lower() == 'mahnob':
        return 'MAHNOB'
    elif db.lower() == 'pmemo':
        return 'PMEmo'
    elif db.lower() == 'ascertain':
        return 'ASCERTAIN'
    elif db.lower() == 'recola':
        return 'RECOLA'
    elif db.lower() == 'liris':
        return 'LIRIS'
    elif db.lower() == 'wesad':
        return 'WESAD'
    else:
        return db

top_dbs = db_freq['Database'].tolist()
custom_labels = [custom_db_label(db) for db in top_dbs]

# Create subplots
fig, axes = plt.subplots(2, 1, figsize=(13, 13), dpi=300)

# Panel A: Stacked bar plot for access type by year
df_access_crosstab.plot(kind='bar', stacked=True, rot=0, ax=axes[0], color=panel_a_colors)
axes[0].text(-0.1, 1.1, 'A', transform=axes[0].transAxes, fontsize=36, fontweight='bold')
axes[0].set_xlabel("Year", fontsize=24)
axes[0].set_ylabel("Number of studies")

# Custom legend for panel A (single line, correct order, custom colors)
legend_elements = [
    Patch(facecolor=panel_a_colors[0], label='open'),
    Patch(facecolor=panel_a_colors[1], label='restricted'),
    Patch(facecolor=panel_a_colors[2], label='both')
]
axes[0].legend(handles=legend_elements, title='Access', loc='upper right', bbox_to_anchor=(1, 1.25), frameon=False, fancybox=True, ncol=3, fontsize=18, title_fontsize=20)

# Panel B: Countplot for database usage (only databases used by more than one paper)
sns.barplot(x='Database', y='Count', data=db_freq, order=top_dbs, ax=axes[1], palette="tab10")
axes[1].text(-0.1, 1.1, 'B', transform=axes[1].transAxes, fontsize=36, fontweight='bold')
axes[1].set_xlabel("Databases", fontsize=24)
axes[1].set_ylabel("Number of studies")
axes[1].set_xticklabels(custom_labels, rotation=90, fontsize=20)

fig.tight_layout()
sns.despine(ax=axes[0])
sns.despine(ax=axes[1])
plt.show()

# %%

# Participants
# Ahora tu tarea es analizar la composicion de la muestra en estos estudios
# Debes buscar el promedio de sample size, y el rango (minimo y maximo sample size en la muestra)
# Debes decirme cuantos papers no reportan la edad.
# Tambien debes decirme cuantos papers no reportan el genero. Y, los que si lo hacen, cual es el porcentaje promedio de inclusion de mujeres en la muestra.
# Debes decirme cuantos papers reportan de algun modo la edad de los participantes (edad promedio, rango, mediana, etc.)
# De los papers que si reportan edad, debes decirme promedio across papers y el rango de edad across papers.
# Por ultimo, tambien debes decirme cuantos papers reportan el pais de origen de la muestra. 
# Y un count de los paises reportados

#%%
# Demographic analysis of participants
print("\nParticipant Demographics Analysis:")

# Load participant data
participants_path = r'.\data\cleaned\Normalized Table - Participants.xlsx'
df_participants = pd.read_excel(participants_path)
df_participants = df_participants.fillna('-')

# --- Sample size analysis ---
# Papers with and without sample size (considering if ANY model reports it)
papers_with_n = df_participants[df_participants['n'] != '-']['paper_id'].unique()
papers_without_n = df_participants[~df_participants['paper_id'].isin(papers_with_n)]['paper_id'].unique()

# For calculations, use all models
participants_n = df_participants[df_participants['n'] != '-']
participants_n['n'] = participants_n['n'].astype(int)

# Mean, median, min, max sample size (across all models)
mean_n = participants_n['n'].mean()
median_n = participants_n['n'].median()
min_n = participants_n['n'].min()
max_n = participants_n['n'].max()

print(f"Number of papers reporting sample size: {len(papers_with_n)}")
print(f"Number of papers NOT reporting sample size: {len(papers_without_n)}")
print(f"Mean sample size: {mean_n:.1f}")
print(f"Median sample size: {median_n:.1f}")
print(f"Sample size range: {min_n} - {max_n}")

# Plot age distribution
sns.histplot(participants_n['n'], bins=20, kde=True)
plt.show()

#%%
# --- Gender reporting analysis ---
# Papers reporting female participants (if ANY model reports it)
papers_with_female = df_participants[df_participants['n_female'] != '-']['paper_id'].unique()
papers_without_female = df_participants[~df_participants['paper_id'].isin(papers_with_female)]['paper_id'].unique()

total_papers = len(df_participants['paper_id'].unique())
percent_without_female = (len(papers_without_female) / total_papers) * 100

print(f"Number of papers NOT reporting number of female participants: {len(papers_without_female)} ({percent_without_female:.1f}%)")
print(f"Number of papers reporting number of female participants: {len(papers_with_female)}")

# Percentage of women in the sample (across all models)
female_data = df_participants[(df_participants['n_female'] != '-') & (df_participants['n'] != '-')]
if not female_data.empty:
    female_data['n_female'] = female_data['n_female'].astype(int)
    female_data['n'] = female_data['n'].astype(int)
    female_data['female_percent'] = female_data['n_female'] / female_data['n'] * 100
    mean_female_percent = female_data['female_percent'].mean()
    min_female = female_data['n_female'].min()
    max_female = female_data['n_female'].max()
    print(f"Average percentage of women in the sample (across all models): {mean_female_percent:.1f}%")
    print(f"Range of female participants: {min_female} - {max_female}")
else:
    print("No valid sample size and n_female data to calculate percentage of women.")


# --- Age reporting analysis ---
# Papers reporting mean age (if ANY model reports it)
papers_with_mean_age = df_participants[df_participants['mean_age'] != '-']['paper_id'].unique()
papers_without_mean_age = df_participants[~df_participants['paper_id'].isin(papers_with_mean_age)]['paper_id'].unique()

# Papers reporting age range (if ANY model reports it)
papers_with_range_age = df_participants[df_participants['range_age'] != '-']['paper_id'].unique()
papers_without_range_age = df_participants[~df_participants['paper_id'].isin(papers_with_range_age)]['paper_id'].unique()

# Papers reporting any age info
papers_reporting_age = set(papers_with_mean_age).union(set(papers_with_range_age))
papers_not_reporting_age = total_papers - len(papers_reporting_age)
percent_not_reporting_age = (papers_not_reporting_age / total_papers) * 100

print(f"Number of papers reporting any age info: {len(papers_reporting_age)}")
print(f"Number of papers NOT reporting any age info: {papers_not_reporting_age} ({percent_not_reporting_age:.1f}%)")

# Mean of reported mean ages (across all models)
mean_age_data = df_participants[df_participants['mean_age'] != '-']
if not mean_age_data.empty:
    mean_age_data['mean_age'] = mean_age_data['mean_age'].astype(float)
    mean_of_means = mean_age_data['mean_age'].mean()
    min_mean_age = mean_age_data['mean_age'].min()
    max_mean_age = mean_age_data['mean_age'].max()
    print(f"Mean of reported mean ages: {mean_of_means:.1f} (range: {min_mean_age}-{max_mean_age})")
    
    # Papers that report mean age
    papers_with_mean_age = mean_age_data['paper_id'].unique()
    
    # Papers that report both mean age and age range
    papers_with_both = df_participants[
        (df_participants['mean_age'] != '-') & 
        (df_participants['range_age'] != '-')
    ]['paper_id'].unique()
    
    # Calculate percentage of papers with mean age but no range
    papers_with_mean_no_range = set(papers_with_mean_age) - set(papers_with_both)
    percentage_no_range = (len(papers_with_mean_no_range) / len(papers_with_mean_age)) * 100
    
    print(f"\nOf the papers that reported average participant age, {percentage_no_range:.1f}% did not provide the age range.")
    
    # Calculate statistics for papers reporting both
    if len(papers_with_both) > 0:
        both_data = df_participants[
            (df_participants['paper_id'].isin(papers_with_both)) & 
            (df_participants['mean_age'] != '-')
        ]
        both_data['mean_age'] = both_data['mean_age'].astype(float)
        mean_with_range = both_data['mean_age'].mean()
        min_with_range = both_data['mean_age'].min()
        max_with_range = both_data['mean_age'].max()
        print(f"In those that did, the mean age was {mean_with_range:.1f} years, with a range from {min_with_range:.2f} to {max_with_range:.1f} years.")
else:
    print("No mean ages reported.")

# Range of ages (across all models)
range_age_data = df_participants[df_participants['range_age'] != '-']
if not range_age_data.empty:
    # Normalizar guiones y limpiar espacios
    def clean_range(val):
        if isinstance(val, str):
            val = val.replace('–', '-').replace('—', '-').replace('‐', '-')
            val = val.replace(',', '.')
            val = re.sub(r'[^0-9\-.]', '', val)  # Solo números, punto, guion
            val = val.strip('-').strip()
        return val

    range_age_data['range_age_clean'] = range_age_data['range_age'].apply(clean_range)
    # Extraer los valores numéricos
    min_vals = []
    max_vals = []
    for val in range_age_data['range_age_clean']:
        if '-' in val:
            parts = val.split('-')
            try:
                min_vals.append(float(parts[0]))
                max_vals.append(float(parts[1]))
            except Exception:
                continue
    if min_vals and max_vals:
        min_range = min(min_vals)
        max_range = max(max_vals)
        print(f"Age range across all papers: {min_range}-{max_range}")
    else:
        print("No valid age ranges found.")
else:
    print("No age ranges reported.")


# --- Country of origin reporting ---
# Papers reporting country (if ANY model reports it)
papers_with_country = df_participants[df_participants['country'] != '-']['paper_id'].unique()
papers_without_country = df_participants[~df_participants['paper_id'].isin(papers_with_country)]['paper_id'].unique()

percent_with_country = (len(papers_with_country) / total_papers) * 100

print(f"Number of papers reporting country of origin: {len(papers_with_country)} ({percent_with_country:.1f}%)")
print(f"Number of papers NOT reporting country of origin: {len(papers_without_country)}")

# Count of reported countries (across all models)
country_data = df_participants[df_participants['country'] != '-']
country_counts = country_data['country'].value_counts()
print("\nReported country counts:")
print(country_counts)

#%%
# SELF REPORT

# Helper functions for network visualization
def nudge(pos, x_shift, y_shift):
    return {n:(x + x_shift, y + y_shift) for n,(x,y) in pos.items()}

# Define boxes for network labels
boxes = dict(facecolor='white', alpha=1)

# Self-report Analysis
print("\nSelf-report Analysis:")

# Load self-report data
df_self_report = pd.read_excel(r'.\data\cleaned\Normalized Table - Self-report.xlsx')
df_self_report = df_self_report.fillna('-')

# Analysis of questionnaire usage
print("\nQuestionnaire Usage Analysis:")
# Clean and standardize questionnaire usage data
df_self_report['use_questionnaire'] = df_self_report['use_questionnaire'].str.replace('x', "Yes")
df_self_report['use_questionnaire'] = df_self_report['use_questionnaire'].str.replace('-', "No")
df_self_report['use_questionnaire'] = df_self_report['use_questionnaire'].str.replace("Relies on  other's questionnaire", "Relies on other's questionnaire")
df_self_report['use_questionnaire'] = df_self_report['use_questionnaire'].str.replace("Relies on other´s questionaire", "Relies on other's questionnaire")

# Get unique questionnaire usage per paper
used_questionnaires = df_self_report.groupby(['paper_id', "use_questionnaire"]).nth(0)
used_questionnaires.reset_index(inplace=True)

# Calculate absolute and percentage counts
questionnaire_counts = used_questionnaires["use_questionnaire"].value_counts()
questionnaire_percentages = used_questionnaires["use_questionnaire"].value_counts(normalize=True).mul(100).round(1)

print("\nQuestionnaire Usage Distribution:")
for usage, count in questionnaire_counts.items():
    percentage = questionnaire_percentages[usage]
    print(f"{usage}: {count} ({percentage}%)")

# Standardized Questionnaire Analysis
print("\nStandardized Questionnaire Analysis:")
# Get unique questionnaire usage per paper
questionnaires = df_self_report.groupby(['paper_id', 'affective_questionnaire_SAM', 'affective_questionnaire_PSS', 
                                       'affective_questionnaire_PANAS', 'affective_questionnaire_DES', 
                                       'affective_questionnaire_affective_grid']).nth(0)
questionnaires.reset_index(inplace=True)

# Melt the questionnaire columns
questionnaire_columns = ['affective_questionnaire_SAM', 'affective_questionnaire_PSS', 
                        'affective_questionnaire_PANAS', 'affective_questionnaire_DES', 
                        'affective_questionnaire_affective_grid']
questionnaires_melted = pd.melt(questionnaires, 
                               id_vars=['paper_id'], 
                               value_vars=questionnaire_columns,
                               var_name='Questionnaire',
                               value_name='Used')

# Clean questionnaire names
questionnaires_melted['Questionnaire'] = questionnaires_melted['Questionnaire'].str.replace('affective_questionnaire_', '')

# Calculate usage statistics
questionnaire_usage = questionnaires_melted[questionnaires_melted['Used'] == 'x']['Questionnaire'].value_counts()
questionnaire_usage_pct = questionnaires_melted[questionnaires_melted['Used'] == 'x']['Questionnaire'].value_counts(normalize=True).mul(100).round(1)

print("\nStandardized Questionnaire Usage:")
for questionnaire, count in questionnaire_usage.items():
    percentage = questionnaire_usage_pct[questionnaire]
    print(f"{questionnaire}: {count} ({percentage}%)")

# Calculate percentage of papers using SAM among those that specified questionnaires
papers_with_questionnaires = used_questionnaires[used_questionnaires["use_questionnaire"] == "Yes"]["paper_id"].unique()
sam_papers = questionnaires[questionnaires['affective_questionnaire_SAM'] == 'x']["paper_id"].unique()
panas_papers = questionnaires[questionnaires['affective_questionnaire_PANAS'] == 'x']["paper_id"].unique()

sam_percentage = (len(sam_papers) / len(papers_with_questionnaires)) * 100
panas_percentage = (len(panas_papers) / len(papers_with_questionnaires)) * 100

print(f"\nPercentage of papers using SAM among those specifying questionnaires: {sam_percentage:.1f}%")
print(f"Percentage of papers using PANAS among those specifying questionnaires: {panas_percentage:.1f}%")

# Analysis of emotional categories
emotional_categories = [
    'anger', 'disgust', 'fear', 'sadness', 'surprise', 'happiness', 
    'pleasant', 'anxiety', 'neutral', 'funny', 'amusement', 'joy',
    'bored', 'calm', 'calmness', 'cheerful', 'concentration', 'confrustion',
    'confusion', 'contempt', 'dejection', 'delight', 'depressed', 'engaged',
    'eureka', 'excited', 'frustration', 'negative', 'nervous', 'positive',
    'pride', 'relieved', 'sorrow', 'tenderness'
]

# Group by paper_id and get first occurrence of each category
emotional_categories_grouped = df_self_report.groupby(['paper_id'] + emotional_categories).nth(0)
emotional_categories_grouped.reset_index(inplace=True)

# Melt the dataframe to get category frequencies
emotional_categories_melted = pd.melt(
    emotional_categories_grouped,
    id_vars=['paper_id'],
    value_vars=emotional_categories,
    var_name='variable',
    value_name='value'
)

# Filter for categories that were used (marked with 'x')
emotional_categories_used = emotional_categories_melted[emotional_categories_melted['value'] == 'x']

# Analysis of emotional dimensions
emotional_dimensions = [
    'valence', 'arousal', 'dominance', 'like_dislike', 'familiarity',
    'stress', 'engagement', 'predictability', 'boredom', 'relaxation',
    'anxiety_dim', 'flow', 'frustration_dim'
]

# Group by paper_id and get first occurrence of each dimension
emotional_dimensions_grouped = df_self_report.groupby(['paper_id'] + emotional_dimensions).nth(0)
emotional_dimensions_grouped.reset_index(inplace=True)

# Melt the dataframe to get dimension frequencies
emotional_dimensions_melted = pd.melt(
    emotional_dimensions_grouped,
    id_vars=['paper_id'],
    value_vars=emotional_dimensions,
    var_name='variable',
    value_name='value'
)

# Filter for dimensions that were used (marked with 'x')
emotional_dimensions_used = emotional_dimensions_melted[emotional_dimensions_melted['value'] == 'x']

# Create adjacency matrices for network analysis
# For categories
df_matrix_cat = df_self_report[emotional_categories].replace(['-', '--', '---', ' ', np.nan], 0).replace('x', 1)
df_matrix_cat = df_matrix_cat.astype(int)  # Convert to integer type
adj_matrix_cat = df_matrix_cat.T.dot(df_matrix_cat)
np.fill_diagonal(adj_matrix_cat.values, 0)

# For dimensions
df_matrix_dim = df_self_report[emotional_dimensions].replace(['-', '--', '---', ' ', np.nan], 0).replace('x', 1)
df_matrix_dim = df_matrix_dim.astype(int)  # Convert to integer type
adj_matrix_dim = df_matrix_dim.T.dot(df_matrix_dim)
np.fill_diagonal(adj_matrix_dim.values, 0)

# Create network graphs
G_cat = nx.DiGraph(adj_matrix_cat)
G_dim = nx.DiGraph(adj_matrix_dim)

# Get edge weights
weights_cat = nx.get_edge_attributes(G_cat, 'weight').values()
weights_dim = nx.get_edge_attributes(G_dim, 'weight').values()

# Create the final 4-panel figure
#plt.style.use('seaborn')
sns.set_context("talk")

# Rename for display
rename_dim = {'anxiety_dim': 'Anxiety', 'frustration_dim': 'Frustration'}

def rename_dimension_label(label):
    return rename_dim.get(label, label.capitalize() if label != 'like_dislike' else 'Preference')

def rename_dimension_label_network(label):
    return rename_dim.get(label, label)

# Create figure with custom GridSpec
fig = plt.figure(figsize=(15, 15), dpi=300)
gs = gridspec.GridSpec(2, 2)

# SUBPLOT A: Categories bar plot
ax1 = plt.subplot(gs[0, 0])
categories_count = emotional_categories_used['variable'].value_counts()
categories = categories_count.index.tolist()
categories = [category.capitalize() for category in categories]
values = categories_count.tolist()
colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
ax1.bar(categories, values, color=colors)
ax1.set_title('Categories', fontweight='bold')
ax1.set_ylabel('Number of models')
# Reduce font size and rotate for better legibility
ax1.set_xticklabels(categories, rotation=60, ha='right', fontsize=10)
ax1.text(-0.1, 1.1, 'A', transform=ax1.transAxes, fontsize=36, fontweight='bold')

# SUBPLOT B: Dimensions bar plot
ax2 = plt.subplot(gs[0, 1])
dimensions_count = emotional_dimensions_used['variable'].replace(rename_dim).value_counts()
dimensions = dimensions_count.index.tolist()
dimensions = [rename_dimension_label(dim) for dim in dimensions]
values2 = dimensions_count.tolist()
colors_b = plt.cm.tab10(np.linspace(0, 1, len(dimensions)))
ax2.bar(dimensions, values2, color=colors_b)
ax2.set_title('Dimensions', fontweight='bold')
ax2.set_ylabel('Number of models')
ax2.set_xticklabels(dimensions, rotation=45, ha='right', fontsize=12)
ax2.text(-0.1, 1.1, 'B', transform=ax2.transAxes, fontsize=36, fontweight='bold')

# SUBPLOT C: Categories network
ax3 = plt.subplot(gs[1, 0])
# Only include categories that appear in more than 2 studies
good_categories = [cat for cat, count in categories_count.items() if count > 2]
G_cat_sub = G_cat.subgraph(good_categories)
shell_nodes = [list(G_cat_sub.nodes())]
pos_C = nx.shell_layout(G_cat_sub, nlist=shell_nodes)
for k in pos_C:
    pos_C[k] = pos_C[k] * 2.0
mapping_cat = {k: k.capitalize() for k in G_cat_sub.nodes()}
G_cat_title = nx.relabel_nodes(G_cat_sub, mapping_cat)
pos_C_title = {mapping_cat[k]: v for k, v in pos_C.items()}
pos_nodes_C = nudge(pos_C_title, 0, 0.25)
# Fixed node size, 25% larger than previous (previously 675, now 844)
fixed_node_size = 844
nx.draw(G_cat_title, pos_C_title, ax=ax3, edgecolors="black", node_color='white', 
        linewidths=3, font_size=10, font_weight="bold", 
        width=[i/4 for i in weights_cat if k in good_categories], arrows=False,
        node_size=fixed_node_size)
# Draw node labels as the number of studies inside the node (font size 25% smaller than before)
for node, (x, y) in pos_C_title.items():
    original = node.lower()
    n_studies = categories_count.get(original, 0)
    ax3.text(x, y, str(n_studies), fontsize=10.5, ha='center', va='center', fontweight='bold', color='black', zorder=10)
nx.draw_networkx_labels(G_cat_title, pos=pos_nodes_C, labels=None, font_size=10, 
                       font_color='k', font_family='sans-serif', 
                       font_weight='normal', alpha=None, bbox=boxes, 
                       horizontalalignment='center', verticalalignment='center', ax=ax3)
ax3.set_title('Categories', fontweight='bold')
ax3.text(-0.1, 1.1, 'C', transform=ax3.transAxes, fontsize=36, fontweight='bold')

# SUBPLOT D: Dimensions network
ax4 = plt.subplot(gs[1, 1])
# Only include dimensions that appear in more than 2 studies
good_dimensions = [dim for dim, count in dimensions_count.items() if count > 2]
G_dim_sub = G_dim.subgraph(good_dimensions)
pos_D = nx.circular_layout(G_dim_sub)
for k in pos_D:
    pos_D[k] = pos_D[k] * 1.4
def dim_label_title(k):
    if k == 'like_dislike':
        return 'Preference'
    label = rename_dimension_label_network(k)
    return label.capitalize()
mapping_dim = {k: dim_label_title(k) for k in G_dim_sub.nodes()}
G_dim_title = nx.relabel_nodes(G_dim_sub, mapping_dim)
pos_D_title = {mapping_dim[k]: v for k, v in pos_D.items()}
pos_nodes_D = nudge(pos_D_title, 0.05, 0.15)
nx.draw(G_dim_title, pos_D_title, ax=ax4, edgecolors="black", node_color='white', 
        linewidths=3, font_size=10, font_weight="bold", 
        width=[i/4 for i in weights_dim if k in good_dimensions], arrows=False,
        node_size=fixed_node_size)
# Draw node labels as the number of studies inside the node (font size 25% smaller than before)
for node, (x, y) in pos_D_title.items():
    if node == 'Preference':
        original = 'like_dislike'
    else:
        original = node.lower()
    n_studies = dimensions_count.get(original, 0)
    ax4.text(x, y, str(n_studies), fontsize=10.5, ha='center', va='center', fontweight='bold', color='black', zorder=10)
nx.draw_networkx_labels(G_dim_title, pos=pos_nodes_D, labels=None, font_size=12, 
                       font_color='k', font_family='sans-serif', 
                       font_weight='normal', alpha=None, bbox=boxes, 
                       horizontalalignment='center', verticalalignment='center', ax=ax4)
ax4.set_title('Dimensions', fontweight='bold')
ax4.text(-0.1, 1.1, 'D', transform=ax4.transAxes, fontsize=36, fontweight='bold')

# Remove spines
sns.despine(ax=ax1)
sns.despine(ax=ax2)
sns.despine(ax=ax3)
sns.despine(ax=ax4)

# Print statistics for the report
print("\nEmotional Categories Analysis:")
print(f"Total number of unique categories used: {len(categories_count)}")
print("\nTop 6 most used categories:")
print(categories_count.head(6))

print("\nEmotional Dimensions Analysis:")
print(f"Total number of unique dimensions used: {len(dimensions_count)}")
print("\nTop 5 most used dimensions:")
print(dimensions_count.head(5))

# Identify the most common categories for the paper
#top_categories = categories_count.head(6).index.tolist()
#print("\nMost common emotional categories:")
#print(", ".join([f"'{cat.capitalize()}'" for cat in top_categories]))

# Tight layout and show
plt.tight_layout()
plt.show()

#%%
# Comprehensive Self-report Analysis
print("\nComprehensive Self-report Analysis:")

# Load self-report data
df_self_report = pd.read_excel(r'.\data\cleaned\Normalized Table - Self-report.xlsx')
df_self_report = df_self_report.fillna('-')

# 1. Questionnaire Usage Analysis
print("\nQuestionnaire Usage Analysis:")

# Clean and standardize questionnaire usage data
df_self_report['use_questionnaire'] = df_self_report['use_questionnaire'].str.replace('x', "Yes")
df_self_report['use_questionnaire'] = df_self_report['use_questionnaire'].str.replace('-', "No")
df_self_report['use_questionnaire'] = df_self_report['use_questionnaire'].str.replace("Relies on  other's questionnaire", "Relies on other's questionnaire")
df_self_report['use_questionnaire'] = df_self_report['use_questionnaire'].str.replace("Relies on other´s questionaire", "Relies on other's questionnaire")

# Get unique questionnaire usage per paper
used_questionnaires = df_self_report.groupby(['paper_id', "use_questionnaire"]).nth(0)
used_questionnaires.reset_index(inplace=True)

# Calculate absolute and percentage counts
questionnaire_counts = used_questionnaires["use_questionnaire"].value_counts()
questionnaire_percentages = used_questionnaires["use_questionnaire"].value_counts(normalize=True).mul(100).round(1)

print("\nQuestionnaire Usage Distribution:")
for usage, count in questionnaire_counts.items():
    percentage = questionnaire_percentages[usage]
    print(f"{usage}: {count} ({percentage}%)")

# 2. Standardized Questionnaire Analysis
print("\nStandardized Questionnaire Analysis:")

# Get unique questionnaire usage per paper
questionnaires = df_self_report.groupby(['paper_id', 'affective_questionnaire_SAM', 'affective_questionnaire_PSS', 
                                       'affective_questionnaire_PANAS', 'affective_questionnaire_DES', 
                                       'affective_questionnaire_affective_grid']).nth(0)
questionnaires.reset_index(inplace=True)

# Melt the questionnaire columns
questionnaire_columns = ['affective_questionnaire_SAM', 'affective_questionnaire_PSS', 
                        'affective_questionnaire_PANAS', 'affective_questionnaire_DES', 
                        'affective_questionnaire_affective_grid']
questionnaires_melted = pd.melt(questionnaires, 
                               id_vars=['paper_id'], 
                               value_vars=questionnaire_columns,
                               var_name='Questionnaire',
                               value_name='Used')

# Clean questionnaire names
questionnaires_melted['Questionnaire'] = questionnaires_melted['Questionnaire'].str.replace('affective_questionnaire_', '')

# Calculate usage statistics
questionnaire_usage = questionnaires_melted[questionnaires_melted['Used'] == 'x']['Questionnaire'].value_counts()
questionnaire_usage_pct = questionnaires_melted[questionnaires_melted['Used'] == 'x']['Questionnaire'].value_counts(normalize=True).mul(100).round(1)

print("\nStandardized Questionnaire Usage:")
for questionnaire, count in questionnaire_usage.items():
    percentage = questionnaire_usage_pct[questionnaire]
    print(f"{questionnaire}: {count} ({percentage}%)")

# Calculate percentage of papers using SAM among those that specified questionnaires
papers_with_questionnaires = used_questionnaires[used_questionnaires["use_questionnaire"] == "Yes"]["paper_id"].unique()
sam_papers = questionnaires[questionnaires['affective_questionnaire_SAM'] == 'x']["paper_id"].unique()
panas_papers = questionnaires[questionnaires['affective_questionnaire_PANAS'] == 'x']["paper_id"].unique()

sam_percentage = (len(sam_papers) / len(papers_with_questionnaires)) * 100
panas_percentage = (len(panas_papers) / len(papers_with_questionnaires)) * 100

print(f"\nPercentage of papers using SAM among those specifying questionnaires: {sam_percentage:.1f}%")
print(f"Percentage of papers using PANAS among those specifying questionnaires: {panas_percentage:.1f}%")

# 3. Emotional Categories Analysis
print("\nEmotional Categories Analysis:")

# Get unique category usage per paper
emotional_categories = [
    'anger', 'disgust', 'fear', 'sadness', 'surprise', 'happiness', 
    'pleasant', 'anxiety', 'neutral', 'funny', 'amusement', 'joy',
    'bored', 'calm', 'calmness', 'cheerful', 'concentration', 'confrustion',
    'confusion', 'contempt', 'dejection', 'delight', 'depressed', 'engaged',
    'eureka', 'excited', 'frustration', 'negative', 'nervous', 'positive',
    'pride', 'relieved', 'sorrow', 'tenderness'
]

categories_grouped = df_self_report.groupby(['paper_id'] + emotional_categories).nth(0)
categories_grouped.reset_index(inplace=True)

# Melt the dataframe to get category frequencies
categories_melted = pd.melt(
    categories_grouped,
    id_vars=['paper_id'],
    value_vars=emotional_categories,
    var_name='Category',
    value_name='Used'
)

# Filter for categories that were used
categories_used = categories_melted[categories_melted['Used'] == 'x']

# Calculate usage statistics
category_counts = categories_used['Category'].value_counts()
category_percentages = categories_used['Category'].value_counts(normalize=True).mul(100).round(1)

print("\nEmotional Categories Usage:")
print(f"Total number of unique categories used: {len(category_counts)}")
print("\nTop 10 most used categories:")
for category, count in category_counts.head(10).items():
    percentage = category_percentages[category]
    print(f"{category.capitalize()}: {count} ({percentage}%)")

# 4. Emotional Dimensions Analysis
print("\nEmotional Dimensions Analysis:")

# Get unique dimension usage per paper
emotional_dimensions = [
    'valence', 'arousal', 'dominance', 'like_dislike', 'familiarity',
    'stress', 'engagement', 'predictability', 'boredom', 'relaxation',
    'anxiety_dim', 'flow', 'frustration_dim'
]

dimensions_grouped = df_self_report.groupby(['paper_id'] + emotional_dimensions).nth(0)
dimensions_grouped.reset_index(inplace=True)

# Melt the dataframe to get dimension frequencies
dimensions_melted = pd.melt(
    dimensions_grouped,
    id_vars=['paper_id'],
    value_vars=emotional_dimensions,
    var_name='Dimension',
    value_name='Used'
)

# Filter for dimensions that were used
dimensions_used = dimensions_melted[dimensions_melted['Used'] == 'x']

# Calculate usage statistics
dimension_counts = dimensions_used['Dimension'].value_counts()
dimension_percentages = dimensions_used['Dimension'].value_counts(normalize=True).mul(100).round(1)

print("\nEmotional Dimensions Usage:")
print(f"Total number of unique dimensions used: {len(dimension_counts)}")
print("\nAll dimensions usage:")
for dimension, count in dimension_counts.items():
    percentage = dimension_percentages[dimension]
    print(f"{dimension.capitalize()}: {count} ({percentage}%)")

# 5. Analysis of Papers Using Both Categories and Dimensions
print("\nAnalysis of Papers Using Both Categories and Dimensions:")

# Get papers using categories
papers_with_categories = df_self_report[df_self_report[emotional_categories].isin(['x']).any(axis=1)]['paper_id'].unique()
papers_with_dimensions = df_self_report[df_self_report[emotional_dimensions].isin(['x']).any(axis=1)]['paper_id'].unique()

# Calculate overlap
papers_with_both = set(papers_with_categories).intersection(set(papers_with_dimensions))
papers_with_only_categories = set(papers_with_categories) - set(papers_with_dimensions)
papers_with_only_dimensions = set(papers_with_dimensions) - set(papers_with_categories)

total_papers = len(df_self_report['paper_id'].unique())

print(f"\nTotal number of papers: {total_papers}")
print(f"Papers using both categories and dimensions: {len(papers_with_both)} ({len(papers_with_both)/total_papers*100:.1f}%)")
print(f"Papers using only categories: {len(papers_with_only_categories)} ({len(papers_with_only_categories)/total_papers*100:.1f}%)")
print(f"Papers using only dimensions: {len(papers_with_only_dimensions)} ({len(papers_with_only_dimensions)/total_papers*100:.1f}%)")


# Emotion Elicitation Techniques Analysis
print("\nEmotion Elicitation Techniques Analysis:")

# Load emotion elicitation techniques data
df_eet = pd.read_excel(r'.\data\cleaned\Normalized Table - Emotion elicitation techniques.xlsx')
df_eet.fillna('-', inplace=True)

# 1. Standardized Techniques Analysis
print("\nStandardized Techniques Analysis:")

# Define standardized techniques and their variants
standardized_techniques = {
    'IAPS': ['IAPS', 'IAPS-guided autobiographical conversation'],
    'TSST': ['TSST', 'Trier social stress test', 'Trier Social Stress Test (TSST)', 'Modified Trier Social Stress Test (TSST)'],
    'SCWT': ['Stroop color-word interference test', 'SCWT'],
    'Rapid-ABC': ['Rapid-ABC play protocol'],
    'IADS': ['International Affective Digitized Sound', 'IADS'],
    'Robin': ['Robin'],
    'AMT': ['Modified Autobiographical Memory Test']
}

# Load and prepare data
df_techniques_no_dup = df_eet.drop_duplicates(subset="paper_id")
df_techniques_no_dup = df_techniques_no_dup.groupby(['paper_id', "technique_name"]).nth(0)
df_techniques_no_dup.reset_index(inplace=True)

# Initialize counters
technique_counts = {tech: 0 for tech in standardized_techniques.keys()}
total_standardized = 0
total_papers = len(df_techniques_no_dup['paper_id'].unique())

# Count occurrences of each standardized technique
for paper_id, row in df_techniques_no_dup.iterrows():
    technique_name = row['technique_name']
    if technique_name != '-':
        found = False
        for tech, variants in standardized_techniques.items():
            if any(variant.lower() in technique_name.lower() for variant in variants):
                technique_counts[tech] += 1
                total_standardized += 1
                found = True
                break

# Calculate percentages
technique_percentages = {tech: (count/total_papers)*100 for tech, count in technique_counts.items()}
total_percentage = (total_standardized/total_papers)*100

print("\nStandardized Techniques Usage:")
print(f"Total papers using standardized techniques: {total_standardized} ({total_percentage:.1f}%)")
print("\nBreakdown by technique:")
for technique, count in technique_counts.items():
    percentage = technique_percentages[technique]
    print(f"{technique}: {count} ({percentage:.1f}%)")

# Create visualization for standardized techniques
plt.figure(figsize=(10, 6))
technique_df = pd.DataFrame({
    'Technique': list(technique_counts.keys()),
    'Count': list(technique_counts.values())
})
sns.barplot(data=technique_df, x='Count', y='Technique')
plt.title('Standardized Techniques Usage')
plt.xlabel('Number of Studies')
plt.tight_layout()
plt.show()

# 2. Multimodal Analysis
print("\nMultimodal Analysis:")
multimodal_counts = df_eet["is_multimodal"].value_counts()
multimodal_percentages = df_eet["is_multimodal"].value_counts(normalize=True).mul(100).round(1)

print("\nMultimodal Usage:")
for modality, count in multimodal_counts.items():
    percentage = multimodal_percentages[modality]
    print(f"{modality}: {count} ({percentage}%)")

# 3. Task Type Analysis
print("\nTask Type Analysis:")
type_task = df_eet.groupby(['paper_id', 'task_type_active', 'task_type_passive']).nth(0)
type_task.reset_index(inplace=True)

# Count papers with no task type specified
no_task_type = len(type_task[(type_task['task_type_active'] == '-') & 
                            (type_task['task_type_passive'] == '-')])
print(f"\nPapers with no task type specified: {no_task_type}")

# Calculate task type distribution
task_type = fn.multi_reversing(type_task, 'model_id', type_task[['task_type_active', 'task_type_passive']])
task_counts = task_type['variable'].value_counts()
task_percentages = task_type['variable'].value_counts(normalize=True).mul(100).round(1)

print("\nTask Type Distribution:")
for task, count in task_counts.items():
    percentage = task_percentages[task]
    print(f"{task}: {count} ({percentage}%)")

# 4. Modality Analysis
print("\nModality Analysis:")
freq_modality = df_eet.groupby(['paper_id', 'is_multimodal', 'modality_visual', 
                               'modality_auditory', 'modality_somatosensory']).nth(0)
freq_modality.reset_index(inplace=True)

# Store paper_id before multi_reversing
paper_ids = freq_modality['paper_id']

df_modality = fn.multi_reversing(freq_modality, 'model_id', 
                                freq_modality[['is_multimodal', 'modality_visual', 
                                             'modality_auditory', 'modality_somatosensory']])

# Add paper_id back to the DataFrame
df_modality['paper_id'] = paper_ids

modality_counts = df_modality['variable'].value_counts()
modality_percentages = df_modality['variable'].value_counts(normalize=True).mul(100).round(1)

print("\nModality Distribution:")
for modality, count in modality_counts.items():
    percentage = modality_percentages[modality]
    print(f"{modality}: {count} ({percentage}%)")

# 5. Visual Modality Analysis
print("\nVisual Modality Analysis:")
visual_modality = df_eet.groupby(['paper_id', 'visual_pictures', 'visual_videos', 
                                 'visual_words', 'visual_other']).nth(0)
visual_modality.reset_index(inplace=True)

# Store paper_id before multi_reversing
paper_ids = visual_modality['paper_id']

df_visual_modality = fn.multi_reversing(visual_modality, 'model_id', 
                                       visual_modality[['visual_pictures', 'visual_videos', 
                                                      'visual_words', 'visual_other']])

# Add paper_id back to the DataFrame
df_visual_modality['paper_id'] = paper_ids

visual_counts = df_visual_modality['variable'].value_counts()
visual_percentages = df_visual_modality['variable'].value_counts(normalize=True).mul(100).round(1)

print("\nVisual Modality Distribution:")
for visual, count in visual_counts.items():
    percentage = visual_percentages[visual]
    print(f"{visual}: {count} ({percentage}%)")

# 6. Comprehensive Techniques Analysis
print("\nComprehensive Techniques Analysis:")
all_techniques = df_eet.groupby(['paper_id', 'visual_pictures', 'visual_videos', 'visual_words', 
                                'visual_other', 'auditory_music', 'auditory_other',
                                'technique_clasif_driving',
                                'technique_clasif_imagination_techniques_or_memory_recall',
                                'technique_clasif_social_interactions',
                                'technique_clasif_virtual_reality', 
                                'technique_clasif_meditation',
                                'technique_clasif_reading', 'technique_clasif_ux',
                                'technique_clasif_tactile_enhanced_multimedia_clips',
                                'technique_clasif_videogame', 
                                'technique_clasif_puzzle']).nth(0)
all_techniques.reset_index(inplace=True)

# Store paper_id before multi_reversing
paper_ids = all_techniques['paper_id']

df_all_techniques = fn.multi_reversing(all_techniques, 'model_id', 
                                      all_techniques[['visual_pictures', 'visual_videos', 
                                                     'visual_words', 'visual_other',
                                                     'auditory_music', 'auditory_other',
                                                     'technique_clasif_driving',
                                                     'technique_clasif_imagination_techniques_or_memory_recall',
                                                     'technique_clasif_social_interactions',
                                                     'technique_clasif_virtual_reality', 
                                                     'technique_clasif_meditation',
                                                     'technique_clasif_reading', 
                                                     'technique_clasif_ux',
                                                     'technique_clasif_tactile_enhanced_multimedia_clips',
                                                     'technique_clasif_videogame', 
                                                     'technique_clasif_puzzle']])

# Add paper_id back to the DataFrame
df_all_techniques['paper_id'] = paper_ids

technique_counts = df_all_techniques['variable'].value_counts()
technique_percentages = df_all_techniques['variable'].value_counts(normalize=True).mul(100).round(1)

print("\nAll Techniques Distribution:")
for technique, count in technique_counts.items():
    percentage = technique_percentages[technique]
    print(f"{technique}: {count} ({percentage}%)")

# 7. Summary Statistics
print("\nSummary Statistics:")
total_papers = len(df_eet['paper_id'].unique())
total_techniques = len(df_all_techniques)
print(f"Total number of papers analyzed: {total_papers}")
print(f"Total number of technique instances: {total_techniques}")

# Calculate average techniques per paper
techniques_per_paper = df_all_techniques.groupby('paper_id').size()
print(f"Average techniques per paper: {techniques_per_paper.mean():.2f}")
print(f"Range of techniques per paper: {techniques_per_paper.min()} - {techniques_per_paper.max()}")


#%%
# Electrodermal Activity (EDA) Analysis
print("\nElectrodermal Activity (EDA) Analysis:")

# Load EDA data
df_eda = pd.read_excel(r'.\data\cleaned\Normalized Table - EDA.xlsx')
df_eda.fillna('-', inplace=True)

# 1. EDA Device Analysis
eda_devices = df_eda[~((df_eda['eda_device_specification'] == '-') &
      (df_eda['eda_device_is_homemade'] == '-'))         
]

# Count papers with no device specified (but some are homemade)
aver = df_eda.groupby(['paper_id', 'eda_device_specification']).nth(0)
aver.reset_index(inplace= True)
no_device_specified = len(aver[aver['eda_device_specification'] == '-'])
print(f"Papers with no device specified: {no_device_specified}")

# Calculate total number of unique papers
total_eda_papers = len(df_eda['paper_id'].unique())

# Percentage of papers with no device specified and not homemade
no_device_percentage = (no_device_specified / total_eda_papers) * 100
print(f"Percentage of papers with no device specified: {no_device_percentage:.1f}%")

# Group by paper_id and get first occurrence of each device
eda_devices = eda_devices.groupby(['paper_id', 'eda_device_specification']).nth(0)
eda_devices.reset_index(inplace=True)

# Count papers with homemade devices
homemade_counts = eda_devices['eda_device_is_homemade'].value_counts()
print("\nHomemade device counts:")
print(homemade_counts)

homemade_percentages = eda_devices['eda_device_is_homemade'].value_counts(normalize=True).mul(100).round(1)
print("\nHomemade device percentages:")
print(homemade_percentages.astype(str) + '%')

# Calculate percentage of homemade devices
homemade_percentage = homemade_percentages.get('x', 0)
print(f"\nPercentage of studies employing custom-made devices: {homemade_percentage}%")

# Map EDA devices to standardized names
mapping_eda = {
    # —— BIOPAC ——————————————————————————
    'BIOPAC': 'BIOPAC', 'BIOPAC MP150': 'BIOPAC',
    'BIOPAC MP150 with EDA100C module': 'BIOPAC',
    'BIOPAC MP160': 'BIOPAC', 'BIOPAC MP35': 'BIOPAC',
    'BIOPAC MP36': 'BIOPAC', 'BIOPAC-MP150': 'BIOPAC',
    "BIOPAC's MP150": 'BIOPAC', 'Biopac': 'BIOPAC',
    'Biopac\nMP36': 'BIOPAC', 'Biopac MP 150 system': 'BIOPAC',
    'Biopac MP150': 'BIOPAC', 'MP150': 'BIOPAC',
    'MP150 BIOPAC': 'BIOPAC', 'MP150 BIOPAC system': 'BIOPAC',
    'MP150 Biopac': 'BIOPAC', 'MP35 Biopac': 'BIOPAC',

    # —— Biosemi ActiveTwo ——————————————————
    'Biosemi ActiveTwo': 'Biosemi ActiveTwo',
    'Biosemi ActiveTwo (GSR channel) \u200b': 'Biosemi ActiveTwo',
    'Biosemi ActiveTwo system\u200b\u200b': 'Biosemi ActiveTwo',
    'Biosemi activeTwo': 'Biosemi ActiveTwo',

    # —— Affectiva Q‑Sensor —————————————
    'Affectiva ': 'Affectiva Q Sensor',
    'Affectiva Q1 sensor': 'Affectiva Q Sensor',
    'Affectiva-QSensors5': 'Affectiva Q Sensor',
    'Four Affectiva Q-sensors': 'Affectiva Q Sensor',
    'Q Sensor by Afectiva': 'Affectiva Q Sensor',

    # —— PowerLab ————————————————————————
    'PowerLab': 'PowerLab',
    'PowerLab (manufactured\nby ADInstruments)': 'PowerLab',

    # —— Shimmer ——————————————————————————
    'Shimmer': 'Shimmer',
    'Shimmer 2R': 'Shimmer',
    'Shimmer 2R GSR module': 'Shimmer',
    'Shimmer 3 GSR+ Unit': 'Shimmer',
    'Shimmer GSR': 'Shimmer',
    'Shimmer GSR+Unit2': 'Shimmer',
    'Shimmer module': 'Shimmer',
    'Shimmer3': 'Shimmer',
    'Shimmer3\nGSR+': 'Shimmer',
    'Shimmer3 EDA+': 'Shimmer',
    'Shimmer3 GSR': 'Shimmer',
    'Shimmer3 GSR+': 'Shimmer',
    'Shimmer3 GSR+ Unit': 'Shimmer',
    'Shimmer3 GSR+ Unit sensor': 'Shimmer',

    # —— Nexus (MindMedia) ——————————————
    'Nexus-10': 'NEXUS',
    'Nexus 4 Biofeedback system3': 'NEXUS',
    'Nexus-32': 'NEXUS',
    'NeXus-32 amplifier': 'NEXUS',

    # —— ProComp Infiniti ——————————————
    'ProComp Infiniti': 'ProComp Infiniti',
    'ProComp Infinity': 'ProComp Infiniti',
    'Procomp Infiniti Encoder': 'ProComp Infiniti',
    'Procomp5 Infiniti': 'ProComp Infiniti',

    # —— BioRadio ————————————————————————
    'BioRadio 150': 'BioRadio 150',
    'BioRadio 150 wireless sensor': 'BioRadio 150',

    # —— Grove GSR ————————————————————————
    'Grove': 'Grove',
    'Grove\n(a standalone LM324 quadruple operational amplifier based on EDA sensor kit)': 'Grove',
    'Grove 1.2': 'Grove',
    'Grove GSR sensor produced by Seeed': 'Grove',

    # —— Homemade & específicos ——————————
    'Custom': 'Homemade (unspecified)',
    '-': 'Homemade (unspecified)',
    'Self-customised skin sensor': 'Homemade (unspecified)',

    'Custom iFidgetCube (Arduino MCU + PPG & EDA sensors)': 'iFidgetCube',
    'Bluno Beetle BLE board with an ATMega328P@16Mhz microprocessor and Bluetooth capability integrated into the board with the TI CC2540 chip.': 'Bluno Beetle BLE',

    'LabVIEW': 'LabVIEW DAQ',

    'Commercial bluetooth GSR sensor': 'Commercial BT GSR',
    'Commercial bluetooth sensor': 'Commercial BT GSR',

    # —— Otros dispositivos individuales ——
    '(BITalino (r)evolution Plugged\nKit BT': 'BITalino (r)evolution Plugged Kit BT',
    'Biosignalplux': 'Biosignalplux',
    'Bodymedia': 'BodyMedia',
    'Empatica E4': 'Empatica',
    'Empatica\u202fE4': 'Empatica',
    'ErgoLAB EDA Wireless Skin Conductance Sensor': 'ErgoLAB Wireless EDA',
    'Ergosensing wristband (ES1)': 'Ergosensing ES1',
    'GSR-2': 'GSR2', 'GSR2': 'GSR2',
    'Gen II integrated wearable device from Analog Devices, Inc': 'Gen II Analog Devices',
    'HKR-11C+': 'HKR-11C+',
    'Microsoft Band 2': 'Microsoft Band 2',
    'Mindfield eSense': 'Mindfield eSense',
    'Other': 'Other',
    'Sociograph': 'Sociograph', 'Sociograph\n': 'Sociograph',
    'Sudologger 3 device': 'Sudologger 3',
    'Thought Technology SA9309M (GSR sensor)': 'Thought Technology',
    'sensors produced by Thought Technology': 'Thought Technology',
    'Varioport': 'Varioport', 'Varioport-B': 'Varioport',
    'e-Health Sensor\nPlatform V2.0': 'e-Health Sensor Platform V2.0',
    'imec': 'imec',
}
#%%

# Replace device specifications with standardized names
eda_devices['eda_device_specification'] = (
    eda_devices['eda_device_specification'].replace(mapping_eda)
)

# Count unique devices
device_counts = eda_devices['eda_device_specification'].value_counts()
print("\nDevice counts:")
print(device_counts)

device_percentages = eda_devices['eda_device_specification'].value_counts(normalize=True).mul(100).round(1)
print("\nDevice percentages:")
print(device_percentages.astype(str) + '%')

# Get the top 3 most used devices
top_devices = device_counts.head(3).index.tolist()
print(f"\nTop 3 most used devices: {', '.join(top_devices)}")

n_dispositivos_eda = eda_devices['eda_device_specification'].nunique()
print(f'Total distinct EDA devices used: {n_dispositivos_eda}')

# Calculate the percentage of studies that did not report the specific device
print(f"\nOn the other hand, {no_device_percentage:.1f}% of the studies did not report the specific device used or clarified whether it was a homemade solution.")

# Print total number of papers with EDA data for reference
print(f"\nTotal number of papers with EDA data: {total_eda_papers}")



# Plot EDA devices
from turtle import width


plt.figure(figsize = (23,16))
sns.set_context('paper')
sns.countplot(y = 'eda_device_specification',
            data = eda_devices,
            order = eda_devices['eda_device_specification'].value_counts().index)
plt.ylabel('Device',
            fontsize = 24,
            fontweight = 'bold')
plt.xlabel('')
plt.yticks(fontsize = 25)
plt.xticks(ticks = range(1,19), fontsize = 23)
plt.show()


# Filter devices with 1 or fewer occurrences
devices_with_others = eda_devices.copy()
device_counts = devices_with_others['eda_device_specification'].value_counts()

low_freq_devices = device_counts[device_counts <= 1].index
mapping_others = {device: 'Others' for device in low_freq_devices}

devices_with_others['eda_device_specification'] = (
    devices_with_others['eda_device_specification'].replace(mapping_others)
)

# Plot EDA devices with "Others" category
sns.set_context('poster')

name_changes = {
    "Affectiva Q Sensor": "Affectiva Q-Sensor",
    "NEXUS": "NeXuS",
}

# Update name changes in DataFrame
devices_with_others['eda_device_specification'].replace(name_changes, inplace=True)

# Update the plot_order list
counts = devices_with_others['eda_device_specification'].value_counts()
plot_order = counts[counts.index != 'Others'].index.tolist() + ['Others']

#Plot
plt.figure(figsize = (20,10))
sns.countplot(y = 'eda_device_specification',
              data = devices_with_others,
              order = plot_order,
              palette = "tab10")
plt.ylabel('EDA Device', fontsize = 23, fontweight = 'bold')
plt.xlabel('Frequency of papers', fontsize = 23)
plt.yticks(fontsize = 24)
plt.xticks(ticks = range(0, counts.max(),5), fontsize = 22)

sns.despine()

plt.show()

#%%
# 2. EDA Location Analysis
print("\nEDA Location Analysis:")
df_eda['location_hemibody'] = df_eda['location_hemibody'].replace({'non-dominant': 'not dominant'})

hemibody = df_eda.groupby(['paper_id', 'location_hemibody']).nth(0)
hemibody.reset_index(inplace=True)

# Count papers with hemibody specified
hemibody_counts = hemibody['location_hemibody'].value_counts()
print("\nHemibody counts:")
print(hemibody_counts)

hemibody_percentages = hemibody['location_hemibody'].value_counts(normalize=True).mul(100).round(1)
print("\nHemibody percentages:")
print(hemibody_percentages.astype(str) + '%')

# Calculate percentage of papers reporting hemibody
hemibody_reported = hemibody[hemibody['location_hemibody'] != "-"]
hemibody_reported_percentage = (len(hemibody_reported) / len(hemibody)) * 100
print(f"\nPercentage of studies reporting hemibody placement: {hemibody_reported_percentage:.1f}%")

# Frequency of hemibody placements when reported
hemibody_reported_counts = hemibody_reported['location_hemibody'].value_counts()
print("\nHemibody placement when reported:")
print(hemibody_reported_counts)

hemibody_reported_percentages = hemibody_reported['location_hemibody'].value_counts(normalize=True).mul(100).round(1)
print("\nHemibody placement percentages when reported:")
print(hemibody_reported_percentages.astype(str) + '%')

# Calculate percentage of left side placement when reported
left_side_percentage = hemibody_reported_percentages.get('left', 0)
print(f"Percentage of left side placement when reported: {left_side_percentage}%")

# Analyze sensor locations
print("\nSensor Location Analysis:")
sensors = df_eda.groupby(['paper_id','is_hands','wrist', 'chest',
                    'finger_thumb', 'finger_index', 'finger_middle', 'finger_ring', 'finger_little',
                     'phalange_proximal', 'phalange_medial','phalange_distal',
                     ]).nth(0)
sensors.reset_index(inplace=True)

sensors_location = df_eda.groupby(['paper_id','is_hands','wrist', 'chest']).nth(0)
sensors_location.reset_index(inplace=True)

# Percentage of papers with no sensor location specified
no_location_percentage = (len(sensors[(sensors['is_hands'] == '-') &
            (sensors['wrist'] == '-') &
            (sensors['chest'] == '-')]) / len(sensors)) * 100
print(f"\nPercentage of papers with no body part specified: {no_location_percentage:.1f}%")

# Calculate percentage of papers reporting body part
body_part_reported_percentage = 100 - no_location_percentage
print(f"Percentage of studies reporting body part placement: {body_part_reported_percentage:.1f}%")

# Frequency of sensor locations
general_place = fn.multi_reversing(sensors, 'model_id',sensors[['is_hands','wrist', 'chest']])
general_place_counts = general_place['variable'].value_counts()
print("\nBody part placement counts:")
print(general_place_counts)

general_place_percentages = general_place['variable'].value_counts(normalize=True).mul(100).round(1)
print("\nBody part placement percentages:")
print(general_place_percentages.astype(str) + '%')

# Calculate percentage of hand placement
hand_percentage = general_place_percentages.get('is_hands', 0)
print(f"Percentage of hand placement when body part reported: {hand_percentage}%")

# Frequency of finger sensors
finger_sensor = fn.multi_reversing(sensors, 'model_id',sensors[['finger_thumb', 'finger_index', 'finger_middle', 'finger_ring', 'finger_little']])
finger_sensor_counts = finger_sensor['variable'].value_counts()
print("\nFinger placement counts:")
print(finger_sensor_counts)

finger_sensor_percentages = finger_sensor['variable'].value_counts(normalize=True).mul(100).round(1)
print("\nFinger placement percentages:")
print(finger_sensor_percentages.astype(str) + '%')

# Calculate percentage of middle and index finger placement
middle_finger_percentage = finger_sensor_percentages.get('finger_middle', 0)
index_finger_percentage = finger_sensor_percentages.get('finger_index', 0)
print(f"Percentage of middle finger placement: {middle_finger_percentage}%")
print(f"Percentage of index finger placement: {index_finger_percentage}%")

# Frequency of phalange sensors
location_phalanges = fn.multi_reversing(sensors, 'model_id',sensors[['phalange_proximal', 'phalange_medial','phalange_distal']])
location_phalanges_counts = location_phalanges['variable'].value_counts()
print("\nPhalange placement counts:")
print(location_phalanges_counts)

location_phalanges_percentages = location_phalanges['variable'].value_counts(normalize=True).mul(100).round(1)
print("\nPhalange placement percentages:")
print(location_phalanges_percentages.astype(str) + '%')

# Calculate total number of papers with EDA data
total_eda_papers = len(df_eda['paper_id'].unique())
print(f"\nTotal number of papers with EDA data: {total_eda_papers}")

#%%
# Rename finger sensors for plotting
finger_sensor['variable'] = finger_sensor['variable'].str.replace('finger_middle','Middle')
finger_sensor['variable'] = finger_sensor['variable'].str.replace('finger_index','Index')
finger_sensor['variable'] = finger_sensor['variable'].str.replace('finger_ring','Ring')
finger_sensor['variable'] = finger_sensor['variable'].str.replace('finger_thumb','Thumb')
finger_sensor['variable'] = finger_sensor['variable'].str.replace('finger_little','Little')

# Set the context for even bigger fonts
sns.set_context("talk")

# Create figure with custom GridSpec
fig = plt.figure(figsize=(16, 8), dpi=300)
gs = gridspec.GridSpec(1, 3)


new_labels_ax1 = ['Left', 'Right', 'Not dominant', 'Dominant']
new_labels_ax2 = ['Hand', 'Wrist', 'Chest']

# SUBPLOT A
ax1 = plt.subplot(gs[0, 0])
plot_order = ['left', 'right', 'not dominant', 'dominant']
sns.countplot(x='location_hemibody', data=hemibody, order=plot_order, ax=ax1, palette="tab10")
ax1.set_title('Hemibody Location', fontweight='bold')
ax1.set_ylabel("Frequency")
ax1.set_xticklabels(new_labels_ax1, rotation=45)
ax1.text(-0.1, 1.1, 'A', transform=ax1.transAxes, fontsize=36, fontweight='bold')
ax1.set_xlabel('')
sns.despine(ax=ax1)

# SUBPLOT B
ax2 = plt.subplot(gs[0, 1])
sns.countplot(x='variable', data=general_place, ax=ax2, palette="tab10")
ax2.set_title('Location of Electrodes in the Body', fontweight='bold')
ax2.set_ylabel("Frequency")
ax2.set_xticklabels(new_labels_ax2, rotation=45)
ax2.text(-0.1, 1.1, 'B', transform=ax2.transAxes, fontsize=36, fontweight='bold')
ax2.set_xlabel('')
sns.despine(ax=ax2)

# SUBPLOT C
ax3 = plt.subplot(gs[0, 2])
sns.countplot(x='variable', data=finger_sensor, ax=ax3, palette="tab10")
ax3.set_title('Location of Electrodes in the Hand', fontweight='bold')
ax3.set_ylabel("Frequency")
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
ax3.text(-0.1, 1.1, 'C', transform=ax3.transAxes, fontsize=36, fontweight='bold')
ax3.set_xlabel('')
sns.despine(ax=ax3)

# Tight layout and show
plt.tight_layout()
plt.show()

#%%
# Statistical learning models analysis
print("\nStatistical Learning Models Analysis:")

# Load EDA data
df_statistical_learning_models  = pd.read_excel(r'.\data\cleaned\Normalized Table - Statistical learning model - Performances.xlsx')
df_statistical_learning_models.fillna('-', inplace=True)

# 1. Affective model analysis
df_statistical_learning_models=df_statistical_learning_models[df_statistical_learning_models['affective_model'].isin(['categorical', 'dimensional'])]
df_statistical_learning_models_0 = df_statistical_learning_models.groupby(['paper_id','affective_model']).nth(0)
df_statistical_learning_models_0.reset_index(inplace=True)
df_statistical_learning_models_0["year"] = df_statistical_learning_models_0["year"].astype(int)
df_statistical_learning_models_0

# Calculate overall percentage of dimensional vs categorical models
affective_model_percentages = df_statistical_learning_models_0['affective_model'].value_counts(normalize=True).mul(100).round(1)
print("\nAffective model distribution:")
print(affective_model_percentages)
print(f"Dimensional models account for {affective_model_percentages['dimensional']}% of all models used")

# Plotting the number of papers per year and affective model
category_order = [2010, 2011, 2012, 2013, 2014, 2015, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
g= sns.countplot(x='year', 
    data= df_statistical_learning_models_0, 
    hue='affective_model', 
    order=category_order)
g.set(xlabel = 'Año', ylabel = 'Cantidad de papers')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 2. Models analysis
def label_model (row):
   if row['is_classifier'] == "x" :
      return 'classifier'
   if row['is_regressor'] == "x" :
      return 'regressor'
   return 'Other'

df_statistical_learning_models['model']  = df_statistical_learning_models.apply(lambda row: label_model(row), axis=1)
model_counts = df_statistical_learning_models['model'].value_counts()
model_percentages = df_statistical_learning_models['model'].value_counts(normalize=True).mul(100).round(1)

print("\nModel type distribution:")
print(model_counts)
print(model_percentages)
print(f"Classification models account for {model_percentages['classifier']}% of all models")
print(f"Regression models account for {model_percentages['regressor']}% of all models")

#%%
df_models = df_statistical_learning_models[["apa_citation",'model', "year", "model_id"]]

# Plotting the number of papers per year and model
category_order = [2010, 2011, 2012, 2013, 2014, 2015, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
g= sns.countplot(x='year', 
    data= df_models, 
    hue='model', 
    order=category_order)
g.set(xlabel = 'Año', ylabel = 'Cantidad de modelos')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Others in the regression algorithms
# Plotting frequency of specific regression algorithms with 'class_other' column
df_algoritmos_regre = fn.multi_reversing_with_other(
    df_statistical_learning_models, 
    'model_id', 
    df_statistical_learning_models.columns[43:60],
    df_statistical_learning_models.columns[59]     # columna "class_other"
)

df_algoritmos_regre['model'] = df_algoritmos_regre['model'].str.replace('regre_', '', regex=False)
df_algoritmos_regre = df_algoritmos_regre[~df_algoritmos_regre['model'].str.strip().eq('-')]

algoritmos_de_regresion = df_algoritmos_regre['model'].unique()

# Others in the classification algorithms
# Plotting frequency of specific classification algorithms with 'class_other' column
df_algoritmos_class = fn.multi_reversing_with_other(
    df_statistical_learning_models, 
    'model_id', 
    df_statistical_learning_models.columns[8:41],
    df_statistical_learning_models.columns[41]     # columna "class_other"
)

df_algoritmos_class['model'] = df_algoritmos_class['model'].str.replace('class_', '', regex=False)
df_algoritmos_class = df_algoritmos_class[~df_algoritmos_class['model'].str.strip().eq('-')]

algoritmos_de_clasificacion = df_algoritmos_class['model'].unique()

# NEW: Plotting the frequency of specific class algorithms used in the studies
class_mapping = {
    'support_vector_machine': 'Support Vector Machine',
    'tree_based_models': 'Tree-based Models',
    'adaboost_dt': 'Boosting-based Models',
    'gradient_boostingclass': 'Boosting-based Models',
    
    'k_nearest_neighbor': 'K-Nearest Neighbors',

    'logistic_regression': 'Linear Models',
    'linear_discriminant_analysis': 'Linear Models',
    'quadratic_discrimant_classifier': 'Linear Models',

    'naive_bayes': 'Naive Bayes',
    'GNB (no se si es Gaussian Naive Bayes o algo de boosted': 'Naive Bayes',

    'regre_fully_connected_neuronal_network_or_multi_layer_perceptron': 'Fully-connected Neural Network',
    'Multi layer perceptron': 'Fully-connected Neural Network',
    'ann': 'Fully-connected Neural Network',
    'DNN': 'Fully-connected Neural Network',
    'Double Deep Q‑learning (DDQ)': 'Fully-connected Neural Network',
    'backpropagation': 'Fully-connected Neural Network',
    'LSQC': 'Fully-connected Neural Network',

    'convolutional_neuronal_network': 'Convolutional Neural Network',
    'cCNN': 'Convolutional Neural Network',
    'AlexNet': 'Convolutional Neural Network',
    'VGG16': 'Convolutional Neural Network',
    'VGG17': 'Convolutional Neural Network',

    'lstm': 'Recurrent Neural Network',
    'recurrent_neuronal_network': 'Recurrent Neural Network',
    'gated_recurrent_units': 'Recurrent Neural Network',

    'probabilistic_neural_network': 'Other / Unclear',
    'radial_basis_function': 'Other / Unclear',
    'Transformer': 'Other / Unclear',
    'PhGP': 'Other / Unclear',
    'spiking_deep_belief_network': 'Other / Unclear',
    'SEL (Stacking Ensemble Learning)': 'Other / Unclear',
    'LSLC': 'Other / Unclear',
    'cellular_neural_networks': 'Other / Unclear',
    'quantum_neural_network': 'Other / Unclear',
    'hmm': 'Other / Unclear',
    '1r_rule': 'Other / Unclear',
    'y el paper no lo aclara)': 'Other / Unclear'
}

# NEW: Plotting the frequency of specific regre algorithms used in the studies
regressor_mapping = {
    'decision_tree': 'Tree-based Models',
    'Random Forest': 'Tree-based Models',
    'boosted_regression_trees': 'Boosting-based Models',

    'linear_regression': 'Linear Models',
    'ridge_regression': 'Linear Models',
    'logistic_regression': 'Linear Models',
    'polynomial_regression': 'Linear Models',

    'knn': 'K-Nearest Neighbors',

    'support_vector_regression': 'Support Vector Regression',

    'fully_connected_neuronal_network_or_multi_layer_perceptron': 'Fully-connected Neural Network',
    'multilayer_regression': 'Fully-connected Neural Network',

    'convolutional_neuronal_network': 'Convolutional Neural Network',
    'VGG16': 'Convolutional Neural Network',
    'ResNet50': 'Convolutional Neural Network',

    'recurrent_neuronal_network': 'Recurrent Neural Network',
    'lstm': 'Recurrent Neural Network'
}

# Apply mappings to create model_grouped columns
df_algoritmos_class['model_grouped'] = df_algoritmos_class['model'].map(class_mapping)
df_algoritmos_regre['model_grouped'] = df_algoritmos_regre['model'].map(regressor_mapping)

#%%
# Display regression algorithms count
sns.countplot(y='model_grouped', data=df_algoritmos_regre, order=df_algoritmos_regre['model_grouped'].value_counts().index)
plt.show()

#%%
# Set Seaborn style and context
sns.set_context("talk")

# Obtener los conteos y ordenarlos, dejando "Other / Unclear" al final
class_counts = df_algoritmos_class['model_grouped'].value_counts()
regre_counts = df_algoritmos_regre['model_grouped'].value_counts()

class_order = class_counts[class_counts.index != 'Other / Unclear'].sort_values(ascending=False).index.tolist()
class_order.append('Other / Unclear')

regre_order = regre_counts[regre_counts.index != 'Other / Unclear'].sort_values(ascending=False).index.tolist()
regre_order.append('Other / Unclear')

# Crear figura con subplots independientes (sin compartir eje Y)
fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=300)

# Subplot A - Clasificadores
sns.countplot(y='model_grouped', data=df_algoritmos_class, order=class_order, ax=axes[0], palette="tab10")
axes[0].set_title("A - Classification Algorithms", loc='left', fontsize=28, fontweight='bold')
axes[0].set_xlabel("Count")
axes[0].set_ylabel("")

# Subplot B - Regresores
sns.countplot(y='model_grouped', data=df_algoritmos_regre, order=regre_order, ax=axes[1], palette="tab10")
axes[1].set_title("B - Regression Algorithms", loc='left', fontsize=28, fontweight='bold')
axes[1].set_xlabel("Count")
axes[1].set_ylabel("")

# Ajuste visual final
plt.tight_layout()
sns.despine()
plt.show()

#%%
# Plotting frequency of all models
df_all_models = df_statistical_learning_models.iloc[:,1:57]
df_all_models.drop(df_all_models.columns[[1,2,3,4,5,6,39,40,41]], axis=1, inplace=True)

df_all_models = fn.multi_reversing(df_all_models, 'model_id', df_all_models.iloc[:,1:])
df_all_models['variable'] = df_all_models['variable'].str.replace('class_','')
df_all_models['variable'] = df_all_models['variable'].str.replace('regre_','')

sns.countplot(x='variable', data=df_all_models, order = getattr(df_all_models, 'variable').value_counts().index, palette="Paired")
plt.xticks(rotation=90)
plt.show()

#%%
# Porcentaje de modelos de regresión
df_algoritmos_regre['model'].value_counts(normalize=True).mul(100).round(2)

# Porcentaje de modelos de clasificación
df_algoritmos_class['model'].value_counts(normalize=True).mul(100).round(2)

df_models = df_statistical_learning_models[["paper_id","apa_citation",'model', "year", "model_id"]]

df_models = df_models.groupby(
        ["paper_id",'model']
        ).nth(0)
df_models.reset_index(inplace=True)


models = df_statistical_learning_models[["paper_id", "year", "affective_model", "model_id"]]

models = models.groupby(
        ["paper_id",'affective_model']
        ).nth(0)
models.reset_index(inplace=True)

models["year"] = models["year"].astype(int)

models_crosstab = pd.crosstab(index=models['year'], columns=models['affective_model'],normalize='index')

# Analyze the chronological evolution of affective models
early_years = models[models['year'] <= 2015]
early_years_counts = early_years['affective_model'].value_counts(normalize=True).mul(100).round(1)
print("\nAffective model distribution in early years (up to 2015):")
print(early_years_counts)

later_years = models[models['year'] > 2015]
later_years_counts = later_years['affective_model'].value_counts(normalize=True).mul(100).round(1)
print("\nAffective model distribution in later years (after 2015):")
print(later_years_counts)

n_models = df_models.groupby(
        ["paper_id",'model']
        ).nth(0)
n_models.reset_index(inplace=True)

n_models["year"] = n_models["year"].astype(int)

n_models_crosstab = pd.crosstab(index=n_models['year'], columns=n_models['model'],normalize='index')


#%%
# Plotting Figure 8: Chronological evolution of emotion model types and algorithm usage in emotion recognition research with EDA over a decade
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np

# Set the context for even bigger fonts
sns.set_context("talk")

# Create figure with custom GridSpec
fig = plt.figure(figsize=(10, 10), dpi=300)
gs = gridspec.GridSpec(2, 1)

# SUBPLOT A
ax1 = plt.subplot(gs[0, 0])
(models_crosstab * 100).plot(kind='bar',  
                            stacked=True,
                            rot=0,
                            ax=ax1,
                            color=['#1f77b4', '#ff7f0e'])
ax1.set_ylim([0, 100])  # Hasta 100%
ax1.set_xticklabels(ax1.get_xticklabels())
plt.xticks(rotation=45)
ax1.set_yticklabels(ax1.get_yticklabels())
ax1.set_xlabel("Year")
ax1.set_ylabel("Percentage of articles (%)")
ax1.legend(title='Affective model', loc='upper right', bbox_to_anchor=(1, 1.4),
          frameon=False, fancybox=True, ncol=2, fontsize=18)
ax1.text(0.0, 1.1, 'A', transform=ax1.transAxes, fontsize=36, fontweight='bold')
# Ensure y-axis goes from 0 to 100
ax1.set_yticks(np.arange(0, 101, 20))
sns.despine(ax=ax1)

# SUBPLOT B
ax2 = plt.subplot(gs[1, 0])
(n_models_crosstab * 100).plot(kind='bar',  
                              stacked=True,
                              rot=0,
                              ax=ax2,
                              color=['#1f77b4', '#ff7f0e'])
ax2.set_ylim([0, 100])  # Hasta 100%
ax2.set_xticklabels(ax2.get_xticklabels())
plt.xticks(rotation=45)
ax2.set_yticklabels(ax2.get_yticklabels())
ax2.set_xlabel("Year")
ax2.set_ylabel("Percentage of articles (%)")
ax2.legend(title='Type of algorithm', loc='upper right', bbox_to_anchor=(1, 1.4),
          frameon=False, fancybox=True, ncol=2, fontsize=18)
ax2.text(0.0, 1.1, 'B', transform=ax2.transAxes, fontsize=36, fontweight='bold')
# Ensure y-axis goes from 0 to 100
ax2.set_yticks(np.arange(0, 101, 20))
sns.despine(ax=ax2)

plt.tight_layout()
plt.show()

#%%
# Plotting Figure 9: Absolute counts of emotion model types and algorithm usage by year
# Create raw counts crosstabs (without normalization)
models_crosstab_counts = pd.crosstab(index=models['year'], columns=models['affective_model'])
n_models_crosstab_counts = pd.crosstab(index=n_models['year'], columns=n_models['model'])

# Set the context for even bigger fonts
sns.set_context("talk")

# Create figure with custom GridSpec
fig = plt.figure(figsize=(10, 10), dpi=300)
gs = gridspec.GridSpec(2, 1)

# SUBPLOT A - Absolute counts of affective models
ax1 = plt.subplot(gs[0, 0])
models_crosstab_counts.plot(kind='bar',  
                           stacked=True,
                           rot=0,
                           ax=ax1,
                           color=['#1f77b4', '#ff7f0e'])
ax1.set_xticklabels(ax1.get_xticklabels())
plt.xticks(rotation=45)
ax1.set_yticklabels(ax1.get_yticklabels())
ax1.set_xlabel("Year")
ax1.set_ylabel("Number of articles")
ax1.legend(title='Affective model', loc='upper right', bbox_to_anchor=(1, 1.4),
          frameon=False, fancybox=True, ncol=2, fontsize=18)
ax1.text(0.0, 1.1, 'A', transform=ax1.transAxes, fontsize=36, fontweight='bold')
sns.despine(ax=ax1)

# SUBPLOT B - Absolute counts of algorithm types
ax2 = plt.subplot(gs[1, 0])
n_models_crosstab_counts.plot(kind='bar',  
                             stacked=True,
                             rot=0,
                             ax=ax2,
                             color=['#1f77b4', '#ff7f0e'])
ax2.set_xticklabels(ax2.get_xticklabels())
plt.xticks(rotation=45)
ax2.set_yticklabels(ax2.get_yticklabels())
ax2.set_xlabel("Year")
ax2.set_ylabel("Number of articles")
ax2.legend(title='Type of algorithm', loc='upper right', bbox_to_anchor=(1, 1.4),
          frameon=False, fancybox=True, ncol=2, fontsize=18)
ax2.text(0.0, 1.1, 'B', transform=ax2.transAxes, fontsize=36, fontweight='bold')
sns.despine(ax=ax2)

plt.tight_layout()
plt.show()

# Calcular y mostrar el total de artículos en cada plot
total_articles_plot_A = models_crosstab_counts.sum().sum()
print(f"\nTotal de artículos en el Plot A (Affective models): {total_articles_plot_A}")

total_articles_plot_B = n_models_crosstab_counts.sum().sum()
print(f"Total de artículos en el Plot B (Algorithm types): {total_articles_plot_B}")

# Análisis de la diferencia entre los totales
print("\n--- Análisis de la diferencia entre totales ---")

# Verificar cuántos papers únicos hay en cada dataframe
unique_papers_models = models['paper_id'].nunique()
unique_papers_n_models = n_models['paper_id'].nunique()
print(f"Papers únicos en 'models' (Affective models): {unique_papers_models}")
print(f"Papers únicos en 'n_models' (Algorithm types): {unique_papers_n_models}")

# Identificar qué papers están en uno pero no en el otro
papers_in_models = set(models['paper_id'].unique())
papers_in_n_models = set(n_models['paper_id'].unique())

papers_only_in_models = papers_in_models - papers_in_n_models
papers_only_in_n_models = papers_in_n_models - papers_in_models

print(f"\nPapers que solo aparecen en 'models' (no en 'n_models'): {len(papers_only_in_models)}")
if len(papers_only_in_models) > 0:
    print("Ejemplos de paper_id:", list(papers_only_in_models)[:5])

print(f"\nPapers que solo aparecen en 'n_models' (no en 'models'): {len(papers_only_in_n_models)}")
if len(papers_only_in_n_models) > 0:
    print("Ejemplos de paper_id:", list(papers_only_in_n_models)[:5])

# Verificar si hay duplicados en los conteos
print("\nVerificando duplicados en los conteos:")
duplicates_in_models = models[models.duplicated(subset=['paper_id'], keep=False)]
print(f"Duplicados en 'models' por paper_id: {len(duplicates_in_models)}")

duplicates_in_n_models = n_models[n_models.duplicated(subset=['paper_id'], keep=False)]
print(f"Duplicados en 'n_models' por paper_id: {len(duplicates_in_n_models)}")

# Verificar valores nulos o faltantes
print("\nVerificando valores nulos o faltantes:")
print(f"Valores nulos en 'models' para affective_model: {models['affective_model'].isna().sum()}")
print(f"Valores nulos en 'n_models' para model: {n_models['model'].isna().sum()}")

#%%
# Percentage of model tpyes used in the studies
n_models["model"].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'

# Percentage of affective models used in the studies
models["affective_model"].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'

# 3. Interpretation analysis

model_interpretion = df_statistical_learning_models[df_statistical_learning_models['model_interpretation'] !='-']
model_interpretation = model_interpretion.groupby(['paper_id', 'model_interpretation']).nth(0)

model_interpretation.reset_index(inplace= True)
model_interpretation.drop_duplicates(subset = ['paper_id'],inplace=True)
print(f' En {len(model_interpretation)} papers se realizan interpretaciones emocionales de los modelos')

model_interpretation_list = model_interpretation["paper_id"].to_list()
model_interpretation_list = [int(a) for a in model_interpretation_list]

df_metadata_filtered = df_metadata[df_metadata['paper_id'].isin(model_interpretation_list)]
df_metadata_filtered.drop_duplicates("paper_id", inplace= True)
df_metadata_filtered[["paper_id", "apa_citation", "year", "source_title"]]


# Contar cuántas veces aparece cada journal
journal_counts = df_metadata_filtered['source_title'].value_counts()

# Filtrar para quedarnos solo con los journals con más de una interpretación
journals_with_multiple = journal_counts[journal_counts > 1].index
df_metadata_filtered = df_metadata_filtered[df_metadata_filtered['source_title'].isin(journals_with_multiple)]

# Plot
titulos = [' ', 'Journal', 'Cantidad']
var_x = "source_title"
df = df_metadata_filtered

g = sns.countplot(y=var_x, data=df, order=getattr(df, var_x).value_counts().index)
g.set(title=titulos[0], xlabel=titulos[1], ylabel=titulos[2])
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# %%
import statsmodels.api as sm
from scipy.stats import permutation_test, ttest_1samp, bootstrap
from statsmodels.regression.linear_model import OLS
df = pd.read_excel('./data/processed/final_melted_df_excel_paired_ALL.xlsx')

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
    129: "Ganapathy et al., 2021",
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

df['diff_acc'] = df['accuracy_arousal'] - df['accuracy_valence']

df['mean_acc'] = np.mean([df['accuracy_arousal'], df['accuracy_valence']], axis=0)

from scipy.stats import linregress
import matplotlib.pyplot as plt
import numpy as np

# Sample data for demonstration (replace with your actual DataFrame)
# import pandas as pd
# df = pd.DataFrame({
#     'paper_id': [...],
#     'N': [...],
#     'accuracy_arousal': [...],
#     'accuracy_valence': [...]
# })

# Calculate the difference in accuracy
df['diff_acc'] = df['accuracy_arousal'] - df['accuracy_valence']

# Unique paper IDs
unique_paper_ids = df['paper_id'].unique()

# Create a color map for paper IDs
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_paper_ids)))

plt.figure(figsize=(22, 9))  # Significantly increased width to accommodate the legend

# Calculate the mean difference in accuracy
mean_diff_acc = np.mean(df['diff_acc'])

# Add a horizontal line to represent the mean difference in accuracy
plt.axhline(mean_diff_acc, color='black', linestyle='-', linewidth=2)

# Add text to indicate the mean value
plt.text(max(df['accuracy_valence']) * 1, mean_diff_acc, f'Mean difference = {mean_diff_acc:.2f}', verticalalignment='bottom', horizontalalignment='right')

# Calculate the correlation coefficient and p-value
slope, intercept, r_value, p_value, std_err = linregress(df['accuracy_valence'], df['diff_acc'])

for i, paper_id in enumerate(unique_paper_ids):
    subset = df[df['paper_id'] == paper_id]
    
    # Check if N == 32 for the subset
    is_N_32 = subset['N'] == 32
    
    # Get the citation for the paper ID
    citation = paper_id_to_citation.get(paper_id, f"Paper ID {paper_id}")
    
    # Plot points with N == 32 using triangles
    if any(is_N_32):
        plt.scatter(subset['accuracy_valence'][is_N_32], subset['diff_acc'][is_N_32], 
                    color=colors[i], marker='^', label=f"{citation}")
    
    # Plot other points using circles
    if any(~is_N_32):
        plt.scatter(subset['accuracy_valence'][~is_N_32], subset['diff_acc'][~is_N_32], 
                    color=colors[i], label=citation)

# Adding horizontal lines
plt.axhline(np.mean(df['diff_acc']) + 1.96 * np.std(df['diff_acc']), linestyle='--')
plt.axhline(np.mean(df['diff_acc']) - 1.96 * np.std(df['diff_acc']), linestyle='--')

# Adding labels and title
plt.xlabel('Accuracy of Valence Models')
plt.ylabel('Difference in Accuracy (Arousal - Valence)')
plt.title('Valence Accuracy vs Difference in Arousal and Valence Accuracy')

# Adjust the layout to maintain the plot size
plt.tight_layout(rect=[0, 0, 0.45, 1])  # Adjusted to leave about 1/3 of figure width for legends

# Get handles and labels for the references
handles, labels = plt.gca().get_legend_handles_labels()

# Single legend for all references with increased font size for better readability
legend1 = plt.legend(handles, labels, title='References', 
                    bbox_to_anchor=(1.01, 1), loc='upper left', ncol=1, 
                    fontsize=9, title_fontsize=10)
plt.gca().add_artist(legend1)
plt.setp(legend1.get_title(), weight='bold')

# Legend for marker types
legend3 = plt.legend([plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markersize=10),
                      plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10)],
                     ['DEAP Database                       ', 'Other Data'], 
                     title='Database', 
                     bbox_to_anchor=(1.01, 0.1), loc='upper left')
plt.setp(legend3.get_title(), weight='bold')

# Remove spines
sns.despine()

# Show the plot
plt.show()
#%%

x = df['diff_acc']

def my_stat(x):
    return ttest_1samp(x, popmean=0).statistic

#%%
permutation_test((x.values,), my_stat, permutation_type='samples')

#%%
bootstrap((x.values,), my_stat)
#%%
X = df[['N', 'year', 'mean_acc']]
X = sm.add_constant(X)

y = df['diff_acc']

model = OLS(y, X)
res = model.fit()
res.summary()

#%%%
rlm_mod = sm.RLM(y, X, M=sm.robust.norms.HuberT())
rlm_res = rlm_mod.fit()
rlm_res.summary()

# %%
# SEGUIR DESDE ACA ->
# Con los datos del meta analisis hacer un boxplot de la diferencia de accuracy entre arousal y valence agrupado por modelo.
# Para eso, voy a tener que hacer un join para entender que modelo es el que se esta usando en cada estudio
# Eso me daria e panel C del plot que inicio Jero
# Luego de eso, tendria que integrar el plot pasado (panel A y B, que hizo Jero) con este panel C.

#%% Tabla 2

modelos_metaanalisis = pd.read_excel(r'.\data\processed\final_melted_df_excel_paired_ALL.xlsx')
tabla_2 = pd.DataFrame()

list_databases = ['MAHNOB', 'MAHNOB', 'MAHNOB', 'DEAP', 'DEAP', 'DEAP', 'DEAP', 'DEAP', 'DEAP',
 'DEAP', 'DEAP', 'DEAP', 'DEAP', 'DEAP', 'DEAP', 'DEAP', 'DEAP', 'DEAP', 'DEAP',
 'DEAP', 'DEAP', 'AMIGOS', 'AMIGOS', 'DEAP', 'DEAP', 'DEAP', 'DEAP', 'DEAP',
 'DEAP', 'DEAP', 'DEAP', 'DEAP', 'DEAP', 'DEAP', 'DEAP', 'DEAP', 'DEAP', 'DEAP',
 'DEAP', 'DEAP', 'DEAP', 'DEAP', 'DEAP', 'DEAP', 'DEAP', 'DEAP', 'DEAP', 'DEAP',
 'DEAP', 'AMIGOS', 'AMIGOS', 'AMIGOS', 'AMIGOS', 'AMIGOS', 'AMIGOS', 'AMIGOS',
 'AMIGOS', 'AMIGOS', 'AMIGOS', 'AMIGOS', 'DEAP', 'PMEmo', 'PMEmo', 'PMEmo',
 'PMEmo', 'PMEmo', 'PMEmo', 'DEAP', 'DEAP', 'DEAP', 'DEAP', 'DEAP', 'DEAP',
 'DEAP', 'DEAP', 'AMIGOS', 'AMIGOS', 'AMIGOS', 'AMIGOS', 'K-EmoCon', 'K-EmoCon',
 'K-EmoCon', 'K-EmoCon', 'ASCERTAIN', 'ASCERTAIN', 'ASCERTAIN', 'DEAP',
 'AMIGOS', 'MAHNOB', 'DEAP and AMIGOS', 'DEAP, AMIGOS AND MAHNOB',
 'DEAP, AMIGOS (training) and MAHNOB (test)', 'DEAP (training) and MANHOB (test)',
 'Private', 'DEAP', 'DEAP', 'DEAP', 'AMIGOS', 'AMIGOS', 'Private', 'Private',
 'Private', 'Private', 'BM-SWU', 'BM-SWU', 'BM-SWU', 'BM-SWU', 'BM-SWU',
 'PAFEW', 'PAFEW', 'PAFEW', 'PAFEW', 'PAFEW', 'DEAP', 'DEAP', 'DEAP', 'DEAP',
 'DEAP', 'AMIGOS', 'AMIGOS', 'AMIGOS', 'AMIGOS', 'AMIGOS', 'VREED', 'VREED',
 'VREED', 'VREED', 'VREED', 'VREED', 'Private', 'DEAP', 'DEAP', 'DEAP', 'DEAP',
 'AMIGOS', 'ASCERTAIN', 'ASCERTAIN', 'ASCERTAIN', 'ASCERTAIN', 'ASCERTAIN',
 'VREED', 'DEAP', 'DEAP', 'DEAP', 'MAHNOB', 'CASE', 'CASE', 'CASE', 'CASE',
 'DEAP', 'MAHNOB', 'AMIGOS', 'AMIGOS', 'DEAP', 'DEAP']

tabla_2["Citation"] = modelos_metaanalisis["apa_citation"]
tabla_2["Year"] = modelos_metaanalisis["year"]
tabla_2["Modelos Machine Learning"] = modelos_metaanalisis["ML-model"]
tabla_2["Database"] = [list_databases]
tabla_2["Arousal Accuracy (%)"] = modelos_metaanalisis["accuracy_arousal"]
tabla_2["Valence Accuracy (%)"] = modelos_metaanalisis["accuracy_valence"]
tabla_2["Mean Accuracy (%)"] =((tabla_2["Arousal Accuracy (%)"]+tabla_2["Valence Accuracy (%)"])/2).round(2)

tabla_2.to_excel("data/processed/TABLE_2_incomplete.xlsx")