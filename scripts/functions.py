import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def multi_reversing(df,col_id, col_values):
    df_raw = pd.melt(df, id_vars = col_id, value_vars = col_values)
    return df_raw[df_raw.value == 'x']

def multi_reversing_n(df,col_id, col_values):
    df_raw = pd.melt(df, id_vars = col_id, value_vars = col_values)
    return df_raw[df_raw.value != 0]

def bar_plot(col, data, titulos):
    var_x = col
    df = data
    g = sns.countplot(x=var_x, data=df, order = getattr(df, var_x).value_counts().index, palette="Paired")
    g.set(title = titulos[0], xlabel = titulos[1], ylabel = titulos[2])
    plt.xticks(rotation=90)
    
    plt.tight_layout()
    plt.show()

def relaciones(weight):
    df_weights_dim = pd.DataFrame(weight)
    wserie = df_weights_dim.value_counts()
    wserie = wserie.to_frame()
    wserie.index.names = ['index']
    wserie.rename(columns = {0:'relations'}, inplace = True)
    wserie = wserie.reset_index()
    
    col1 = wserie.columns[0]
    col2 = wserie.columns[1]

    # iterar sobre las filas del dataframe
    for i, row in wserie.iterrows():
        print(f"Interacciones con {row[col1]} conexiones se observaron {row[col2]/2} veces.")
        

def multi_reversing_with_other(df, col_id, col_values, col_other):
    # 1. Derretir columnas binarias tipo 'x'
    df_raw = pd.melt(df, id_vars=col_id, value_vars=col_values)
    df_raw = df_raw[df_raw['value'] == 'x']
    df_raw = df_raw[[col_id, 'variable']].rename(columns={'variable': 'model'})

    # 2. Expandir 'regre_other' en filas separadas
    df_other = df[[col_id, col_other]].dropna()
    df_other['model'] = df_other[col_other].str.split(',\s*')  # separa por coma y espacios
    df_other = df_other.explode('model')[[col_id, 'model']]

    # 3. Unir ambos
    df_result = pd.concat([df_raw, df_other], ignore_index=True)

    return df_result