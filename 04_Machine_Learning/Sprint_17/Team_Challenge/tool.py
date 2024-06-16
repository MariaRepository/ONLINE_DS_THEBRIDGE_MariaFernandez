import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import f_oneway

def plot_features_num_classification(df, target_col="", columns=[], pvalue=0.05):
    """
    Descripción: Realiza un análisis de clasificación entre una columna objetivo y las columnas numéricas de un DataFrame,
    filtrando aquellas que tienen un valor p bajo según el test de ANOVA.

    Argumentos:
    df (DataFrame): El DataFrame que contiene los datos.
    target_col (str): El nombre de la columna objetivo que se usará en el análisis de clasificación.
    columns (list): Una lista de nombres de columnas a considerar. Si está vacía, se considerarán todas las columnas numéricas del DataFrame.
    pvalue (float): El valor p máximo para considerar una columna como estadísticamente significativa.

    Retorna:
    list: pairplot del dataframe considerando la columna designada por "target_col" y aquellas incluidas en "columns" que cumplan los requisitos indicados.
    """
    # Comprobar si la lista de columnas está vacía
    if not columns:
        # Obtener todas las columnas numéricas del DataFrame
        columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Verificar si el target_col no está vacío
    if not target_col:
        raise ValueError("El argumento 'target_col' no puede estar vacío.")
    
    # Verificar si el target_col es una columna categórica
    if target_col not in df.select_dtypes(include=['object', 'category', 'int64']).columns:
        raise ValueError("El argumento 'target_col' debe ser una columna categórica o numérica discreta.")
    
    # Filtrar columnas basadas en el test de ANOVA y pvalue
    filtered_columns = []
    for col in columns:
        if col != target_col:
            unique_values = df[target_col].unique()
            groups = [df[df[target_col] == val][col] for val in unique_values]
            f_stat, p_val = f_oneway(*groups)
            if p_val < pvalue:
                filtered_columns.append(col)
            else:
                print(f"La columna numérica '{col}' no es significativa (p_value: {p_val} >= {pvalue}).")
        
    # Obtener valores únicos de target_col
    unique_target_values = df[target_col].unique()
    
    # Dividir los valores únicos de target_col en grupos de máximo cinco para pairplot
    for i in range(0, len(unique_target_values), 5):
        current_target_values = unique_target_values[i:i+5]
        df_filtered = df[df[target_col].isin(current_target_values)]
        
        # Dividir las columnas en grupos de máximo cinco para pairplot
        for j in range(0, len(filtered_columns), 4):
            sns.pairplot(df_filtered[filtered_columns[j:j+4] + [target_col]], hue=target_col, kind='reg', diag_kind='kde')
            plt.show()
    
    return filtered_columns

# Ejemplo de uso
# plot_features_num_classification(df_inmo, target_col="median_house_value", columns=[], pvalue=0.05)
