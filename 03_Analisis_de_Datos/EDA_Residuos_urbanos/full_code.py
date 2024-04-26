import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importar data sets
df_3= pd.read_csv("../data/3.protection_GDP.csv", sep=',', encoding='latin1')
# fuente_3 = "Gross value added in environmental good and services. EUROSTAT"
#https://ec.europa.eu/eurostat/databrowser/view/sdg_12_61__custom_10871879/default/table?lang=en

df_4 = pd.read_csv("../data/4.tasa_reciclaje_europea.csv", sep=';', encoding='latin1')
# fuente_4 = "Recycling rate of municipal waste.EUROSTAT"
#https://ec.europa.eu/eurostat/databrowser/view/cei_wm011/default/table?lang=en&category=cei.cei_wm

df_5 = pd.read_csv("../data/5.residuos_europa.csv", sep=';', encoding='latin1')
# fuente_5 = "Generation of municipal waste per capita. EUROSTAT"
#https://ec.europa.eu/eurostat/databrowser/view/cei_pc031/default/table?lang=en&category=cei.cei_pc

df_6= pd.read_csv("../data/conversion_eurostat_country.csv", sep=';', encoding='latin1')
# data set creado por mí para obtener una equivalencia de siglas de país con nombre de país.

df_7= pd.read_csv("../data/7.GDP_per_capita.csv", sep=',', encoding='latin1')
# fuente_7 = "Real GDP per capita. EUROSTAT"
#https://ec.europa.eu/eurostat/databrowser/view/sdg_08_10/default/table

# Generar copia de seguridad
df3bu=df_3.copy()
df4bu=df_4.copy()
df5bu=df_5.copy()
df6bu=df_6.copy()
df7bu=df_7.copy()

'''Se comienza el data Cleaning de los data set 3 y 7 y su unión'''
# Lista de columnas a eliminar
columnas_a_eliminar = ['DATAFLOW','LAST UPDATE', 'freq', 'nace_r2', 'ceparema', 'na_item', 'ty','unit', 'OBS_FLAG']

# Eliminar las columnas especificadas
df_3 = df_3.drop(columns=columnas_a_eliminar)

# Cambiar el nombre de la columna OBS_VALUE a otra más explicativa
df_3.rename(columns={'OBS_VALUE': 'Valor añadido ambiental respecto al PIB %'}, inplace=True)

# Lista de columnas a eliminar
columnas_a_eliminar = ['DATAFLOW','LAST UPDATE', 'freq', 'na_item','unit', 'OBS_FLAG']

# Eliminar las columnas especificadas
df_7 = df_7.drop(columns=columnas_a_eliminar)

# Cambiar el nombre de la columna OBS_VALUE a otra más explicativa
df_7.rename(columns={'OBS_VALUE': 'PIB per capita'}, inplace=True)

# Fusionar los DataFrames
df_37 = pd.merge(df_7, df_3, on=['geo', 'TIME_PERIOD'], how='left')


'''Se comienza con el data Cleaning del data set 4'''
# Lista de columnas a eliminar
columnas_a_eliminar = ['DATAFLOW','LAST UPDATE', 'freq', 'unit', 'OBS_FLAG','wst_oper']

# Eliminar las columnas especificadas
df_4 = df_4.drop(columns=columnas_a_eliminar)

# Fusionar los DataFrames df_4 y df_5 en función de la columna "geo"
df_4 = pd.merge(df_4, df_6[['geo', 'country','group']], on='geo', how='left')

# Cambiar el nombre de la columna OBS_VALUE a otra más explicativa
df_4.rename(columns={'OBS_VALUE': 'tasa reciclaje %'}, inplace=True)

# Lista de columnas a eliminar
columnas_a_eliminar = ['DATAFLOW', 'LAST UPDATE', 'freq', 'unit', 'OBS_FLAG', 'wst_oper']

# Eliminar las columnas especificadas
df_5 = df_5.drop(columns=columnas_a_eliminar)

# Cambiar el nombre de la columna OBS_VALUE a residuos_kg_hab
df_5.rename(columns={'OBS_VALUE': 'residuos_kg_hab'}, inplace=True)

# Fusionar los DataFrames
df_eurostat = pd.merge(df_4, df_5, on=['geo', 'TIME_PERIOD'], how='left')

# Generar copia de seguridad
df_eurostat_copy1=df_eurostat.copy()

'''Seguimos trabajando sobre el data frame agrupado: nulos, países con pocos datos se eliminan, creación de columna Group con zonas geográficas'''
# Contar el número de datos en la columna "geo" para cada país
conteo_datos_por_pais = df_eurostat.groupby('country')['geo'].count()

# Obtener los nombres de los países que tienen al menos 22 datos en la columna "geo"
paises_con_suficientes_datos = conteo_datos_por_pais[conteo_datos_por_pais >= 22].index.tolist()

# Filtrar el DataFrame para mantener solo las filas de los países con suficientes datos
df_eurostat_filtrado = df_eurostat[df_eurostat['country'].isin(paises_con_suficientes_datos)]

# Agrupar los datos por la columna "group" y obtener los países correspondientes a cada grupo
for group, group_data in df_eurostat_filtrado.groupby("group"):
    # Obtener los países únicos para el grupo actual
    paises_grupo = group_data["country"].unique()
    
    # Imprimir el nombre del grupo y los países correspondientes
    print(f"Grupo: {group}")
    print(", ".join(paises_grupo))
    print()  # Imprimir una línea en blanco entre grupos

'''Data Frame final: últimos data frames a unir y modificaciones'''

# Fusionar los DataFrames df_eurostat_frltrado y df_37 en función de la columna "geo"
df_eurostat_full = pd.merge(df_eurostat_filtrado, df_37, on=['geo', 'TIME_PERIOD'], how='left')

# Eliminar las filas donde el valor de la columna "group" sea el texto "NAN"
df_eurostat_full = df_eurostat_full[df_eurostat_full['group'] != 'NAN']

# Hacer copia de seguridad
df_eurostat_full_con_2022= df_eurostat_full.copy()

# Filtrar el DataFrame para excluir el periodo de tiempo 2022
df_eurostat_full = df_eurostat_full[df_eurostat_full['TIME_PERIOD'] != 2022]

'''Limpieza: estudio de outliers'''

# Generar categorías
columnas_categoricas_euro= ["country",'TIME_PERIOD','group']
columnas_numericas_euro = ["tasa reciclaje %","residuos_kg_hab","Valor añadido ambiental respecto al PIB %", "PIB per capita"]

'''Visualización para outliers'''
import sys
sys.path.insert(0, '../utils')

from bootcampviztools import plot_combined_graphs
plot_combined_graphs(df_eurostat_full, columnas_numericas_euro)

'''Incorporación de columna percentiles generación residuos'''
# Calcular la media de residuos por habitante por país
media_residuos_por_pais = df_eurostat_full.groupby('country')['residuos_kg_hab'].mean()

# Calcular los percentiles de la media de residuos por habitante en todo el conjunto de datos
percentiles = media_residuos_por_pais.quantile([0.25, 0.50, 0.75])

# Función para asignar el grupo basado en los percentiles y calcular el rango de residuos
def asignar_grupo(media):
    if media <= percentiles[0.25]:
        grupo = 'Grupo 1 perc.0-0.25'
        rango = f'{media_residuos_por_pais.min():.2f} - {percentiles[0.25]:.2f}'
    elif percentiles[0.25] < media <= percentiles[0.50]:
        grupo = 'Grupo 2 perc.0.25-0.50'
        rango = f'{percentiles[0.25]:.2f} - {percentiles[0.50]:.2f}'
    elif percentiles[0.50] < media <= percentiles[0.75]:
        grupo = 'Grupo 3 perc.0.50-0.75'
        rango = f'{percentiles[0.50]:.2f} - {percentiles[0.75]:.2f}'
    else:
        grupo = 'Grupo 4 perc.0.75-1'
        rango = f'{percentiles[0.75]:.2f} - {media_residuos_por_pais.max():.2f}'
    return grupo, rango

# Aplicar la función para asignar el grupo correspondiente a cada país
grupos, rangos = zip(*media_residuos_por_pais.apply(asignar_grupo))

''' Crear un DataFrame con los resultados'''
df_grupos_pais_residuos = pd.DataFrame({
    'country': media_residuos_por_pais.index,
    'Grupo residuos/hab': grupos,
    'Rango de residuos (kg/hab)': rangos
})
df_grupos_pais_residuos

df_eurostat_full = df_eurostat_full.merge(df_grupos_pais_residuos, on='country', how='left')

columnas_categoricas_euro= ["country",'TIME_PERIOD','group','Grupo residuos/hab','Rango de residuos (kg/hab)']
columnas_numericas_euro = ["tasa reciclaje %","residuos_kg_hab","Valor añadido ambiental respecto al PIB %", "PIB per capita"]

'''Analisis univariante'''
#Estadísticos min,max y otros
df_eurostat_full.describe()

# Calcular el rango para cada columna numérica en df_eurostat1221
rango = df_eurostat_full.describe().loc["max"] - df_eurostat_full.describe().loc["min"]

# Calcular el rango intercuartílico (IQR) para la columna seleccionada
import sys
sys.path.insert(0, '../utils')

from formulas import get_IQR
resultado_IQR = get_IQR(df_eurostat_full, columnas_numericas_euro)

# Calcular la variabilidad para las columnas seleccionadas
def variabilidad(df, col):
    df_var = df.describe().loc[["std","mean"]].T
    df_var['variabilidad'] = df_var["std"] / df_var["mean"]
    return df_var

resultado_variabilidad = variabilidad(df_eurostat_full[columnas_numericas_euro], 'variabilidad')
print(resultado_variabilidad)

# Crear un DataFrame con los resultados
tabla_resultados = pd.DataFrame({
    'Rango': rango,
    'IQR': resultado_IQR,
    'Variabilidad': resultado_variabilidad['variabilidad']
})

# Mostrar la tabla de resultados
print(tabla_resultados)

# Ver histogramas 2000-2021 y 2018-2021
def plot_histo_den(df, columns):
    num_cols = len(columns)
    num_rows = num_cols // 2 + num_cols % 2
    fig, axes = plt.subplots(num_rows, 2, figsize=(12, 6 * num_rows))
    axes = axes.flatten()

    for i, column in enumerate(columns):
        if df[column].dtype in ['int64', 'float64']:
            sns.histplot(df[column], kde=True, ax=axes[i])	
            axes[i].set_title(f'Histograma y KDE de {column}')

    # Ocultar ejes vacíos
    for j in range(i + 1, num_rows * 2):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

plot_histo_den(df_eurostat_full,columnas_numericas_euro)

df_eurostat_full_1821 = df_eurostat_full[(df_eurostat_full['TIME_PERIOD'] >= 2018) & (df_eurostat_full['TIME_PERIOD'] <= 2021)]
plot_histo_den(df_eurostat_full_1821,columnas_numericas_euro)

'''Análisis bivariante'''
df_eurostat_full.groupby("group")["tasa reciclaje %"].describe()
df_eurostat_full_1821.groupby("group")["tasa reciclaje %"].describe()
df_eurostat_full.groupby("group")["residuos_kg_hab"].describe()
df_eurostat_full_1821.groupby("group")["residuos_kg_hab"].describe()
df_eurostat_full.groupby("group")["Valor añadido ambiental respecto al PIB %"].describe()
df_eurostat_full_1821.groupby("group")["Valor añadido ambiental respecto al PIB %"].describe()
df_eurostat_full.groupby("group")["PIB per capita"].describe()
df_eurostat_full_1821.groupby("group")["PIB per capita"].describe()

'''Análisis bivariante: Correlación'''
# Calcular la correlación entre la tasa de reciclaje y los residuos por habitante (kg) por año
correlation_recycling_residues = df_eurostat_full.groupby('TIME_PERIOD')['tasa reciclaje %'].corr(df_eurostat_full['residuos_kg_hab'])

# Calcular la correlación entre la tasa de reciclaje y el PIB per cápita por año
correlation_recycling_PIB = df_eurostat_full.groupby('TIME_PERIOD')['tasa reciclaje %'].corr(df_eurostat_full['PIB per capita'])

# Crear un DataFrame con los resultados de correlación
df_correlation_by_year = pd.DataFrame({
    'Correlación Tasa-Residuos': correlation_recycling_residues,
    'Correlación Tasa-PIB': correlation_recycling_PIB
})

# Especificar la ruta del archivo Excel
excel_file_path = "../resultados/correlacion_reciclaje_por_anio.xlsx"

# Guardar el DataFrame en un archivo Excel
df_correlation_by_year.to_excel(excel_file_path)

print(f"Los resultados se han guardado en el archivo: {excel_file_path}")

# Calcular la correlación entre la tasa de reciclaje y los residuos por habitante (kg) por año
correlation_recycling_residues = df_eurostat_full_1821.groupby('TIME_PERIOD')['tasa reciclaje %'].corr(df_eurostat_full_1821['residuos_kg_hab'])

# Calcular la correlación entre la tasa de reciclaje y el PIB per cápita por año
correlation_recycling_PIB = df_eurostat_full_1821.groupby('TIME_PERIOD')['tasa reciclaje %'].corr(df_eurostat_full_1821['PIB per capita'])

# Crear un DataFrame con los resultados de correlación
df_correlation_by_year = pd.DataFrame({
    'Correlación Tasa-Residuos': correlation_recycling_residues,
    'Correlación Tasa-PIB': correlation_recycling_PIB
})

# Especificar la ruta del archivo Excel
excel_file_path = "../resultados/correlacion_reciclaje_por_18_21.xlsx"

# Guardar el DataFrame en un archivo Excel
df_correlation_by_year.to_excel(excel_file_path)

print(f"Los resultados se han guardado en el archivo: {excel_file_path}")

# Hacer gráfico de dispersión
import sys
sys.path.insert(0, '../utils')

from bootcampviztools import grafico_dispersion_con_correlacion
grafico_dispersion_con_correlacion(df_eurostat_full, "residuos_kg_hab", "tasa reciclaje %",50, mostrar_correlacion = True)


# Hacer gráfico de dispersión
import sys
sys.path.insert(0, '../utils')

from bootcampviztools import grafico_dispersion_con_correlacion
grafico_dispersion_con_correlacion(df_eurostat_full_1821, "residuos_kg_hab", "tasa reciclaje %",50, mostrar_correlacion = True)

'''Pearson'''
# Pearson
df_eurostat_full[["residuos_kg_hab", "tasa reciclaje %"]].corr()
from scipy.stats import pearsonr
pearsonr(df_eurostat_full["residuos_kg_hab"], df_eurostat_full["tasa reciclaje %"], alternative= "less")

df_eurostat_full[["residuos_kg_hab", "tasa reciclaje %"]].corr()
pearsonr(df_eurostat_full_1821["residuos_kg_hab"], df_eurostat_full_1821["tasa reciclaje %"], alternative= "less")

'''Correlación por grupo de países 2000-2021'''
import sys
sys.path.insert(0, '../utils')
from bootcampviztools import grafico_dispersion_con_correlacion

# Iterar sobre cada valor único en la columna "group"
for group_value in df_eurostat_full['group'].unique():
    # Filtrar el DataFrame para el valor de "group" actual
    df_group = df_eurostat_full[df_eurostat_full['group'] == group_value]
    
    # Imprimir el nombre del grupo como parte del título del gráfico
    print("Grupo:", group_value)
    
    # Crear el gráfico de dispersión con correlación para la columna "residuos_kg_hab" y "tasa reciclaje %"
    grafico_dispersion_con_correlacion(df_group, "residuos_kg_hab", "tasa reciclaje %", 50, mostrar_correlacion=True)

'''Correlación por grupo de países 2018-2021'''

# Iterar sobre cada valor único en la columna "group"
for group_value in df_eurostat_full_1821['group'].unique():
    # Filtrar el DataFrame para el valor de "group" actual
    df_group = df_eurostat_full_1821[df_eurostat_full['group'] == group_value]
    
    # Imprimir el nombre del grupo como parte del título del gráfico
    print("Grupo:", group_value)
    
    # Crear el gráfico de dispersión con correlación para la columna "residuos_kg_hab" y "tasa reciclaje %"
    grafico_dispersion_con_correlacion(df_group, "residuos_kg_hab", "tasa reciclaje %", 50, mostrar_correlacion=True)


'''Correlación pearson por grupo de países 2000-2021'''

# Definir una función para calcular el coeficiente de correlación de Pearson y p-valor para cada grupo
def calcular_pearson_por_grupo(df):
    # Calcular el coeficiente de correlación de Pearson y p-valor
    coef_pearson, p_valor = pearsonr(df["residuos_kg_hab"], df["tasa reciclaje %"])
    return coef_pearson, p_valor

# Calcular el coeficiente de correlación de Pearson para cada grupo
resultados_pearson_por_grupo = df_eurostat_full.groupby("group").apply(calcular_pearson_por_grupo)

'''Correlación pearson por grupo de países 2018-2021'''

# Definir una función para calcular el coeficiente de correlación de Pearson y p-valor para cada grupo
def calcular_pearson_por_grupo(df):
    # Calcular el coeficiente de correlación de Pearson y p-valor
    coef_pearson, p_valor = pearsonr(df["residuos_kg_hab"], df["tasa reciclaje %"])
    return coef_pearson, p_valor

# Calcular el coeficiente de correlación de Pearson para cada grupo
resultados_pearson_por_grupo = df_eurostat_full_1821.groupby("group").apply(calcular_pearson_por_grupo)

# Mostrar los resultados
resultados_pearson_por_grupo


'''Correlación pearson por países 2000-2021'''

# Definir una función para calcular el coeficiente de correlación de Pearson y p-valor para cada país en el grupo 
def calcular_pearson_por_pais(df):
    # Calcular el coeficiente de correlación de Pearson y p-valor
    coef_pearson, p_valor = pearsonr(df["residuos_kg_hab"], df["tasa reciclaje %"])
    return coef_pearson, p_valor

# Mostrar los resultados
resultados_pearson_por_grupo

'''Tabla resumen por pais'''
# Función para calcular el coeficiente de correlación de Pearson y p-valor para cada país
def calcular_pearson_por_pais(df):
    # Calcular el coeficiente de correlación de Pearson y p-valor
    coef_pearson, p_valor = pearsonr(df["residuos_kg_hab"], df["tasa reciclaje %"])
    # Obtener la tasa de reciclaje promedio por país
    tasa_reciclaje = df["tasa reciclaje %"].mean()
    # Obtener la tasa de reciclaje promedio por país
    residuos_habitante = df["residuos_kg_hab"].mean()
    # Obtener la tasa de reciclaje promedio por país
    PIB_habitante = df["PIB per capita"].mean()
    # Crear una Serie con los resultados
    resultados = pd.Series({"Pearson": coef_pearson, "p-valor": p_valor, "Tasa de reciclaje (%)": tasa_reciclaje, "Residuos kg habitante": residuos_habitante,"PIB habitante":PIB_habitante})
    return resultados

# Calcular el coeficiente de correlación de Pearson para cada país en el DataFrame completo
resultados_pearson_por_pais = df_eurostat_full.groupby("country").apply(calcular_pearson_por_pais)

# Mostrar los resultados
resultados_pearson_por_pais

# Especificar la ruta del archivo Excel
excel_file_path = "../resultados/correlacion_reciclaje_residuo_pais.xlsx"

# Guardar el DataFrame en un archivo Excel
resultados_pearson_por_pais.to_excel(excel_file_path, index=False)

# Imprimir comprobación guardado
print("Guardado")

'''Correlación por grupo de percentiles'''
# Definir una función para calcular el coeficiente de correlación de Pearson y p-valor para cada grupo
def calcular_pearson_por_grupo(df):
    # Calcular el coeficiente de correlación de Pearson y p-valor
    coef_pearson, p_valor = pearsonr(df["residuos_kg_hab"], df["tasa reciclaje %"])
    return coef_pearson, p_valor

# Calcular el coeficiente de correlación de Pearson para cada grupo
resultados_pearson_por_grupo = df_eurostat_full.groupby("Grupo residuos/hab").apply(calcular_pearson_por_grupo)

# Mostrar los resultados
print("Resultados 2000-2021")
print(resultados_pearson_por_grupo)

# Calcular el coeficiente de correlación de Pearson para cada grupo
resultados_pearson_por_grupo18 = df_eurostat_full_1821.groupby("Grupo residuos/hab").apply(calcular_pearson_por_grupo)

# Mostrar los resultados
print("Resultados 2018-2021")
print(resultados_pearson_por_grupo18)


'''Análisis correlación con Valor añadido ambiente'''
# Filtrar el DataFrame para incluir solo las filas donde la columna "Valor añadido ambiental respecto al PIB %" no sea NaN
df_eurostat_11_21_filtrado = df_eurostat_full[df_eurostat_full["Valor añadido ambiental respecto al PIB %"].notna()]

grafico_dispersion_con_correlacion(df_eurostat_11_21_filtrado, "residuos_kg_hab", "Valor añadido ambiental respecto al PIB %",50, mostrar_correlacion = True)

df_eurostat_11_21_filtrado[["residuos_kg_hab", "Valor añadido ambiental respecto al PIB %"]].corr()

from scipy.stats import pearsonr

pearsonr(df_eurostat_11_21_filtrado["residuos_kg_hab"], df_eurostat_11_21_filtrado["Valor añadido ambiental respecto al PIB %"], alternative= "less")



'''Análisis correlación residuos con Valor añadido ambiente por grupo de país'''

# Definir una función para calcular el coeficiente de correlación de Pearson y p-valor para cada grupo
def calcular_pearson_por_grupo(df):
    # Calcular el coeficiente de correlación de Pearson y p-valor
    coef_pearson, p_valor = pearsonr(df["residuos_kg_hab"], df[ "Valor añadido ambiental respecto al PIB %"])
    return coef_pearson, p_valor

# Calcular el coeficiente de correlación de Pearson para cada grupo
resultados_pearson_por_grupo = df_eurostat_11_21_filtrado.groupby("group").apply(calcular_pearson_por_grupo)

# Mostrar los resultados
resultados_pearson_por_grupo

'''Análisis correlación tasa reciclaje con Valor añadido ambiente'''

grafico_dispersion_con_correlacion(df_eurostat_11_21_filtrado, "tasa reciclaje %", "Valor añadido ambiental respecto al PIB %",50, mostrar_correlacion = True)
df_eurostat_11_21_filtrado[[ "tasa reciclaje %", "Valor añadido ambiental respecto al PIB %"]].corr()
pearsonr(df_eurostat_11_21_filtrado[ "tasa reciclaje %"], df_eurostat_11_21_filtrado["Valor añadido ambiental respecto al PIB %"], alternative= "less")

# Definir una función para calcular el coeficiente de correlación de Pearson y p-valor para cada grupo
def calcular_pearson_por_grupo(df):
    # Calcular el coeficiente de correlación de Pearson y p-valor
    coef_pearson, p_valor = pearsonr(df["tasa reciclaje %"], df[ "Valor añadido ambiental respecto al PIB %"])
    return coef_pearson, p_valor

# Calcular el coeficiente de correlación de Pearson para cada grupo
resultados_pearson_por_grupo = df_eurostat_11_21_filtrado.groupby("group").apply(calcular_pearson_por_grupo)

# Mostrar los resultados
resultados_pearson_por_grupo


'''Análisis correlación tasa reciclaje con Valor añadido ambiente para un grupo de países y ver detalle de país'''

# Definir una función para calcular el coeficiente de correlación de Pearson y p-valor para cada país dentro del grupo "South"
def calcular_pearson_por_pais(df):
    # Calcular el coeficiente de correlación de Pearson y p-valor
    coef_pearson, p_valor = pearsonr(df["tasa reciclaje %"], df["Valor añadido ambiental respecto al PIB %"])
    return coef_pearson, p_valor

# Filtrar el DataFrame para incluir solo el grupo "South"
df_grupo_s = df_eurostat_11_21_filtrado[df_eurostat_11_21_filtrado["group"] == "South"]

# Calcular el coeficiente de correlación de Pearson para cada país dentro del grupo "South"
resultados_pearson_grupo_s = df_grupo_s.groupby("country").apply(calcular_pearson_por_pais)

# Mostrar los resultados
resultados_pearson_grupo_s


'''Análisis correlación tasa reciclaje con PIB p capita 2000-2021'''

# Grafico o de dispersión con correlación
grafico_dispersion_con_correlacion(df_eurostat_full, "tasa reciclaje %", "PIB per capita", 50, mostrar_correlacion=True)

# Calcular el coeficiente de correlación de Pearson utilizando el método corr() de Pandas
correlacion_pearson = df_eurostat_full[["tasa reciclaje %", "PIB per capita"]].corr()

# Calcular el coeficiente de correlación de Pearson y el p-valor utilizando scipy.stats
coef_pearson, p_valor = pearsonr(df_eurostat_full["tasa reciclaje %"], df_eurostat_full["PIB per capita"])
print("Coeficiente de correlación de Pearson:", coef_pearson)
print("P-valor:", p_valor)


'''Análisis correlación tasa reciclaje con PIB p capita 2018-2021'''

# Grafico o de dispersión con correlación
grafico_dispersion_con_correlacion(df_eurostat_full_1821, "tasa reciclaje %", "PIB per capita", 50, mostrar_correlacion=True)

# Calcular el coeficiente de correlación de Pearson utilizando el método corr() de Pandas
correlacion_pearson = df_eurostat_full_1821[["tasa reciclaje %", "PIB per capita"]].corr()

# Calcular el coeficiente de correlación de Pearson y el p-valor utilizando scipy.stats
coef_pearson, p_valor = pearsonr(df_eurostat_full_1821["tasa reciclaje %"], df_eurostat_full_1821["PIB per capita"])
print("Coeficiente de correlación de Pearson:", coef_pearson)
print("P-valor:", p_valor)



'''Análisis correlación tasa reciclaje con PIB p capita por grupo de países'''

# Definir una función para calcular el coeficiente de correlación de Pearson y p-valor para cada grupo
def calcular_pearson_por_grupo(df):
    # Calcular el coeficiente de correlación de Pearson y p-valor
    coef_pearson, p_valor = pearsonr(df["tasa reciclaje %"], df[ "PIB per capita"])
    return coef_pearson, p_valor

# Calcular el coeficiente de correlación de Pearson para cada grupo
resultados_pearson_por_grupob = df_eurostat_full.groupby("group").apply(calcular_pearson_por_grupo)

# Mostrar los resultados
print("Resultado 2000-2021")
print(resultados_pearson_por_grupob)

# Calcular el coeficiente de correlación de Pearson para cada grupo
resultados_pearson_por_grupo18b = df_eurostat_full.groupby("group").apply(calcular_pearson_por_grupo)


# Calcular el coeficiente de correlación de Pearson para cada grupo
resultados_pearson_por_grupo18b = df_eurostat_full_1821.groupby("group").apply(calcular_pearson_por_grupo)
# Mostrar los resultados
print("Resultado 2018-2021")
print(resultados_pearson_por_grupo18b)




'''Análisis correlación residuos habitante con PIB p capita 2000-2021'''

# Grafico de dispersión con correlación
grafico_dispersion_con_correlacion(df_eurostat_full, "residuos_kg_hab", "PIB per capita", 50, mostrar_correlacion=True)

# Calcular el coeficiente de correlación de Pearson utilizando el método corr() de Pandas
correlacion_pearson = df_eurostat_full[["residuos_kg_hab", "PIB per capita"]].corr()

# Calcular el coeficiente de correlación de Pearson y el p-valor utilizando scipy.stats
coef_pearson, p_valor = pearsonr(df_eurostat_full["residuos_kg_hab"], df_eurostat_full["PIB per capita"])
print("Coeficiente de correlación de Pearson:", coef_pearson)
print("P-valor:", p_valor)


'''Análisis correlación residuos habitante con PIB p capita 2018-2021'''

# Grafico de dispersión con correlación
grafico_dispersion_con_correlacion(df_eurostat_full_1821, "residuos_kg_hab", "PIB per capita", 50, mostrar_correlacion=True)

# Calcular el coeficiente de correlación de Pearson utilizando el método corr() de Pandas
correlacion_pearson = df_eurostat_full_1821[["residuos_kg_hab", "PIB per capita"]].corr()

# Calcular el coeficiente de correlación de Pearson y el p-valor utilizando scipy.stats
coef_pearson, p_valor = pearsonr(df_eurostat_full_1821["residuos_kg_hab"], df_eurostat_full_1821["PIB per capita"])
print("Coeficiente de correlación de Pearson:", coef_pearson)
print("P-valor:", p_valor)



'''Análisis correlación residuos habitante con PIB p capita 2000-2021 por grupo de países'''

# Definir una función para calcular el coeficiente de correlación de Pearson y p-valor para cada grupo
def calcular_pearson_por_grupo(df):
    # Calcular el coeficiente de correlación de Pearson y p-valor
    coef_pearson, p_valor = pearsonr(df["residuos_kg_hab"], df[ "PIB per capita"])
    return coef_pearson, p_valor

# Calcular el coeficiente de correlación de Pearson para cada grupo
resultados_pearson_por_grupo = df_eurostat_full.groupby("group").apply(calcular_pearson_por_grupo)

# Mostrar los resultados
resultados_pearson_por_grupo


'''Análisis correlación residuos habitante con PIB p capita 2018-2021 por grupo de países'''

# Definir una función para calcular el coeficiente de correlación de Pearson y p-valor para cada grupo
def calcular_pearson_por_grupo(df):
    # Calcular el coeficiente de correlación de Pearson y p-valor
    coef_pearson, p_valor = pearsonr(df["residuos_kg_hab"], df[ "PIB per capita"])
    return coef_pearson, p_valor

# Calcular el coeficiente de correlación de Pearson para cada grupo
resultados_pearson_por_grupo = df_eurostat_full_1821.groupby("group").apply(calcular_pearson_por_grupo)

# Mostrar los resultados
resultados_pearson_por_grupo

'''Impresión de boxplots'''
from bootcampviztools import plot_grouped_boxplots
plot_grouped_boxplots(df_eurostat_full, "group", "residuos_kg_hab")


'''Outliers Tasa reciclaje 1'''
# Filtrar el DataFrame para incluir solo el grupo "West"
df_west = df_eurostat_full[df_eurostat_full["group"] == "West"]

# Identificar los países del grupo "West" con una tasa de reciclaje inferior al 30%
paises_reciclaje_bajo_west = df_west[df_west["tasa reciclaje %"] < 30]["country"].unique()

# Mostrar los países del grupo "West" con una tasa de reciclaje inferior al 30%
print("Países del grupo West con tasa de reciclaje inferior al 30%:")
print(paises_reciclaje_bajo_west)

# Visualizar los datos usando plot_grouped_boxplots
plot_grouped_boxplots(df_eurostat_full, "group", "tasa reciclaje %")



'''Outliers Tasa reciclaje 2'''

# Filtrar el DataFrame para incluir solo el grupo "North"
df_north = df_eurostat_full[df_eurostat_full["group"] == "North"]

# Identificar los países del grupo "North" con una tasa de reciclaje superior al 50%
paises_reciclaje_alto_north = df_north[df_north["tasa reciclaje %"] > 50]["country"].unique()

# Mostrar los países del grupo "North" con una tasa de reciclaje superior al 50%
print("Países del grupo North con tasa de reciclaje superior al 50%:")
print(paises_reciclaje_alto_north)

# Visualizar los datos usando plot_grouped_boxplots
plot_grouped_boxplots(df_eurostat_full, "group", "tasa reciclaje %")



'''Outliers PIB'''

plot_grouped_boxplots(df_eurostat_full, "group", "PIB per capita")
# Filtrar el DataFrame para incluir solo el grupo "West"
df_west = df_eurostat_full[df_eurostat_full["group"] == "West"]

# Identificar los países con un PIB per cápita superior a 80000 en el grupo "West"
paises_pib_alto_west = df_west[df_west["PIB per capita"] > 80000]["country"].unique()

# Mostrar los países con un PIB per cápita superior a 80000 en el grupo "West"
print("Países con PIB per cápita superior a 90000 en el grupo West:")
print(paises_pib_alto_west)



'''Outliers Valor Añadido ambiental'''

plot_grouped_boxplots(df_eurostat_11_21_filtrado, "group", "Valor añadido ambiental respecto al PIB %")
# Filtrar el DataFrame para incluir solo los grupos "East" y "South"
df_east_south = df_eurostat_11_21_filtrado[df_eurostat_11_21_filtrado["group"].isin(["East", "South"])]

# Identificar los países con un valor añadido ambiental superior al 3.5% en cada grupo
paises_valor_alto_east = df_east_south[df_east_south["group"] == "East"][df_east_south["Valor añadido ambiental respecto al PIB %"] > 3.5]["country"].unique()
paises_valor_alto_south = df_east_south[df_east_south["group"] == "South"][df_east_south["Valor añadido ambiental respecto al PIB %"] > 3.5]["country"].unique()

# Mostrar los países con valor añadido ambiental superior al 3.5% en cada grupo
print("Países con valor añadido ambiental superior al 3.5% en el grupo East:")
print(paises_valor_alto_east)

print("\nPaíses con valor añadido ambiental superior al 3.5% en el grupo South:")
print(paises_valor_alto_south)



'''Resumen correlaciones Pearson'''

# Función para calcular el coeficiente de correlación de Pearson y p-valor para cada país
def calcular_pearson_por_pais(df):
    # Calcular el coeficiente de correlación de Pearson y p-valor
    coef_pearson, p_valor = pearsonr(df["PIB per capita"], df["tasa reciclaje %"])
    # Obtener la tasa de reciclaje promedio por país
    tasa_reciclaje = df["tasa reciclaje %"].mean()
    # Obtener la tasa de reciclaje promedio por país
    residuos_habitante = df["residuos_kg_hab"].mean()
    # Obtener la tasa de reciclaje promedio por país
    PIB_habitante = df["PIB per capita"].mean()
    # Crear una Serie con los resultados
    resultados = pd.Series({"Pearson": coef_pearson, "p-valor": p_valor, "Tasa de reciclaje (%)": tasa_reciclaje, "Residuos kg habitante": residuos_habitante,"PIB habitante":PIB_habitante})
    return resultados

# Calcular el coeficiente de correlación de Pearson para cada país en el DataFrame completo
resultados_pearson_por_paisPIB = df_eurostat_full.groupby("country").apply(calcular_pearson_por_pais)

# Mostrar los resultados
resultados_pearson_por_paisPIB


# Especificar la ruta del archivo Excel
excel_file_path = "../resultados/correlacion_PIBreciclaje_por_pais.xlsx"

# Guardar el DataFrame en un archivo Excel
resultados_pearson_por_paisPIB.to_excel(excel_file_path)

print(f"Los resultados se han guardado en el archivo: {excel_file_path}")



'''Correlaciones ANOVA '''

from scipy import stats

# Obtener los valores únicos de la columna categórica
grupos = df_eurostat_full['group'].unique() 
tasa_pais = [df_eurostat_full[df_eurostat_full['group'] == grupo]['residuos_kg_hab'] for grupo in grupos] 

f_val, p_val = stats.f_oneway(*tasa_pais) 
print("Valor F Residuos por grupo 2000-2021:", f_val)
print("Valor p Residuos por grupo 2000-2021:", p_val)

# Obtener los valores únicos de la columna categórica
tasa_pais = [df_eurostat_full_1821[df_eurostat_full_1821['group'] == grupo]['residuos_kg_hab'] for grupo in grupos] 
f_val, p_val = stats.f_oneway(*tasa_pais) 
print("Valor F Residuos por grupo 2018-2021:", f_val)
print("Valor p Residuos por grupo 2018-2021:", p_val)


'''Correlaciones ANOVA por grupo'''

tasa_pais = [df_eurostat_full[df_eurostat_full['group'] == grupo]["tasa reciclaje %"] for grupo in grupos] 

f_val, p_val = stats.f_oneway(*tasa_pais) 
print("Valor F Tasa reciclaje 2000-2021 por grupo:", f_val)
print("Valor p Tasa reciclaje 2000-2021 por grupo:", p_val)
# Obtener los valores únicos de la columna categórica
grupos = df_eurostat_full['group'].unique() 
tasa_pais = [df_eurostat_full_1821[df_eurostat_full_1821['group'] == grupo]["tasa reciclaje %"] for grupo in grupos] 

f_val, p_val = stats.f_oneway(*tasa_pais) 
print("Valor F Tasa reciclaje 2018-2021 por grupo :", f_val)
print("Valor p 2018-2021 por grupo:", p_val)

# Obtener los valores únicos de la columna categórica
grupos = df_eurostat_11_21_filtrado['group'].unique() 
tasa_pais = [df_eurostat_11_21_filtrado[df_eurostat_11_21_filtrado['group'] == grupo]["Valor añadido ambiental respecto al PIB %"] for grupo in grupos] 

f_val, p_val = stats.f_oneway(*tasa_pais) 
print("Valor F Valor añadido ambiental 2011-2021:", f_val)
print("Valor p Valor añadido ambiental 2011-2021:", p_val)

# Obtener los valores únicos de la columna categórica
grupos = df_eurostat_full['group'].unique() 
tasa_pais = [df_eurostat_full[df_eurostat_full['group'] == grupo]["PIB per capita"] for grupo in grupos] 

f_val, p_val = stats.f_oneway(*tasa_pais) 
print("Valor F PIB per capita 2000-2021:", f_val)
print("Valor p PIB per capita 2000-2021:", p_val)

# Obtener los valores únicos de la columna categórica
grupos = df_eurostat_full['group'].unique() 
tasa_pais = [df_eurostat_full_1821[df_eurostat_full_1821['group'] == grupo]["PIB per capita"] for grupo in grupos] 

f_val, p_val = stats.f_oneway(*tasa_pais) 
print("Valor F PIB per capita 2018-2021:", f_val)
print("Valor p PIB per capita 2018-2021:", p_val)

# Obtener los valores únicos de la columna categórica
grupos = df_eurostat['TIME_PERIOD'].unique() 
tasa_pais = [df_eurostat[df_eurostat['TIME_PERIOD'] == grupo]["tasa reciclaje %"] for grupo in grupos] 

f_val, p_val = stats.f_oneway(*tasa_pais) 
print("Valor F time period tasa de reciclaje:", f_val)
print("Valor p time period tasa de reciclaje:", p_val)


'''Series temporales'''
'''Reciclaje'''
# Agrupar los datos por 'TIME_PERIOD' y calcular la media de 'tasa reciclaje %' para cada período de tiempo
grupo_tiempo = df_eurostat_full.groupby('TIME_PERIOD')['tasa reciclaje %'].mean()

# Trazar la serie temporal de la tasa de reciclaje promedio a lo largo del tiempo
plt.plot(grupo_tiempo.index, grupo_tiempo.values, marker='o', linestyle='-', label='Tasa de reciclaje promedio')

# Etiquetas y título del gráfico
plt.xlabel('Año')
plt.ylabel('Tasa de reciclaje (%)')
plt.title('Tasa de reciclaje promedio a lo largo del tiempo')
plt.legend()

# Mostrar el gráfico
plt.show()


'''Valor ambiental'''
# Agrupar los datos por 'TIME_PERIOD' y calcular la media de 'tasa reciclaje %' para cada período de tiempo
grupo_tiempo = df_eurostat_full.groupby('TIME_PERIOD')['Valor añadido ambiental respecto al PIB %'].mean()

# Trazar la serie temporal de la tasa de reciclaje promedio a lo largo del tiempo
plt.plot(grupo_tiempo.index, grupo_tiempo.values, marker='o', linestyle='-', label='Valor añadido ambiental respecto al PIB %')

# Etiquetas y título del gráfico
plt.xlabel('Año')
plt.ylabel('Valor añadido ambiental respecto al PIB (%)')
plt.title('Valor añadido ambiental respecto al PIB %promedio a lo largo del tiempo')
plt.legend()

# Mostrar el gráfico
plt.show()

'''Residuos'''
# Trazar la serie temporal a lo largo del tiempo
plt.plot(grupo_tiempo.index, grupo_tiempo.values, marker='o', linestyle='-', label='Tasa residuos habitante promedio')

# Etiquetas y título del gráfico
plt.xlabel('Año')
plt.ylabel('Tasa de reciclaje (%)')

plt.title('Tasa de residuos promedio 2018-2021')
plt.legend()

# Especificar el formato del eje x
plt.xticks(grupo_tiempo.index, [int(year) for year in grupo_tiempo.index])

# Mostrar el gráfico
plt.show()

'''PIB per capita''''
# Agrupar los datos por 'TIME_PERIOD' y calcular la media para cada período de tiempo
grupo_tiempo = df_eurostat_full.groupby('TIME_PERIOD')['PIB per capita'].mean()

# Trazar la serie temporal a lo largo del tiempo
plt.plot(grupo_tiempo.index, grupo_tiempo.values, marker='o', linestyle='-', label='PIB per capita promedio')

# Etiquetas y título del gráfico
plt.xlabel('Año')
plt.ylabel('PIB per capita (%)')
plt.title('PIB per capita promedio a lo largo del tiempo')
plt.legend()

# Mostrar el gráfico
plt.show()

'''Serie por grupo geo tasa reciclaje'''
# Agrupar los datos por 'group' y 'TIME_PERIOD' y calcular la media de 'tasa reciclaje %' para cada grupo en cada período de tiempo
grupo_pais_tiempo = df_eurostat_full.groupby(['group', 'TIME_PERIOD'])['tasa reciclaje %'].mean().unstack(level=0)

# Trazar la serie temporal de la tasa de reciclaje promedio para cada grupo de países
grupo_pais_tiempo.plot(marker='o', linestyle='-')

# Etiquetas y título del gráfico
plt.xlabel('Año')
plt.ylabel('Tasa de reciclaje (%)')
plt.title('Tasa de reciclaje por grupo de países a lo largo del tiempo')
plt.legend(title='Grupo')

# Mostrar el gráfico
plt.show()

'''Serie por grupo geo residuos'''
# Agrupar los datos por 'group' y 'TIME_PERIOD' y calcular la media de 'tasa reciclaje %' para cada grupo en cada período de tiempo
grupo_pais_tiempo = df_eurostat_full.groupby(['group', 'TIME_PERIOD'])['residuos_kg_hab'].mean().unstack(level=0)
# Trazar la serie temporal de la tasa de reciclaje promedio para cada grupo de países
grupo_pais_tiempo.plot(marker='o', linestyle='-')

# Etiquetas y título del gráfico
plt.xlabel('Año')
plt.ylabel('Residuos kg habitante')
plt.title('Residuos habitante por grupo de países a lo largo del tiempo')
plt.legend(title='Grupo')

# Mostrar el gráfico
plt.show()


'''Serie por grupo PIB'''
# Agrupar los datos por 'group' y 'TIME_PERIOD' y calcular la media de 'tasa reciclaje %' para cada grupo en cada período de tiempo
grupo_pais_tiempo = df_eurostat_full.groupby(['group', 'TIME_PERIOD'])['PIB per capita'].mean().unstack(level=0)

# Trazar la serie temporal de la tasa de reciclaje promedio para cada grupo de países
grupo_pais_tiempo.plot(marker='o', linestyle='-')

# Etiquetas y título del gráfico
plt.xlabel('Año')
plt.ylabel('PIB per capita')
plt.title('PIB per capita por grupo de países a lo largo del tiempo')
plt.legend(title='Grupo')

# Mostrar el gráfico
plt.show()

'''Variaciones'''
'''Variacion Residuos'''
# Calcular la variación anual de los residuos por habitante respecto al año anterior
variacion_anual_residuos = df_eurostat_full.groupby('TIME_PERIOD')['residuos_kg_hab'].mean().pct_change()

# Crear la figura y los ejes del gráfico
plt.figure(figsize=(10, 6))

# Graficar la variación anual de los residuos por habitante
plt.plot(variacion_anual_residuos.index, variacion_anual_residuos.values * 100, marker='o', linestyle='-', color='b')

# Añadir etiquetas y título al gráfico
plt.xlabel('Año')
plt.ylabel('Variación anual de residuos por habitante (%)')
plt.title('Variación anual de residuos por habitante respecto al año anterior')

# Mostrar la cuadrícula
plt.grid(True)

# Mostrar el gráfico
plt.show()



'''Variacion PIB p capita'''

# Calcular la variación anual respecto al año anterior
variacion_anual_PIB = df_eurostat_full.groupby('TIME_PERIOD')['PIB per capita'].mean().pct_change()

# Crear la figura y los ejes del gráfico
plt.figure(figsize=(10, 6))

# Graficar la variación anual 
plt.plot(variacion_anual_PIB.index, variacion_anual_PIB.values * 100, marker='o', linestyle='-', color='b')

# Añadir etiquetas y título al gráfico
plt.xlabel('Año')
plt.ylabel('Variación anual de PIB per capita  (%)')
plt.title('Variación anual de PIB per capita  respecto al año anterior')

# Mostrar la cuadrícula
plt.grid(True)

# Mostrar el gráfico
plt.show()


'''Variacion Ambiental'''

# Calcular la variación anual
variacion_anual_ambiental = df_eurostat_full.groupby('TIME_PERIOD')['Valor añadido ambiental respecto al PIB %'].mean().pct_change()

# Crear la figura y los ejes del gráfico
plt.figure(figsize=(10, 6))

# Graficar la variación anual
plt.plot(variacion_anual_ambiental.index, variacion_anual_ambiental.values * 100, marker='o', linestyle='-', color='b')

# Añadir etiquetas y título al gráfico
plt.xlabel('Año')
plt.ylabel('Variación anual de Valor añadido ambiental respecto al PIB %')
plt.title('Variación anual de Valor añadido ambiental respecto al PIB respecto al año anterior')

# Mostrar la cuadrícula
plt.grid(True)

# Mostrar el gráfico
plt.show()



'''Variacion DATA FRAME'''

# Calcular la variación anual para cada columna
df_variaciones = pd.DataFrame({
    'Año': df_eurostat_full['TIME_PERIOD'].unique(),
    'Variación Tasa de Reciclaje (%)': df_eurostat_full.groupby('TIME_PERIOD')['tasa reciclaje %'].mean().pct_change() * 100,
    'Variación Residuos por Habitante (%)': df_eurostat_full.groupby('TIME_PERIOD')['residuos_kg_hab'].mean().pct_change() * 100,
    'Variación PIB per Cápita (%)': df_eurostat_full.groupby('TIME_PERIOD')['PIB per capita'].mean().pct_change() * 100,
    #'Variación Valor Añadido Ambiental (%)': df_eurostat_full.groupby('TIME_PERIOD')['Valor añadido ambiental respecto al PIB %'].mean().pct_change() * 100
})

# Configurar el gráfico
plt.figure(figsize=(10, 6))

# Trazar la variación anual de la tasa de reciclaje
plt.plot(df_variaciones['Año'], df_variaciones['Variación Tasa de Reciclaje (%)'], label='Tasa de Reciclaje')

# Trazar la variación anual de los residuos por habitante
plt.plot(df_variaciones['Año'], df_variaciones['Variación Residuos por Habitante (%)'], label='Residuos por Habitante')

# Trazar la variación anual del PIB per Cápita
plt.plot(df_variaciones['Año'], df_variaciones['Variación PIB per Cápita (%)'], label='PIB per Cápita')

# Configurar título y etiquetas de los ejes
plt.title('Variación Anual de Indicadores')
plt.xlabel('Año')
plt.ylabel('Variación (%)')

# Establecer el eje x para que muestre cada año individualmente
plt.xticks(df_variaciones['Año'], rotation=90)

# Mostrar la leyenda
plt.legend()

# Mostrar el gráfico
plt.grid(True)
plt.tight_layout()
plt.show()

# Mostrar el DataFrame con las variaciones anuales para cada tipo de indicador
df_variaciones

# Especificar la ruta del archivo Excel
excel_file_path = "../resultados/variaciones_gral.xlsx"

# Guardar el DataFrame en un archivo Excel
df_variaciones.to_excel(excel_file_path)

print(f"Los resultados se han guardado en el archivo: {excel_file_path}")



'''Variaciones por grupo''''
# Calcular la variación anual para cada grupo y cada columna
def calcular_variaciones_por_grupo(df, group_column):
    variaciones_por_grupo = {}
    for group_value in df[group_column].unique():
        variaciones = {
            'Año': df[df[group_column] == group_value]['TIME_PERIOD'].unique(),
            f'Variación Tasa de Reciclaje (%) - {group_value}': df[df[group_column] == group_value].groupby('TIME_PERIOD')['tasa reciclaje %'].mean().pct_change() * 100,
            f'Variación Residuos por Habitante (%) - {group_value}': df[df[group_column] == group_value].groupby('TIME_PERIOD')['residuos_kg_hab'].mean().pct_change() * 100,
            f'Variación PIB per Cápita (%) - {group_value}': df[df[group_column] == group_value].groupby('TIME_PERIOD')['PIB per capita'].mean().pct_change() * 100,
            #f'Variación Valor Añadido Ambiental (%) - {group_value}': df[df[group_column] == group_value].groupby('TIME_PERIOD')['Valor añadido ambiental respecto al PIB %'].mean().pct_change() * 100
        }
        variaciones_por_grupo[group_value] = pd.DataFrame(variaciones)
    return variaciones_por_grupo

# Función para graficar y mostrar las variaciones anuales para cada grupo
def graficar_variaciones_por_grupo(variaciones_por_grupo):
    for group_value, df_variaciones in variaciones_por_grupo.items():
        # Configurar el gráfico
        plt.figure(figsize=(10, 6))

        # Trazar la variación anual de la tasa de reciclaje
        plt.plot(df_variaciones['Año'], df_variaciones[f'Variación Tasa de Reciclaje (%) - {group_value}'], label='Tasa de Reciclaje')

        # Trazar la variación anual de los residuos por habitante
        plt.plot(df_variaciones['Año'], df_variaciones[f'Variación Residuos por Habitante (%) - {group_value}'], label='Residuos por Habitante')

        # Trazar la variación anual del PIB per Cápita
        plt.plot(df_variaciones['Año'], df_variaciones[f'Variación PIB per Cápita (%) - {group_value}'], label='PIB per Cápita')

        # Configurar título y etiquetas de los ejes
        plt.title(f'Variación Anual de Indicadores - {group_value}')
        plt.xlabel('Año')
        plt.ylabel('Variación (%)')

        # Establecer el eje x para que muestre cada año individualmente
        plt.xticks(df_variaciones['Año'], rotation=90)

        # Mostrar la leyenda
        plt.legend()

        # Mostrar el gráfico
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Mostrar el DataFrame con las variaciones anuales para cada tipo de indicador
        print(f'Variaciones anuales para {group_value}')
        df_variaciones
        print('\n')

# Calcular las variaciones para cada grupo
variaciones_por_grupo = calcular_variaciones_por_grupo(df_eurostat_full, 'group')

# Graficar y mostrar las variaciones para cada grupo
graficar_variaciones_por_grupo(variaciones_por_grupo)



'''Variaciones dentro de un grupo de países por país'''
# Filtrar el DataFrame para el grupo "South"
df_group_south = df_eurostat_full[df_eurostat_full['group'] == 'South']

# Calcular la variación anual para cada columna dentro del grupo "South"
df_variaciones_south = pd.DataFrame({
    'Año': df_group_south['TIME_PERIOD'].unique(),
    'Variación Tasa de Reciclaje (%)': df_group_south.groupby('TIME_PERIOD')['tasa reciclaje %'].mean().pct_change() * 100,
    'Variación Residuos por Habitante (%)': df_group_south.groupby('TIME_PERIOD')['residuos_kg_hab'].mean().pct_change() * 100,
    'Variación PIB per Cápita (%)': df_group_south.groupby('TIME_PERIOD')['PIB per capita'].mean().pct_change() * 100
})

# Configurar el gráfico
plt.figure(figsize=(10, 6))

# Trazar la variación anual de la tasa de reciclaje para cada país dentro del grupo "South"
for country in df_group_south['country'].unique():
    plt.plot(df_variaciones_south['Año'], df_group_south[df_group_south['country'] == country]['tasa reciclaje %'].pct_change() * 100, label=country)

# Configurar título y etiquetas de los ejes
plt.title('Variación Anual de Tasa de Reciclaje para Países del Grupo South')
plt.xlabel('Año')
plt.ylabel('Variación (%)')

# Establecer el eje x para que muestre cada año individualmente
plt.xticks(df_variaciones_south['Año'], rotation=90)

# Mostrar la leyenda
plt.legend()

# Mostrar el gráfico
plt.grid(True)
plt.tight_layout()
plt.show()

# Mostrar el DataFrame con las variaciones anuales para cada tipo de indicador
df_variaciones_south


# Especificar la ruta del archivo Excel
excel_file_path = "../resultados/variaciones_south.xlsx"

# Guardar el DataFrame en un archivo Excel
df_variaciones_south.to_excel(excel_file_path)

print(f"Los resultados se han guardado en el archivo: {excel_file_path}")


'''Guardar tabla resumen sencilla'''
# Seleccionar las columnas relevantes
columnas_relevantes = ['country', 'group', 'TIME_PERIOD', 'tasa reciclaje %', 'PIB per capita', 'residuos_kg_hab']

# Crear el DataFrame con los datos deseados
df_datos_pais = df_eurostat_full[columnas_relevantes]

# Mostrar el DataFrame
df_datos_pais

# Definir la ruta del archivo Excel
ruta_excel = "../resultados/datos_paises.xlsx"

# Guardar el DataFrame en un archivo Excel
df_datos_pais.to_excel(ruta_excel, index=False)

# Confirmar que se ha guardado correctamente
print(f"guardado")


'''Histogramas por grupos geo'''
from bootcampviztools import  plot_grouped_histograms
plot_grouped_histograms(df_eurostat_full,"group","residuos_kg_hab",group_size=10)
plot_grouped_histograms(df_eurostat_full,"group","tasa reciclaje %",group_size=10)



'''Análisis multivariante'''

# Bubble plot
from bootcampviztools import bubble_plot
bubble_plot(df_eurostat_full, "group", "tasa reciclaje %","residuos_kg_hab",scale = 10)



#  Scatterplot

# Obtener los datos para el gráfico
tasa_reciclaje = df_eurostat_full["tasa reciclaje %"]
residuos_kg_hab = df_eurostat_full["residuos_kg_hab"]

# Definir el tamaño del área para las ciudades como el PIB per capita
PIB_per_capita = df_eurostat_full["PIB per capita"]

# Crear el gráfico de dispersión
plt.figure(figsize=(10, 10))
scatter = plt.scatter(tasa_reciclaje, residuos_kg_hab, c=PIB_per_capita, cmap="viridis", s=100, linewidth=0, alpha=0.5)

# Añadir etiquetas y barra de colores
plt.xlabel("Tasa de reciclaje (%)")
plt.ylabel("Residuos por habitante (kg)")
plt.colorbar(label="PIB per cápita")

# Añadir título
plt.title("Relación entre Tasa de Reciclaje, Residuos por Habitante y PIB per cápita")

# Añadir leyenda
plt.legend(*scatter.legend_elements(), title="PIB per cápita")


# Squarify
import squarify

# Agrupar los datos por grupo y calcular la suma de la tasa de reciclaje
datos = df_eurostat_full.groupby('group', as_index=False)['tasa reciclaje %'].sum()

plt.figure(figsize=(10, 8))
squarify.plot(sizes=datos['tasa reciclaje %'], label=datos['group'], alpha=0.6)
plt.title("Tasa de reciclaje por grupo")
plt.axis("off")
plt.show()

# Join plot
sns.jointplot(x=df_eurostat_full["residuos_kg_hab"],
              y=df_eurostat_full["tasa reciclaje %"],
              color='red',
              height=5)


# Pairplot
sns.pairplot(df_eurostat_full)

# Heatmap
matriz_corr = df_eurostat_full.corr(numeric_only= True)

plt.figure(figsize=(5,5))
sns.heatmap(matriz_corr,
            vmin=-1,
            vmax=1,
            cmap=sns.diverging_palette(145, 280, s=85, l=25, n=7),
            square=True,
            linewidths=.1,
            annot=True)




''''Guardar data frame'''
# Especificar la ruta del archivo Excel
excel_file_path = "../data/full.xlsx"

# Guardar el DataFrame en un archivo Excel
df_eurostat_full.to_excel(excel_file_path)

print(f"Los resultados se han guardado en el archivo: {excel_file_path}")

# Mostrar el gráfico
plt.show()
