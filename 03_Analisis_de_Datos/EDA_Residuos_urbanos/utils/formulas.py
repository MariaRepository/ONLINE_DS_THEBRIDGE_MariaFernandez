import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re  # Utilizado para cambiar columna obj a num

# CAMBIAR COLUMNA OBJ A NUM:
def cambiar_a_numero(df, columna):
    def limpiar_valor(valor):
        if isinstance(valor, str):
            # Reemplaza los puntos por nada, para eliminar los separadores de miles
            valor = valor.replace('.', '')
            # Elimina todos los caracteres que no sean dígitos
            valor = re.sub(r'\D', '', valor)
        return valor

    df[columna] = pd.to_numeric(df[columna].apply(limpiar_valor), errors="coerce")
    return df


# INDICE INTERCUARTILICO
def get_IQR(df,col):
    return df[col].quantile(0.75) -  df[col].quantile(0.25)

# COEFICIENTE VARIACION
def variabilidad(df,col):
    df_var=df.describe().loc[["std","mean"]].T
    df_var[col]=df_var["std"]/df_var["mean"]
    return df_var