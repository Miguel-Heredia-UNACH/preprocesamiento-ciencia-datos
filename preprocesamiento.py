import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

#funcion para limpiar datos y dejar claras las columnas.
def limpiar_datos(df):
    df =df.drop_duplicates() # esto elimina filas que estén duplicadas.
    df = df. fillna(df.median( numeric_only=True)) #rellena los valores que falten con las medianas de cada columna
    
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col].astype(str)) 
    
    # con este bucle for hacemos una traducción de variables categóricas a numéricas, así nuestro modelo las usa correctamente.
        
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df) # con esto hacemos una estandarización de datos.
    return df