"""
rl_prepare_data.py

Prepara los archivos procesados por la etapa de feature engineering
(2_Procesamiento_Datos_Stock/processed_data) para que estén listos
para el entrenamiento y evaluación de los agentes RL.

Comportamiento:
- Lee todos los archivos que terminan en "_processed.csv" dentro de
    la carpeta `../2_Procesamiento_Datos_Stock/processed_data`.
- Valida que cada archivo contenga las columnas necesarias.
- Filtra las columnas relevantes para RL y elimina filas con NaNs.
- Guarda los CSV resultantes en la carpeta `clean_data` con sufijo
    `_rl.csv` (ej: XAUUSD_D1_rl.csv).

Supuestos y notas:
- No modifica la lógica ni los valores de las columnas; solo filtra
    y exporta.
- Los nombres de columnas esperados están en `FINAL_COLS`.
- Este script es idempotente: borra `clean_data` antes de recrearla.
"""

import os
import shutil
import pandas as pd

# Estas son las features que validamos
CLEAN_FEATURES = [
    'retorno_log',
    'precio_vwap_ratio',
    'MACD_Histograma',
    'volumen_cambio',
    'SMA_50',
    'EMA_26',
    'High',
    'Low',
    'Open',
    'SMA_200'
    #,'Fed_Rate',
    #'VIX',
    #'US10Y_diff',
    #'VWAP'
]

# Columnas extra que el entorno necesita para calcular recompensas
# y para referencia. NO serán parte del 'estado' (observation).
NECESSARY_COLS = ['Date', 'Close']

# La lista final de columnas que queremos en nuestros archivos listos para RL
FINAL_COLS = NECESSARY_COLS + CLEAN_FEATURES


def procesar_y_filtrar_datos():
    """
    Lee los archivos procesados por la etapa de procesamiento de datos
    y genera archivos listos para RL.

    Pasos realizados:
    1. Busca archivos que terminen en "_processed.csv" en
       ../2_Procesamiento_Datos_Stock/processed_data
    2. Carga cada CSV y valida que contenga todas las columnas listadas
       en `FINAL_COLS`.
    3. Filtra las columnas a `FINAL_COLS` y elimina filas con valores NaN.
    4. Guarda el resultado en `clean_data` con sufijo `_rl.csv`.

    Salida:
        Archivos CSV en la carpeta `clean_data`. No retorna valor.

    Notas:
    - El script borra la carpeta `clean_data` existente para evitar
      restos de ejecuciones previas.
    - Si faltan columnas en un archivo, se omite y se imprime una advertencia.
    - El formato final esperado por los entornos RL es: Date, Close, [features...]
    """
    
    # Ruta del script actual (dentro de 3_Modelado_Reinforcement_Learning)
    ruta_script = os.path.dirname(os.path.abspath(__file__))
    
    # --- Rutas Relativas (más robusto) ---
    # Carpeta de origen (donde están los 25+ features)
    carpeta_origen = os.path.join(ruta_script, "..", "2_Procesamiento_Datos_Stock", "processed_data")
    
    # Carpeta de destino (donde guardaremos los archivos para RL)
    carpeta_destino = os.path.join(ruta_script, "clean_data")

    # Comprobar si la carpeta de origen existe
    if not os.path.exists(carpeta_origen):
        print(f" Error: La carpeta de origen no existe en la ruta esperada: {carpeta_origen}")
        return

    try:
        # Si ya existe en destino, la eliminamos para un inicio limpio
        if os.path.exists(carpeta_destino):
            shutil.rmtree(carpeta_destino)
            print(f"  Carpeta de destino antigua eliminada: {carpeta_destino}")
        
        # Crear la carpeta de destino vacía
        os.makedirs(carpeta_destino)
        print(f" Nueva carpeta de destino creada: {carpeta_destino}")

        print("\n" + "="*70)
        print("INICIANDO FILTRADO DE ARCHIVOS PARA RL")
        print("="*70)
        
        archivos_procesados = 0
        
        # --- Bucle de Filtrado con Renombrado ---
        for archivo in os.listdir(carpeta_origen):
            # Procesar solo los archivos que nos interesan
            if archivo.endswith("_processed.csv"):
                ruta_origen_archivo = os.path.join(carpeta_origen, archivo)
                
                # --- CAMBIO: Generar nuevo nombre de archivo ---
                # Ejemplo: XAUUSD_D1_processed.csv -> XAUUSD_D1_rl_data.csv
                nombre_base = archivo.replace("_processed.csv", "")
                nuevo_nombre = f"{nombre_base}_rl.csv"
                ruta_destino_archivo = os.path.join(carpeta_destino, nuevo_nombre)
                # -----------------------------------------------
                
                try:
                    # 1. Cargar el archivo con todos los features
                    df = pd.read_csv(ruta_origen_archivo)
                    
                    # 2. Validar que todas las columnas necesarias existan
                    columnas_faltantes = [col for col in FINAL_COLS if col not in df.columns]
                    if columnas_faltantes:
                        print(f"  Advertencia: Omitiendo {archivo}")
                        print(f"   Faltan columnas: {columnas_faltantes}")
                        continue
                    
                    # 3. Filtrar el DataFrame
                    df_filtrado = df[FINAL_COLS].copy()
                    
                    # 4. Asegurarnos de no tener NaNs (crítico para RL)
                    filas_antes = len(df_filtrado)
                    df_filtrado.dropna(inplace=True)
                    filas_despues = len(df_filtrado)
                    
                    if filas_antes > filas_despues:
                        print(f"    {archivo}: Se eliminaron {filas_antes - filas_despues} filas con NaN.")

                    df_filtrado.reset_index(drop=True, inplace=True)
                    
                    # 5. Guardar el archivo filtrado con el nuevo nombre
                    df_filtrado.to_csv(ruta_destino_archivo, index=False)
                    print(f" Procesado: {archivo}")
                    print(f"   → Guardado como: {nuevo_nombre}")
                    print(f"   → Filas finales: {len(df_filtrado):,}")
                    archivos_procesados += 1
                    
                except Exception as e:
                    print(f" Error al procesar el archivo {archivo}: {e}")

        print("\n" + "="*70)
        print(f"PROCESO COMPLETADO")
        print("="*70)
        print(f" Archivos procesados: {archivos_procesados}")
        print(f" Datos listos para RL en: {carpeta_destino}")
        print(f" Formato de salida: [TICKER]_rl_data.csv")
        print(f" Features incluidas: {len(CLEAN_FEATURES)} + Date + Close")
        print("="*70)

    except Exception as e:
        print(f" Error general durante el proceso: {e}")

if __name__ == "__main__":
    procesar_y_filtrar_datos()