import os
import pandas as pd
import numpy as np
# Importación de las funciones de indicadores técnicos
from technical_indicators import (
    calcular_sma,
    calcular_ema,
    calcular_rsi,
    calcular_macd,
    calcular_mfi,
    calcular_obv,
    calcular_vwap
)

"""
processing_stocks_MT5.py

Procesa archivos CSV descargados desde MT5 con series temporales OHLCV.

Entrada esperada:
- Archivos CSV en `../1_Recoleccion_Datos/stock_data/raw_data` con columna `time` y columnas OHLC (`open`,`high`,`low`,`close`) y `tick_volume`.

Salida:
- CSV procesados en `processed_data/` con sufijo `_processed.csv` que contienen retornos, indicadores técnicos y features listos para modelado/entrenamiento.

Notas importantes:
- Convención de nombres: se esperan sufijos de timeframe en el nombre de archivo (_D1, _H1, _M15).
- Este script usa funciones de `technical_indicators` (por ejemplo `calcular_macd` devuelve `(macd, signal)`).
"""
# ═══════════════════════════════════════════════════════════════════════════════
#                               CONFIGURACIÓN DE RUTAS
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "..", "1_Recoleccion_Datos", "stock_data", "raw_data")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_data")
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
#                           PARÁMETROS DE INDICADORES
# ═══════════════════════════════════════════════════════════════════════════════

PARAMS = {
    # Ventanas para Media Móvil Simple (SMA)
    'sma_ventanas': [20, 50, 200], 
    # Ventanas para Media Móvil Exponencial (EMA)
    'ema_ventanas': [12, 26], 
    # Ventana para Índice de Fuerza Relativa (RSI)
    'rsi_ventana': 14, 
    # Parámetros (Rápida, Lenta, Señal) para Convergencia/Divergencia de Medias Móviles (MACD)
    'macd_params': (12, 26, 9), 
    # Ventana para Índice de Flujo de Dinero (MFI)
    'mfi_ventana': 14
}

# Las ventanas por defecto fueron elegidas por convención (20/50/200 para SMA,
# 12/26/9 para MACD) — ajustar según el activo y timeframe.
# ═══════════════════════════════════════════════════════════════════════════════
#                           PROCESAMIENTO DE ARCHIVOS MT5
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print(" INICIANDO PROCESAMIENTO DE DATOS MT5")
print("=" * 80)

archivos_procesados = 0
archivos_con_error = 0

# Flujo principal por archivo:
# 1) Cargar y limpiar columnas básicas
# 2) Verificar/filtrar temporalidad para H1/M15 (detectar si el CSV contiene
#    datos con resolución incorrecta) -> filtrar desde la primera fecha correcta
# 3) Calcular retornos e indicadores técnicos
# 4) Generar features adicionales y target (`y_target`)
# 5) Eliminar filas con NaN en columnas críticas y guardar CSV procesado
for archivo in os.listdir(RAW_DATA_DIR):
    if archivo.endswith(".csv") and any(tf in archivo for tf in ["_D1", "_H1", "_M15"]):
        ticker = archivo.replace(".csv", "")
        ruta_archivo = os.path.join(RAW_DATA_DIR, archivo)
        
        try:
            print(f"\n{'─' * 80}")
            print(f" Procesando: {ticker}")
            print(f"{'─' * 80}")

            # ═══════════════════════════════════════════════════════════════════
            #                          CARGA Y LIMPIEZA INICIAL
            # ═══════════════════════════════════════════════════════════════════
            
            df = pd.read_csv(ruta_archivo, parse_dates=["time"])

            # Renombrar columnas
            df.rename(columns={
                "time": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "tick_volume": "Volume"
            }, inplace=True)

            print(f"    Filas iniciales: {len(df)}")

            # Eliminar columnas innecesarias
            if "real_volume" in df.columns:
                df.drop(columns=["real_volume"], inplace=True)
            
            if "spread" in df.columns:
                df.drop(columns=["spread"], inplace=True)

            # Renombrar 'tick_volume' a 'Volume' para consistencia
            df.rename(columns={"tick_volume": "Volume"}, inplace=True)

            # Convertir columnas numéricas
            columnas_numericas = ["Open", "High", "Low", "Close", "Volume"]
            df[columnas_numericas] = df[columnas_numericas].apply(pd.to_numeric, errors='coerce')
            
            # Verificar que tenemos datos válidos
            if df["Close"].isnull().all():
                print(f"    ERROR: No hay datos válidos de precios para {ticker}")
                archivos_con_error += 1
                continue
            
            # Ordenar por fecha
            df.sort_values("Date", inplace=True)
            df.reset_index(drop=True, inplace=True)
            
            # ═══════════════════════════════════════════════════════════════════
            #           FILTRAR DATOS H1/M15 CON TEMPORALIDAD INCORRECTA
            # ═══════════════════════════════════════════════════════════════════
            
            if "_H1" in ticker or "_M15" in ticker:
                print(f"     Verificando temporalidad para {ticker}...")
                
                # Detectar si hay múltiples barras en la misma hora (señal de datos diarios)
                df['hour'] = df['Date'].dt.hour
                df['date_only'] = df['Date'].dt.date
                
                # Contar barras por día
                barras_por_dia = df.groupby('date_only').size()
                
                # Si hay días con menos de 10 barras, probablemente son datos diarios
                # (umbral heurístico — 24 barras sería lo esperado para H1;
                # usamos 10 para ser tolerantes a días con sesiones cortas o datos faltantes)
                dias_incorrectos = barras_por_dia[barras_por_dia < 10]
                
                if len(dias_incorrectos) > 0:
                    primera_fecha_correcta = barras_por_dia[barras_por_dia >= 10].index.min()
                    
                    # `primera_fecha_correcta` es la primera fecha donde hay al menos
                    # 10 barras en el día — a partir de ahí asumimos temporalidad consistente.
                    if pd.notna(primera_fecha_correcta):
                        filas_antes_filtro = len(df)
                        df = df[df['date_only'] >= primera_fecha_correcta].copy()
                        df.reset_index(drop=True, inplace=True)
                        
                        print(f"      Datos con temporalidad incorrecta detectados")
                        print(f"     Filtrado desde: {primera_fecha_correcta}")
                        print(f"     Filas eliminadas: {filas_antes_filtro - len(df)}")
                    else:
                        print(f"      No se pudo determinar fecha correcta, se mantienen todos los datos")
                else:
                    print(f"     Temporalidad correcta desde el inicio")
                
                # Limpiar columnas auxiliares usadas para el chequeo de temporalidad
                df.drop(columns=['hour', 'date_only'], inplace=True)
            else:
                print(f"     Archivo D1 detectado - sin filtrado de temporalidad")
            
            print(f"    Precio inicial: ${df['Close'].iloc[0]:.2f}")
            print(f"    Precio final: ${df['Close'].iloc[-1]:.2f}")

            # ═══════════════════════════════════════════════════════════════════
            #                        CÁLCULO DE RETORNOS
            # ═══════════════════════════════════════════════════════════════════
            
            # Retornos y volatilidad (ventana 20 por convención)
            df["retorno_simple"] = df["Close"].pct_change()
            df["retorno_log"] = np.log(df["Close"] / df["Close"].shift(1))
            df["volatilidad_20"] = df["retorno_log"].rolling(window=20).std()

            # ═══════════════════════════════════════════════════════════════════
            #                    INDICADORES DE TENDENCIA
            # ═══════════════════════════════════════════════════════════════════
            
            print("    Calculando indicadores de tendencia...")
            for ventana in PARAMS['sma_ventanas']:
                df[f"SMA_{ventana}"] = calcular_sma(df, ventana=ventana)
            
            for ventana in PARAMS['ema_ventanas']:
                df[f"EMA_{ventana}"] = calcular_ema(df, ventana=ventana)

            # ═══════════════════════════════════════════════════════════════════
            #                    INDICADORES DE MOMENTUM
            # ═══════════════════════════════════════════════════════════════════
            
            print("    Calculando indicadores de momentum...")
            df["RSI"] = calcular_rsi(df, ventana=PARAMS['rsi_ventana'])
            df["MACD"], df["MACD_Signal"] = calcular_macd(
                df, 
                rapida=PARAMS['macd_params'][0],
                lenta=PARAMS['macd_params'][1],
                signal=PARAMS['macd_params'][2]
            )
            df["MACD_Histograma"] = df["MACD"] - df["MACD_Signal"]

            # ═══════════════════════════════════════════════════════════════════
            #                    INDICADORES DE VOLUMEN
            # ═══════════════════════════════════════════════════════════════════
            
            print("    Calculando indicadores de volumen...")
            df["MFI"] = calcular_mfi(df, ventana=PARAMS['mfi_ventana'])
            df["OBV"] = calcular_obv(df)
            df["VWAP"] = calcular_vwap(df)
            df["precio_vwap_ratio"] = (df["Close"] - df["VWAP"]) / df["VWAP"]

            # ═══════════════════════════════════════════════════════════════════
            #                    FEATURES ADICIONALES PARA RL
            # ═══════════════════════════════════════════════════════════════════
            
            # ═══════════════════════════════════════════════════════════════════
            #                    FEATURES ADICIONALES PARA RL
            # Descripción de columnas generadas:
            # - `SMA_20_50_cruce`: indicador binario si SMA_20 > SMA_50
            # - `SMA_20_50_distancia`: distancia relativa entre SMA_20 y SMA_50
            # - `volumen_cambio`: cambio porcentual del volumen entre barras
            # - `rango_diario`: rango intradía normalizado por precio de cierre
            # - `y_target`: target de entrenamiento = retorno simple del siguiente período
            # (se usa shift(-1) para predecir el siguiente paso temporal)
            df["SMA_20_50_cruce"] = (df["SMA_20"] > df["SMA_50"]).astype(int)
            df["SMA_20_50_distancia"] = (df["SMA_20"] - df["SMA_50"]) / df["SMA_50"]
            df["volumen_cambio"] = df["Volume"].pct_change()
            df["rango_diario"] = (df["High"] - df["Low"]) / df["Close"]

            # Target: retorno del día siguiente
            df["y_target"] = df["retorno_simple"].shift(-1)

            # ═══════════════════════════════════════════════════════════════════
            #                      LIMPIEZA FINAL DE NaN
            # ═══════════════════════════════════════════════════════════════════
            
            # Validamos indicadores técnicos
            # `columnas_criticas` lista las columnas mínimas requeridas para que una
            # fila sea utilizable en modelado. Si falta alguna de estas columnas
            # (NaN) eliminamos la fila porque las features derivadas serán incompletas.
            columnas_criticas = [
                "Close", "retorno_simple", "retorno_log",
                "SMA_20", "SMA_50", "RSI", "MACD", "MACD_Signal",
                "MFI", "OBV", "VWAP"
            ]
            
            filas_antes = len(df)
            df.dropna(subset=columnas_criticas, inplace=True)
            df.reset_index(drop=True, inplace=True)
            filas_despues = len(df)
            
            if filas_antes - filas_despues > 0:
                print(f"    Filas con NaN en indicadores técnicos eliminadas: {filas_antes - filas_despues}")
        
            print(f"    Filas finales: {len(df)}")

            # ═══════════════════════════════════════════════════════════════════
            #                      GUARDAR ARCHIVO PROCESADO
            # ═══════════════════════════════════════════════════════════════════
            
            # Guardado: nombre `{ticker}_processed.csv`. Se exportan todas las
            # columnas actuales; el índice no se guarda (`index=False`).
            ruta_salida = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_processed.csv")
            df.to_csv(ruta_salida, index=False)
            print(f"     Guardado en: {ruta_salida}")

            archivos_procesados += 1

        except Exception as e:
            print(f"    ERROR al procesar {ticker}: {str(e)}")
            archivos_con_error += 1
            continue

# ═══════════════════════════════════════════════════════════════════════════════
#                              RESUMEN FINAL
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print(" PROCESAMIENTO COMPLETADO")
print("=" * 80)
print(f" Archivos procesados exitosamente: {archivos_procesados}")
if archivos_con_error > 0:
    print(f" Archivos con errores: {archivos_con_error}")
print(f" Directorio de salida: {PROCESSED_DATA_DIR}")
print("=" * 80)