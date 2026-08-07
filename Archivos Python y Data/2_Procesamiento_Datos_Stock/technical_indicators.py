import pandas as pd

"""
╔═══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                           COMPARATIVA DE INDICADORES TÉCNICOS COMPLETA                               ║
╠════════════╦════════════════════╦═════════════════════════════════════════════════════════════════════╣
║ Indicador  ║ Tipo               ║ Señales Típicas                                                     ║
╠════════════╬════════════════════╬═════════════════════════════════════════════════════════════════════╣
║ SMA        ║ Tendencia          ║ Cruce de SMA rápida con lenta                                       ║
║            ║                    ║ — SMA corta > SMA larga → compra                                    ║
║            ║                    ║ — SMA corta < SMA larga → venta                                     ║
╠════════════╬════════════════════╬═════════════════════════════════════════════════════════════════════╣
║ RSI        ║ Momentum           ║ Sobrecompra / Sobreventa                                            ║
║            ║                    ║ — RSI > 70 → sobrecomprado → venta                                  ║
║            ║                    ║ — RSI < 30 → sobrevendido → compra                                  ║
║            ║                    ║ — RSI cruzando 50 puede marcar tendencia                             ║
╠════════════╬════════════════════╬═════════════════════════════════════════════════════════════════════╣
║ MACD       ║ Momentum+Tendencia ║ Cruces de MACD con línea de señal                                   ║
║            ║                    ║ — MACD > Señal → impulso alcista                                    ║
║            ║                    ║ — MACD < Señal → impulso bajista                                    ║
║            ║                    ║ — MACD cruza eje 0 → cambio de tendencia                            ║
╠════════════╬════════════════════╬═════════════════════════════════════════════════════════════════════╣
║ MFI        ║ Momentum+Volumen   ║ Similar a RSI pero con volumen                                      ║
║            ║                    ║ — MFI > 80 → sobrecomprado con volumen                              ║
║            ║                    ║ — MFI < 20 → sobrevendido con volumen                               ║
║            ║                    ║ — Divergencias más confiables que RSI                               ║
╠════════════╬════════════════════╬═════════════════════════════════════════════════════════════════════╣
║ OBV        ║ Volumen Acumulado  ║ Confirmación de tendencias                                          ║
║            ║                    ║ — OBV ↑ + Precio ↑ → tendencia fuerte                               ║
║            ║                    ║ — Divergencias OBV-Precio → reversión                               ║
╠════════════╬════════════════════╬═════════════════════════════════════════════════════════════════════╣
║ VWAP       ║ Precio+Volumen     ║ Nivel de valor justo del día                                        ║
║            ║                    ║ — Precio > VWAP → compras dominan                                   ║
║            ║                    ║ — Precio < VWAP → ventas dominan                                    ║
╠════════════╬════════════════════╬═════════════════════════════════════════════════════════════════════╣
║ Aplicación ║ Todos los activos  ║ Combinables entre sí para mayor robustez                            ║
║            ║ (stocks, crypto,   ║ Ideal para RL: diversidad de señales                                ║
║            ║ forex, commodities)║ Sin alta correlación entre indicadores                              ║
╚════════════╩════════════════════╩═════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════════════════
#                           INDICADORES DE TENDENCIA
# ═══════════════════════════════════════════════════════════════════════════════

def calcular_sma(df, columna='Close', ventana=20):
    """
    Media Móvil Simple (Simple Moving Average)
    
    Parámetros:
        df: DataFrame con datos OHLCV
        columna: Columna sobre la que calcular (default: 'Close')
        ventana: Período de la media móvil (default: 20)
    
    Retorna:
        Series con valores de SMA
    """
    return df[columna].rolling(window=ventana).mean()


def calcular_ema(df, columna='Close', ventana=20):
    """
    Media Móvil Exponencial (Exponential Moving Average)
    
    Parámetros:
        df: DataFrame con datos OHLCV
        columna: Columna sobre la que calcular (default: 'Close')
        ventana: Período de la media móvil (default: 20)
    
    Retorna:
        Series con valores de EMA
    """
    return df[columna].ewm(span=ventana, adjust=False).mean()


# ═══════════════════════════════════════════════════════════════════════════════
#                           INDICADORES DE MOMENTUM
# ═══════════════════════════════════════════════════════════════════════════════

def calcular_rsi(df, columna='Close', ventana=14):
    """
    Relative Strength Index (Índice de Fuerza Relativa)
    
    Mide la velocidad y magnitud de los cambios de precio.
    
    Parámetros:
        df: DataFrame con datos OHLCV
        columna: Columna sobre la que calcular (default: 'Close')
        ventana: Período del RSI (default: 14)
    
    Retorna:
        Series con valores de RSI (0-100)
        
    Señales:
        RSI > 70: Sobrecomprado (posible venta)
        RSI < 30: Sobrevendido (posible compra)
    """
    delta = df[columna].diff()
    ganancia = delta.clip(lower=0)
    perdida = -delta.clip(upper=0)
    media_ganancia = ganancia.rolling(window=ventana).mean()
    media_perdida = perdida.rolling(window=ventana).mean()
    rs = media_ganancia / media_perdida
    return 100 - (100 / (1 + rs))


def calcular_macd(df, columna='Close', rapida=12, lenta=26, signal=9):
    """
    Moving Average Convergence Divergence
    
    Diferencia entre dos medias móviles exponenciales (rápida y lenta).
    
    Parámetros:
        df: DataFrame con datos OHLCV
        columna: Columna sobre la que calcular (default: 'Close')
        rapida: Período EMA rápida (default: 12)
        lenta: Período EMA lenta (default: 26)
        signal: Período de la línea de señal (default: 9)
    
    Retorna:
        Tupla (macd, signal_line)
        
    Señales:
        MACD > Signal: Impulso alcista
        MACD < Signal: Impulso bajista
        MACD cruza 0: Cambio de tendencia
    """
    ema_rapida = df[columna].ewm(span=rapida, adjust=False).mean()
    ema_lenta = df[columna].ewm(span=lenta, adjust=False).mean()
    macd = ema_rapida - ema_lenta
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


# ═══════════════════════════════════════════════════════════════════════════════
#                      INDICADORES DE VOLUMEN (MOMENTUM + VOLUMEN)
# ═══════════════════════════════════════════════════════════════════════════════

def calcular_mfi(df, ventana=14):
    """
    Money Flow Index - RSI con volumen incorporado
    
    Similar al RSI pero considera el volumen de transacciones.
    Más robusto para detectar reversiones.
    
    Parámetros:
        df: DataFrame con datos OHLCV (requiere High, Low, Close, Volume)
        ventana: Período del MFI (default: 14)
    
    Retorna:
        Series con valores de MFI (0-100)
        
    Señales:
        MFI > 80: Sobrecomprado con volumen alto (señal fuerte de venta)
        MFI < 20: Sobrevendido con volumen alto (señal fuerte de compra)
        Divergencias precio-MFI: Señal de reversión
    """
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    
    # Separar flujos positivos y negativos
    positive_flow = pd.Series(0.0, index=df.index)
    negative_flow = pd.Series(0.0, index=df.index)
    
    price_diff = typical_price.diff()
    positive_flow[price_diff > 0] = money_flow[price_diff > 0]
    negative_flow[price_diff < 0] = money_flow[price_diff < 0]
    
    # Calcular ratio
    positive_mf = positive_flow.rolling(window=ventana).sum()
    negative_mf = negative_flow.rolling(window=ventana).sum()
    
    # Evitar división por cero
    mfi = 100 - (100 / (1 + positive_mf / negative_mf.replace(0, 1e-10)))
    return mfi


def calcular_obv(df):
    """
    On-Balance Volume - Volumen acumulado direccional
    
    Acumula volumen basado en la dirección del precio.
    Útil para confirmar tendencias.
    
    Parámetros:
        df: DataFrame con datos OHLCV (requiere Close, Volume)
    
    Retorna:
        Series con valores de OBV
        
    Señales:
        OBV ascendente + Precio ascendente: Tendencia alcista fuerte
        OBV descendente + Precio descendente: Tendencia bajista fuerte
        Divergencias OBV-Precio: Posible reversión
    """
    obv = pd.Series(0.0, index=df.index)
    obv.iloc[0] = df['Volume'].iloc[0]
    
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + df['Volume'].iloc[i]
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - df['Volume'].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv


def calcular_vwap(df):
    """
    Volume Weighted Average Price - Precio promedio ponderado por volumen
    
    Precio promedio del día ponderado por el volumen de cada transacción.
    Usado como referencia de valor justo.
    
    Parámetros:
        df: DataFrame con datos OHLCV (requiere High, Low, Close, Volume)
    
    Retorna:
        Series con valores de VWAP
        
    Señales:
        Precio > VWAP: Compras dominantes (momentum alcista)
        Precio < VWAP: Ventas dominantes (momentum bajista)
    """
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    return (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()


# ═══════════════════════════════════════════════════════════════════════════════
#                      FUNCIÓN AUXILIAR - CALCULAR TODOS
# ═══════════════════════════════════════════════════════════════════════════════

def calcular_todos_indicadores(df, sma_ventanas=[20, 50], rsi_ventana=14, 
                                macd_params=(12, 26, 9), mfi_ventana=14):
    """
    Calcula todos los indicadores técnicos sobre un DataFrame.
    
    Parámetros:
        df: DataFrame con datos OHLCV (requiere: Open, High, Low, Close, Volume)
        sma_ventanas: Lista de ventanas para SMA (default: [20, 50])
        rsi_ventana: Ventana para RSI (default: 14)
        macd_params: Tupla (rapida, lenta, signal) para MACD (default: 12, 26, 9)
        mfi_ventana: Ventana para MFI (default: 14)
    
    Retorna:
        DataFrame original con columnas adicionales de indicadores
        
    Columnas añadidas:
        - SMA_20, SMA_50: Medias móviles simples
        - RSI: Índice de fuerza relativa
        - MACD, MACD_Signal: MACD y su línea de señal
        - MFI: Money Flow Index
        - OBV: On-Balance Volume
        - VWAP: Volume Weighted Average Price
    """
    df_copy = df.copy()
    
    # Indicadores de tendencia
    for ventana in sma_ventanas:
        df_copy[f'SMA_{ventana}'] = calcular_sma(df_copy, ventana=ventana)
    
    # Indicadores de momentum
    df_copy['RSI'] = calcular_rsi(df_copy, ventana=rsi_ventana)
    df_copy['MACD'], df_copy['MACD_Signal'] = calcular_macd(
        df_copy, 
        rapida=macd_params[0], 
        lenta=macd_params[1], 
        signal=macd_params[2]
    )
    
    # Indicadores de volumen
    df_copy['MFI'] = calcular_mfi(df_copy, ventana=mfi_ventana)
    df_copy['OBV'] = calcular_obv(df_copy)
    df_copy['VWAP'] = calcular_vwap(df_copy)
    
    return df_copy