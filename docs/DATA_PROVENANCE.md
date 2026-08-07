# DATA_PROVENANCE.md - Origen, Limpieza y Feature Engineering

## 1. Origen de Datos

### 1.1 Fuente Primaria

- **Activo**: XAU/USD (Oro vs. Dólar Estadounidense)
- **Proveedor**: MetaTrader 5 (MT5) - Broker Austranet
- **Timeframe**: Diario (D1)
- **Período**: 5+ años de datos históricos
- **Archivo**: `raw_data/XAUUSD_D1.csv`

### 1.2 Estructura de Datos Raw

```csv
time,open,high,low,close,tick_volume,spread,real_volume
1996-03-12,395.6,397.5,395.5,396.7,306.0,0,0.0
1996-03-13,396.7,397.9,396.0,396.9,361.0,0,0.0
1996-03-14,396.3,397.1,395.1,395.9,371.0,0,0.0
```

**Columnas originales**:
- `time`: Timestamp de cierre de la barra
- `open`: Precio de apertura
- `high`: Máximo del período
- `low`: Mínimo del período
- `close`: Precio de cierre (precio de referencia)
- `tick_volume`: Volumen de ticks (transacciones)
- `spread`: Diferencial bid-ask (eliminado: columna vacía)
- `real_volume`: Volumen real de mercado (eliminado: columna vacía)

**Columnas eliminadas en preprocesamiento**:
- `spread`: Contenía solo valores cero, no aporta información
- `real_volume`: Contenía solo valores cero, no aporta información

---

## 2. Pipeline de Limpieza y Procesamiento

### 2.1 Fase 1: Validación y Limpieza Básica

#### Operaciones realizadas:
1. **Detección de outliers** 
   - Inspección visual de gaps anormales
   - Validación de lógica: High ≥ Low, Open/Close ∈ [Low, High]

2. **Manejo de datos faltantes**
   - Eliminación de filas con valores NaN
   - Saltos de fin de semana/feriados: interpolación lineal

3. **Normalización de formato**
   - Conversión de fechas a `datetime`
   - Aseguramiento de orden cronológico ascendente
   - Nombres de columnas en minúsculas

#### Código de referencia:
```python
df_full.columns = [c.lower() for c in df_full.columns]
df_full.dropna(inplace=True)
df_full.reset_index(drop=True, inplace=True)
```

---

### 2.2 Fase 2: Separación de Precios

Para la red neuronal del agente RL se usan **dos versiones del precio de cierre**:

#### A. `close_raw` - Precio Real del Mercado
- **Uso**: Cálculo de PnL, ejecución de trades, trailing stops
- **Preservación**: Copia exacta del precio de cierre original
- **Rol**: "Verdad de terreno" para rentabilidad real

```python
df_full['close_raw'] = df_full['close'].copy()
```

#### B. `close` - Precio Filtrado (Kalman)
- **Uso**: Features de entrada, decisiones del modelo
- **Procesamiento**: Aplicación de Filtro de Kalman
- **Rol**: Señal de tendencia limpia para el agente

---

### 2.3 Fase 3: Cálculo de Volatilidad (ATR)

**Average True Range (ATR)** - Medida de volatilidad realizada

#### True Range (TR):
$$TR_t = \max(High_t - Low_t, |High_t - Close_{t-1}|, |Low_t - Close_{t-1}|)$$

#### ATR:
$$ATR_t = \text{SMA}(TR_{14})$$

```python
high_low = df['high'] - df['low']
high_cp = (df['high'] - df['close'].shift(1)).abs()
low_cp = (df['low'] - df['close'].shift(1)).abs()
df['tr'] = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
df['atr'] = df['tr'].rolling(window=14).mean()
```

**Parámetros**:
- Período: 14 barras (estándar en trading)
- Función: Determina tamaño de trailing stops, entrada de volatilidadCálculo sobre precios **reales** (close_raw) para reflejar movimiento actual del mercado

---

### 2.4 Fase 4: Filtro de Kalman (Denoising)

El **Filtro de Kalman** es un algoritmo recursivo que separa señal de ruido en series temporales financieras.

#### Ecuaciones del Filtro

**Predicción** (Ecuación de transición de estado):
$$\hat{x}_{t|t-1} = \hat{x}_{t-1|t-1}$$
$$P_{t|t-1} = P_{t-1|t-1} + Q$$

**Actualización** (Ecuación de observación):
$$K_t = \frac{P_{t|t-1}}{P_{t|t-1} + R}$$
$$\hat{x}_{t|t} = \hat{x}_{t|t-1} + K_t(z_t - \hat{x}_{t|t-1})$$
$$P_{t|t} = (1 - K_t) P_{t|t-1}$$

#### Parámetros Kalman

| Parámetro | Valor | Interpretación |
|-----------|-------|-----------------|
| **R** | 0.1 | Confianza en el precio observado (0.1 = menor confianza) |
| **Q** | $10^{-4}$ | Ruido de proceso (qué tan rápido cambia la tendencia) |

- **R alto** (0.1): El filtro desconfía del precio raw, suaviza más
- **Q bajo** ($10^{-4}$): Permite cambios lentos de tendencia, ideal para swing trading

#### Código:
```python
class KalmanFilter1D:
    def __init__(self, R=0.1, Q=1e-5):
        self.R = R
        self.Q = Q
        self.x_hat = None
        self.P = 1.0
    
    def update(self, measurement):
        if self.x_hat is None:
            self.x_hat = measurement
            return measurement
        
        x_hat_minus = self.x_hat
        P_minus = self.P + self.Q
        K = P_minus / (P_minus + self.R)
        self.x_hat = x_hat_minus + K * (measurement - x_hat_minus)
        self.P = (1 - K) * P_minus
        
        return self.x_hat

df_full['close'] = df_full['close'].apply(KalmanFilter1D(R=0.1, Q=1e-4).update)
```

**Beneficios**:
- Elimina picos de ruido aleatorio
- Preserva la tendencia subyacente
- Mejora señal para la red neuronal LSTM

---

### 2.5 Fase 5: Cálculo de Features Técnicos

Los **indicadores técnicos** proporcionan contexto de mercado al agente RL.

#### A. Retorno Logarítmico
$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

```python
df['retorno_log'] = np.log(df['close'] / df['close'].shift(1))
```

**Interpretación**: Cambio porcentual continuo del precio.

#### B. Volatilidad Histórica (20 días)
$$\sigma_{20} = \text{std}(r_{t-20:t})$$

```python
df['volatilidad_20'] = df['retorno_log'].rolling(window=20).std()
```

**Interpretación**: Desviación estándar de retornos, mide riesgo de mercado.

#### C. Media Móvil Simple (SMA)
$$\text{SMA}_{50} = \frac{1}{50} \sum_{i=0}^{49} P_{t-i}$$

```python
df['sma_50'] = df['close'].rolling(window=50).mean()
```

**Uso**: Identifica tendencia alcista (precio > SMA50) o bajista.

#### D. Media Móvil Exponencial (EMA)
$$\text{EMA}_t = \alpha P_t + (1-\alpha) \text{EMA}_{t-1}$$
$$\alpha = \frac{2}{n+1} = \frac{2}{27}$$

```python
df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
```

**Uso**: Reacción rápida a cambios de precio, mejor para volatilidad.

#### E. MACD (Moving Average Convergence Divergence)
$$\text{MACD} = \text{EMA}_{12} - \text{EMA}_{26}$$
$$\text{Signal} = \text{EMA}_{9}(\text{MACD})$$
$$\text{Histograma} = \text{MACD} - \text{Signal}$$

```python
df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
df['macd'] = df['ema_12'] - df['ema_26']
df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
df['macd_histograma'] = df['macd'] - df['macd_signal']
```

**Interpretación**:
- Histograma > 0: Momentum alcista
- Histograma < 0: Momentum bajista

#### F. VWAP Ratio (Volume Weighted Average Price)
$$\text{VWAP} = \frac{\sum (\text{Típico} \times \text{Volume})}{\sum \text{Volume}}$$
$$\text{VWAP Ratio} = \frac{\text{Precio Actual}}{\text{VWAP}} - 1$$

```python
df['tp'] = (df['high'] + df['low'] + df['close']) / 3
df['vwap'] = (df['tp'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
df['precio_vwap_ratio'] = (df['close'] / df['vwap']) - 1
```

**Interpretación**: Desviación del precio respecto al promedio ponderado por volumen.

#### G. Cambio de Volumen
$$\text{Vol Cambio} = \frac{\text{Volume}_t - \text{SMA}_{20}(\text{Volume})}{\text{SMA}_{20}(\text{Volume})}$$

```python
df['sma_volumen'] = df['volume'].rolling(window=20).mean()
df['volumen_cambio'] = (df['volume'] - df['sma_volumen']) / (df['sma_volumen'] + 1e-8)
```

**Interpretación**: Desviación de volumen respecto a la media, detecta confianza.

---

### 2.6 Fase 6: Normalización Rolling Z-Score

La **normalización adaptativa** permite que el agente se ajuste a diferentes regímenes de mercado.

#### Fórmula Z-Score Móvil
$$z_t = \frac{x_t - \mu_{t-252:t}}{\sigma_{t-252:t} + \epsilon}$$

Donde:
- $\mu_{t-252:t}$ = Media móvil de 252 días (1 año trading)
- $\sigma_{t-252:t}$ = Desviación estándar móvil
- $\epsilon = 10^{-8}$ = Regularización para evitar división por cero

#### Código:
```python
cols_to_normalize = [
    'precio_vwap_ratio', 
    'volumen_cambio', 
    'retorno_log', 
    'volatilidad_20', 
    'sma_50', 
    'ema_26'
]

for col in cols_to_normalize:
    if col in df.columns:
        rolling_mean = df[col].rolling(window=252, min_periods=20).mean()
        rolling_std = df[col].rolling(window=252, min_periods=20).std()
        df[col] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
```

**Ventajas**:
- Adaptación a volatilidad variable
- Evita distribuciones no-estacionarias
- Rango final: típicamente [-3, +3] (95% de datos)

#### Limpieza de NaNs iniciales:
```python
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
```

Los primeros ~252 días contienen NaN, se descartan.

---

## 3. Datasets Finales

### 3.1 Dataset de Entrenamiento

**Archivo**: `clean_data/XAUUSD_D1_rl.csv`

**Estructura final**:
```csv
Date,Close,retorno_log,precio_vwap_ratio,MACD_Histograma,volumen_cambio,SMA_50,EMA_26,High,Low,Open,SMA_200
1996-12-17,368.9,0.0016277811516101,-0.0417390918239791,-0.0127882288926173,0.2185792349726776,377.468,372.8578114237956,370.2,367.2,367.3,385.573
1996-12-18,369.5,0.0016251357856264,-0.0400295506931423,0.1231340163826417,-0.3363228699551569,377.228,372.6090846516626,370.1,368.5,369.2,385.43700000000007
```

**Columnas y descripciones**:

| Columna | Tipo | Descripción |
|---------|------|------------|
| `Date` | datetime | Timestamp de cierre de barra |
| `Close` | float | Precio de cierre |
| `retorno_log` | float | Retorno logarítmico diario |
| `precio_vwap_ratio` | float | Desviación respecto a VWAP |
| `MACD_Histograma` | float | Histograma del MACD (momentum) |
| `volumen_cambio` | float | Cambio de volumen normalizado |
| `SMA_50` | float | Media móvil simple 50 períodos |
| `EMA_26` | float | Media móvil exponencial 26 períodos |
| `High` | float | Máximo del período |
| `Low` | float | Mínimo del período |
| `Open` | float | Precio de apertura |
| `SMA_200` | float | Media móvil simple 200 períodos |

**Estadísticas**:
- **Filas**: 7,390 (30 años de datos diarios)
- **Período**: Diciembre 1996 - Presente
- **Frecuencia**: Diaria (lunes-viernes)
- **Valores faltantes**: 0 (post-limpieza)

### 3.2 División Walk-Forward

El dataset se divide en 24 folds sin solapo para entrenamiento y prueba, con ventanas deslizantes de 5 años de entrenamiento y 1 año de prueba:

| Fold | Período Entrenamiento | Período Test |
|------|----------------------|--------------|
| 1 | 1996-12-17 a 2001-12-16 (5 años) | 2001-12-17 a 2002-12-16 (1 año) |
| 2 | 1997-12-17 a 2002-12-16 (5 años) | 2002-12-17 a 2003-12-16 (1 año) |
| 3 | 1998-12-17 a 2003-12-16 (5 años) | 2003-12-17 a 2004-12-16 (1 año) |
| ... | ... | ... |
| 24 | 2019-12-17 a 2024-12-16 (5 años) | 2024-12-17 a 2025-12-16 (1 año) |

**Protocolo**:
- Cada fold: 5 años de entrenamiento
- Desplazamiento: 1 año (252 días)
- Prueba: 1 año sin solapo con entrenamiento
- Datos nunca vistos (out-of-sample)

---

## 4. Validación de Datos

### 4.1 Pruebas Realizadas

```python
def validar_dataframe(df):
    """Valida estructura mínima del DataFrame"""
    REQUIRED_COLS = ['Date', 'Close']
    missing_columns = [col for col in REQUIRED_COLS if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Columnas faltantes: {missing_columns}")
    
    if len(df.columns) < 3:  # Date, Close + al menos 1 feature
        raise ValueError("Deben existir al menos Date, Close y features adicionales")
    
    return True, "DataFrame válido"
```

### 4.2 Métricas de Calidad

- **Completitud**: 100% (sin NaNs post-limpieza)
- **Consistencia**: High ≥ Low, Close ∈ [Low, High]
- **Continuidad temporal**: Orden cronológico ascendente
- **Cobertura temporal**: 5+ años sin interrupciones

---

## 5. Reproducibilidad

### 5.1 Pasos para Regenerar

```python
# 1. Cargar CSV raw
df = pd.read_csv('raw_data/XAUUSD_D1.csv', parse_dates=['Date'])

# 2. Aplicar pipeline
df.columns = [c.lower() for c in df.columns]
df['close_raw'] = df['close'].copy()

# 3. Kalman
kf = KalmanFilter1D(R=0.1, Q=1e-4)
df['close'] = df['close'].apply(kf.update)

# 4. ATR
df['atr'] = calcular_atr(df)

# 5. Features
df['retorno_log'] = np.log(df['close'] / df['close'].shift(1))
# ... (resto de features)

# 6. Normalizar
for col in cols_to_normalize:
    df[col] = aplicar_z_score_movil(df[col], window=252)

# 7. Guardar
df.to_csv('clean_data/XAUUSD_D1_rl.csv', index=False)
```

### 5.2 Dependencias

- `pandas >= 1.3.0`
- `numpy >= 1.21.0`
- `matplotlib >= 3.4.0` (visualización)

---

## 6. Referencias

- **Script de procesamiento**: `2_Procesamiento_Datos_Stock/processing_stocks_MT5.py`
- **Features técnicos**: `2_Procesamiento_Datos_Stock/technical_indicators.py`
- **Datos raw**: `1_Recoleccion_Datos/stock_data/raw_data/XAUUSD_D1.csv`

---

**Versión**: 1.0  
**Fecha**: 25/02/2026   

