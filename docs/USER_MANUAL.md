# USER_MANUAL.md - Guía de Uso del Agente de RL para Trading

## Tabla de Contenidos

1. [Instalación y Configuración](#instalación-y-configuración)
2. [Preparar Datos](#preparar-datos)
3. [Ejecutar el Agente](#ejecutar-el-agente)
4. [Cambiar Parámetros](#cambiar-parámetros)
5. [Entender los Logs](#entender-los-logs)
6. [Archivos de Salida](#archivos-de-salida)
7. [Troubleshooting](#troubleshooting)
8. [Ejemplos Prácticos](#ejemplos-prácticos)
9. [Contacto y Soporte](#contacto-y-soporte)

---

## Instalación y Configuración

### 1.1 Requisitos del Sistema

```bash
Python >= 3.10
RAM >= 4 GB
Espacio en disco >= 1 GB (para datos + modelos)
```

### 1.2 Instalación de Dependencias

```bash
# 1. Navegar al directorio del proyecto
cd Gold-RL-Austranet

# 2. Crear entorno virtual (recomendado)
python -m venv venv

# 3. Activar entorno virtual
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate

# 4. Instalar dependencias
pip install -r requirements.txt
```

### 1.3 Verificar Instalación

```bash
# Verificar que todas las librerías se instalaron correctamente
python -c "import stable_baselines3; import gymnasium; import pandas; import torch; print('Todas las dependencias instaladas')"
```

### 1.4 Estructura del Proyecto

```
Gold-RL-Austranet/
├── docs/                          # Documentación
│   ├── MODEL_SPEC.md              # Especificación del modelo
│   ├── DATA_PROVENANCE.md         # Origen y limpieza de datos
│   ├── BACKTEST_RESULTS.md        # Resultados de backtesting
│   └── USER_MANUAL.md             # Este archivo
│
├── Archivos Python y Data/
│   ├── 1_Recoleccion_Datos/       # Descarga de datos raw
│   ├── 2_Procesamiento_Datos_Stock/  # Limpieza y features
│   └── 3_Modelado_Reinforcement_Learning/
│       ├── env/                   # Environment de trading
│       ├── training/              # Scripts de entrenamiento
│       ├── evaluation/            # Scripts de evaluación
│       ├── clean_data/            # Datos procesados
│       └── results/               # Salida de backtest
│
├── frontend_demo/                 # Dashboard (opcional)
├── requirements.txt               # Dependencias
└── README.md
```

---

## Ejecutar el Agente

### 2.0 Preparar Datos (Requerido por primera vez)

Antes de entrenar, debes descargar y procesar los datos históricos.

#### 2.0.1 Descargar Datos desde MetaTrader 5

```bash
# Navegar al directorio de descarga
cd "Archivos Python y Data/1_Recoleccion_Datos/stock_data"

# Ejecutar el script de descarga
python download_historical_stocks_price_MT5_D1_blocks.py
```

**Requisitos:**
- MetaTrader 5 instalado y una cuenta (demo o real) configurada
- Librería `MetaTrader5`: `pip install MetaTrader5`
- Variables de entorno configuradas: `MT5_LOGIN`, `MT5_PASSWORD` y `MT5_SERVER`

En PowerShell, configúralas solo para la sesión actual antes de ejecutar:

```powershell
$env:MT5_LOGIN = "tu_login"
$env:MT5_PASSWORD = "tu_password"
$env:MT5_SERVER = "tu_servidor"
```

**Output esperado:**
```
════════════════════════════════════════════════════════════════
   Conectado a la cuenta #<MT5_LOGIN>
════════════════════════════════════════════════════════════════

Descargando histórico de XAUUSD (86400) hacia atrás...
════════════════════════════════════════════════════════════════
   → 5000 barras obtenidas | Total acumulado: 5,000
   → 5000 barras obtenidas | Total acumulado: 10,000
   → 3500 barras obtenidas | Total acumulado: 13,500
Fin del historial alcanzado (tamaño de bloque menor al CHUNK_SIZE).
════════════════════════════════════════════════════════════════
DESCARGA COMPLETA: 13,500 barras
Desde: 1996-12-17 00:00:00
Hasta: 2025-02-21 00:00:00
════════════════════════════════════════════════════════════════
```

**Salida:** `raw_data/XAUUSD_D1.csv`

#### 2.0.2 Procesar Datos y Calcular Indicadores Técnicos

```bash
# Navegar al directorio de procesamiento
cd "Archivos Python y Data/2_Procesamiento_Datos_Stock"

# Ejecutar el script de procesamiento
python processing_stocks_MT5.py
```

**Requisitos:**
- Datos brutos descargados en paso 2.0.1
- Librerías: `pandas`, `numpy`

**Qué hace:**
1. Lee los archivos raw (`XAUUSD_D1.csv`, `XAUUSD_H1.csv`, `XAUUSD_M15.csv`)
2. Calcula 25+ indicadores técnicos:
   - **Tendencia:** SMA(20, 50, 200), EMA(12, 26)
   - **Momentum:** RSI, MACD, MFI
   - **Volumen:** OBV, VWAP
   - **Volatilidad:** Rolling Z-Score, ATR
3. Crea features adicionales (cruces SMA, cambios de volumen, rangos)
4. Elimina filas con datos faltantes
5. Guarda archivos procesados con 25+ features

**Nota:** De estos 25+ indicadores calculados, el entrenamiento RL utiliza **10 features finales** (6 efectivos + 4 infraestructurales) seleccionados automáticamente.

**Output esperado:**
```
════════════════════════════════════════════════════════════════
 INICIANDO PROCESAMIENTO DE DATOS MT5
════════════════════════════════════════════════════════════════

────────────────────────────────────────────────────────────────
 Procesando: XAUUSD_D1
────────────────────────────────────────────────────────────────
    Filas iniciales: 7,500
    Archivo D1 detectado - sin filtrado de temporalidad
    Precio inicial: $395.20
    Precio final: $2,085.50
    Calculando indicadores de tendencia...
    Calculando indicadores de momentum...
    Calculando indicadores de volumen...
    Filas finales: 7,450
     Guardado en: processed_data/XAUUSD_D1_processed.csv

════════════════════════════════════════════════════════════════
 PROCESAMIENTO COMPLETADO
════════════════════════════════════════════════════════════════
 Archivos procesados exitosamente: 1
 Directorio de salida: processed_data/
════════════════════════════════════════════════════════════════
```

**Salida:** `processed_data/XAUUSD_D1_processed.csv` (25+ features)

### 2.1 Ejecución Completa (Entrenamiento + Backtesting)

Este es el flujo principal que entrena el agente con Walk-Forward Validation.

```bash
# Navegar al directorio de training
cd "Archivos Python y Data/3_Modelado_Reinforcement_Learning"

# Ejecutar el script de entrenamiento
python training/train_rppo_trading_walk_forward_validation_D1.py
```

**Qué hace**:
1. Carga datos limpios desde `clean_data/XAUUSD_D1_rl.csv` (25+ features disponibles)
2. Filtra a **10 features finales** (6 efectivos + 4 infraestructurales):
   - **Efectivos:** precio_vwap_ratio, volumen_cambio, retorno_log, macd_histograma, sma_50, ema_26
   - **Infraestructurales:** date, close (Kalman), close_raw, atr
3. Aplica Filtro de Kalman al precio y normalización Rolling Z-Score (window=252)
4. Ejecuta 24 folds de Walk-Forward Validation (período 1996-2025)
5. Para cada fold:
   - Entrena RecurrentPPO en 5 años de datos con 6 features efectivos
   - Prueba en 1 año de datos nunca vistos (out-of-sample)
   - Registra todas las operaciones
6. Guarda trades en `results/walk_forward/trades_rppo.csv`

**Tiempo de ejecución**: ~5-6 horas (depende del CPU, 24 folds)

**Output esperado (Cifras de referencia)**:
```
INICIANDO VALIDACIÓN WALK-FORWARD
   Ventana de Entrenamiento: 1260 días
   Ventana de Observación:   30 días
================================================================================

ITERACIÓN 1: Probando en 2023-01-01 -> 2023-12-31
   Entrenando con 1260 días...
   [CONFIG] Entrenando PPO con PARÁMETROS FINALES (ÓPTIMOS)...
   Resultado Fold: ROI 8.23% | Balance: $10823.00 | Trades: 28

ITERACIÓN 2: Probando en 2023-01-01 -> 2023-12-31
   Entrenando con 1260 días...
   Resultado Fold: ROI 6.41% | Balance: $11516.00 | Trades: 31

...

VALIDACIÓN WALK-FORWARD COMPLETADA
================================================================================
Balance Inicial: $10000.00
Balance Final:   $18543.00
ROI Acumulado:   85.43%
Total Trades:    147
Hit Rate Global: 60.17%
Detalle guardado en: results/walk_forward/trades_rppo.csv
```

### 2.2 Ejecución Rápida (Solo Prueba)

Si solo se quiere probar sin entrenar, para analizar resultados de los trades de un modelo anterior:

```bash
# Ejecutar evaluación en datos de test
python evaluation/analyze_walk_forward_validation.py
```

**Requisito**: Debes haber ejecutado entrenamiento antes (genera `trades_rppo.csv`)

### 2.3 Ejecución en Segundo Plano

Para ejecutar sin bloquear la terminal:

```bash
# En Windows PowerShell:
Start-Process python -ArgumentList "training/train_rppo_trading_walk_forward_validation_D1.py" -NoNewWindow

# En bash (Linux/macOS):
nohup python training/train_rppo_trading_walk_forward_validation_D1.py > training.log 2>&1 &
```

---

## Cambiar Parámetros

### 3.1 Parámetros Principales

Los parámetros se configuran en el script de entrenamiento. Abre:

```
Archivos Python y Data/3_Modelado_Reinforcement_Learning/training/train_rppo_trading_walk_forward_validation_D1.py
```

**Sección: CONFIGURACIÓN DEL EXPERIMENTO**

```python
# ═══════════════════════════════════════════════════════════════════════════════
#                        CONFIGURACIÓN DEL EXPERIMENTO
# ═══════════════════════════════════════════════════════════════════════════════

RANDOM_SEED = 42                    # Semilla para reproducibilidad
DATA_PATH = os.path.join(...)       # Ruta a datos limpios

# CONFIGURACIÓN WALK-FORWARD
TRAIN_WINDOW_SIZE = 252 * 5         # 5 años de entrenamiento
TEST_WINDOW_SIZE = 252 * 1          # 1 año de prueba
STEP_SIZE = 252 * 1                 # Desplazamiento de 1 año
WINDOW_SIZE_OBS = 30                # Ventana de features para LSTM

TOTAL_TIMESTEPS_PER_FOLD = 300_000  # Timesteps de entrenamiento
INITIAL_BALANCE = 10000             # Capital inicial
```

### 3.2 Parámetros del Entorno de Trading

En la sección donde se crea el entorno:

```python
env_train = crear_entorno_discreto(
    df=df_train,
    initial_balance=INITIAL_BALANCE,      # Capital en USD
    transaction_cost=0.0005,               # Comisión (0.05%)
    leverage=3.0,                         # Apalancamiento
    short_enabled=False,                  # Permitir cortos
    eval_mode=False,                      # Mezclar datos (train)
    position_pct=0.25,                    # % base posición (legacy)
    window_size=WINDOW_SIZE_OBS,          # Ventana temporal
    scaling_params=None,                  # Parámetros norm externos
    normalize_internal=False              # Usa datos pre-normalizados
)
```

**Parámetros explicados**:

| Parámetro | Valor Default | Rango Recomendado | Descripción |
|-----------|---------------|-------------------|------------|
| `transaction_cost` | 0.0005 | 0.0001-0.001 | Comisión como % (0.05%) |
| `leverage` | 3.0 | 1.0-5.0 | Multiplicador de exposición |
| `short_enabled` | False | True/False | ¿Permitir posiciones cortas? |
| `window_size` | 30 | 10-60 | Memoria temporal del LSTM |
| `position_pct` | 0.25 | 0.1-0.5 | % capital por operación (legacy) |

### 3.3 Parámetros del Algoritmo PPO

```python
model = RecurrentPPO( 
    "MlpLstmPolicy",
    env_train,
    verbose=0,
    
    # Parámetros PPO
    learning_rate=1.0379252144886538e-05,  # Velocidad de aprendizaje
    ent_coef=0.011618001142365253,          # Coeficiente de entropía
    gamma=0.99,                              # Descuento de recompensas
    n_steps=1024,                            # Steps antes de actualización
    batch_size=128,                          # Tamaño de minibatch
    
    policy_kwargs=dict(
        net_arch=[256, 128, 64],            # Arquitectura MLP
        lstm_hidden_size=64,                 # Tamaño de LSTM
        n_lstm_layers=1,                     # Número de capas LSTM
    ),
    
    device="cpu",                            # "cpu" o "cuda" si se tiene GPU
    seed=RANDOM_SEED
)
```

**Parámetros avanzados**:

| Parámetro | Valor Default | Efecto de aumentar |
|-----------|---------------|-------------------|
| `learning_rate` | 1e-5 | Convergencia más rápida (pero menos estable) |
| `ent_coef` | 0.0116 | Más exploración (menos explotación) |
| `gamma` | 0.99 | Valora más recompensas futuras |
| `lstm_hidden_size` | 64 | Memoria más grande (computación más lenta) |
| `batch_size` | 128 | Gradientes más suaves (menos ruido) |

### 3.4 Cambios Típicos y Sus Efectos

#### A. Aumentar Rendimiento (Más Riesgo)

```python
# Aumentar leverage
leverage=5.0,                           # De 3x a 5x

# Más entrenamiento
TOTAL_TIMESTEPS_PER_FOLD = 500_000,    # De 300k a 500k

# Menos conservador
ent_coef=0.02,                         # De 0.0116 a 0.02
```

**Resultado**: Mayor CAGR, pero drawdown potencialmente mayor.

#### B. Reducir Riesgo (Menos Rendimiento)

```python
# Reducir leverage
leverage=1.5,                          # De 3x a 1.5x

# Ventana de observación más grande
window_size=60,                        # De 30 a 60 (más memoria)

# Mayor conservadurismo
learning_rate=5e-6,                    # De 1e-5 a 5e-6
```

**Resultado**: Drawdown menor, CAGR también menor.

#### C. Entrenar Más Rápido (Menos Datos)

```python
# Reducir tamaño de ventanas
TRAIN_WINDOW_SIZE = 252 * 2,           # De 5 años a 2 años
TEST_WINDOW_SIZE = 252 * 0.5,          # De 1 año a 6 meses

# Timesteps menores
TOTAL_TIMESTEPS_PER_FOLD = 100_000,    # De 300k a 100k
```

**Resultado**: Ejecución 5x más rápida, pero menos datos para aprender.

---

## Entender los Logs

### 4.1 Estructura de Logs

El script produce logs en la consola y en archivos. Ejemplo:

```
[2026-02-25 10:30:45] INFO: Datos cargados: 2018 registros 
[2026-02-25 10:30:46] INFO: Aplicando Filtro de Kalman (Denoising)...
[2026-02-25 10:30:47] INFO: Aplicando Rolling Z-Score (Window 252)...
[2026-02-25 10:31:02] INFO: INICIANDO VALIDACIÓN WALK-FORWARD

ITERACIÓN 1: Probando en 2023-01-01 -> 2023-12-31
   Entrenando con 1260 días...
   [CONFIG] Entrenando PPO con PARÁMETROS FINALES (ÓPTIMOS)...
   
   Resultado Fold: ROI 8.23% | Balance: $10823.00 | Trades: 28
```

### 4.2 Interpretación de Métricas en Logs

```
ROI 8.23%           → Retorno en ese fold (8.23% de ganancia)
Balance: $10823.00  → Capital final del fold
Trades: 28          → Número de operaciones cerradas
```

### 4.3 Alertas y Errores Comunes

#### Error: "ModuleNotFoundError: No module named 'env'"

```
Solución:
1. Asegúrate de estar en el directorio correcto
2. Ejecuta: python -c "import sys; print(sys.path)"
3. Verifica que 3_Modelado_Reinforcement_Learning está en el PATH
```

#### Error: "CUDA out of memory"

```
Solución:
1. Cambia device a CPU
   device="cpu"  # En lugar de "cuda"

2. Reduce tamaño de batch
   batch_size=64  # De 128 a 64

3. Reduce LSTM
   lstm_hidden_size=32  # De 64 a 32
```

#### Warning: "NaN en observación"

```
Causa: Features mal normalizadas o división por cero
Solución:
1. Verifica que normalize_internal=False
2. Asegúrate que datos están en clean_data/XAUUSD_D1_rl.csv
3. Regenera datos: python 2_Procesamiento_Datos_Stock/processing_stocks_MT5.py
```

---

## Archivos de Salida

### 5.1 Archivo Principal: trades_rppo.csv

**Ubicación**: `results/walk_forward/trades_rppo.csv`

**Estructura**:

```csv
entry_step,exit_step,entry_date,exit_date,type,entry_price,exit_price,position_size,notional_size,pnl_usd,pnl_pct,fee,net_pnl
100,108,2023-01-15,2023-01-23,LONG,1951.50,1958.20,500.00,1500.00,335.00,67.00,4.50,330.50
108,115,2023-01-23,2023-01-30,LONG,1958.20,1962.45,450.00,1350.00,190.00,42.22,4.05,185.95
...
```

**Columnas explicadas**:

| Columna | Tipo | Descripción |
|---------|------|------------|
| `entry_step` | int | Paso (barra) de entrada |
| `exit_step` | int | Paso de salida |
| `entry_date` | datetime | Fecha de entrada |
| `exit_date` | datetime | Fecha de salida |
| `type` | str | "LONG", "SHORT", "NEUTRAL" |
| `entry_price` | float | Precio de entrada (USD/oz) |
| `exit_price` | float | Precio de salida (USD/oz) |
| `position_size` | float | Margen usado (USD) |
| `notional_size` | float | Exposición total con leverage (USD) |
| `pnl_usd` | float | Ganancia/pérdida bruta (USD) |
| `pnl_pct` | float | Ganancia/pérdida % sobre margen |
| `fee` | float | Comisión total (USD) |
| `net_pnl` | float | PnL neto después de comisión |

### 5.2 Análisis de Trades

Para analizar los trades generados:

```python
import pandas as pd

df_trades = pd.read_csv('results/walk_forward/trades_rppo.csv')

# Estadísticas básicas
print(f"Total trades: {len(df_trades)}")
print(f"Trades ganadores: {len(df_trades[df_trades['net_pnl'] > 0])}")
print(f"Hit rate: {len(df_trades[df_trades['net_pnl'] > 0]) / len(df_trades):.2%}")

# PnL total
print(f"PnL total: ${df_trades['net_pnl'].sum():.2f}")
print(f"Ganancia promedio: ${df_trades[df_trades['net_pnl'] > 0]['net_pnl'].mean():.2f}")
print(f"Pérdida promedio: ${df_trades[df_trades['net_pnl'] < 0]['net_pnl'].mean():.2f}")

# Profit factor
ganancias = df_trades[df_trades['net_pnl'] > 0]['net_pnl'].sum()
pérdidas = abs(df_trades[df_trades['net_pnl'] < 0]['net_pnl'].sum())
print(f"Profit factor: {ganancias / pérdidas:.2f}")
```

### 5.3 Archivos Auxiliares

```
Archivos Python y Data/3_Modelado_Reinforcement_Learning/
├── results/
│   └── walk_forward/
│       ├── trades_rppo.csv              # Trades principales
│       └── portfolio_history.csv        # Historial de patrimonio (si se genera)
│
└── models/
    └── rppo_fold_*.zip                  # Modelos entrenados (sb3 format)
```

---

## Troubleshooting

### 6.1 El script tarda mucho

```
Causa: Está entrenando correctamente, tarda ~30 minutos - 1 hr. por fold

Solución (si es DEMASIADO lento):
1. Reducir TOTAL_TIMESTEPS_PER_FOLD a 100_000 o menos
2. Reducir batch_size a 64
```

### 6.2 Resultados muy variables entre ejecuciones

```
Causa: Falta de semilla fija

Solución:
1. Asegúrate que RANDOM_SEED = 42 está definido
2. Verifica que set_global_seed(RANDOM_SEED) se llama
3. No uses datos nuevos entre ejecuciones
```

### 6.3 Memoria insuficiente (RAM)

```
Síntomas: Proceso se "cuelga" o sale con error de memoria

Soluciones:
1. Reducir tamaño de batch: batch_size=32
2. Reducir lstm_hidden_size: lstm_hidden_size=32
3. Usar walk-forward con ventanas más pequeñas
4. Ejecutar en máquina con >8GB RAM
```

### 6.4 GPU CUDA errors

```
Error: "CUDA out of memory" o "CUDA runtime error"

Solución:
1. Cambia device a CPU:
   device="cpu"
   
2. Si quieres usar GPU:
   - Verifica: nvidia-smi (debe mostrar GPU disponible)
   - Reduce batch_size y lstm_hidden_size
   - Reinstala: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 6.5 Drawdown muy alto / Resultados negativos

```
Posibles causas:
1. Parámetros muy agresivos (leverage=5.0, learning_rate muy alto)
2. Período de test con mercado en contra

Soluciones:
1. Reduce leverage a 1.5
2. Aumenta ent_coef a 0.02 (más exploración)
3. Regenera datos y reintenta
```

---

## Ejemplos Prácticos

### 7.1 Entrenar con Diferentes Configuraciones

#### Ejemplo 1: Configuración Conservadora (Bajo Riesgo)

```python
# Editar train_rppo_trading_walk_forward_validation_D1.py

RANDOM_SEED = 42
TRAIN_WINDOW_SIZE = 252 * 2        # 2 años (menos datos = menos riesgo)
TEST_WINDOW_SIZE = 252 * 0.5       # 6 meses
TOTAL_TIMESTEPS_PER_FOLD = 200_000

# En crear_entorno_discreto:
env_train = crear_entorno_discreto(
    df=df_train,
    initial_balance=10000,
    transaction_cost=0.001,         # Comisión más alta
    leverage=1.5,                   # Bajo apalancamiento
    window_size=30,
    ...
)

# En RecurrentPPO:
model = RecurrentPPO(
    ...,
    learning_rate=5e-6,             # Aprendizaje más lento
    ent_coef=0.01,                  # Menos exploración
    ...
    policy_kwargs=dict(
        lstm_hidden_size=32,        # Red más pequeña
        ...
    )
)

# Ejecutar
python training/train_rppo_trading_walk_forward_validation_D1.py
```

**Resultado esperado**: CAGR ~5%, Max DD ~15%, Sharpe ~1.2

#### Ejemplo 2: Configuración Agresiva (Alto Rendimiento)

```python
TRAIN_WINDOW_SIZE = 252 * 5        # 5 años
TOTAL_TIMESTEPS_PER_FOLD = 500_000 # Más entrenamiento

env_train = crear_entorno_discreto(
    ...,
    transaction_cost=0.0001,        # Comisión baja
    leverage=5.0,                   # Alto apalancamiento
    ...
)

model = RecurrentPPO(
    ...,
    learning_rate=2e-5,             # Aprendizaje más rápido
    ent_coef=0.02,                  # Más exploración
    ...
    policy_kwargs=dict(
        lstm_hidden_size=128,       # Red más grande
        net_arch=[512, 256, 128],
        ...
    )
)
```

**Resultado esperado**: CAGR ~18%, Max DD ~50%, Sharpe ~2.8

---

## Contacto y Soporte

Para problemas específicos:

1. **Consultar DATA_PROVENANCE.md**: Para entender los datos
2. **Revisar MODEL_SPEC.md**: Para entender el modelo
3. **Consultar BACKTEST_RESULTS.md**: Para resultados esperados
4. **Ejecutar análisis**: `python evaluation/analyze_trades_walk_forward_validation.py` para estadísticas y visualizaciones de trades

---

**Versión**: 1.0  
**Fecha**: 25/02/2026
