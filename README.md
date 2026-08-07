# Trading Algorítmico con Reinforcement Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)
![Stable-Baselines3](https://img.shields.io/badge/SB3-2.7.0-green.svg)

**Agente de trading autónomo usando Recurrent PPO (RPPO) para operar en mercados financieros**

[Características](#características) • [Instalación](#instalación-rápida) • [Uso](#uso) • [Arquitectura](#arquitectura-del-proyecto) • [Resultados](#resultados)

</div>

---

## Descripción del Proyecto

Este proyecto implementa un **agente de trading inteligente** basado en **Reinforcement Learning (RL)** capaz de:

- Aprender estrategias de trading de forma autónoma
- Operar en XAU/USD
- Gestionar riesgo con **Dynamic Position Sizing (DPS)**
- Adaptarse a diferentes condiciones de mercado
- Superar estrategias clásicas (Buy & Hold, Trend Following)

### Algoritmo Principal: Recurrent PPO (RPPO)

- **Memoria LSTM**: Recuerda patrones históricos de mercado
- **Recompensa Multi-Horizonte**: Optimiza ganancias a corto (1 día) y largo plazo (60 días)
- **Filtro de Kalman**: Reduce ruido en precios (método del paper)
- **Rolling Z-Score**: Normalización adaptativa para diferentes regímenes de mercado

---

## Características

### Inteligencia del Agente

| Componente | Descripción |
|------------|-------------|
| **Algoritmo** | Recurrent PPO (LSTM + PPO) |
| **Memoria** | 30 días de contexto histórico |
| **Features RL** | 6 indicadores técnicos efectivos + 4 infraestructurales |
| **Recompensa** | Multi-horizonte (1d, 5d, 20d, 60d) |
| **Gestión de Riesgo** | Dynamic Position Sizing (DPS) basado en confianza |

#### Features Utilizadas (10 totales)

**Features Efectivos (Entrenamiento RL): 6**
- `precio_vwap_ratio` - Ratio entre precio actual y VWAP
- `volumen_cambio` - Cambio porcentual de volumen
- `retorno_log` - Retorno logarítmico diario
- `macd_histograma` - Histograma del MACD (momentum)
- `sma_50` - Media móvil simple de 50 períodos (tendencia corta)
- `ema_26` - Media móvil exponencial de 26 períodos (tendencia exponencial)

Estas features están desfasadas en t-1 con respecto a la data real, por lo que el agente en su entrenamiento nunca posee data leakage.

**Features Infraestructurales (Soporte/Contexto): 4**
- `date` - Timestamp (logs y periodos)
- `close` - Precio filtrado con Kalman (visión IA de tendencia)
- `close_raw` - Precio real del mercado (ejecución de trades y PnL)
- `atr` - Average True Range (gestión de riesgo y volatilidad)

### Métricas de Evaluación

- **CAGR** (Compound Annual Growth Rate)
- **Sharpe Ratio** (retorno ajustado por riesgo)
- **Max Drawdown** (máxima pérdida histórica)
- **Hit Rate** (% de trades ganadores)
- **Profit Factor** (ganancia bruta / pérdida bruta)

### Sistema de Gestión de Riesgo

```python
# El agente ajusta dinámicamente el tamaño de la posición
Confianza Alta (>80%) → Apuesta 50% del capital
Confianza Media (60-80%) → Apuesta 35% del capital
Confianza Baja (<60%) → Apuesta 15% del capital
```

---

## Instalación Rápida

### Requisitos Previos

- Python 3.10 o superior
- GPU NVIDIA (opcional)
- MetaTrader 5 (solo para descarga de datos nuevos)

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/T1Draco/Gold-RL-Austranet.git
cd Gold-RL-Austranet
```

### Paso 2: Crear Entorno Virtual

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

### Paso 3: Instalar Dependencias

#### Opción A: Con GPU NVIDIA

```bash
# Instalar dependencias base
pip install -r requirements.txt

# Instalar PyTorch con CUDA
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Verificar GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

#### Opción B: Sin GPU (Solo CPU)

```bash
pip install -r requirements.txt
```

---

## Uso

### 0. Descargar datos desde MT5

Antes de comenzar con el procesamiento o el entrenamiento, debes obtener los
datos históricos brutos desde MetaTrader 5. El repositorio incluye un script
en `Archivos Python y Data/1_Recoleccion_Datos/stock_data` que se conecta a tu
cuenta MT5 y descarga los bloques de barras en un CSV.

**Requisitos para la descarga:**

1. MT5 instalado y una cuenta (demo o real) configurada.
2. Paquete Python `MetaTrader5` (`pip install MetaTrader5`).
3. Actualiza las credenciales dentro del script o, idealmente, utiliza variables
de entorno (`MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`).

```bash
cd "Archivos Python y Data/1_Recoleccion_Datos/stock_data"
python download_historical_stocks_price_MT5_D1_blocks.py
```

El archivo resultante se guarda en `raw_data/XAUUSD_D1.csv` (columnas OHLCV).

### 1. Preparar Datos para RL

Los datos ya están procesados en `clean_data/`. Si necesitas reprocesar desde cero:

```bash
cd "Archivos Python y Data/3_Modelado_Reinforcement_Learning"
python rl_prepare_data.py
```

**Entrada:** `../2_Procesamiento_Datos_Stock/processed_data/XAUUSD_D1_processed.csv` (25 features)  
**Salida:** `clean_data/XAUUSD_D1_rl.csv` (12 features normalizados)

### 2. Procesamiento y Feature Engineering

Antes de preparar los datos para RL, debes procesar los datos brutos descargados de MT5.
El script calcula 25+ indicadores técnicos y prepara el conjunto de features inicial.

**Requisitos:**
- Datos brutos en `Archivos Python y Data/1_Recoleccion_Datos/stock_data/raw_data/`
- Paquetes Python: `pandas`, `numpy`

```bash
cd "Archivos Python y Data/2_Procesamiento_Datos_Stock"
python processing_stocks_MT5.py
```

**Proceso:**
1. Carga archivos CSV brutos (`XAUUSD_D1.csv`, `XAUUSD_H1.csv`, `XAUUSD_M15.csv`)
2. Calcula retornos (log y simple)
3. Genera indicadores técnicos:
   - **Tendencia:** SMA(20, 50, 200), EMA(12, 26)
   - **Momentum:** RSI, MACD, MFI
   - **Volumen:** OBV, VWAP, precio_vwap_ratio
   - **Volatilidad:** rolling Z-score
4. Crea features adicionales (cruces SMA, cambios de volumen, rango diario)
5. Elimina filas con valores faltantes en columnas críticas
6. Guarda los datos procesados en `processed_data/XAUUSD_D1_processed.csv`

**Salida esperada:**
```
════════════════════════════════════════════════════════════════
 PROCESAMIENTO COMPLETADO
════════════════════════════════════════════════════════════════
 Archivos procesados exitosamente: 3
 Directorio de salida: processed_data/
════════════════════════════════════════════════════════════════
```

**Salida:** `processed_data/XAUUSD_D1_processed.csv` (25 features técnicos)

### 3. Entrenar el Agente (Walk-Forward Validation)

```bash
cd "Archivos Python y Data/3_Modelado_Reinforcement_Learning"
python -m training.train_rppo_trading_walk_forward_validation_D1
```

**Salida esperada:**
```
════════════════════════════════════════════════════════════════
Fold 1/24 - Período: 1996-12-17 a 2001-12-16 (Train) -> 2001-12-17 a 2002-12-16 (Test)
  Datos cargados: 5,230 registros
  [PAPER] Filtro de Kalman aplicado...
  [PAPER] Rolling Z-Score normalizado (window=252)...
  Entrenando 300,000 timesteps...
  ✓ Fold 1 completado - ROI: 18.45% | Balance: $11,845.23 | Trades: 47

Fold 2/24 - Período: 1997-12-17 a 2002-12-16 (Train) -> 2002-12-17 a 2003-12-16 (Test)
  ...
```

**Duración Aproximada:**
- **Con GPU NVIDIA (recomendado):** 5-6 horas para 24 folds
- **Sin GPU (CPU solo):** 24-30 horas para 24 folds
- **Por fold individual:** 12-15 minutos (GPU) o 1 hora (CPU)

### 4. Evaluar Resultados

```bash
# Análisis completo de métricas globales
python -m evaluation.analyze_walk_forward_validation

# Análisis detallado de operaciones individuales
python -m evaluation.analyze_trades_walk_forward_validation
```

Esto genera:
- **Outputs en consola:**
  - Tabla comparativa: RL vs Buy & Hold vs Trend Following
  - Métricas clave: CAGR, Sharpe, Profit Factor, Hit Rate
  - Análisis de reporte de trades
  
- **Archivos PNG:**
  - `board_validacion_rl_vs_benchmarks.png` - Curvas de equidad y drawdowns
  - `board_trades.png` - Distribución de PnL, duraciones, posiciones
  
- **Archivo CSV:**
  - `results/walk_forward/trades_rppo.csv` - Detalle completo de operaciones

---

## Arquitectura del Proyecto

```
Gold-RL-Austranet/
│
├── Archivos Python y Data/               # Core: Datos y Entrenamiento RL
│   │
│   ├── 1_Recoleccion_Datos/              # Descarga de datos brutos
│   │   └── stock_data/
│   │       ├── download_historical_stocks_price_MT5_D1_blocks.py
│   │       └── raw_data/                 # CSVs descargados de MT5 (1996-2025)
│   │           └── XAUUSD_D1.csv         # ~7,500 filas
│   │
│   ├── 2_Procesamiento_Datos_Stock/      # Feature Engineering (25+ indicadores)
│   │   ├── processing_stocks_MT5.py      # Cálculo de indicadores
│   │   ├── technical_indicators.py       # RSI, MACD, MFI, OBV, VWAP, ATR
│   │   ├── IC.py                         # Information Coefficient
│   │   └── processed_data/
│   │       └── XAUUSD_D1_processed.csv   # Dataset con 25+ features
│   │
│   └── 3_Modelado_Reinforcement_Learning/
│       │
│       ├── env/                          # Entorno de Trading (Gymnasium)
│       │   ├── __init__.py
│       │   └── multi_horizon_based_reward_env_v4.py
│       │
│       ├── training/                     # Entrenamiento 24-Fold WF
│       │   ├── __init__.py
│       │   └── train_rppo_trading_walk_forward_validation_D1.py
│       │
│       ├── evaluation/                   # Análisis de Resultados
│       │   ├── analyze_walk_forward_validation.py     # Métricas globales
│       │   └── analyze_trades_walk_forward_validation.py # Análisis por trade
│       │
│       ├── clean_data/                   # Datos procesados para RL
│       │   └── XAUUSD_D1_rl.csv         # 12 features normalizados
│       │
│       ├── results/                      # Salidas de Validación
│       │   └── walk_forward/
│       │       └── trades_rppo.csv       # Historial completo de trades
│       │
│       ├── hyperparameters_optuna.py     # HPO opcional
│       └── rl_prepare_data.py            # Filtro: 25 → 12 features
│
├── frontend_demo/                        # [OPCIONAL] Visualización Online
│   ├── src/
│   │   ├── App.jsx                       # Componente principal
│   │   ├── App.css                       # Estilos
│   │   ├── main.jsx
│   │   ├── index.css
│   │   ├── components/
│   │   │   ├── TradingChart.jsx          # Gráfico con lightweight-charts
│   │   │   └── DashboardStats.jsx        # Métricas y estadísticas
│   │   ├── hooks/
│   │   │   └── useDataLoader.js          # Carga CSV de trades
│   │   └── assets/
│   │
│   ├── public/
│   │   ├── data/
│   │   │   ├── trades_rppo.csv           # CSV de trades
│   │   │   └── XAUUSD_D1_rl.csv          # CSV de precios
│   │   └── index.html
│   │
│   ├── package.json                      # Stack: React 19 + Vite + Lightweight Charts
│   ├── vite.config.js
│   ├── eslint.config.js
│   └── README.md
│
├── docs/                                 # Documentación Técnica
│   ├── README.md                         # Índice central de docs
│   ├── MODEL_SPEC.md                     # Especificación del modelo
│   ├── DATA_PROVENANCE.md                # Pipeline de datos
│   ├── BACKTEST_RESULTS.md               # Resultados de validación
│   ├── USER_MANUAL.md                    # Guía de uso
│   └── FUTURE_WORK_AND_SCALABILITY.md    # Roadmap futuro
│
├── bats/                                 # Scripts batch para Windows
│   └── run_download_historical_stocks_price.bat
│
├── README.md                             # Este archivo
├── requirements.txt                      # Dependencias Python
└── .gitignore
```

### Flujo de Datos y Procesamiento

!

```
┌─────────────────┐
│  MT5 Data       │  (1996-12-17 hasta 2025-12-17)
│  XAUUSD_D1.csv  │  7,390 filas de OHLCV
└────────┬────────┘
         │
         ▼
┌──────────────────────────┐
│  Feature Engineering     │  (Etapa 2)
│  25+ Indicadores:        │
│  - SMA, EMA, MACD, RSI   │  Dataset: processed_data/
│  - ATR, OBV, VWAP, etc  │  12 columnas
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Preparación RL          │  (Etapa 3)
│  Kalman Filter +         │
│  Z-Score Normalization   │  Dataset: clean_data/
│  Feature Selection       │  10 features finales
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Entrenamiento RL        │  (Etapa 4)
│  RecurrentPPO + LSTM     │  24 Folds WF
│  Timesteps: 300K/fold    │  ~5-6 horas (GPU)
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Evaluación              │  (Etapa 5)
│  Walk-Forward Analysis   │  Métricas + Gráficos
│  Generación de Trades    │  CSV de operaciones
└────────┬─────────────────┘
         │
         ├──────────────────────────────────────────┐
         │                                          │
         ▼                                          ▼
┌──────────────────────┐              ┌──────────────────────┐
│  Console Output      │              │  Frontend Demo       │
│  (CLI Dashboard)     │              │  (Browser UI)        │
│  - Tabla de métricas │              │  - Gráficos interact.│
│  - Comparativa RL vs │              │  - Estadísticas      │
│    Benchmarks        │              │  - Análisis de trades│
└──────────────────────┘              └──────────────────────┘
                                              ▲
                                              │
                                       [Opcional]
                                       Ver detalles
                                       en navegador
```

---

## Pipeline del Proyecto

### Etapa 1: Recolección de Datos

```bash
# Descargar datos históricos de MetaTrader 5
cd "Archivos Python y Data/1_Recoleccion_Datos/stock_data"
python download_historical_stocks_price_MT5_D1_blocks.py
```

**Salida:** `raw_data/XAUUSD_D1.csv` con columnas OHLCV

### Etapa 2: Procesamiento y Feature Engineering

```bash
# Calcular indicadores técnicos
cd "Archivos Python y Data/2_Procesamiento_Datos_Stock"
python processing_stocks_MT5.py
```

**Proceso:**
1. Carga datos brutos
2. Calcula 25+ indicadores técnicos:
   - **Tendencia:** SMA(20, 50, 200), EMA(12, 26)
   - **Momentum:** RSI, MACD, MFI
   - **Volumen:** OBV, VWAP
3. Merge con datos macroeconómicos (Fed Rate, US10Y, VIX)
4. Guarda en `processed_data/`

### Etapa 3: Preparación para RL

```bash
# Filtrar features y normalizar
cd "Archivos Python y Data/3_Modelado_Reinforcement_Learning"
python rl_prepare_data.py
```

**Proceso:**
1. Filtra 10 features finales de los 25+ indicadores (selección manual):
   - `retorno_log`
   - `precio_vwap_ratio`
   - `MACD_Histograma`
   - `volumen_cambio`
   - `SMA_50`
   - `EMA_26`
   - `High`
   - `Low`
   - `Open`
   - `SMA_200`
2. Aplica **Filtro de Kalman** para denoising
3. Aplica **Rolling Z-Score** para normalización adaptativa
4. Guarda en `clean_data/`

### Etapa 4: Entrenamiento

```bash
python -m training.train_rppo_trading_walk_forward_validation_D1
```

**Período de Datos Disponibles:** 17 de Diciembre de 1996 - 17 de Diciembre de 2025

**Validación Walk-Forward (24 Folds):**
```
Período Total: 1996-12-17 hasta 2025-12-17 (29 años de datos)

Ejemplos de ventanas:
[Ventana 1:  Train: 1996-12-17 a 2001-12-16 | Test: 2001-12-17 a 2002-12-16]
[Ventana 2:  Train: 1997-12-17 a 2002-12-16 | Test: 2002-12-17 a 2003-12-16]
[Ventana 3:  Train: 1998-12-17 a 2003-12-16 | Test: 2003-12-17 a 2004-12-16]
...
[Ventana 24: Train: 2019-12-17 a 2024-12-16 | Test: 2024-12-17 a 2025-12-16]

Cada fold:
- Entrenamiento: 5 años (1,260 días de trading)
- Testing: 1 año (252 días de trading)
- Step rolling: 1 año (252 días)
```

**Duración del Entrenamiento:**
- Con GPU NVIDIA: ~5-6 horas (todos los 24 folds)
- Sin GPU: ~24-30 horas (recomendado usar GPU)

**Configuración del Agente:**
- **Algoritmo:** RecurrentPPO (LSTM + PPO)
- **Memoria:** 30 días de contexto histórico
- **Timesteps por Fold:** 300,000
- **Learning Rate:** 1.04e-05
- **Gamma:** 0.99
- **Entropía:** 0.0116
- **Período de Datos:** 1996-2025 (29 años)

### Etapa 5: Evaluación y Análisis

```bash
# Análisis completo de validación walk-forward
python -m evaluation.analyze_walk_forward_validation

# Análisis detallado de operaciones individuales
python -m evaluation.analyze_trades_walk_forward_validation
```

**Archivos Generados:**
- `board_validacion_rl_vs_benchmarks.png` - Dashboard con curvas de equidad y drawdowns
- `board_trades.png` - Distribución de PnL, duraciones, posiciones
- Impresión en consola de tabla comparativa de métricas

**Análisis Incluidos:**
1. **Curvas de Equidad**: RL vs Buy & Hold vs Trend Following
2. **Drawdown Histórico**: Visualización de riesgo por estrategia
3. **Distribución de PnL**: Análisis de trades ganadores/perdedores
4. **Duraciones de Trades**: Histograma de velas por operación
5. **Métricas Clave**: CAGR, Sharpe, Calmar Ratio, Profit Factor, Hit Rate

---

## Visualización Online (Frontend Demo - Opcional)

Si prefieres analizar los resultados de forma **interactiva en el navegador** en lugar de solo gráficos estáticos, el proyecto incluye una aplicación React completamente funcional.

**Requisito importante:** Esta sección requiere que Node.js y `npm`/`yarn` estén instalados en tu sistema. Consulta la sección `FRONTEND (Node.js)` en `requirements.txt` para instrucciones de instalación y versiones recomendadas.

### Instalación y Ejecución

```bash
cd frontend_demo

# Instalar dependencias
npm install

# Ejecutar servidor de desarrollo
npm run dev
```

**Acceso:** `http://localhost:5173` (se abre automáticamente)

### Tecnología del Frontend

| Componente | Versión | Función |
|-----------|---------|---------|
| **React** | 19.2 | Framework UI |
| **Vite** | 7.2 | Build tool (desarrollo rápido) |
| **Lightweight-Charts** | 5.1 | Gráficos financieros profesionales |
| **PapaParse** | 5.5 | Parsing de archivos CSV |

### Flujo de Datos (Frontend)

```
┌─────────────────────────────────────────────────┐
│  Frontend Demo (Navegador)                      │
│                                                 │
│  1. Carga CSV desde public/data/:               │
│     • trades_rppo.csv (operaciones realizadas)  │
│     • XAUUSD_D1_rl.csv (datos históricos)       │
│                                                 │
│  2. Renderiza Gráficos Interactivos:            │
│     • Candlestick con operaciones overlay       │
│     • Equidad acumulada                         │
│     • Drawdown máximo                           │
│                                                 │
│  3. Estadísticas Dinámicas (clickeable):        │
│     • ROI, Sharpe, Win Rate, Profit Factor      │
│     • Operaciones filtradas por rango           │
│     • Zoom temporal y análisis de períodos      │
└─────────────────────────────────────────────────┘
```

### Archivos Necesarios

Los datos se leen automáticamente de:

```
frontend_demo/
├── public/
│   └── data/
│       ├── trades_rppo.csv         ← Genera entrenamiento  
│       └── XAUUSD_D1_rl.csv        ← Prepara RL data
└── src/
    ├── App.jsx
    └── components/
        ├── TradingChart.jsx        ← Candlestick + trades
        └── DashboardStats.jsx      ← Métricas
```

**Nota:** Los archivos en `public/data/` se actualizan automáticamente después de entrenar (`train_rppo_...`) y evaluar (`analyze_trades_...`).

### Alternativa: Visualización por Consola (100% Python)

Si no deseas instalar Node.js, todos los análisis se ejecutan directamente en Python con salida en consola + PNG:

```bash
python -m evaluation.analyze_walk_forward_validation
python -m evaluation.analyze_trades_walk_forward_validation
```

**Diferencias:**
- CLI: Salida en texto + imágenes PNG estáticas
- Web: Gráficos interactivos en tiempo real (zoom, hover, selección de rangos)

---

## Configuración del Entorno de Trading

### Parámetros del Entorno

```python
# env/multi_horizon_based_reward_env_v4.py

INITIAL_BALANCE = 10000       # Capital inicial
TRANSACTION_COST = 0.0001     # Comisión por operación (0.01%)
LEVERAGE = 5.0                # Apalancamiento máximo (test)
SHORT_ENABLED = False         # Solo operaciones LONG
WINDOW_SIZE = 30              # Días de memoria
```

### Sistema de Recompensa Multi-Horizonte

```python
reward = (
    0.20 * reward_1d +    # Corto plazo (día a día)
    0.25 * reward_5d +    # Medio plazo (semanal)
    0.30 * reward_20d +   # Largo plazo (mensual)
    0.25 * reward_60d     # Muy largo plazo (trimestral)
)
```

### Penalizaciones

```python
# 1. Drawdown exponencial (evita pérdidas grandes)
drawdown_penalty = 100.0 * (current_dd_pct ** 2)

# 2. Overtrading (evita operaciones frecuentes)
if steps_since_last_change < 20:
    overtrading_penalty = -0.10 * (20 - steps_since_last_change) / 20

# 3. Holding bonus (premia posiciones ganadoras largas)
if unrealized_pnl > 0 and steps_in_position >= 30:
    holding_bonus = 0.15
```

---

## Resultados Típicos

### Métricas Esperadas (Walk-Forward Validation)

| Métrica | Objetivo | Agente RL | Buy & Hold | Trend Following |
|---------|----------|-----------|------------|-----------------|
| **CAGR** | > 8.0% | 12-18% | 8-12% | 5-10% |
| **Sharpe Ratio** | > 1.2 | 1.2-1.8 | 0.8-1.2 | 0.6-1.0 |
| **Max Drawdown** | < 20% | 12-18% | 20-30% | 15-25% |
| **Profit Factor** | > 1.2 | 1.5-2.0 | N/A | 1.1-1.4 |
| **Hit Rate** | > 50% | 55-65% | N/A | 45-55% |

---

## Tecnologías Utilizadas

### Core

- **Python 3.10+**: Lenguaje principal
- **PyTorch 2.5.1**: Framework de Deep Learning
- **Stable-Baselines3 2.7.0**: Algoritmos de RL
- **SB3-Contrib 2.7.0**: RecurrentPPO
- **Gymnasium 1.2.2**: API de entornos de RL

### Análisis de Datos

- **Pandas 2.2.3**: Manipulación de datos
- **NumPy 2.2.4**: Computación numérica
- **SciPy 1.15.2**: Algoritmos científicos

### Visualización

- **Matplotlib 3.10.1**: Gráficos
- **Seaborn 0.13.2**: Visualización estadística

### Frontend (Visualización Online)

- **Node.js 18+**: Entorno de JavaScript para el servidor local del frontend
- **React 19.2**: Biblioteca UI principal
- **Vite 7.2**: Bundler/development server rápido
- **Lightweight-Charts 5.1**: Gráficos financieros interactivos
- **PapaParse 5.5**: Parsing de CSV en el navegador

### Optimización

- **Optuna 4.6.0**: Hyperparameter tuning (opcional)

### Datos

- **MetaTrader 5 5.0.4510**: Descarga de datos de forex/commodities

---

## Metodología Científica

### Walk-Forward Validation

```
Train Window: 5 años (1,260 días)
Test Window: 1 año (252 días)
Step Size: 1 año (252 días)

[════════Train════════][Test]
    [════════Train════════][Test]
        [════════Train════════][Test]
            [════════Train════════][Test]
```

**Ventajas:**
- Evita overfitting
- Simula trading en tiempo real
- Validación robusta en datos no vistos

### Feature Engineering (Paper-Based)

**1. Filtro de Kalman (Denoising):**
```python
# Reduce ruido en precios sin lag
Q = 1e-4  # Ruido del proceso
R = 0.1   # Ruido de medición
```

**2. Rolling Z-Score (Normalización Adaptativa):**
```python
# Se adapta a diferentes regímenes de mercado
window = 252  # 1 año de trading
z_score = (x - rolling_mean) / rolling_std
```

**3. Feature Selection:**
- De 25+ indicadores → 10 features finales cuidadosamente seleccionadas
- Lista: retorno_log, precio_vwap_ratio, MACD_Histograma, volumen_cambio, SMA_50,
  EMA_26, High, Low, Open, SMA_200
- Baja correlación entre ellas
- Validación empírica durante fases previas

---

## Troubleshooting

### Error: "CUDA out of memory"

**Solución:**
```python
# En train_rppo_trading_walk_forward_validation_D1.py
# Reducir batch_size:
batch_size=64  # En lugar de 128
```

### Error: "ModuleNotFoundError: No module named 'env'"

**Solución:**
```bash
# Asegúrate de ejecutar como módulo desde la carpeta correcta
cd "Archivos Python y Data/3_Modelado_Reinforcement_Learning"
python -m training.train_rppo_trading_walk_forward_validation_D1
```

### Error: "DataFrame too short"

**Causa:** Datos insuficientes después del filtrado

**Solución:**
```bash
# Verificar que el archivo tiene suficientes filas
python -c "import pandas as pd; df = pd.read_csv('clean_data/XAUUSD_D1_processed.csv'); print(f'Filas: {len(df)}')"

# Debe tener al menos 1,500 filas
```

### Warning: "FutureWarning: fillna with 'method' is deprecated"

**Solución:** Actualizar el código de procesamiento:
```python
# Reemplazar:
df['VIX'] = df['VIX'].fillna(method='ffill')

# Por:
df['VIX'] = df['VIX'].ffill()
```

---

## Documentación Técnica

Este proyecto incluye documentación completa en la carpeta `/docs/`:

| Documento | Público | Duración | Propósito |
|-----------|---------|----------|-----------|
| [MODEL_SPEC.md](/docs/MODEL_SPEC.md) | Investigadores, ML Engineers | 30 min | Especificación técnica del agente RL, arquitectura LSTM, reward shaping |
| [DATA_PROVENANCE.md](/docs/DATA_PROVENANCE.md) | Data Scientists, Traders | 20 min | Pipeline completo: descarga → limpieza → feature engineering → normalización |
| [BACKTEST_RESULTS.md](/docs/BACKTEST_RESULTS.md) | Traders, PM | 15 min | Resultados de validación (24 folds), métricas vs benchmarks, análisis de drawdown |
| [USER_MANUAL.md](/docs/USER_MANUAL.md) | Todos | 25 min | Guía paso-a-paso, instalación, entrenamiento, troubleshooting |
| [FUTURE_WORK_AND_SCALABILITY.md](/docs/FUTURE_WORK_AND_SCALABILITY.md) | Product Managers, Architects | 30 min | Roadmap: MT5, macro variables, multi-asset, transformer models |
| [INDEX (docs/README.md)](/docs/README.md) | Todos | 5 min | Hub central con rutas de lectura recomendadas por rol |

### Rutas Recomendadas de Lectura

**Por Rol:**

- **Traders**: [USER_MANUAL.md](/docs/USER_MANUAL.md) (25m) → [BACKTEST_RESULTS.md](/docs/BACKTEST_RESULTS.md) (15m)  
- **ML Engineers**: [MODEL_SPEC.md](/docs/MODEL_SPEC.md) (30m) → [DATA_PROVENANCE.md](/docs/DATA_PROVENANCE.md) (20m)  
- **Data Scientists**: [DATA_PROVENANCE.md](/docs/DATA_PROVENANCE.md) (20m) → [MODEL_SPEC.md](/docs/MODEL_SPEC.md) (30m)  
- **DevOps / Deployment**: [USER_MANUAL.md](/docs/USER_MANUAL.md) (25m) → [FUTURE_WORK_AND_SCALABILITY.md](/docs/FUTURE_WORK_AND_SCALABILITY.md) (30m)  
- **Directivos/PM**: [BACKTEST_RESULTS.md](/docs/BACKTEST_RESULTS.md) (15m) → [FUTURE_WORK_AND_SCALABILITY.md](/docs/FUTURE_WORK_AND_SCALABILITY.md) (30m)

**Comprensión Completa:** Todo en orden (2-3 horas total)

---

## Roadmap

### Versión 1.0 (Actual)
- [x] Implementación de RPPO con LSTM
- [x] Dynamic Position Sizing (DPS)
- [x] Walk-Forward Validation (24 años, 24 folds)
- [x] Análisis comparativo vs benchmarks
- [x] Documentación técnica completa

### Próxima Fase: Deployment (Q2-Q3 2026)
- [ ] Integración con MetaTrader 5
- [ ] API REST para predicciones en tiempo real
- [ ] Exportación de parámetros a formato MT5
- [ ] Variables macroeconómicas (tasas, VIX, DXY)
- [ ] Expert Advisor (EA) en MQL5

### Fase Avanzada (Q4 2026 - Q2 2027)
- [ ] Arquitectura Transformer + attention mechanism
- [ ] Multi-agent ensemble (especializado por activo)
- [ ] Multi-timeframe trading (D1, H4, H1, M15)
- [ ] Soporte para otros activos (Silver, Oil, Copper)
- [ ] Modelo versioning con MLOps (MLFlow, DVC)

### Producción (Q3 2027+)
- [ ] Transformer-based architecture con mejor generalización
- [ ] Meta-learning para adaptación rápida a cambios de régimen
- [ ] Trading en tiempo real con APIs de brokers
- [ ] Sistema de alertas y monitoreo 24/7
- [ ] Auditoría de robustez y stress testing

---

## Ejemplo de Salida

### Consola (Python Output)

Al ejecutar `analyze_walk_forward_validation.py`, verás una tabla muy parecida a la que genera el script:

```
┌─────────────────────────────────────────────────────────────────┐
│ MÉTRICA         │  RL AGENT  │  BUY & HOLD  │  TREND FOLLOW  │
├─────────────────────────────────────────────────────────────────┤
│ CAGR            │   12.47%   │    8.92%     │     5.34%      │
│ Sharpe          │    1.82    │    0.94      │     0.65       │
│ Max Drawdown    │  -18.34%   │   -38.92%    │    -64.21%     │
│ Calmar Ratio    │    1.56    │    0.87      │     0.42       │
│ Profit Factor   │    2.34    │     -        │     1.12       │
│ Hit Rate        │   54.32%   │     -        │    48.21%      │
└─────────────────────────────────────────────────────────────────┘

Gráficos generados:
- `board_validacion_rl_vs_benchmarks.png` (curvas de equidad y drawdown)

```

---

---

## Autores

**Gold-RL-Austranet**
- Martín P.
- Maria Q.
- Francisco A.

Universidad Católica del Norte, Coquimbo, Chile

---

## Licencia

Copyright (c) 2026 Gold-RL-Austranet - Universidad Católica del Norte

**Código Propietario - Todos los derechos reservados.**

Este software es propiedad de Gold-RL-Austranet y su uso está restringido exclusivamente al cliente Academis.

Contacto: martin.puebla.rivera@gmail.com

---

## Disclaimer

Este software es una herramienta de investigación y desarrollo. **NO constituye asesoramiento financiero ni recomendación de inversión**.

### Advertencias importantes:

- **Riesgo de pérdida de capital**: El trading algorítmico puede resultar en pérdidas significativas
- **No garantías**: Los resultados pasados no garantizan rendimientos futuros
- **Uso bajo responsabilidad propia**: El usuario asume todos los riesgos asociados
- **Requiere supervisión**: Este sistema debe ser monitoreado y recalibrado dado un número apropiado de años

**Gold-RL-Austranet no asume responsabilidad por pérdidas derivadas del uso de este software.**

Consulte a un asesor financiero profesional certificado antes de implementar cualquier estrategia de trading.

---

## Referencias

1. Deng, Y., Bao, F., Kong, Y., and Ren, Z. (2016). "Deep direct reinforcement learning for financial signal representation and trading." IEEE Transactions on Neural Networks and Learning Systems, 28(3):1–12.

2. Kili, A., Raouyane, B., Rachdi, M., and Bellafkih, M. (2025). "Kalman-Enhanced Deep Reinforcement Learning for Noise-Resilient Algorithmic Trading in Volatile Gold Markets." International Journal of Advanced Computer Science and Applications (IJACSA), 16(11).

3. Théate, T. and Ernst, D. (2021). "An application of deep reinforcement learning to algorithmic trading." Expert Systems with Applications, 173:114632.

---

## Contacto

Para consultas técnicas, soporte o licenciamiento:

- **Email:** martin.puebla.rivera@gmail.com
- **Institución:** Universidad Católica del Norte, Chile

**Nota:** Este es un repositorio privado. El acceso está restringido a personal autorizado.

---

<div align="center">

**Desarrollado por Gold-RL-Austranet**

</div>
