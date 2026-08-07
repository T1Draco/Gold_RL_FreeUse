# MODEL_SPEC.md - Especificación del Agente de RL para Trading de Oro

## 1. Descripción General

**Gold-RL-Austranet** es un agente de Reinforcement Learning basado en **RecurrentPPO con LSTM** entrenado para operar en el mercado de XAU/USD (Oro) con frecuencia diaria (D1).

- **Algoritmo**: RecurrentPPO (PPO con memoria LSTM)
- **Framework**: Stable-Baselines3 (sb3_contrib)
- **Activo**: XAU/USD (Oro)
- **Timeframe**: Diario (D1)
- **Período de Entrenamiento**: Walk-Forward Validation (24 folds, período 1996-2025)
- **Estrategia**: Swing Trading discreto con gestión dinámica de riesgo

---

## 2. Espacios de Acción y Observación

### 2.1 Espacio de Acciones (Discreto)

El agente puede ejecutar 3 acciones discretas:

| Acción | ID | Descripción |
|--------|----|----|
| **LONG** | 2 | Abre/mantiene posición alcista |
| **NEUTRAL** | 1 | Cierra posición o no opera |
| **SHORT** | 0 | Abre/mantiene posición bajista (deshabilitada en versión actual) |

**Nota**: Actualmente `short_enabled=False`, por lo que el agente solo puede ir LONG o NEUTRAL.

### 2.2 Espacio de Observación

La observación completa consta de **dos secciones**:

#### A. Estado del Mercado (Ventana Temporal de Features)
- **Tamaño de ventana**: 30 días históricos
- **10 Features totales** (6 efectivos + 4 infraestructurales):

** Features Efectivos para Entrenamiento ML (6):**
  1. `precio_vwap_ratio`: Ratio entre precio actual y VWAP
  2. `volumen_cambio`: Cambio relativo de volumen
  3. `retorno_log`: Retorno logarítmico diario
  4. `macd_histograma`: Histograma del MACD (momentum)
  5. `sma_50`: Media móvil simple de 50 períodos (tendencia corta)
  6. `ema_26`: Media móvil exponencial de 26 períodos (tendencia exponencial)

** Features Infraestructurales (4):**
  7. `date`: Timestamp (necesario para logs y periodos)
  8. `close`: Precio filtrado con Kalman (visión IA para tendencia)
  9. `close_raw`: Precio real del mercado (ejecución de trades)
  10. `atr`: Average True Range (volatilidad y gestión de riesgo)

#### B. Estado de la Cuenta
```
[pos_short, pos_neutral, pos_long, entry_price_norm, unrealized_pnl_norm, net_worth_norm, cash_ratio]
```
- **pos_short/neutral/long**: One-hot encoding de la posición actual
- **entry_price_norm**: Precio de entrada normalizado (0 si no hay posición)
- **unrealized_pnl_norm**: PnL no realizado normalizado por capital inicial
- **net_worth_norm**: Patrimonio neto en escala logarítmica
- **cash_ratio**: Proporción de efectivo disponible

**Tamaño total de observación**: (6 features efectivos × 30 ventana) + 7 variables = 187 dimensiones
*(Los 4 features infraestructurales se usan como contexto pero no se envían a la red como features)*

**Normalización** (solo features efectivos):
- **Rolling Z-Score**: Media/std móviles de 252 días
- **Filtro de Kalman**: Aplicado a `close` para denoising de tendencia
- **Rango final**: [-10.0, 10.0] (clipped para estabilidad)

---

## 3. Sistema de Recompensas Multi-Horizonte

El sistema de recompensas balancea el rendimiento en múltiples timeframes:

### 3.1 Recompensas Base (por horizonte temporal)

$$\text{reward}_{1d} = 100 \times \log\left(\frac{NW_t}{NW_{t-1}}\right)$$

$$\text{reward}_{5d} = 30 \times \log\left(\frac{NW_t}{NW_{t-5}}\right)$$

$$\text{reward}_{20d} = 15 \times \log\left(\frac{NW_t}{NW_{t-20}}\right)$$

$$\text{reward}_{60d} = 8 \times \log\left(\frac{NW_t}{NW_{t-60}}\right)$$

Donde $NW_t$ = Net Worth (patrimonio neto) en el tiempo $t$.

### 3.2 Recompensa Combinada

$$\text{reward} = 0.10 \times r_{1d} + 0.35 \times r_{5d} + 0.40 \times r_{20d} + 0.15 \times r_{60d}$$

**Ponderación**:
- **10%** corto plazo (1 día): volatilidad intradiaria
- **35%** mediano plazo (5 días): tactical swing trades
- **40%** largo plazo (20 días): tendencias principales
- **15%** muy largo plazo (60 días): dirección general

### 3.3 Penalizaciones

#### A. Penalización por Drawdown
```
if drawdown_pct > 5%:
    penalty = -0.1 × log(drawdown_pct / 5%)
```
Evita que el agente acumule pérdidas excesivas.

#### B. Penalización por Overtrading
```
if action_changed < 10 steps:
    penalty = -0.05
```
Desalienta cambios de posición demasiado frecuentes.

#### C. Bonificación por Tendencia
```
if position == LONG and price > entry_price:
    bonus = +0.02
```
Recompensa mantener posiciones rentables.

---

## 4. Mecánica de Trading

### 4.1 Gestión de Posiciones

#### Apertura de Posición
- **Dynamic Position Sizing (DPS)**: Tamaño basado en confianza del modelo
  - Confianza baja (0.5): 10% del balance → margen requerido
  - Confianza alta (1.0): 75% del balance → margen requerido

- **Fórmula DPS**:
  $$\text{margen} = \text{balance} \times \left(0.10 + 0.65 \times \frac{\text{confidence} - 0.5}{0.5}\right)$$

#### Precio de Ejecución
- **Entrada**: Precio filtrado por Kalman (close)
- **Salida**: Precio raw del mercado (close_raw)
- **PnL Real**: Calculado con close_raw para consistencia

#### Cierre de Posición
- **Cierre manual**: Acción NEUTRAL
- **Trailing Stop**: Para posiciones LONG
  - Stop = `highest_price_since_entry - (3.0 × ATR)`
  - Se activa automáticamente si el precio cae demasiado

### 4.2 Costos Operacionales

| Parámetro | Valor | Descripción |
|-----------|-------|------------|
| **Transaction Cost** | 0.01% (0.0001) | Comisión por apertura/cierre |
| **Apalancamiento (Leverage)** | 3.0x | Exposición = margen × 3 |
| **Capital Inicial** | $10,000 USD | Balance de partida |

---

## 5. Arquitectura Neural

### 5.1 Policy Network (RecurrentPPO)

```
Input (217 dims)
    ↓
MLP Encoder: [256] → ReLU
    ↓
         [128] → ReLU
    ↓
         [64] → ReLU
    ↓
LSTM Layer: 64 hidden units, 1 layer
    ↓
Policy Head: 3 outputs (softmax)
    ↓
Actions: [LONG, NEUTRAL, SHORT]
```

### 5.2 Value Network

```
Input (217 dims)
    ↓
MLP: [256] → ReLU
    ↓
    [128] → ReLU
    ↓
    [64] → ReLU
    ↓
Value Head: 1 output (scalar)
    ↓
State Value Estimate
```

### 5.3 Parámetros de Entrenamiento

| Parámetro | Valor |
|-----------|-------|
| **Algoritmo** | RecurrentPPO (sb3_contrib) |
| **Learning Rate** | $1.04 \times 10^{-5}$ |
| **Entropy Coefficient** | 0.0116 |
| **Discount Factor (γ)** | 0.99 |
| **N Steps (Rollouts)** | 1,024 |
| **Batch Size** | 128 |
| **Timesteps per Fold** | 300,000 |
| **Total Folds (Walk-Forward)** | 24 |

---

## 6. Metodología Walk-Forward

### 6.1 Estructura

```
Período Total: 30 años (1996-2025, 7,390 días)

Fold 1:  Train [1996-12-17:2001-12-16]  → Test [2001-12-17:2002-12-16]  (1 año)
Fold 2:  Train [1997-12-17:2002-12-16]  → Test [2002-12-17:2003-12-16]  (1 año)
Fold 3:  Train [1998-12-17:2003-12-16]  → Test [2003-12-17:2004-12-16]  (1 año)
...
Fold 24: Train [2019-12-17:2024-12-16]  → Test [2024-12-17:2025-12-16]  (1 año)

Total: 24 folds con desplazamiento anual
```

### 6.2 Protocolo

1. **Entrenamiento (In-Sample)**
   - 5 años de datos históricos por fold
   - Mezcla aleatoria de episodios
   - 300,000 timesteps de aprendizaje

2. **Prueba (Out-of-Sample)**
   - 1 año de datos nunca vistos
   - Sin re-entrenamiento
   - Evaluación determinística
   - Memoria LSTM mantenida entre pasos

3. **Rebalance**
   - Balance final de Fold N → Capital inicial de Fold N+1
   - Cadena de trades continua a través de 24 años

---

## 7. Preparación de Datos

### 7.1 Pipeline de Limpieza

1. **Filtro de Kalman** (denoising)
   - $R = 0.1$ (confianza en medición)
   - $Q = 10^{-4}$ (ruido de proceso)
   - Aplicado a `close` para obtener señal limpia

2. **ATR (Average True Range)**
   - Período: 14 barras
   - Usado para: trailing stops y volatilidad

3. **Normalización Rolling Z-Score**
   - Ventana: 252 días (1 año de trading)
   - Features: VWAP ratio, volumen, retorno, MACD, SMA50, EMA26
   - Fórmula: $z = \frac{x - \mu}{\sigma + \epsilon}$ (con $\epsilon = 10^{-8}$)

### 7.2 Features Finales (10 Totales: 6 Efectivos + 4 Infraestructurales)

** Features Efectivos para Entrenamiento ML (6):**

| Feature | Período/Parámetro | Descripción |
|---------|-------------------|------------|
| `precio_vwap_ratio` | VWAP | Ratio precio/VWAP |
| `volumen_cambio` | Volumen | Cambio porcentual |
| `retorno_log` | Cierre | $\ln(P_t / P_{t-1})$ |
| `macd_histograma` | (12, 26, 9) | Histograma MACD (momentum) |
| `sma_50` | 50 días | Tendencia mediano plazo |
| `ema_26` | 26 días | Tendencia exponencial |

** Features Infraestructurales (Soporte/Contexto) (4):**

| Feature | Propósito | Descripción |
|---------|----------|------------|
| `date` | Metadata | Timestamp (logs y periodos) |
| `close` | Señal IA | Precio filtrado con Kalman (tendencia) |
| `close_raw` | Ejecución | Precio real del mercado (PnL) |
| `atr` | Riesgo | Volatilidad (Average True Range, período 14) |

---

## 8. Métricas de Rendimiento

---

## 8. Métricas de Rendimiento

### 8.1 Métricas Primarias

$$\text{Sharpe Ratio} = \frac{\text{Retorno Medio}}{\text{Volatilidad}}$$

$$\text{CAGR} = \left(\frac{\text{Balance Final}}{\text{Capital Inicial}}\right)^{1/n} - 1$$

$$\text{Max Drawdown} = \min_t \frac{NW_t - \text{Peak}(NW)}{\text{Peak}(NW)}$$

$$\text{Hit Rate} = \frac{\text{# Trades Ganadores}}{\text{# Trades Totales}}$$

$$\text{Profit Factor} = \frac{\text{Ganancias Brutas}}{\text{Pérdidas Brutas}}$$

### 8.2 Desempeño Logrado

| Métrica | Valor (3x) | Valor (5x) |
|---------|-----------|-----------|
| Sharpe Ratio | 2.59 | 2.80 |
| CAGR | 4.65% | 14.01% |
| Max Drawdown | -20.15% | -45.58% |
| Hit Rate | 59.39% | 60.17% |
| Profit Factor | 2.13 | 2.11 |

---

## 9. Referencias y Documentación

- **Environment**: `env/multi_horizon_based_reward_env_v4.py`
- **Training Script**: `training/train_rppo_trading_walk_forward_validation_D1.py`
- **Evaluación**: `evaluation/analyze_walk_forward_validation.py`
- **Datos**: `clean_data/XAUUSD_D1_rl.csv`

---

**Versión**: 1.0  
**Fecha**: 25/02/2026
