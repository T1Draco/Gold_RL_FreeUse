# Trabajo Futuro y Escalabilidad

**Fecha de última actualización:** Febrero 2026  
**Estado del Proyecto:** Fase de Investigación y Experimentación

---

## Tabla de Contenidos

1. [Contexto Actual](#contexto-actual)
2. [Deployment en Plataformas de Trading](#deployment-en-plataformas-de-trading)
3. [Expansión de Variables Externas](#expansión-de-variables-externas)
4. [Mejoras Técnicas y Arquitectura](#mejoras-técnicas-y-arquitectura)
5. [Optimizaciones de Rendimiento](#optimizaciones-de-rendimiento)
6. [Monitoreo y Robustez](#monitoreo-y-robustez)
7. [Expansión a Otros Activos](#expansión-a-otros-activos)
8. [Roadmap de Implementación](#roadmap-de-implementación)

---

## Contexto Actual

### Etapa Actual: Validación de Concepto (PoC)

El presente proyecto representa una **etapa temprana de investigación y desarrollo** enfocada en:

- **Validar la viabilidad** de aplicar Reinforcement Learning (RecurrentPPO + LSTM) a trading de oro (XAU/USD)
- **Establecer un workflow básico** reproducible para experimentación
- **Demostrar rentabilidad** en validación walk-forward (24 años de datos, 1996-2025)
- **Implementar infraestructura mínima** para recolección, procesamiento y entrenamiento de datos

### Limitaciones de la Etapa Actual

#### Scope Técnico
- **Solo variables técnicas**: SMA, EMA, MACD, ATR, volumen relativo, VWAP, Z-Score
- **Activo único**: XAU/USD a timeframe D1 (diario)
- **Horizonte único**: Decisiones basadas en estado de 30 días
- **Acción discreta**: LONG, NEUTRAL, SHORT (sin posiciones parciales)

#### Infraestructura
- **Procesamiento local**: Cálculos ejecutados en máquinas individuales
- **Datos MT5**: Dependencia de plataforma propietaria Austranet
- **Modelo no exportable**: Pesos guardados en formato PyTorch, no interpretables en MT5
- **Sin API trading**: Ejecución manual o integración ad-hoc

#### Validación
- **No incluye costos reales**: Spreads, comisiones, slippage
- **Ejecución perfecta**: Asume que las órdenes se ejecutan exactamente al precio esperado
- **Sin eventos extremos**: Datos no incluyen comportamientos anómalos recientes

---

## Deployment en Plataformas de Trading

### 1. Integración con MetaTrader 5 (MT5)

#### Problema Actual
El modelo entrenado existe como:
- Archivo `.pt` con pesos de PyTorch
- Código Python con lógica de preprocessing
- **No es directamente ejecutable en MT5**

#### Solución: Exportar a MQL5

**Opción A: Wrapper en MQL5 + Python**

```
MT5 EA (MQL5)
    ↓
Python API Server (HTTP/WebSocket)
    ↓
Modelo PyTorch + Preprocessing
    ↓
Orden a MT5
```

**Pasos de Implementación:**
1. Crear servidor REST/WebSocket en Python que expone el modelo
2. Desarrollar Expert Advisor (EA) en MQL5 que:
   - Recopila datos de mercado en tiempo real
   - Envía observaciones al servidor Python
   - Recibe acciones del agente
   - Ejecuta órdenes

**Stack Recomendado:**
```python
# Dependencias nuevas
- Flask o FastAPI para API
- python-socketio para WebSocket
- MetaTrader5 Python API
```

**Ejemplo de Estructura EA:**
```mql5
// trading_agent_ea.mq5
input string PythonServerURL = "http://localhost:5000";
input int ModelPort = 5000;

OnTick() {
    // 1. Recopilar datos últimas 30 velas
    // 2. Hacer POST request a servidor Python
    // 3. Recibir acción (LONG/NEUTRAL/SHORT)
    // 4. Ejecutar orden correspondiente
}
```

**Opción B: Convertir Modelo a ONNX/TensorRT**

Para reducir latencia y dependencias:
1. Exportar modelo PyTorch a ONNX (Open Neural Network Exchange)
2. Usar ONNX Runtime (compatible con múltiples plataformas)
3. Implementar preprocessing en C++ para máxima velocidad
4. Integrar en MT5 como DLL

**Ventajas:**
- Latencia <10ms por predicción
- Sin dependencias Python en MT5
- Modelo inmutable y seguro

### 2. Exportación de Parámetros del Agente

#### Formato de Configuración Estándar

Crear archivo de configuración `.yaml` o `.json`:

```yaml
# agent_config.yaml
model:
  type: "RecurrentPPO"
  lstm_hidden_units: 64
  mlp_layers: [256, 128, 64]
  learning_rate: 1.04e-05
  
training:
  timesteps_per_fold: 300000
  total_folds: 24
  train_window_years: 5
  test_window_years: 1
  
environment:
  initial_balance: 10000
  leverage: 3.0 # o 5.0 para variante alta riesgo
  actions: ["LONG", "NEUTRAL", "SHORT"]
  observation_history: 30 # días
  
risk_management:
  trailing_stop_long: 0.02 # 2% bajo el máximo
  max_position_size: 0.05 # 5% del balance
  
preprocessing:
  kalman_filter_r: 0.1
  kalman_filter_q: 0.0001
  zscore_window: 252
  
indicators:
  - name: "SMA_50"
  - name: "SMA_200"
  - name: "EMA_26"
  - name: "MACD_Signal"
  - name: "ATR_14"
  # ... resto de indicadores
```

#### Checkpoints del Modelo

Guardar archivos de estado en intervalos regulares:

```
models/
├── fold_001/
│   ├── model.pt              # Pesos del modelo
│   ├── config.yaml           # Configuración
│   ├── optimizer_state.pt    # Estado del optimizador
│   ├── training_log.csv      # Métricas de entrenamiento
│   └── validation_metrics.json
├── fold_002/
│   └── ...
└── latest_checkpoint.pt      # Último checkpoint global
```

---

## Expansión de Variables Externas

### 1. Variables Macroeconómicas

#### Datos de Tasas de Interés

**Tasas Clave a Incorporar:**
- Federal Funds Rate (FED) - Tasas de interés de EE.UU.
- ECB Rate (BCE) - Tasas Banco Central Europeo
- BOE Rate (Banco de Inglaterra)
- Spread USD/EUR

**Fuentes de Datos:**
- FRED (Federal Reserve Economic Data): https://fred.stlouisfed.org/
- API OpenBB (acceso gratuito a datos económicos)
- Trading View API
- Investing.com API

**Impacto en Oro:**
- Tasas más altas → Debilidad del oro (mayor coste de oportunidad)
- Tasas más bajas → Fortaleza del oro (activo sin rendimiento favorecido)

#### Inflación y CPI

```python
# Nuevas features
features = {
    'CPI_USA_YoY': cpi_change,           # Cambio anual CPI USA
    'CPI_EUR_YoY': cpi_eur_change,       # Cambio anual CPI EUR
    'PCE_USA': pce_rate,                 # Personal Consumption Expenditures
    'Real_Rates': fed_rate - inflation,  # Tasa real = nominal - inflación
}
```

#### Índice del Dólar (DXY)

**Significancia:** Oro cotiza inversamente al dólar
```python
# Feature: correlación inversa
features['DXY_Normalized'] = -normalize(dxy_price)  # Negativa por correlación inversa
features['DXY_Momentum'] = dxy_price.diff(periods=5)
```

### 2. Datos de Volatilidad

#### VIX (Volatility Index)

El VIX mide volatilidad esperada del S&P 500, correlaciona con demanda de oro seguro.

```python
# Nuevas features VIX
features = {
    'VIX_Level': vix_current,
    'VIX_MA_20': vix.rolling(20).mean(),
    'VIX_Percentile_252': vix.rolling(252).apply(lambda x: stats.percentileofscore(x, x.iloc[-1])),
    'VIX_Regime': 'high' if vix > 20 else 'low',  # Regímenes de volatilidad
}
```

**Interpretación:**
- VIX alto → Demanda de activos seguros (favorece oro)
- VIX bajo → Ambiente riesgoso favorece equities (debilita oro)

#### Volatilidad Implícita del Oro (GOLD VIX)

Si está disponible, usar directamente. Si no, derivar de opciones.

### 3. Sentimiento de Mercado

#### Sentiment Index

Incorporar índices de sentimiento:
- Fear & Greed Index
- Crypto Fear Index (proxy para apetito de riesgo global)
- CNN Fear & Greed Meter

```python
# API de sentimiento
import requests

def fetch_fear_greed_index():
    url = "https://api.alternative.me/fng/?limit=1&date_format=us"
    response = requests.get(url)
    data = response.json()['data'][0]
    return {
        'sentiment_score': int(data['value']),  # 0-100
        'sentiment_label': data['value_classification'],  # Extreme Fear/Fear/Greed/Extreme Greed
    }

# Normalizar e incluir en features
sentiment = fetch_fear_greed_index()
features['Market_Sentiment_Score'] = sentiment['sentiment_score'] / 100  # 0-1
features['Is_Fear_Regime'] = int(sentiment['sentiment_score'] < 30)
```

### 4. Eventos Económicos y Calendarios

#### Calendario Económico

Detectar días de reportes económicos importantes:

```python
# Integración con calendario económico
from datetime import datetime
import pandas as pd

ECONOMIC_EVENTS = {
    'NFP': {'country': 'USA', 'importance': 'high', 'day': 'first_friday'},
    'CPI': {'country': 'USA', 'importance': 'high', 'day': 'mid_month'},
    'ECB_Rate': {'country': 'EUR', 'importance': 'high', 'frequency': 'quarterly'},
    'BOE_Rate': {'country': 'GBR', 'importance': 'high', 'frequency': 'monthly'},
}

def get_upcoming_events(days_ahead=7):
    """Obtener eventos programados para los próximos N días"""
    pass

def is_high_volatility_event_today(date):
    """Retorna True si hay evento importante hoy"""
    # Reducir posiciones 4 horas antes de eventos críticos
    # o evitar completamente (no hacer trades)
```

**Uso en Modelo:**
- Feature binaria: `is_high_impact_event_day`
- Reducir tamaño de posición en días de eventos
- Considerar mayor volatilidad esperada

### 5. Datos Geopolíticos

#### Índices de Incertidumbre

```python
# EPU (Economic Policy Uncertainty) Index
# https://www.policyuncertainty.com/
# Oro típicamente sube en períodos de incertidumbre política

features = {
    'EPU_USA': economic_policy_uncertainty,
    'Geopolitical_Risk_Index': gpr_index,  # https://www.matteoiacoviello.com/gpr.htm
}

# Interpretación:
# EPU alta → Mayor demanda de oro como safe-haven
# GPR alta → Tensiones geopolíticas, oro favorecido
```

### 6. Datos de Cripto (Correlación Moderna)

El oro correlaciona con activos de riesgo más que antes:

```python
# Nuevas features cripto
features = {
    'BTC_Price_Normalized': normalize(btc_price),
    'BTC_Dominance': btc_market_cap / total_crypto_market_cap,
    'Altcoin_Season_Index': altcoin_volume_ratio,  # Apetito por riesgo
}

# Interpretación:
# BTC sube, oro bajo volatilidad → Ambiente risk-on
# BTC baja, oro sube → Flight-to-safety
```

---

## Mejoras Técnicas y Arquitectura

### 1. Multi-Agent Ensemble

En lugar de un único agente, usar múltiples agentes especializados:

```
┌─ Agente Scalp (5-min trades)
├─ Agente Swing (1-4 días)
├─ Agente Posición (1-4 semanas)
└─ Meta-Agent (coordina los 3)
```

**Beneficios:**
- Capturar múltiples horizones temporales
- Reducir riesgo de un único modelo
- Mejor robustez ante cambios de mercado

### 2. Modelo Transformer en lugar de LSTM

Actualizar de LSTM a arquitectura Transformer:

```python
# Actual: LSTM(64 units)
# Futuro: Transformer con Multi-Head Attention

class TransformerAgent(nn.Module):
    def __init__(self, d_model=64, num_heads=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(217, d_model)  # Input dimension
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, batch_first=True),
            num_layers=num_layers
        )
        self.actor = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # 3 acciones
        )
        
    def forward(self, x):
        # x: [batch, seq_len, features]
        x = self.embedding(x)
        x = self.transformer(x)
        x = x[:, -1, :]  # Última vela
        return self.actor(x)
```

**Ventajas:**
- Mejor atención a múltiples horizones temporales
- Mayor paralelización en GPU
- Mejor captura de dependencias a largo plazo

### 3. Integración con Ray Tune para HPO

Optimización distribuida de hiperparámetros:

```python
from ray import tune
from ray.tune import CLIReporter

config = {
    "learning_rate": tune.loguniform(1e-6, 1e-3),
    "entropy_coef": tune.loguniform(0.001, 0.1),
    "gamma": tune.uniform(0.99, 0.999),
    "lstm_units": tune.choice([32, 64, 128]),
}

analysis = tune.run(
    train_rppo,
    config=config,
    num_samples=100,  # 100 combinaciones aleatorias
    scheduler=ASHAScheduler(),
    search_alg=OptunaSearch(),
)
```

### 4. Model Versioning y MLOps

Implementar versionado de modelos:

```
mlflow/
├── experiments/
│   ├── exp_001_baseline/
│   │   ├── run_001_lr_1e5/
│   │   │   ├── model.pkl
│   │   │   ├── params.json
│   │   │   └── metrics.json
│   │   └── run_002_lr_5e5/
│   └── exp_002_with_macro_vars/
└── registry/
    ├── gold_agent_v1
    └── gold_agent_v2
```

---

## Optimizaciones de Rendimiento

### 1. Reducción de Latencia

#### Predicción en Tiempo Real

**Latencia actual:** ~100ms (Python puro)
**Objetivo:** <10ms

**Estrategias:**
1. **Caché de cálculos:** Reutilizar indicadores si los datos no cambian
2. **Batch processing:** Procesar múltiples símbolos juntos
3. **GPU acceleration:** Usar CUDA para inferencia
4. **Quantization:** Reducir precisión de pesos (float32 → int8)

```python
# Quantizar modelo PyTorch
from torch.quantization import quantize_dynamic

quantized_model = quantize_dynamic(
    model,
    {nn.Linear},
    dtype=torch.qint8
)
# Resultado: 4x más rápido, 4x menos memoria
```

### 2. Computación Paralela

Usar múltiples CPU cores:

```python
from multiprocessing import Pool
from functools import partial

def process_fold(fold_id):
    # Training de un fold
    return metrics

# Entrenar múltiples folds en paralelo
with Pool(processes=4) as pool:
    results = pool.map(process_fold, range(24))
```

### 3. Caching Inteligente

```python
# Caché LRUD para indicadores recientes
from functools import lru_cache

@lru_cache(maxsize=1000)
def calculate_sma(prices_tuple, window):
    prices = np.array(prices_tuple)
    return np.convolve(prices, np.ones(window)/window, mode='valid')[-1]
```

---

## Monitoreo y Robustez

### 1. Dashboard en Tiempo Real

Implementar monitoring en vivo con Grafana + Prometheus:

```yaml
# prometheus.yml
global:
  scrape_interval: 10s

scrape_configs:
  - job_name: 'trading_agent'
    static_configs:
      - targets: ['localhost:8000']
```

**Métricas a Monitorear:**
- Rendimiento de agente (Sharpe diario, DD, PF)
- Latencia de predicciones
- Tasa de ejecución de órdenes
- Desviaciones de backtest vs live (slippage, spread real)
- Balance y equity curve

### 2. Circuit Breaker

Detener trading automáticamente si:
- DD excede umbral (ej: -10% en 24h)
- Sharpe intradiario baja bajo cierto nivel
- Error en conexión a MT5
- Cambio abrupto en correlaciones de mercado

```python
class CircuitBreaker:
    def __init__(self, dd_threshold=-0.10, sharpe_threshold=0.5):
        self.dd_threshold = dd_threshold
        self.sharpe_threshold = sharpe_threshold
        self.is_active = True
    
    def check_health(self, daily_returns, equity_curve):
        # Calcular DD y Sharpe
        mdd = calculate_max_drawdown(equity_curve)
        sharpe = calculate_sharpe(daily_returns)
        
        if mdd < self.dd_threshold or sharpe < self.sharpe_threshold:
            self.is_active = False
            alert_user()
            return False
        return True
```

### 3. A/B Testing en Vivo

Ejecutar versiones A (en producción) y B (nueva) en paralelo:

```
Live Trading:
├─ 50% volumen → Agent V1 (actual)
├─ 50% volumen → Agent V2 (experimental)
└─ Comparar PnL, Sharpe, DD de ambos

Criterios de switch:
- V2 supera V1 por 2+ Sharpe points durante 2+ semanas
- V2 DD no excede V1 DD
- V2 muestra consistencia en 3+ pares/timeframes
```

---

## Expansión a Otros Activos

### 1. Multi-Commodity Trading

Entrenar agentes especializados para:
- Silver (XAGUSD) - Alta correlación con oro
- Crude Oil (XTIUSD, XTIOUSD)
- Natural Gas (XNGUSD)
- Copper (XCUUSD)

**Arquitectura:**
```
Master Agent (Meta-Model)
├─ Gold Sub-Agent (especializado en XAU)
├─ Silver Sub-Agent (especializado en XAG)
├─ Oil Sub-Agent
└─ Copper Sub-Agent
```

### 2. Multi-Timeframe

Actualmente solo D1. Expandir a:
- H4 (4 horas) - Swing trading
- H1 (1 hora) - Day trading
- M15 (15 minutos) - Scalping

**Estructura de datos:**
```python
observation_space = {
    'D1': [217 features],   # Daily
    'H4': [217 features],   # 4-hourly
    'H1': [217 features],   # Hourly
}

# Total: 651 features
# Usar diferentes capas de atención para cada timeframe
```

### 3. Pares Correlacionados

Explorar trading de spread entre correlacionados:

```python
# Feature: Ratio Gold/Silver
spread = np.log(xau_price / xag_price)
features['AU_AG_Spread'] = spread
features['AU_AG_Spread_ZScore'] = zscore(spread, window=252)

# Trading: Long spread si undervalued, short si overvalued
```

---

## Roadmap de Implementación

### **Q1 2026 - Consolidación Actual**
- [x] Validar workflow básico con 24 folds
- [ ] Documentar completamente el codebase
- [ ] Setup de CI/CD (GitHub Actions)
- [ ] Crear suite de tests unitarios

### **Q2 2026 - Integración Básica MT5**
- [ ] Crear API REST en FastAPI
- [ ] Desarrollar EA básico en MQL5
- [ ] Pruebas en cuenta demo MT5
- [ ] Exportar modelo a ONNX
- [ ] Crear archivo de configuración estándar

### **Q3 2026 - Variables Macroeconómicas**
- [ ] Integración con FRED API
- [ ] Agregar tasas de interés como features
- [ ] Agregar VIX y volatilidad
- [ ] Reentrenar modelo con nuevas variables
- [ ] Backtest con datos históricos macro

### **Q4 2026 - Robustez y Monitoreo**
- [ ] Implementar Circuit Breaker
- [ ] Setup Grafana + Prometheus
- [ ] Dashboard de monitoreo en tiempo real
- [ ] Logging centralizado (ELK Stack)
- [ ] Alertas automáticas

### **Q1 2027 - Arquitectura Avanzada**
- [ ] Migrar a Transformer architecture
- [ ] Implementar multi-agent ensemble
- [ ] Optimizar con Ray Tune (HPO distribuido)
- [ ] Quantization del modelo
- [ ] Pruebas de latencia <10ms

### **Q2 2027 - Expansión Multi-Activo**
- [ ] Entrenar agentes para Silver, Oil
- [ ] Implementar multi-timeframe (H4, H1, M15)
- [ ] Trading de spreads (correlacionados)
- [ ] Coordinar trades entre activos

### **Q3 2027 - Production Ready**
- [ ] Auditoría de seguridad (model robustness)
- [ ] Load testing (1000+ trades/hora)
- [ ] Backup y disaster recovery
- [ ] Documentación de deployment
- [ ] Training del equipo operacional

### **Q4 2027 - Go Live**
- [ ] Deployment en cuenta real (pequeño volumen)
- [ ] Aumentar volumen gradualmente (ramping)
- [ ] Monitoreo diario de desviaciones
- [ ] Ajustes finos según performance real

---

## Consideraciones de Riesgo y Mitigation

### 1. Overfitting Temporal

**Riesgo:** Modelo ajustado a período específico (1996-2025) que no generaliza

**Mitigación:**
- Walk-forward validation (ya implementado)
- Monte Carlo permutation testing
- Stress testing con datos sintéticos
- Regular retraining cada mes

### 2. Cambio de Régimen de Mercado

**Riesgo:** Patrón técnico cambia (ej: nueva relación con tasas de interés)

**Mitigación:**
- Incluir variables macroeconómicas (desacoplar de solo tecnicales)
- Adaptive learning: reentrenar cada trimestre
- Ensemble de modelos (no confiar en uno)
- Análisis de correlaciones dinámicas

### 3. Riesgo Operacional

**Riesgo:** Fallo en integración MT5, conexión internet, latencia

**Mitigación:**
- Redundancia de conexiones (2+ servidores)
- Heartbeat monitoring
- Fallback a estrategia manual
- Testing exhaustivo antes de live
- Pequeños volúmenes iniciales

### 4. Riesgo Regulatorio

**Riesgo:** Algoritmos de trading pueden estar regulados en algunas jurisdicciones

**Mitigación:**
- Consultar con abogados de compliance
- Documentar decisiones del modelo (interpretabilidad)
- Auditoría regular de trades
- Cumplimiento con MiFID II, CFTC, etc.

---

## Recursos y Referencias

### Librerías Recomendadas

```python
# Datos macroeconómicos
pip install pandas-datareader fred opensearch

# APIs de trading
pip install MetaTrader5 ccxt

# MLOps
pip install mlflow wandb

# Monitoreo
pip install prometheus-client grafana

# Optimización
pip install optuna ray[tune]

# Exportación de modelos
pip install onnx onnxruntime

# Análisis
pip install backtrader zipline-reloaded
```

### Fuentes de Datos

| Fuente | Tipo | Cobertura | Costo |
|--------|------|-----------|-------|
| FRED | Macro | USA | Gratis |
| Trading View | Precios/Técnico | Global | Pro ($15-80/mes) |
| Investing.com | Calendarios/Datos | Global | Gratis |
| OpenBB | Datos financieros | Acciones/Cripto/Macro | Gratis/Pro |
| IQFeed | Datos de opciones | USA | ~$100/mes |
| dxFeed | Datos en vivo | Global | Custom |

### Plataformas Alternativas a MT5

- **cTrader** - Integración C#, más flexible
- **Interactive Brokers** - API Python robusta
- **Oanda** - API REST simples
- **Crypto exchanges** (Binance, Kraken) - Si expandir a cripto

---

## Conclusiones y Próximos Pasos

### Estado Actual
El proyecto ha validado exitosamente que **es posible usar RL para trading de oro** con resultados atractivos (Sharpe 2.80 con 5x leverage). Sin embargo, esta es una **solución de laboratorio** que requiere trabajos importantes antes de producción.

### Prioridades Inmediatas
1. **Integración MT5** (Q2 2026): Sin esto, no hay ejecución real
2. **Variables macro** (Q3 2026): Mejorar robustez del modelo
3. **Monitoreo y alertas** (Q4 2026): Necesario para trading real
4. **Stress testing** (Q1 2027): Validar ante extremos de mercado

### Oportunidades de Diferenciación
- Multi-agent ensemble (menos correlacionado que competencia)
- Macro-informed RL (pocos traders combinan RL + economía)
- Open-source community contributions (publicar en GitHub)
- Extensión a multi-commodity (escala horizontal)

### Disclaimer
Este documento es una **hoja de ruta** indicativa. El mercado de oro es volátil y no hay garantía de rentabilidad futura. Cualquier implementación debe incluir:
- Risk management riguroso
- Supervisión humana constante
- Validación antes de cada nueva versión
- Capital adecuado para soportar drawdowns

---

**Última actualización:** Febrero 2026  
**Mantenedor:** Equipo de Desarrollo Gold-RL-Austranet  
**Versión:** 1.0
