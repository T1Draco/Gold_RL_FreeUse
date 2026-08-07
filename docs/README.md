# Documentación - Gold-RL-Austranet

**Centro de documentación técnica del proyecto de Reinforcement Learning para trading de oro**

---

## Índice de Documentación

### 1. [Especificación del Modelo](MODEL_SPEC.md)
**Enfoque:** Arquitectura técnica detallada del agente RL

- Descripción del algoritmo RecurrentPPO
- Arquitectura de la red neuronal (LSTM + MLP)
- Espacio de observación y acciones
- Sistema de recompensas multi-horizonte
- Metodología walk-forward validation

**Para quién:** Developers, Research, Análisis técnico del modelo

---

### 2. [Datos y Proveniencia](DATA_PROVENANCE.md)
**Enfoque:** Pipeline completo de datos desde MT5 hasta modelo

- Fuente de datos: MetaTrader 5 (Austranet)
- Estructura de datos brutos (8 columnas OHLCV)
- Pipeline de limpieza 6-fases
- Indicadores técnicos calculados (25+)
- Filtro de Kalman para denoising
- Normalización Z-Score adaptativa
- Dataset final para RL (12 columnas seleccionadas)

**Para quién:** Data Engineers, Analistas de datos, QA testing

---

### 3. [Resultados de Backtesting](BACKTEST_RESULTS.md)
**Enfoque:** Validación exhaustiva del desempeño histórico

- Metodología walk-forward (24 folds, 1996-2025)
- Resultados agregados por período
- Comparativa: RL vs Buy & Hold vs Trend Following
- Análisis de drawdown y risk-adjusted returns
- Métricas clave (CAGR, Sharpe, Profit Factor, Hit Rate)
- Validación de robustez en datos no vistos
- Análisis de estabilidad temporal

**Para quién:** Traders, Portfolio managers, Stakeholders de inversión

---

### 4. [Manual de Usuario](USER_MANUAL.md)
**Enfoque:** Guía práctica de instalación, ejecución y troubleshooting

- Requisitos del sistema (Python, PyTorch, CUDA)
- Instalación paso-a-paso
- Comandos de ejecución (training, evaluación, análisis)
- Interpretación de logs y resultados
- Parámetros configurables
- Troubleshooting de errores comunes
- Guía de contacto y soporte

**Para quién:** Usuarios operacionales, DevOps, Support técnico

---

### 5. [Trabajo Futuro y Escalabilidad](FUTURE_WORK_AND_SCALABILITY.md)
**Enfoque:** Visión a futuro, plan de desarrollo y mejoras

#### Secciones Principales:

**Deployment en Producción**
- Integración con MetaTrader 5 (2 opciones: API REST vs ONNX)
- Exportación de parámetros del modelo
- Expert Advisor (EA) en MQL5
- Archivos de configuración estándar

**Variables Macroeconómicas**
- Tasas de interés (FED, ECB, BOE)
- Inflación y CPI
- Índice del Dólar (DXY)
- Volatilidad (VIX, VIX del Oro)
- Sentimiento de mercado
- Calendarios económicos
- Riesgo geopolítico

**Mejoras Técnicas**
- Multi-agent ensemble
- Arquitectura Transformer vs LSTM
- Optimización de hiperparámetros (Ray Tune)
- MLOps y versionado de modelos

**Optimizaciones de Rendimiento**
- Reducción de latencia (<10ms)
- Computación paralela
- Caching inteligente
- Quantization de modelos

**Robustez y Monitoreo**
- Circuit breaker automático
- Dashboard en tiempo real (Grafana)
- A/B testing en vivo
- Métricas de monitoreo

**Expansión a Otros Activos**
- Silver (XAGUSD)
- Oil, Copper, Natural Gas
- Multi-timeframe trading
- Trading de spreads

**Roadmap Temporal**
- Q1 2026: Consolidación
- Q2 2026: Integración MT5
- Q3 2026: Variables macro
- Q4 2026: Robustez y monitoreo
- Q1 2027: Arquitectura avanzada
- Q2 2027: Multi-asset
- Q3 2027: Production ready
- Q4 2027: Go live

**Para quién:** Product managers, Architects, C-level decision makers, R&D

---

## Lectura Recomendada

### Para Entender el Proyecto Rápidamente (30 minutos)
1. Leer [README principal](/README.md) - Visión general
2. Secciones "Resultados Típicos" en [BACKTEST_RESULTS.md](BACKTEST_RESULTS.md)
3. Secciones "Contexto Actual" en [FUTURE_WORK_AND_SCALABILITY.md](FUTURE_WORK_AND_SCALABILITY.md)

### Para Implementar el Proyecto (2-3 horas)
1. [USER_MANUAL.md](USER_MANUAL.md) - Instalación y ejecución
2. [DATA_PROVENANCE.md](DATA_PROVENANCE.md) - Entender los datos
3. [MODEL_SPEC.md](MODEL_SPEC.md) - Arquitectura técnica

### Para Desarrollar Futuras Mejoras (4-5 horas)
1. [MODEL_SPEC.md](MODEL_SPEC.md) - Base técnica
2. [BACKTEST_RESULTS.md](BACKTEST_RESULTS.md) - Baseline de performance
3. [FUTURE_WORK_AND_SCALABILITY.md](FUTURE_WORK_AND_SCALABILITY.md) - Visión completa

### Para Decisiones de Inversión (1-2 horas)
1. [BACKTEST_RESULTS.md](BACKTEST_RESULTS.md) - Resultados y comparativas
2. [FUTURE_WORK_AND_SCALABILITY.md](FUTURE_WORK_AND_SCALABILITY.md) - Plan de riesgos y roadmap
3. [USER_MANUAL.md](USER_MANUAL.md) - Sección "Disclaimer"

---

## Matriz de Documentos vs Audiencia

| Audiencia | Docs Primarios | Docs Secundarios |
|-----------|----------------|------------------|
| **Traders/Investors** | BACKTEST_RESULTS | FUTURE_WORK, USER_MANUAL |
| **Data Scientists** | MODEL_SPEC | DATA_PROVENANCE, BACKTEST_RESULTS |
| **DevOps/Deployment** | USER_MANUAL | FUTURE_WORK (Deployment section) |
| **Product/Strategy** | FUTURE_WORK | MODEL_SPEC, BACKTEST_RESULTS |
| **Nuevos Contribuidores** | README, USER_MANUAL | MODEL_SPEC, DATA_PROVENANCE |

---

## Preguntas Frecuentes (FAQ)

### ¿Por dónde empiezo?
→ Lee [README principal](/README.md), luego [USER_MANUAL.md](USER_MANUAL.md)

### ¿Cómo están los datos?
→ Ver [DATA_PROVENANCE.md](DATA_PROVENANCE.md) para pipeline completo

### ¿Cuál es el rendimiento esperado?
→ Consulta [BACKTEST_RESULTS.md](BACKTEST_RESULTS.md) para resultados históricos

### ¿Cómo implemento esto en MT5?
→ Lee sección "Deployment en Plataformas de Trading" en [FUTURE_WORK_AND_SCALABILITY.md](FUTURE_WORK_AND_SCALABILITY.md)

### ¿Qué hay en el roadmap?
→ Consulta [FUTURE_WORK_AND_SCALABILITY.md](FUTURE_WORK_AND_SCALABILITY.md) sección "Roadmap de Implementación"

### ¿Tengo un error, cómo lo resuelvo?
→ Ver sección "Troubleshooting" en [USER_MANUAL.md](USER_MANUAL.md)

---

## Control de Versión de Documentación

| Versión | Fecha | Cambios Principales |
|---------|-------|---------------------|
| 1.0 | 25/02/2026 | Documentación inicial completa (5 archivos) |
| 1.1 | TBD | Agregar sección de deployment MT5 |
| 2.0 | TBD | Documentación de variables macro y multi-asset |

---

## Contribución a la Documentación

Si identificas:
- **Imprecisiones técnicas:** Contactar con el Equipo de Desarrollo para revisar y corregir
- **Falta de claridad:** Propone cambios en las secciones pertinentes
- **Información desactualizada:** Informa para actualizar

**Responsable de mantenimiento:** Equipo de Desarrollo Gold-RL

---

## Contacto y Soporte

Para preguntas sobre cualquier documento:
- **Technical Issues:** martin.puebla.rivera@gmail.com

---

<div align="center">

**Centro de Documentación - Gold-RL-Austranet**  
*Última actualización: 25/02/2026*

</div>
