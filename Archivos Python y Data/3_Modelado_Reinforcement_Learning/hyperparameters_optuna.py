"""
hyperparameters_optuna.py

Script de optimización de hiperparámetros para RecurrentPPO (PPO + LSTM) usando Optuna.

Propósito:
- Buscar automáticamente los mejores hiperparámetros que maximicen el Sharpe Ratio
  en un conjunto de validación out-of-sample.

Flujo:
1. Carga y preprocesa los datos (Kalman denoising + Rolling Z-Score).
2. Divide en train (70%) y validación (30%).
3. Para cada trial (N_TRIALS=50):
   - Sugiere hiperparámetros (learning_rate, ent_coef, gamma, net_width, net_depth,
     lstm_hidden_size, n_lstm_layers, n_steps, batch_size).
   - Entrena RecurrentPPO en el set de entrenamiento.
   - Evalúa en el set de validación (sin entrenamiento adicional).
   - Calcula Sharpe Ratio como métrica de optimización.
4. Guarda los mejores parámetros en `results/optuna/`.

Supuestos importantes:
- Los datos ya vienen pre-procesados (Kalman filtering + Rolling Z-Score).
- normalize_internal=False en los entornos (features ya normalizadas).
- Métrica objetivo: Sharpe Ratio anualizado.
- Penalización: si total_trades < 10 en validación, retorna -1.0.

Dependencias:
- optuna (pip install optuna)
- sb3_contrib (RecurrentPPO)
- PyTorch, NumPy, Pandas
"""

import pandas as pd
import numpy as np
import os
import sys
import optuna
from sb3_contrib import RecurrentPPO  # ← CAMBIO CRÍTICO
import torch
import random

# Configuración de rutas
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from env.multi_horizon_based_reward_env_v4 import crear_entorno_discreto, validar_dataframe

# ═══════════════════════════════════════════════════════════════════════════════
#                        FUNCIONES DE PREPROCESSING (MATCH CON TRAINING)
# ═══════════════════════════════════════════════════════════════════════════════

class KalmanFilter1D:
    """Filtro de Kalman 1D para denoising de precios"""
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

def aplicar_kalman_a_columna(series, R=0.1, Q=1e-5):
    kf = KalmanFilter1D(R=R, Q=Q)
    return series.apply(kf.update)

def set_global_seed(seed_value):
    """Establece semilla global para reproducibilidad"""
    if seed_value is None:
        return
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ═══════════════════════════════════════════════════════════════════════════════
#                        CONFIGURACIÓN FIJA
# ═══════════════════════════════════════════════════════════════════════════════

DATA_PATH = os.path.join(project_root, "3_Modelado_Reinforcement_Learning/clean_data", "XAUUSD_D1_rl.csv")
WINDOW_SIZE_OBS = 30  # ← MATCH CON TRAINING
INITIAL_BALANCE = 10000
RANDOM_SEED = 42

# Parámetros del Entorno (Fijos)
TRAIN_COST = 0.0005
VAL_COST = 0.0001
SHORT_ENABLED = False

# Configuración de Optuna
N_TRIALS = 50
TIMESTEPS_PER_TRIAL = 250_000  # ← Aumentado para LSTM (necesita más datos)

# ═══════════════════════════════════════════════════════════════════════════════
#                        PREPARACIÓN DE DATOS (MATCH CON TRAINING)
# ═══════════════════════════════════════════════════════════════════════════════

def cargar_y_procesar_datos():
    """
    Carga datos brutos y aplica el MISMO preprocessing que en training.

    Fases:
    1. Kalman denoising: suaviza el precio para reducir ruido de mercado.
    2. Rolling Z-Score: normaliza features con ventana móvil (252 días)
       para adaptarse a diferentes regímenes.
    3. Feature selection: selecciona 6 features clave.

    Returns:
        tuple: (df_train, df_val)
            - df_train: 70% de los datos (entrenamiento).
            - df_val: 30% de los datos (validación out-of-sample).
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"No se encuentra: {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    
    # --- FASE 1: DENOISING CON KALMAN ---
    print("   [PREPROCESO] Aplicando Filtro de Kalman...")
    Q_H1 = 1e-4
    df['Close_Filtered'] = aplicar_kalman_a_columna(df['Close'], R=0.1, Q=Q_H1)
    
    # Recalcular features con precio limpio
    df["retorno_log"] = np.log(df["Close_Filtered"] / df["Close_Filtered"].shift(1))
    df["volatilidad_20"] = df["retorno_log"].rolling(window=20).std()
    
    if 'VWAP' in df.columns:
        df["precio_vwap_ratio"] = (df["Close_Filtered"] - df["VWAP"]) / df["VWAP"]
    
    # Reemplazar Close original por filtrado
    df['Close'] = df['Close_Filtered']
    df.drop(columns=['Close_Filtered'], inplace=True, errors='ignore')
    
    # --- FASE 2: ROLLING Z-SCORE NORMALIZATION ---
    print("   [PREPROCESO] Aplicando Rolling Z-Score...")
    cols_to_normalize = ['precio_vwap_ratio', 'volumen_cambio', 'retorno_log', 'volatilidad_20']
    
    for col in cols_to_normalize:
        rolling_mean = df[col].rolling(window=252, min_periods=20).mean()
        rolling_std = df[col].rolling(window=252, min_periods=20).std()
        df[col] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
    
    # Limpiar NaNs
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # --- FEATURE SELECTION FINAL ---
    features_finales = [
        'Date', 'Close',
        'precio_vwap_ratio',
        'volumen_cambio',
        'retorno_log',
        'MACD_Histograma'
    ]
    df = df[[col for col in features_finales if col in df.columns]].copy()
    
    # Split Train/Val (70/30)
    split_idx = int(len(df) * 0.7)
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_val = df.iloc[split_idx:].reset_index(drop=True)
    
    return df_train, df_val

# Cargar datos una vez
DF_TRAIN, DF_VAL = cargar_y_procesar_datos()
print(f"✓ Datos procesados. Train: {len(DF_TRAIN)} | Val: {len(DF_VAL)}")

# ═══════════════════════════════════════════════════════════════════════════════
#                        FUNCIÓN AUXILIAR: SHARPE RATIO
# ═══════════════════════════════════════════════════════════════════════════════

def calcular_sharpe_ratio(df_history):
    """
    Calcula el Sharpe Ratio anualizado de una serie de net worth.

    Fórmula: Sharpe = (media_retornos / std_retornos) * sqrt(252 días trading)

    Args:
        df_history (pd.DataFrame): DataFrame con columna 'net_worth'.

    Returns:
        float: Sharpe Ratio anualizado. Retorna -10.0 si std=0 (sin volatilidad).
    """

# ═══════════════════════════════════════════════════════════════════════════════
#                        FUNCIÓN OBJECTIVE (OPTUNA) - RECURRENT PPO
# ═══════════════════════════════════════════════════════════════════════════════

def objective(trial):
    """
    Función objetivo para Optuna — optimizar hiperparámetros de RecurrentPPO.

    Hiperparámetros a optimizar:
    - learning_rate: tasa de aprendizaje del algoritmo (log-uniform 1e-6 a 1e-4).
    - ent_coef: coeficiente de entropía para exploración (log-uniform 0.001 a 0.05).
    - gamma: factor de descuento (categórico: 0.98, 0.99, 0.995).
    - net_width: ancho de las capas MLP (64, 128, 256).
    - net_depth: profundidad de las capas MLP (1 a 3).
    - lstm_hidden_size: tamaño del estado oculto del LSTM (32, 64, 128).
    - n_lstm_layers: número de capas LSTM (1 a 2).
    - n_steps: número de pasos antes de actualizar (512, 1024, 2048).
    - batch_size: tamaño de mini-batch (64, 128, 256).

    Proceso:
    1. Sugiere hiperparámetros.
    2. Entrena RecurrentPPO en el set de entrenamiento.
    3. Evalúa en validación (con memoria LSTM y cálculo de confianza DPS).
    4. Calcula Sharpe Ratio y penaliza si hay pocos trades.

    Returns:
        float: Sharpe Ratio (a maximizar). Retorna -1.0 o -1000.0 en casos de error.
    """
    # 1. HIPERPARÁMETROS A OPTIMIZAR
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.001, 0.05, log=True)
    gamma = trial.suggest_categorical("gamma", [0.98, 0.99, 0.995])
    
    # Arquitectura MLP (antes/después del LSTM)
    net_width = trial.suggest_categorical("net_width", [64, 128, 256])
    net_depth = trial.suggest_int("net_depth", 1, 3)
    
    # Parámetros LSTM
    lstm_hidden_size = trial.suggest_categorical("lstm_hidden_size", [32, 64, 128])
    n_lstm_layers = trial.suggest_int("n_lstm_layers", 1, 2)
    
    # Parámetros de entrenamiento
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    # 2. CREAR ENTORNOS (Sin normalización interna - datos ya procesados)
    # Los datos ya vienen normalizados, así que normalize_internal=False.
    env_train = crear_entorno_discreto(
        df=DF_TRAIN,
        initial_balance=INITIAL_BALANCE,
        transaction_cost=TRAIN_COST,
        leverage=1.0,
        short_enabled=SHORT_ENABLED,
        eval_mode=False,
        position_pct=0.25,
        window_size=WINDOW_SIZE_OBS,
        scaling_params=None,
        normalize_internal=False  # ← Datos ya normalizados
    )
    
    env_val = crear_entorno_discreto(
        df=DF_VAL,
        initial_balance=INITIAL_BALANCE,
        transaction_cost=VAL_COST,
        leverage=1.0,
        short_enabled=SHORT_ENABLED,
        eval_mode=True,
        position_pct=0.25,
        window_size=WINDOW_SIZE_OBS,
        scaling_params=None,
        normalize_internal=False
    )

    # 3. CONSTRUIR ARQUITECTURA DE RED
    # net_arch especifica el número de capas y neuronas por capa
    # Ej: net_arch=[256, 256] significa 2 capas con 256 neuronas cada una
    # Todas las capas se aplican antes del LSTM
    net_arch = [net_width] * net_depth

    # 4. CREAR MODELO RECURRENTPPO
    # RecurrentPPO (PPO with Recurrent Neural Network) usa LSTM para mantener memoria
    # La memoria se actualiza a cada paso, permitiendo que el modelo considere la historia
    try:
        model = RecurrentPPO(
            "MlpLstmPolicy",  # ← Policy con LSTM: MLP → LSTM → Output
            env_train,
            learning_rate=learning_rate,  # Velocidad de aprendizaje
            ent_coef=ent_coef,             # Coeficiente de entropía (balance exploración/explotación)
            gamma=gamma,                   # Factor de descuento (importancia del futuro)
            n_steps=n_steps,               # Pasos antes de actualizar pesos
            batch_size=batch_size,         # Tamaño de mini-batch durante actualizaciones
            policy_kwargs=dict(
                net_arch=net_arch,         # Capas MLP: [width, width, ..., width]
                lstm_hidden_size=lstm_hidden_size,  # Dimensión del estado oculto LSTM
                n_lstm_layers=n_lstm_layers,        # Número de capas LSTM (1 o 2)
            ),
            device="cpu",
            seed=RANDOM_SEED,
            verbose=0  # Sin output durante entrenamiento
        )

        # 4.1 ENTRENAR MODELO
        # total_timesteps es el número de pasos de entrenamiento (250,000)
        model.learn(total_timesteps=TIMESTEPS_PER_TRIAL)
        
        # 5. EVALUAR EN VALIDACIÓN (CON DPS Y MEMORIA LSTM)
        # - DPS: Dynamic Position Sizing — ajustar tamaño posición según confianza
        # - LSTM Memory: se reutiliza memoria (h, c) entre pasos para decisiones secuenciales
        obs, _ = env_val.reset()
        done = False
        all_net_worths = [INITIAL_BALANCE]
        
        # 5.1 INICIALIZAR MEMORIA LSTM
        # shape: (n_lstm_layers, batch_size=1, lstm_hidden_size)
        # h (hidden state) y c (cell state) mantienen información temporal
        memory_shape = (n_lstm_layers, 1, lstm_hidden_size)
        lstm_states = (
            torch.zeros(memory_shape, device=model.device),  # h (hidden state)
            torch.zeros(memory_shape, device=model.device),  # c (cell state)
        )
        episode_starts = np.array([True])  # Primera predicción es inicio de episodio
        
        while not done:
            with torch.no_grad():
                # 5.2 PREDECIR ACCIÓN CON MEMORIA LSTM
                # La acción se calcula usando:
                # - Observación actual (historia de precios/indicadores)
                # - Memoria LSTM del paso anterior (contexto temporal)
                # El estado LSTM se actualiza en cada paso (recuerda tendencias pasadas)
                actions_tensor, lstm_states = model.predict(
                    obs,
                    state=lstm_states,          # Memoria anterior (h, c)
                    episode_start=episode_starts,
                    deterministic=False         # Con exploración
                )
                action = actions_tensor.item()
                
                # 5.3 CALCULAR CONFIANZA PARA DPS (Dynamic Position Sizing)
                # La confianza es P(action_elegida | observation, lstm_state)
                # Rango: [0.33, 1.0] (en Discrete-3, min es 1/3 por uniform random)
                # Valores cercanos a 1.0 → alta certeza → mayor posición
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(model.device)
                episode_starts_float = torch.as_tensor(episode_starts, dtype=torch.float32).to(model.device)
                
                # Asegurar que lstm_states son tensores (a veces vienen en numpy)
                h, c = lstm_states
                if isinstance(h, np.ndarray):
                    h = torch.as_tensor(h, dtype=torch.float32, device=model.device)
                    c = torch.as_tensor(c, dtype=torch.float32, device=model.device)
                    lstm_states = (h, c)
                
                # Obtener distribución de probabilidades sobre acciones
                distribution_result = model.policy.get_distribution(
                    obs_tensor,
                    lstm_states=lstm_states,
                    episode_starts=episode_starts_float
                )
                distribution_object = distribution_result[0]
                # Softmax convierte logits en probabilidades [0, 1]
                action_probs = torch.softmax(distribution_object.distribution.logits, dim=-1)
                # Extraer probabilidad de la acción tomada
                confidence = action_probs[0, action].item()
            
            # 5.4 EJECUTAR PASO EN ENTORNO DE VALIDACIÓN
            # Pasar acción y confianza (DPS) al entorno
            # El entorno calcula:
            # - Nueva observación (precios/indicadores del siguiente paso)
            # - Reward (basado en multi-horizonte: 1d, 5d, 20d, 60d)
            # - done (fin de episodio = fin de datos de validación)
            # - info (estadísticas: net_worth, total_trades, etc.)
            obs, _, done, _, info = env_val.step(action, confidence=confidence)
            all_net_worths.append(info['net_worth'])
            episode_starts = np.array([done])  # Señal para reset de LSTM si done=True
        
        # 6. CALCULAR SHARPE RATIO (MÉTRICA OBJETIVO)
        # Sharpe mide rentabilidad ajustada por riesgo:
        # Sharpe = (retorno_medio / std_retorno) * sqrt(252)
        # Valores > 1.0 indican buena relación riesgo-retorno
        # Valores < 0.5 indican pobre desempeño
        df_history = pd.DataFrame({'net_worth': all_net_worths})
        sharpe_ratio = calcular_sharpe_ratio(df_history)
        
        # 7. PENALIZACIÓN POR POCOS TRADES
        # Una estrategia que hace muy pocos trades puede estar "dormida" (no operando)
        # Requerimos mínimo 10 trades en validación como prueba de actividad
        if info['total_trades'] < 10:
            return -1.0  # Penalización severa
            
        return sharpe_ratio  # Retornar Sharpe para que Optuna lo maximice

    except Exception as e:
        # En caso de error (crash del modelo, NaN, etc.), retornar penalización
        print(f"Error en trial: {e}")
        return -1000.0  # Penalización muy severa para descartar este trial

# ═══════════════════════════════════════════════════════════════════════════════
#                        EJECUCIÓN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    set_global_seed(RANDOM_SEED)
    
    # ══════════════════════════════════════════════════════════════════════════════
    # 1. IMPRIMIR CONFIGURACIÓN INICIAL
    # Muestra las opciones de búsqueda antes de iniciar la optimización
    # ══════════════════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print(" OPTIMIZACIÓN DE HIPERPARÁMETROS - RECURRENT PPO")
    print("="*80)
    print(f"  Objetivo: Maximizar Sharpe Ratio")
    print(f"  Arquitectura: MLP + LSTM (MlpLstmPolicy)")
    print(f"  Datos: Pre-procesados con Kalman + Rolling Z-Score")
    print(f"  Trials: {N_TRIALS} | Timesteps/Trial: {TIMESTEPS_PER_TRIAL:,}")
    print("="*80 + "\n")
    
    # ══════════════════════════════════════════════════════════════════════════════
    # 2. CREAR ESTUDIO DE OPTUNA
    # - TPEsampler: Tree-structured Parzen Estimator (Bayesian optimization)
    # - direction='maximize': buscamos maximizar el Sharpe Ratio
    # - seed=RANDOM_SEED: reproducibilidad entre ejecuciones
    # ══════════════════════════════════════════════════════════════════════════════
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
    )
    
    # ══════════════════════════════════════════════════════════════════════════════
    # 3. EJECUTAR OPTIMIZACIÓN
    # - n_trials=N_TRIALS (50): evaluar 50 combinaciones de hiperparámetros
    # - Cada trial entrena RecurrentPPO durante TIMESTEPS_PER_TRIAL (250k) pasos
    # - La función objective() retorna el Sharpe Ratio de validación
    # ══════════════════════════════════════════════════════════════════════════════
    study.optimize(objective, n_trials=N_TRIALS)

    # ══════════════════════════════════════════════════════════════════════════════
    # 4. MOSTRAR MEJORES PARÁMETROS ENCONTRADOS
    # ══════════════════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print(" MEJORES PARÁMETROS ENCONTRADOS")
    print("="*80)
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")
    print(f"\n  Mejor Sharpe Ratio: {study.best_value:.4f}")
    print("="*80)
    
    # ══════════════════════════════════════════════════════════════════════════════
    # 5. GUARDAR RESULTADOS PARA ANÁLISIS POSTERIOR
    # - optuna_rppo_results.csv: histórico de todos los 50 trials
    #   (incluye parámetros testeados, objetivo, valores de hiperparámetros)
    # - best_params_rppo.txt: parámetros óptimos en formato simple
    # ══════════════════════════════════════════════════════════════════════════════
    results_dir = os.path.join(project_root, "results", "optuna")
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        # Guardar histórico de trials para análisis posterior (Excel, matplotlib, etc.)
        df_results = study.trials_dataframe()
        df_results.to_csv(os.path.join(results_dir, "optuna_rppo_results.csv"), index=False)
        
        # Guardar solo los mejores parámetros en archivo de texto para copia manual
        # hacia train_rppo_trading_walk_forward_validation_D1.py
        with open(os.path.join(results_dir, "best_params_rppo.txt"), "w") as f:
            f.write("BEST HYPERPARAMETERS (RecurrentPPO)\n")
            f.write("="*50 + "\n\n")
            f.write("Copiar estos valores a train_rppo_trading_walk_forward_validation_D1.py\n\n")
            for param, value in study.best_params.items():
                f.write(f"{param}: {value}\n")
            f.write(f"\nSharpe Ratio: {study.best_value:.4f}\n")
        
        print(f"\n✓ Resultados guardados en: {results_dir}")
        print(f"  - optuna_rppo_results.csv: histórico de todos los trials")
        print(f"  - best_params_rppo.txt: parámetros óptimos")
        
    except Exception as e:
        print(f"\n Error al guardar resultados: {e}")
        print("Copia los parámetros manualmente desde la consola.")