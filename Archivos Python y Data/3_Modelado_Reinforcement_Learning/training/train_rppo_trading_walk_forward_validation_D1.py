"""
train_rppo_trading_walk_forward_validation_D1.py

Script principal de entrenamiento usando validación Walk-Forward para un agente de RL (RecurrentPPO)
en trading de XAU/USD con frecuencia D1 (diaria).

Flujo principal:
1. Carga datos limpios desde `XAUUSD_D1_rl.csv`.
2. Aplica denoising (Kalman) y normalización (Rolling Z-Score) a las features.
3. Divide en ventanas de entrenamiento/test usando estrategia walk-forward.
4. Para cada fold:
   - Entrena un modelo RecurrentPPO en la ventana de entrenamiento (in-sample).
   - Prueba en la ventana de test sin re-entrenamiento (out-of-sample).
   - Guarda todas las operaciones en `results/walk_forward/trades_rppo.csv`.

Características clave:
- Filtro de Kalman para denoising de precios.
- Rolling Z-Score (window=252) para normalización adaptativa.
- RecurrentPPO con LSTM interno para memoria temporal.
- Dynamic Position Sizing (DPS) con confianza del modelo.
- Trailing Stop para posiciones LONG.

Dependencias:
- stable_baselines3 (sb3_contrib: RecurrentPPO)
- PyTorch, NumPy, Pandas, Matplotlib
- Módulo local `env.multi_horizon_based_reward_env_v4` (entorno de trading)
"""

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
#from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
import torch
import random # Necesario para la función
import os
from collections import namedtuple # <-- NUEVA IMPORTACIÓN NECESARIA

# Definición de la estructura de estado LSTM esperada por sb3_contrib
LSTMCustomStates = namedtuple("LSTMCustomStates", ["pi", "vf"])

def set_global_seed(seed_value):
    """
    Establece la semilla para todos los generadores de números aleatorios 
    (Python, NumPy, PyTorch) para garantizar la reproducibilidad.
    """
    if seed_value is None:
        print("Advertencia: No se estableció ninguna semilla, la ejecución no será reproducible.")
        return

    # 1. Semillas de Python (Base)
    random.seed(seed_value)
    
    # 2. Semillas de NumPy
    np.random.seed(seed_value)
    
    # 3. Semillas de PyTorch
    torch.manual_seed(seed_value)
    # torch.cuda.manual_seed_all(seed_value) # Si usaras GPU
    
    # 4. Semillas del entorno (Necesario para SB3)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    # 5. Configuración de determinismo para PyTorch (puede ralentizar la ejecución)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Semilla global configurada a: {seed_value}")

# --- CLASE DE FILTRO DE KALMAN (Necesaria para Denoising de Precios) ---
class KalmanFilter1D:
    """
    Filtro de Kalman Simple para series de tiempo financieras (1D).
    Parámetros ajustados para el trading de XAU/USD (ajustes suaves).
    """
    def __init__(self, R=0.1, Q=1e-5):
        # R (Ruido de Medición): Desconfianza del precio observado (alto para trading).
        self.R = R  
        # Q (Ruido del Proceso): Qué tan rápido permitimos que la tendencia cambie.
        self.Q = Q  
        self.x_hat = None 
        self.P = 1.0     

    def update(self, measurement):
        if self.x_hat is None:
            self.x_hat = measurement
            return measurement

        # 1. Predicción (Ecuación 11 simplificada del paper)
        x_hat_minus = self.x_hat
        P_minus = self.P + self.Q

        # 2. Actualización (Ecuación 12 simplificada del paper)
        K = P_minus / (P_minus + self.R) 
        self.x_hat = x_hat_minus + K * (measurement - x_hat_minus)
        self.P = (1 - K) * P_minus

        return self.x_hat

def aplicar_kalman_a_columna(series, R=0.1, Q=1e-5):
    kf = KalmanFilter1D(R=R, Q=Q)
    return series.apply(kf.update)
# --------------------------------------------------------------------------


# Configuración de rutas
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

# Importamos el entorno con la lógica DPS y Multi-Horizonte
from env.multi_horizon_based_reward_env_v4 import crear_entorno_discreto, validar_dataframe

# ═══════════════════════════════════════════════════════════════════════════════
#                        CONFIGURACIÓN DEL EXPERIMENTO
# ═══════════════════════════════════════════════════════════════════════════════

RANDOM_SEED = 42 # 42 es un clásico.

DATA_PATH = os.path.join(project_root, "clean_data", "XAUUSD_D1_rl.csv") 

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN WALK-FORWARD
# ══════════════════════════════════════════════════════════════════════════════
# El walk-forward divide el historial en ventanas deslizantes.
# Cada iteración entrena en una ventana y prueba en la siguiente sin solapo.
TRAIN_WINDOW_SIZE = 252 * 5  # 5 años de entrenamiento (días de trading)
TEST_WINDOW_SIZE = 252 * 1   # 1 año de prueba (out-of-sample)
STEP_SIZE = 252 * 1          # Desplazamiento de 1 año por iteración
WINDOW_SIZE_OBS = 30         # Ventana de features para contexto LSTM (memoria temporal)

# Configuración del Agente (PPO) - Usando los valores óptimos encontrados
TOTAL_TIMESTEPS_PER_FOLD = 300_000 # Aumentado para asegurar convergencia (vs. 100k)
INITIAL_BALANCE = 10000

# Carpetas de salida
WF_RESULTS_DIR = os.path.join(project_root, "results", "walk_forward")
os.makedirs(WF_RESULTS_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
#                        LÓGICA WALK-FORWARD
# ═══════════════════════════════════════════════════════════════════════════════

def ejecutar_walk_forward():
    """
    Ejecuta la validación walk-forward con entrenamiento y prueba iterativos.

    Proceso:
    1. Carga y valida el CSV de datos limpios.
    2. Aplica transformaciones (Kalman, normalización).
    3. Itera sobre ventanas de entrenamiento/test sin solapo.
    4. Para cada fold:
       - Entrena RecurrentPPO en datos de entrenamiento.
       - Evalúa en datos de prueba con la memoria LSTM.
       - Recolecta trades cerrados.
    5. Guarda todos los trades en `results/walk_forward/trades_rppo.csv`.
    """
    # 1. Cargar Datos
    if not os.path.exists(DATA_PATH):
        print(f" Error: No se encuentran los datos en {DATA_PATH}")
        return
    
    df_full = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    total_rows = len(df_full)
    print(f" Datos cargados: {total_rows} registros ({df_full['Date'].min().date()} a {df_full['Date'].max().date()})")

    # Validar estructura del DataFrame
    es_valido, msj = validar_dataframe(df_full)
    if not es_valido:
        raise ValueError(msj)
    
    # ════════════════════════════════════════════════════════════════════════════════
    # FASE 1: DENOISING Y RECALCULO DE FEATURES
    # ════════════════════════════════════════════════════════════════════════════════
    # Objetivo: Generar features limpias y normalizadas para la red neuronal.
    # Paso clave: 'close' es filtrado (para tendencia IA) pero 'close_raw'
    # se preserva (para ejecución y PnL real).
    
    # 0. Normalizar nombres de columnas a minúsculas
    df_full.columns = [c.lower() for c in df_full.columns]
    col_close = 'close'
    col_high = 'high'
    col_low = 'low'

    # 1. GUARDAR PRECIO REAL ANTES DE FILTRAR
    # 'close_raw' = precio real del mercado (usado para ejecución en step())
    # 'close' = precio filtrado (usado para features de tendencia)
    df_full['close_raw'] = df_full[col_close].copy()

    # 2. Calcular el True Range (TR) y ATR sobre precios REALES
    # Esto asegura que la volatilidad refleje el movimiento real del mercado
    high_low = df_full[col_high] - df_full[col_low]
    high_cp = (df_full[col_high] - df_full[col_close].shift(1)).abs()
    low_cp = (df_full[col_low] - df_full[col_close].shift(1)).abs()
    df_full['tr'] = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    df_full['atr'] = df_full['tr'].rolling(window=14).mean()

    # 3. Aplicar Filtro de Kalman al precio de cierre
    # Reduce ruido de mercado para una señal más clara de tendencia
    print("   [PAPER] Aplicando Filtro de Kalman (Denoising)...")
    Q_H1 = 1e-4
    df_full['close'] = aplicar_kalman_a_columna(df_full['close'], R=0.1, Q=Q_H1)
    
    # 4. Recalcular Features (usando el 'close' ya limpio)
    df_full["retorno_log"] = np.log(df_full["close"] / df_full["close"].shift(1))
    df_full["volatilidad_20"] = df_full["retorno_log"].rolling(window=20).std()
    
    # 5. DEFINICIÓN DE LA LISTA (Ahora sí existe para ser usada después)
    features_finales_nn = [
        'date', 'close', 'close_raw', 'atr',
        'sma_200',                 # <--- NUEVA FEATURE AÑADIDA
        'precio_vwap_ratio',       
        'volumen_cambio',          
        'retorno_log',             
        'macd_histograma',           
        'sma_50',
        'ema_26'
    ]

    # Filtrado dinámico (esto asegura que sma_200 viaje al entorno)
    df_full = df_full[[col for col in features_finales_nn if col in df_full.columns]].copy()

    # ════════════════════════════════════════════════════════════════════════════════
    # FASE 2: ROLLING Z-SCORE NORMALIZATION (Adaptabilidad)
    # ════════════════════════════════════════════════════════════════════════════════
    # Normalización adaptativa: cada feature se normaliza con media/std
    # calculadas en una ventana móvil (252 días). Esto permite que la red
    # se adapte a diferentes regímenes de mercado.
    
    cols_to_normalize = [
        'precio_vwap_ratio', 
        'volumen_cambio', 
        'retorno_log', 
        'volatilidad_20', 
        'sma_50', 
        'ema_26'
    ]

    print("   [PAPER] Aplicando Rolling Z-Score (Window 252)...")

    for col in cols_to_normalize:
        if col in df_full.columns:
            rolling_mean = df_full[col].rolling(window=252, min_periods=20).mean()
            rolling_std = df_full[col].rolling(window=252, min_periods=20).std()
            # Normalizar: z = (x - media) / std. Epsilon (1e-8) evita división por cero.
            df_full[col] = (df_full[col] - rolling_mean) / (rolling_std + 1e-8)
        
    # Limpiar NaNs generados al inicio por el Rolling Window
    df_full.dropna(inplace=True)
    df_full.reset_index(drop=True, inplace=True)
    
    # ════════════════════════════════════════════════════════════════════
    #  FEATURE SELECTION FINAL (Solo 6 Alpha Features + Infraestructura)
    # ════════════════════════════════════════════════════════════════════
    
    features_finales_nn = [
            'date',            # Necesaria para logs y periodos
            'close',           # Visión IA (Kalman)
            'close_raw',       # Realidad (PnL)
            'atr',             # Gestión de Riesgo
            'precio_vwap_ratio',
            'volumen_cambio',
            'retorno_log',
            'macd_histograma',
            'sma_50',
            'ema_26'
        ]
    
    # Filtrado dinámico
    df_full = df_full[[col for col in features_finales_nn if col in df_full.columns]].copy()    
    
    print(f"\n [FILTRO FINAL] Red neuronal usará {len(features_finales_nn) - 2} features.")


    # 2. Definir Iteraciones
    # Aquí nos aseguramos de que el start_index esté después de la limpieza de NaNs
    total_rows = len(df_full)
    start_index = TRAIN_WINDOW_SIZE 
    
    all_trades = []
    current_balance = INITIAL_BALANCE
    iteration = 1
    
    print(f"\n INICIANDO VALIDACIÓN WALK-FORWARD")
    print(f"   Ventana de Entrenamiento: {TRAIN_WINDOW_SIZE} días")
    print(f"   Ventana de Observación:   {WINDOW_SIZE_OBS} días")
    print("="*80)

    while start_index + TEST_WINDOW_SIZE <= total_rows:
        # Definir índices de corte
        train_start = start_index - TRAIN_WINDOW_SIZE 
        
        train_end = start_index
        test_end = start_index + TEST_WINDOW_SIZE
        
        # Cortar DataFrames
        df_train = df_full.iloc[train_start:train_end].reset_index(drop=True)
        df_test = df_full.iloc[train_end:test_end].reset_index(drop=True)
        
        periodo_test_str = f"{df_test['date'].iloc[0].date()} -> {df_test['date'].iloc[-1].date()}"
        
        print(f"\n ITERACIÓN {iteration}: Probando en {periodo_test_str}")
        print(f"   Entrenando con {len(df_train)} días...")

        # --- A. ENTRENAMIENTO (In-Sample) ---
        env_train = crear_entorno_discreto(
            df=df_train,
            initial_balance=INITIAL_BALANCE,
            transaction_cost=0.0005,  
            leverage=3.0,            
            short_enabled=False,     
            eval_mode=False, 
            position_pct=0.25,
            window_size=WINDOW_SIZE_OBS,
            
            scaling_params=None,      
            normalize_internal=False  # <--- CRÍTICO: USA DATOS PRE-NORMALIZADOS
        )
        
        # trained_scaling_params ya no se usa aquí.
        
        print("   [CONFIG] Entrenando PPO con PARÁMETROS FINALES (ÓPTIMOS)...")
        
        # Construcción de la arquitectura de red: net_width=128, net_depth=2 => [128, 64]
        # Usamos una arquitectura sencilla para el MLP antes del LSTM
        net_arch_rp = [256, 128, 64] 

        # --- CAMBIO CRÍTICO: PPO -> RecurrentPPO ---
        model = RecurrentPPO( 
                    "MlpLstmPolicy", # Policy que usa MLP + LSTM (Estándar para RecurrentPPO)
                    env_train, verbose=0, 
                    
                    # --- PARÁMETROS RPPO ---
                    learning_rate=1.0379252144886538e-05,     
                    ent_coef=0.011618001142365253,
                    gamma=0.99,
                    n_steps=1024,           # Rollouts más cortos (H1)
                    batch_size=128,         
                    
                    policy_kwargs=dict(
                        net_arch=net_arch_rp, # Capas MLP antes y después del LSTM
                        lstm_hidden_size=64,  # <-- TAMAÑO DEL NÚCLEO DE MEMORIA (sugerencia del paper)
                        n_lstm_layers=1,      # Solo una capa LSTM para empezar
                    ), 
                    device="cpu",
                    seed=RANDOM_SEED)
                   
        model.learn(total_timesteps=TOTAL_TIMESTEPS_PER_FOLD)
        
        # --- B. PRUEBA (Out-of-Sample) ---
        env_test = crear_entorno_discreto(
            df=df_test,
            initial_balance=current_balance, 
            transaction_cost=0.0001,  
            leverage=3.0,             
            short_enabled=False,      
            eval_mode=True, 
            position_pct=0.25,
            window_size=WINDOW_SIZE_OBS,
            
            scaling_params=None,      
            normalize_internal=False
        )
        
        obs, _ = env_test.reset()
        done = False
        
        # ═══════════════════════════════════════════════════════════════
        # INICIALIZACIÓN DE LA MEMORIA (VOLVER A LA TUPLA SIMPLE (h, c))
        # ═══════════════════════════════════════════════════════════════
        
        LSTM_HIDDEN_SIZE = 64 
        LSTM_LAYERS = 1
        memory_shape = (LSTM_LAYERS, 1, LSTM_HIDDEN_SIZE)
        
        # Inicializar estados de memoria (h y c) - Tupla simple esperada por forward()
        lstm_states = (
            torch.zeros(memory_shape, device=model.device), 
            torch.zeros(memory_shape, device=model.device),
        )
        
        episode_starts = np.array([True])
        
        # Simulación RPPO
        while not done:
            
            # --- CONVERSIÓN CRÍTICA: episode_starts a FLOAT ---
            # Aseguramos que la bandera de inicio sea el tipo correcto (float)
            episode_starts_float = torch.as_tensor(episode_starts, dtype=torch.float32).to(model.device)
            # ----------------------------------------------------
            
            with torch.no_grad():
                # --- MÉTODO ROBUSTO: Usar el método interno de predicción ---
                # ... (Llamada a model.predict)
                actions_tensor, lstm_states_out = model.predict(
                    obs, 
                    state=lstm_states, 
                    episode_start=episode_starts, 
                    deterministic=False 
                )
                
                # actions_tensor es un array de NumPy (posiblemente 0-dimensional). 
                # Usamos .item() para obtener el valor escalar.
                action_final = actions_tensor.item() 
                
                # --- El resto del código de Confidence permanece igual ---
                
                # --- Cálculo de la Confianza (DPS) ---
                
                # 1. Preparar el tensor de observación 
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(model.device)
                
                # --------------------------------------------------------------------------------------
                # **NUEVA CORRECCIÓN: ASEGURAR QUE lstm_states ES UN TENSOR ANTES DE ENTRAR A LA RED**
                # --------------------------------------------------------------------------------------
                h, c = lstm_states
                if isinstance(h, np.ndarray):
                    # Si es un array de NumPy, lo convertimos de vuelta a Tensor de PyTorch
                    h = torch.as_tensor(h, dtype=torch.float32, device=model.device)
                    c = torch.as_tensor(c, dtype=torch.float32, device=model.device)
                    lstm_states = (h, c) # Actualizar lstm_states para que sean Tensors

                # 2. Obtener la distribución del modelo
                # El resultado es una tupla: (distribution_object, nuevo_estado_lstm_para_distribucion)
                distribution_result_tuple = model.policy.get_distribution(
                    obs_tensor, 
                    lstm_states=lstm_states, # La memoria simple (h, c)
                    episode_starts=episode_starts_float 
                )
                
                # 3. Acceder al objeto de distribución (es el primer elemento de la tupla)
                distribution_object = distribution_result_tuple[0]
                
                # 4. Calcular probabilidades: 
                #    Acceder al objeto Categorical de PyTorch (.distribution) 
                #    y de ahí a los logits (.logits).
                action_probs = torch.softmax(distribution_object.distribution.logits, dim=-1)
                
                # 5. Extraer la confianza para la acción ya elegida (action_final viene de model.predict)
                confidence = action_probs[0, action_final].item()

            # 2. Actualizar el estado de memoria para el próximo paso
            lstm_states = lstm_states_out # El resultado de model.predict es la nueva tupla (h', c')
            
            # 3. Actualizar la bandera episode_starts (para el siguiente paso)
            episode_starts = np.array([done]) 

            # 4. Pasar la acción y la confianza al entorno
            obs, _, done, _, info = env_test.step(action_final, confidence=confidence)

        # Recolectar resultados
        trades_fold = env_test.get_closed_trades()
        if not trades_fold.empty:
            all_trades.append(trades_fold)
            
        end_balance_fold = info['net_worth']
        roi_fold = (end_balance_fold - current_balance) / current_balance
        
        print(f"   Resultado Fold: ROI {roi_fold:.2%} | Balance: ${end_balance_fold:.2f} | Trades: {len(trades_fold)}")
        
        current_balance = end_balance_fold
        start_index += STEP_SIZE
        iteration += 1

    # ═══════════════════════════════════════════════════════════════════════════════
    #                        RESULTADOS FINALES
    # ═══════════════════════════════════════════════════════════════════════════════
    
    print("\n" + "="*80)
    print(" VALIDACIÓN WALK-FORWARD COMPLETADA")
    print("="*80)
    
    final_roi = (current_balance - INITIAL_BALANCE) / INITIAL_BALANCE
    print(f" Balance Inicial: ${INITIAL_BALANCE:.2f}")
    print(f" Balance Final:   ${current_balance:.2f}")
    print(f" ROI Acumulado:   {final_roi:.2%}")
    
    if all_trades:
        all_trades_df = pd.concat(all_trades)
        trades_path = os.path.join(WF_RESULTS_DIR, "trades_rppo.csv")
        all_trades_df.to_csv(trades_path, index=False)
        
        # Métricas simples
        win_trades = all_trades_df[all_trades_df['net_pnl'] > 0]
        hit_rate = len(win_trades) / len(all_trades_df)
        print(f" Total Trades:    {len(all_trades_df)}")
        print(f" Hit Rate Global: {hit_rate:.2%}")
        print(f" Detalle guardado en: {trades_path}")
    else:
        print(" No se realizaron operaciones.")

if __name__ == "__main__":
    # Asegura que todas las librerías usen la misma semilla
    set_global_seed(RANDOM_SEED) 
    ejecutar_walk_forward()