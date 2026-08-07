"""
Trading Environment con Acciones Discretas y Recompensas Multi-Horizonte

Este módulo implementa un entorno de trading compatible con Gymnasium que utiliza:
- Acciones discretas (SHORT, NEUTRAL, LONG)
- Sistema de recompensas basado en múltiples horizontes temporales (1d, 5d, 20d, 60d)
- Gestión de posiciones con apalancamiento configurable
- Penalizaciones por drawdown exponencial y overtrading
- Windowing de features para memoria temporal del agente
- Normalización flexible de features (interna o externa)

Expectativas de entrada:
- `df` (pd.DataFrame): debe contener al menos columnas de precios y features.
    Se recomienda al menos las columnas: 'date' (o 'Date'), 'close' (precio filtrado),
    opcionalmente 'close_raw' (precio de ejecución), 'atr' y medias móviles usadas por la política
    (ej. 'sma_50'). Las columnas son sensibles a mayúsculas/minúsculas en este módulo.

Salida y utilidades:
- El entorno expone `observation_space` y `action_space` compatibles con Gymnasium.
- Métodos auxiliares: `get_portfolio_history()` y `get_closed_trades()` devuelven DataFrames
    con el historial y las operaciones cerradas respectivamente.

Dependencias y requisitos:
- Python 3.10+ (uso de `|` en type hints), numpy, pandas y gymnasium.
- Recomendación: usar `logging` en lugar de `print` para despliegues en producción.

Notas de seguridad/operación:
- Normalización: si `normalize_internal=False`, se asume que las features ya vienen
    preprocesadas/normalizadas. Para evaluaciones reproducibles, calcule `scaling_params`
    en train y páselos al entorno de evaluación.
"""

import numpy as np
import pandas as pd
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from collections import deque


class TradingEnvironmentDiscrete(Env):
    """
    Entorno de trading discreto con gestión avanzada de riesgo y recompensas multi-horizonte.
    
    Este entorno permite entrenar agentes de RL para trading mediante:
    - Observaciones que incluyen ventanas temporales de features del mercado
    - Variables de cuenta normalizadas (posición, PnL, net worth, etc.)
    - Sistema de recompensas que balancea rendimiento de corto y largo plazo
    - Penalizaciones diseñadas para evitar drawdowns excesivos y overtrading
    
    Attributes:
        window_size (int): Tamaño de la ventana temporal para features históricos
        normalize_internal (bool): Si True, normaliza features internamente; si False, asume features pre-normalizados
        features (list): Lista de columnas del DataFrame que se usan como features de entrada
        scaling_params (dict): Parámetros de normalización (media y std) por feature
        observation_space (Box): Espacio de observaciones del entorno
        action_space (Discrete): Espacio de acciones (0=SHORT, 1=NEUTRAL, 2=LONG)
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, df, 
                 initial_balance=10000, 
                 transaction_cost=0.0001, 
                 leverage=1.0, 
                 short_enabled=True, 
                 eval_mode=False, 
                 render_mode=None,
                 position_pct=0.25,
                 holding_penalty=0.0,
                 window_size=10,  
                 scaling_params=None,
                 normalize_internal=True):
        """
        Inicializa el entorno de trading.
        
        Args:
            df (pd.DataFrame): DataFrame con datos de mercado. Debe contener 'Date' y 'Close',
                             más cualquier número de features adicionales
            initial_balance (float): Capital inicial en USD
            transaction_cost (float): Costo de transacción como porcentaje (ej: 0.0001 = 0.01%)
            leverage (float): Apalancamiento máximo permitido
            short_enabled (bool): Si True, permite posiciones cortas (SHORT)
            eval_mode (bool): Si False, mezcla aleatoriamente el DataFrame; si True, mantiene orden
            render_mode (str): Modo de renderizado ('human', 'rgb_array', o None)
            position_pct (float): Porcentaje base del balance a usar por operación (legacy, reemplazado por DPS)
            holding_penalty (float): Penalización opcional por mantener posiciones
            window_size (int): Número de pasos históricos a incluir en la observación
            scaling_params (dict): Parámetros de normalización externos. Si None, se calculan internamente
            normalize_internal (bool): Si True, aplica normalización z-score; si False, usa valores raw del DataFrame
        """
        super().__init__()
    
        self.window_size = window_size
        self.normalize_internal = normalize_internal 
        
        # Configuración del entorno
        self.df = df.copy()
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.leverage = leverage
        self.short_enabled = short_enabled
        self.eval_mode = eval_mode
        self.render_mode = render_mode
        self.position_pct = position_pct 
        self.holding_penalty = holding_penalty

        # Mezcla aleatoria del DataFrame en modo entrenamiento
        if not self.eval_mode:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

        if len(self.df) < 2:
            raise ValueError(f"DataFrame demasiado corto. Se requieren al menos 2 filas.")

        # Detección automática de features
        # Excluye columnas no relacionadas con el estado del mercado
        
        # AGREGAMOS 'Close_Raw' A LA LISTA DE EXCLUSIÓN
        # Deben coincidir exactamente con los nombres en minúsculas
        # `NON_STATE_COLS` define columnas que no se consideran features de mercado.
        # Importante: la detección es sensible a mayúsculas/minúsculas — adapte si
        # su DataFrame usa 'Date'/'Close' en mayúsculas. Aquí se asumen nombres en minúscula.
        NON_STATE_COLS = ['date', 'close', 'close_raw', 'atr']

        # Construimos la lista de features excluyendo las columnas de estado
        # NO_STATE_COLS. Las columnas restantes se tratarán como features de mercado
        # y se normalizarán (si `normalize_internal=True`) usando `scaling_params`.
        self.features = [col for col in df.columns if col not in NON_STATE_COLS]
        
        if len(self.features) == 0:
            raise ValueError("No se encontraron features. El DataFrame debe contener columnas además de 'Date' y 'Close'.")
        if 'Close' in self.features:
            raise ValueError("ERROR CRÍTICO: 'Close' detectado como feature. Esto puede causar data leakage.")

        # Configuración de normalización
        if scaling_params is not None:
            # Usa parámetros externos (útil para test con parámetros de train)
            self.scaling_params = scaling_params
            print(f"Entorno cargado con {len(self.features)} features (Escalado FIJO externo).")
        else:
            # Calcula parámetros de normalización desde el DataFrame
            self.scaling_params = {}
            for feature in self.features:
                data = df[feature].dropna()
                mean = data.mean()
                std = data.std()
                if std == 0: 
                    std = 1.0  # Evita división por cero
                self.scaling_params[feature] = {'mean': mean, 'std': std}
            print(f"Entorno cargado con {len(self.features)} features (Escalado APRENDIDO interno).")
        
        # Definición de espacios
        n_features = len(self.features)
        account_vars_size = 7  # [pos_short, pos_neutral, pos_long, entry_price_norm, unrealized_pnl_norm, net_worth_norm, cash_ratio]
        # Tamaño total de observación: (features * ventana temporal) + variables de cuenta
        total_obs_size = (n_features * self.window_size) + account_vars_size

        # Observación (estructura):
        # - Primera sección: ventana temporal de features ordenada cronológicamente
        #   y aplanada: shape = (window_size * n_features,)
        # - Segunda sección: variables de cuenta en el siguiente orden:
        #   [pos_short (1/0), pos_neutral (1/0), pos_long (1/0), entry_price_norm,
        #    unrealized_pnl_norm, net_worth_norm, cash_ratio]

        
        self.observation_space = Box(low=-10.0, high=10.0, shape=(total_obs_size,), dtype=np.float64)
        self.window_buffer = deque(maxlen=self.window_size)
        
        # Espacio de acciones discreto: 3 acciones posibles
        self.action_space = Discrete(3)
        self.ACTION_SHORT = 0
        self.ACTION_NEUTRAL = 1
        self.ACTION_LONG = 2
        self.action_names = {0: "SHORT", 1: "NEUTRAL", 2: "LONG"}
        
        # Límites de pasos en el episodio
        self.min_step = 1
        self.max_step = len(self.df) - 1
        
        # Variables de estado del entorno (se inicializan en reset())
        self.current_step = None
        self.balance = None
        self.net_worth = None
        self.position_size_usd = None
        self.position = None
        self.entry_price = None
        self.previous_net_worth = None
        self.total_trades = None
        
        # Inicialización del estado
        self.reset()

        # Historial de net worth para cálculo de recompensas multi-horizonte
        self.net_worth_history_5d = deque(maxlen=5)
        self.net_worth_history_20d = deque(maxlen=20)
        self.net_worth_history_60d = deque(maxlen=60)

        # Variables para control de overtrading
        self.last_action = self.ACTION_NEUTRAL
        self.steps_since_last_change = 0
    
    def get_scaling_params(self):
        """
        Devuelve los parámetros de normalización calculados.
        
        Útil para transferir parámetros de un entorno de entrenamiento
        a un entorno de evaluación/test, garantizando consistencia.
        
        Returns:
            dict: Diccionario con estructura {feature: {'mean': float, 'std': float}}
        """
        return self.scaling_params

    def _normalize_feature(self, feature_name, value):
        """
        Normaliza un valor de feature usando z-score.
        
        Si normalize_internal es False, devuelve el valor sin modificar
        (asumiendo que ya viene normalizado del DataFrame).
        
        Args:
            feature_name (str): Nombre del feature a normalizar
            value (float): Valor raw del feature
            
        Returns:
            float: Valor normalizado (z-score) o valor raw si normalize_internal=False
        """
        if not self.normalize_internal:
            return value

        params = self.scaling_params.get(feature_name, {'mean': 0.0, 'std': 1.0})
        return (value - params['mean']) / params['std']
    
    def reset(self, seed=None, options=None):
        """
        Reinicia el entorno a su estado inicial.
        
        Inicializa todas las variables de estado, limpia el historial,
        y prepara el buffer de ventana temporal con los primeros pasos.
        
        Args:
            seed (int, optional): Semilla para reproducibilidad
            options (dict, optional): Opciones adicionales (no implementadas)
            
        Returns:
            tuple: (observación inicial, info dict)
        """
        super().reset(seed=seed)
        
        # Reinicio de variables de cuenta
        self.current_step = self.min_step
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.position_size_usd = 0.0
        self.position = self.ACTION_NEUTRAL
        self.entry_price = 0.0
        self.previous_net_worth = self.net_worth
        self.total_trades = 0
        
        # Reinicio de historial
        self.portfolio_history = []
        self.closed_trades = []
        self.last_position_change_step = self.min_step

        self.highest_price_since_entry = 0.0  # Rastreador para Long
        self.atr_multiplier = 3.0             # Factor de holgura (3x ATR)

        # Guardar estado inicial en historial
        self._append_to_history(self.current_step - 1, self.df.iloc[self.current_step - 1]['close'])

        # Pre-llenado del buffer de ventana temporal
        # Si estamos en el paso 0, repetimos el primer dato; si no, tomamos los N pasos anteriores
        first_step_idx = self.current_step - 1
        self.window_buffer.clear()
        
        for i in range(self.window_size):
            idx = max(0, first_step_idx - (self.window_size - 1) + i)
            feat_vec = self._get_feature_vector(idx)
            self.window_buffer.append(feat_vec)

        # Reinicio de estructuras para recompensas multi-horizonte
        self.net_worth_history_5d = deque(maxlen=5)
        self.net_worth_history_20d = deque(maxlen=20)
        self.net_worth_history_60d = deque(maxlen=60)
        
        # Reinicio de variables de control de trading
        self.last_action = self.ACTION_NEUTRAL
        self.steps_since_last_change = 0
        self.steps_in_position = 0
            
        return self._get_observation(), self._get_info()

    def _get_feature_vector(self, step_index):
        """
        Extrae y normaliza el vector de features de una fila específica.
        
        Args:
            step_index (int): Índice de la fila en el DataFrame
            
        Returns:
            np.ndarray: Array con features normalizados de la fila
        """
        row = self.df.iloc[step_index]
        vals = []
        for feature in self.features:
            val = self._normalize_feature(feature, row[feature])
            vals.append(val)
        return np.array(vals, dtype=np.float32)

    def _get_observation(self):
        """
        Construye la observación actual del entorno.
        
        La observación consta de dos partes:
        1. Estado del mercado: Ventana temporal de features históricos (aplanada)
        2. Estado de cuenta: Variables de posición, PnL, net worth, etc.
        
        Returns:
            np.ndarray: Vector de observación completo, clipped a [-10, 10]
        """
        # 1. Obtener features del paso actual
        current_feat = self._get_feature_vector(self.current_step - 1)
        
        # 2. Aplanar la ventana temporal: De (window_size, n_features) a vector 1D
        market_state = np.array(self.window_buffer).flatten()
        
        # 3. Construir variables de cuenta
        row = self.df.iloc[self.current_step - 1]
        current_price = row['close']
        
        # One-hot encoding de la posición actual
        position_encoded = [1.0 if self.position == i else 0.0 for i in range(3)]
        
        # Precio de entrada normalizado (0 si no hay posición)
        entry_price_norm = (self.entry_price / current_price - 1.0) if self.entry_price > 0 else 0.0
        
        safe_net_worth = max(self.net_worth, 1e-8)
        net_worth_norm = np.log(safe_net_worth / self.initial_balance)
        
        # Ratio de efectivo disponible
        cash_ratio = self.balance / self.net_worth if self.net_worth > 0 else 0.0
        
        # PnL no realizado normalizado
        unrealized_pnl_norm = (self._calculate_unrealized_pnl(current_price) / self.initial_balance)
        
        account_state = np.array(
            position_encoded + [entry_price_norm, unrealized_pnl_norm, net_worth_norm, cash_ratio], 
            dtype=np.float64
        )
        
        # Concatenar estado de mercado y estado de cuenta
        full_obs = np.concatenate((market_state, account_state))
        
        # Clip y manejo de NaNs para estabilidad numérica
        return np.clip(np.nan_to_num(full_obs), -10.0, 10.0)

    def _calculate_unrealized_pnl(self, current_price):
        """
        Calcula el PnL no realizado de la posición actual.
        
        Args:
            current_price (float): Precio actual del activo
            
        Returns:
            float: PnL no realizado en USD, considerando apalancamiento
        """
        if self.position == self.ACTION_NEUTRAL or self.entry_price == 0: 
            return 0.0
            
        price_change_pct = (current_price - self.entry_price) / self.entry_price
        
        # Para LONG: ganamos si el precio sube
        # Para SHORT: ganamos si el precio baja
        if self.position == self.ACTION_LONG:
            pnl = price_change_pct * self.position_size_usd
        else:  # SHORT
            pnl = -price_change_pct * self.position_size_usd
            
        # El PnL considera `position_size_usd` como exposición nocional; se aplica
        # `leverage` para reflejar la ganancia/pérdida sobre el capital apalancado.
        return pnl * self.leverage

    def _close_position(self, current_price):
        """
        Cierra la posición actual y actualiza el balance.
        
        Calcula el PnL realizado, las comisiones totales, y devuelve
        el margen más las ganancias/pérdidas al balance disponible.
        
        Args:
            current_price (float): Precio al que se cierra la posición
        """
        # Efectos secundarios importantes:
        # - Actualiza `self.balance` (se suma el margen liberado + PnL - comisiones)
        # - Agrega un registro en `self.closed_trades`
        # - Resetea variables de posición (`position`, `entry_price`, `position_size_usd`, `entry_fee`)
        # Nota: este método no devuelve valor; los cambios se reflejan en el estado del entorno.
        if self.position == self.ACTION_NEUTRAL: 
            return

        # 1. Calcular PnL realizado
        realized_pnl = self._calculate_unrealized_pnl(current_price)
        
        # 2. Calcular comisiones
        # La comisión se paga sobre el valor nocional total (valor de mercado de la posición)
        exit_fee = self.position_size_usd * self.transaction_cost 
        total_fee = self.entry_fee + exit_fee 
        
        # 3. Recuperar el margen
        # El margen es el capital real bloqueado: valor_nocional / apalancamiento
        # Ejemplo: Posición de $300 con leverage 3x requiere $100 de margen
        margin_used = self.position_size_usd / self.leverage
        
        # Lo que retorna al balance: margen + ganancia/pérdida - comisión de salida
        # (La comisión de entrada ya se restó al abrir)
        closing_return = margin_used + realized_pnl - exit_fee
        
        # 4. Registro del trade cerrado
        exit_idx = self.current_step
        if exit_idx >= len(self.df): 
            exit_idx = len(self.df) - 1
        entry_idx = self.last_position_change_step

        trade_info = {
            'entry_step': entry_idx,
            'exit_step': exit_idx,
            'entry_date': self.df.iloc[entry_idx]['date'] if 'date' in self.df.columns else None,
            'exit_date': self.df.iloc[exit_idx]['date'] if 'date' in self.df.columns else None,
            'type': self.action_names[self.position],
            'entry_price': self.entry_price,
            'exit_price': current_price,
            'position_size': margin_used,  # Margen real arriesgado
            'notional_size': self.position_size_usd,  # Tamaño total de la posición
            'pnl_usd': realized_pnl,
            'pnl_pct': (realized_pnl / margin_used) * 100 if margin_used > 0 else 0.0,
            'fee': total_fee,
            'net_pnl': realized_pnl - total_fee
        }
        self.closed_trades.append(trade_info)

        # 5. Actualizar balance
        self.balance += closing_return
        
        # 6. Resetear variables de posición
        self.position = self.ACTION_NEUTRAL
        self.entry_price = 0.0
        self.position_size_usd = 0.0
        self.entry_fee = 0.0

    def _open_position(self, action, current_price, confidence=1.0):
        """
        Abre una nueva posición con tamaño dinámico basado en confianza (DPS).
        
        Implementa Dynamic Position Sizing (DPS): ajusta el tamaño de la posición
        según la confianza del modelo (probabilidad de la acción predicha).
        
        Args:
            action (int): Acción a ejecutar (SHORT o LONG)
            current_price (float): Precio actual del activo
            confidence (float): Nivel de confianza del modelo [0.5, 1.0]
                               - 0.5: mínima confianza -> usa 15% del balance
                               - 1.0: máxima confianza -> usa 50% del balance
        """
        # DPS (Dynamic Position Sizing) — notas:
        # - `confidence` se clampa en [0.5, 1.0] y se mapea linealmente a un porcentaje
        #   de riesgo entre `min_pct` y `max_pct`.
        # - `trade_amount_usd` representa el margen que se bloqueará (capital propio);
        #   `notional_value_usd` = `trade_amount_usd * leverage` es la exposición total.
        # - Se deduce la comisión de entrada (`transaction_fee`) del balance al abrir.
        # - Validaciones: se evita abrir posiciones si no hay liquidez suficiente
        #   (balance < margin + comisión) o si `trade_amount_usd` es muy pequeño.
        if action == self.ACTION_NEUTRAL: 
            return
        
        # Ajuste del porcentaje de capital basado en confianza
        # Rango: 15% (baja confianza) a 50% (alta confianza)
        clamped_confidence = np.clip(confidence, 0.5, 1.0)
        
        min_pct = 0.10  # Mínimo capital a arriesgar
        max_pct = 0.75  # Máximo capital a arriesgar
        
        # Interpolación lineal: conf=0.5 -> 15%, conf=1.0 -> 50%
        adjusted_pct = (max_pct - min_pct) * (clamped_confidence - 0.5) / 0.5 + min_pct
        
        # Calcular margen requerido (capital a bloquear)
        trade_amount_usd = self.balance * adjusted_pct
        
        # Validación de tamaño mínimo
        if trade_amount_usd < 1.0: 
            return
        
        # Calcular valor nocional (exposición total con apalancamiento)
        notional_value_usd = trade_amount_usd * self.leverage
        
        # Calcular comisión de entrada
        transaction_fee = notional_value_usd * self.transaction_cost
        
        # Verificación de liquidez: se necesita margen + comisión
        if self.balance < trade_amount_usd + transaction_fee:
            return

        # Ejecución de la apertura
        self.entry_fee = transaction_fee 
        self.balance -= (trade_amount_usd + transaction_fee)  # Bloquear margen y pagar comisión
        self.position_size_usd = notional_value_usd  # Exposición total al mercado
        self.position = action
        self.entry_price = current_price
        
        # Actualizar estadísticas
        self.total_trades += 1
        self.last_position_change_step = self.current_step
        self.steps_in_position = 0

    def step(self, action, confidence=1.0):
        """
        Ejecuta un paso en el entorno.
        
        Procesa la acción del agente, actualiza el estado del mercado,
        calcula recompensas multi-horizonte con penalizaciones, y
        determina si el episodio ha terminado.
        
        Args:
            action (int): Acción a ejecutar (0=SHORT, 1=NEUTRAL, 2=LONG)
            confidence (float): Nivel de confianza del modelo para DPS
            
        Returns:
            tuple: (observación, recompensa, terminated, truncated, info)
                - observación (np.ndarray): Estado actual del entorno
                - recompensa (float): Recompensa del paso
                - terminated (bool): True si el episodio terminó
                - truncated (bool): Siempre False (no implementado)
                - info (dict): Información adicional del paso
        """
        action = int(action)

        # 0. VALIDACIÓN DE ACCIONES
        if action not in [0, 1, 2]: 
            action = self.ACTION_NEUTRAL
        if not self.short_enabled and action == self.ACTION_SHORT: 
            action = self.ACTION_NEUTRAL
        
        row = self.df.iloc[self.current_step] 
        
        # --- PRECIOS DIFERENCIADOS ---
        # Close_Raw para dinero, Close (filtrado) para observación/tendencia
        execution_price = row['close_raw'] if 'close_raw' in row else row['close']
        current_atr = row['atr'] if 'atr' in row else (execution_price * 0.01)
            
        if execution_price <= 1e-8:
             return (self._get_observation(), -1.0, True, False, {'error': 'Price error'})

        # 1. LÓGICA DE TRAILING STOP (CHANDELIER EXIT)
        # Solo aplica si ya estamos en una posición LONG
        if self.position == self.ACTION_LONG:
            # Actualizamos el pico máximo alcanzado desde que entramos
            self.highest_price_since_entry = max(self.highest_price_since_entry, execution_price)
            
            # El stop se sitúa a 3 veces la volatilidad (ATR) por debajo del pico máximo
            # Nota: `atr_multiplier` se define en `reset()` como 3.0, pero aquí
            # se usa 2.5*ATR en la condición de trailing. Hay una ligera
            # inconsistencia que se documenta aquí — si se desea armonizar,
            # sustituir 2.5 por `self.atr_multiplier` o ajustar la variable.
            trailing_stop = self.highest_price_since_entry - (2.5 * current_atr)
            
            # Si el precio REAL toca o cruza el stop, forzamos salida NEUTRAL
            if execution_price < trailing_stop:
                action = self.ACTION_NEUTRAL

        # 2. GESTIÓN DE POSICIONES (Ejecución real)
        if action == self.ACTION_NEUTRAL:
            if self.position != self.ACTION_NEUTRAL: 
                self._close_position(execution_price)
                self.highest_price_since_entry = 0.0 # Reset al salir
        elif action != self.position:
            # Si cambiamos de posición (o abrimos), inicializamos el rastreador de pico
            self._open_position(action, execution_price, confidence=confidence)
            self.highest_price_since_entry = execution_price 
        else:
            if self.position != self.ACTION_NEUTRAL: 
                self.steps_in_position += 1
            
        # 3. ACTUALIZAR PATRIMONIO (NET WORTH)
        if self.position == self.ACTION_NEUTRAL: 
            self.net_worth = self.balance
        else:
            unrealized_pnl = self._calculate_unrealized_pnl(execution_price) 
            self.net_worth = self.balance + self.position_size_usd + unrealized_pnl

        # 4. SISTEMA DE RECOMPENSAS MULTI-HORIZONTE
        # Calculamos los retornos logarítmicos basados en Net Worth Real
        if self.previous_net_worth > 1e-8 and self.net_worth > 1e-8:
            reward_1d = np.log(self.net_worth / self.previous_net_worth) * 100.0
        else:
            reward_1d = 0.0
        
        self.net_worth_history_5d.append(self.net_worth)
        self.net_worth_history_20d.append(self.net_worth)
        self.net_worth_history_60d.append(self.net_worth)
        
        # Definir un valor mínimo de seguridad para evitar log(0)
        eps = 1e-8

        # Reemplaza tus cálculos de recompensa por esto:
        if len(self.net_worth_history_5d) >= 5 and self.net_worth_history_5d[0] > eps:
            reward_5d = np.log(max(self.net_worth, eps) / self.net_worth_history_5d[0]) * 30.0
        else:
            reward_5d = 0.0

        if len(self.net_worth_history_20d) >= 20 and self.net_worth_history_20d[0] > eps:
            reward_20d = np.log(max(self.net_worth, eps) / self.net_worth_history_20d[0]) * 15.0
        else:
            reward_20d = 0.0

        if len(self.net_worth_history_60d) >= 60 and self.net_worth_history_60d[0] > eps:
            reward_60d = np.log(max(self.net_worth, eps) / self.net_worth_history_60d[0]) * 8.0
        else:
            reward_60d = 0.0
        
        # Ponderación optimizada para Swing
        # Pesos y factores de recompensa (documentación):
        # - reward_1d se escala por 100.0 para magnificar cambios intradiarios
        # - reward_5d se escala por 30.0, reward_20d por 15.0, reward_60d por 8.0
        # - La combinación final usa ponderaciones: 10% (1d), 35% (5d), 40% (20d), 15% (60d)
        # Ajustar estos multiplicadores y pesos según los objetivos de horizonte del agente.
        reward = (0.10 * reward_1d) + (0.35 * reward_5d) + (0.40 * reward_20d) + (0.15 * reward_60d)
        
        # 5. PENALIZACIONES Y BONOS
        # A. Drawdown: Castigamos el incremento del drawdown (más suave que la exponencial)
        peak = max(max(self.net_worth_history_60d), self.net_worth) if self.net_worth_history_60d else self.net_worth
        current_dd_pct = (peak - self.net_worth) / peak
        
        if current_dd_pct > 0.05: # Solo penalizar si el DD supera el 5%
            reward -= (current_dd_pct * 10.0)

        # B. Overtrading
        if action != self.position and self.steps_since_last_change < 10:
            reward -= 0.1 * (10 - self.steps_since_last_change) / 10
        
        # C. Tendencia (Uso de precio filtrado para la "visión" de tendencia)
        if self.position == self.ACTION_LONG:
            # Bono por estar a favor de la tendencia limpia
            if row['close'] > row['sma_50']:
                reward += 0.01
            # Bonus por mantener una posición ganadora (Holding Bonus)
            if (execution_price - self.entry_price) > current_atr:
                reward += 0.02

        # 6. ACTUALIZAR ESTADO PARA SIGUIENTE PASO
        self.previous_net_worth = self.net_worth
        self.steps_since_last_change = 0 if action != self.position else self.steps_since_last_change + 1
        self.current_step += 1
        
        # 7. TERMINACIÓN
        terminated = self.current_step >= len(self.df)
        if self.net_worth < self.initial_balance * 0.3: # Stop Out al 30%
            terminated = True
            reward = -20.0
        
        if terminated and self.position != self.ACTION_NEUTRAL:
            self._close_position(execution_price)
            self.net_worth = self.balance

        # En el método step(), sección de terminación:
        if self.net_worth < self.initial_balance * 0.1: # Si pierde el 90%
            terminated = True
            reward = -100.0 # Castigo masivo para que el gradiente se aleje de aquí

        # 8. INFO Y BUFFER
        info = self._get_info() 
        self._append_to_history(self.current_step - 1, execution_price)
        
        if not terminated:
            self.window_buffer.append(self._get_feature_vector(self.current_step - 1))
            
        return (self._get_observation(), reward, terminated, False, info)
    def _get_info(self):
        """
        Genera diccionario de información del paso actual.
        
        Returns:
            dict: Información sobre el estado actual del entorno
                - step: Paso actual del episodio
                - date: Fecha del dato actual (si existe en DataFrame)
                - price: Precio actual del activo
                - net_worth: Patrimonio neto actual
                - roi: Retorno de inversión porcentual
                - total_trades: Número total de operaciones realizadas
                - position: Posición actual como string
        """
        row = self.df.iloc[self.current_step - 1]
        # Mostrar precio real
        execution_price = row['close_raw'] if 'close_raw' in row else row['close']
        current_date = self.df.iloc[self.current_step - 1]['date'] if 'date' in self.df.columns else None
        return {
            'step': self.current_step,
            'date': current_date,
            'price': execution_price,
            'net_worth': self.net_worth,
            'roi': ((self.net_worth - self.initial_balance) / self.initial_balance) * 100,
            'total_trades': self.total_trades,
            'position': self.action_names[self.position]
        }

    def _append_to_history(self, step_index, current_price):
        """
        Agrega el estado actual al historial del portafolio.
        
        Args:
            step_index (int): Índice del paso a registrar
            current_price (float): Precio actual del activo
        """
        # Asegurar que el índice esté dentro de los límites del DataFrame
        safe_idx = max(0, min(step_index, len(self.df) - 1))
        self.portfolio_history.append({
            'step': self.current_step,
            'date': self.df.iloc[safe_idx]['date'] if 'date' in self.df.columns else None,
            'price': current_price,
            'net_worth': self.net_worth,
        })

    def render(self):
        """
        Renderiza el estado actual del entorno.
        
        En modo 'human', imprime información del paso actual en consola.
        """
        if self.render_mode == 'human':
            info = self._get_info()
            print(f"Step {info['step']}: Net Worth=${info['net_worth']:.2f} | ROI={info['roi']:.2f}%")

    def close(self):
        """
        Limpia recursos del entorno.
        
        Cierra cualquier figura de matplotlib abierta.
        """
        pass
        
    def get_portfolio_history(self):
        """
        Obtiene el historial completo del portafolio.
        
        Returns:
            pd.DataFrame: DataFrame con el historial de net worth paso a paso
        """
        return pd.DataFrame(self.portfolio_history)
        
    def get_closed_trades(self):
        """
        Obtiene el detalle de todas las operaciones cerradas.
        
        Returns:
            pd.DataFrame: DataFrame con información detallada de cada trade cerrado
                - entry_step, exit_step: Pasos de entrada/salida
                - entry_date, exit_date: Fechas de entrada/salida
                - type: Tipo de operación (LONG/SHORT)
                - entry_price, exit_price: Precios de entrada/salida
                - position_size: Margen utilizado
                - notional_size: Valor nocional total
                - pnl_usd: PnL en USD
                - pnl_pct: PnL porcentual
                - fee: Comisiones pagadas
                - net_pnl: PnL neto después de comisiones
        """
        return pd.DataFrame(self.closed_trades)


def crear_entorno_discreto(df, 
                          initial_balance=10000, 
                          transaction_cost=0.0001, 
                          leverage=1.0, 
                          short_enabled=True, 
                          eval_mode=False, 
                          render_mode=None, 
                          position_pct=0.25, 
                          holding_penalty=0.0, 
                          scaling_params=None, 
                          window_size=10, 
                          normalize_internal=True):
    """
    Función factory para crear instancias del entorno de trading.
    
    Esta función simplifica la creación del entorno y permite
    una configuración más limpia desde scripts externos.
    
    Args:
        df (pd.DataFrame): DataFrame con datos de mercado
        initial_balance (float): Capital inicial en USD
        transaction_cost (float): Costo de transacción como porcentaje
        leverage (float): Apalancamiento máximo
        short_enabled (bool): Permite posiciones cortas
        eval_mode (bool): Modo evaluación (no mezcla datos)
        render_mode (str): Modo de renderizado
        position_pct (float): Porcentaje base de posición (legacy)
        holding_penalty (float): Penalización por mantener posiciones
        scaling_params (dict): Parámetros de normalización externos
        window_size (int): Tamaño de ventana temporal
        normalize_internal (bool): Aplica normalización interna
        
    Returns:
        TradingEnvironmentDiscrete: Instancia del entorno configurada
    """
    return TradingEnvironmentDiscrete(
        df=df, 
        initial_balance=initial_balance, 
        transaction_cost=transaction_cost, 
        leverage=leverage, 
        short_enabled=short_enabled, 
        eval_mode=eval_mode, 
        render_mode=render_mode,
        position_pct=position_pct,
        holding_penalty=holding_penalty,
        window_size=window_size,
        scaling_params=scaling_params,
        normalize_internal=normalize_internal
    )


def validar_dataframe(df):
    """
    Valida que el DataFrame tenga la estructura mínima requerida.
    
    Verifica que existan las columnas esenciales ('Date' y 'Close')
    y al menos un feature adicional para el estado del mercado.
    
    Args:
        df (pd.DataFrame): DataFrame a validar
        
    Returns:
        tuple: (bool, str)
            - bool: True si el DataFrame es válido, False si no
            - str: Mensaje descriptivo del resultado de la validación
    """
    REQUIRED_COLS = ['Date', 'Close']
    missing_columns = [col for col in REQUIRED_COLS if col not in df.columns]
    
    if missing_columns:
        return False, f"Faltan columnas esenciales: {missing_columns}"
    
    if len(df.columns) < 3:
        return False, "El DataFrame debe tener al menos una feature además de 'Date' y 'Close'."
        
    return True, "DataFrame válido (Estructura Flexible)"