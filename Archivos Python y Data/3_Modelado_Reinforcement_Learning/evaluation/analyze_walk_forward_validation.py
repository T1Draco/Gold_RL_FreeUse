import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

"""
analyze_walk_forward_validation.py

Analiza los resultados del entrenamiento walk-forward comparando el desempeño
del agente RL contra benchmarks pasivos (Buy & Hold, Trend Following).

Comportamiento:
- Lee el CSV de trades generados por el agente RL (`trades_rppo.csv`).
- Lee el CSV de precios históricos limpios (`XAUUSD_D1_rl.csv`).
- Reconstruye la curva de equity del agente a partir del registro de PnLs.
- Calcula benchmarks simples (Buy & Hold, Media Móvil Cruce).
- Computa métricas estándar de rendimiento (CAGR, Sharpe, Max Drawdown,
  Calmar Ratio, Profit Factor, Hit Rate).
- Genera un dashboard comparativo (3 gráficos) con las curvas de equidad
  y drawdowns.
- Imprime tabla resumen en consola.

Entradas esperadas:
- `trades_rppo.csv`: CSV con registro de cada operación ejecutada.
  Columnas mínimas:
    * entry_date, exit_date (datetime)
    * entry_step, exit_step (índices en serie temporal)
    * net_pnl (float, ganancia/pérdida en USD)
    * notional_size (float, tamaño de posición)
- `XAUUSD_D1_rl.csv`: CSV de precios diarios XAU/USD.
  Columnas mínimas:
    * Date (datetime, índice de serie temporal)
    * Close (float, precio de cierre)

Salidas:
- Impresión en consola de tabla comparativa con métricas (CAGR, Sharpe, MDD, etc.)
- Archivo `board_validacion_rl_vs_benchmarks.png` con:
    * Gráfico 1: Comparativa de curvas de equidad (RL vs B&H vs Trend Follow)
    * Gráfico 2: Drawdowns normalizados de las tres estrategias

Parámetros configurables:
- INITIAL_BALANCE: Balance inicial para simulación (default: $10,000)
- RISK_FREE_RATE: Tasa libre de riesgo para cálculo de Sharpe (default: 0.0)
- TRADING_DAYS_YEAR: Días de trading anuales para anualización (default: 252)
- SMA_WINDOW: Ventana de media móvil corta para trend follower (default: 50)
- LMA_WINDOW: Ventana de media móvil larga para trend follower (default: 200)

Supuestos:
- Los datos de trades ya están filtrados para período de validación
  (no incluyen período de entrenamiento).
- Las fechas en ambos CSVs usan formato ISO (YYYY-MM-DD).
- El índice de fechas es continuo (se asume que datos faltantes son fin de semana).

Notas técnicas:
- CAGR (Compound Annual Growth Rate): Retorno anualizado geométrico.
- Sharpe: Exceso de retorno anualizado / volatilidad anualizada.
  Usa media aritmética de retornos diarios (aprox. correcta para CAGR corto).
- Max Drawdown: Máxima caída porcentual desde un pico a un valle.
- Calmar Ratio: CAGR / |Max Drawdown| (mayor es mejor, >1 es bueno).
- Hit Rate: Porcentaje de operaciones ganadoras (net_pnl > 0).
- Profit Factor: Suma ganancias / Suma pérdidas (>1.5 es bueno).
"""

# ═══════════════════════════════════════════════════════════════════════════════
#                         CONFIGURACIÓN Y RUTAS
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRADES_FILE = os.path.join(BASE_DIR, "..", "results", "walk_forward", "trades_rppo.csv")
BRUTE_DATA_FILE = os.path.join(BASE_DIR, "..", "clean_data", "XAUUSD_D1_rl.csv")

# Parámetros de simulación y cálculo de métricas
INITIAL_BALANCE = 10000        # Balance inicial en USD para simulaciones
RISK_FREE_RATE = 0.0           # Tasa libre de riesgo (para Sharpe)
TRADING_DAYS_YEAR = 252        # Días de trading anuales (estándar)

# Parámetros para el benchmark de Cruce de Medias Móviles
SMA_WINDOW = 50                # Ventana de media móvil corta
LMA_WINDOW = 200               # Ventana de media móvil larga

# ═══════════════════════════════════════════════════════════════════════════════
#                      FUNCIONES DE CÁLCULO DE MÉTRICAS
# ═══════════════════════════════════════════════════════════════════════════════


def calcular_max_drawdown(equity_curve):
    """
    Calcula el máximo drawdown (caída máxima) de una curva de equity.

    El drawdown es la caída porcentual desde un pico (máximo histórico) hasta
    un valle (mínimo posterior). Esta función calcula el drawdown para cada
    punto en la serie y retorna el máximo (más negativo).

    Args:
        equity_curve (iterable): Serie de valores de equity (floats).
            Ejemplo: [10000, 10500, 10200, 11000, ...]

    Returns:
        tuple: (max_drawdown, drawdown_array)
            - max_drawdown (float): Máximo drawdown como porcentaje negativo.
              Ejemplo: -0.25 representa un drawdown del 25%.
            - drawdown_array (np.ndarray): Array con drawdown en cada punto.
              Usado para visualización.

    Ejemplo:
        >>> equity = [10000, 12000, 10000, 11000]
        >>> mdd, dd_curve = calcular_max_drawdown(equity)
        >>> print(mdd)  # -0.1666... (caída de ~16.67% desde 12000 a 10000)
        >>> print(dd_curve)  # [0.0, 0.0, -0.1666..., -0.0909...]
    """
    equity = np.array(equity_curve)
    if len(equity) == 0: return 0.0, np.array([])
    peak = np.maximum.accumulate(equity)
    with np.errstate(divide='ignore', invalid='ignore'):
        drawdown = (equity - peak) / peak
    drawdown = np.nan_to_num(drawdown)
    max_mdd = np.min(drawdown)
    return max_mdd, drawdown


def calcular_profit_factor(df):
    """
    Calcula el Profit Factor de una serie de trades.

    Profit Factor = Suma de ganancias / Suma de pérdidas (en valor absoluto).

    Un PF > 1.0 indica ganancias positivas netas.
    Interpretación:
    - PF < 1.0: Estrategia con pérdidas
    - PF = 1.0 a 1.5: Apenas rentable
    - PF = 1.5 a 2.0: Bueno
    - PF > 2.0: Muy bueno
    - PF > 3.0: Excelente

    Args:
        df (pd.DataFrame): DataFrame con columna 'net_pnl' (gains/losses).

    Returns:
        float: Profit Factor. Si no hay pérdidas, retorna gross_profit / 1e-8
        para evitar división por cero.

    Ejemplo:
        >>> trades = pd.DataFrame({'net_pnl': [100, -50, 200, -30]})
        >>> pf = calcular_profit_factor(trades)
        >>> print(pf)  # (100+200) / (50+30) = 300/80 = 3.75
    """
    gross_profit = df[df['net_pnl'] > 0]['net_pnl'].sum()
    gross_loss = np.abs(df[df['net_pnl'] < 0]['net_pnl'].sum())
    return gross_profit / (gross_loss if gross_loss != 0 else 1e-8)


def calcular_retornos_anualizados(final_balance, initial_balance, duration_years, equity_curve):
    """
    Calcula CAGR (Compound Annual Growth Rate) y Sharpe Ratio anualizado.

    Fórmulas:
    - CAGR = (final_balance / initial_balance) ^ (1 / duration_years) - 1
    - Sharpe = (Retorno Anualizado - Tasa Libre de Riesgo) / Volatilidad Anualizada
    - Retorno anualizado se calcula con media aritmética de retornos diarios
    - Volatilidad se anualiza multiplicando por sqrt(TRADING_DAYS_YEAR)

    Args:
        final_balance (float): Balance final de la estrategia.
        initial_balance (float): Balance inicial (típicamente $10,000).
        duration_years (float): Duración del período en años.
        equity_curve (iterable): Serie de valores de equity diarios.

    Returns:
        tuple: (cagr, sharpe)
            - cagr (float): Retorno anualizado compuesto (ej: 0.15 para 15%)
            - sharpe (float): Ratio de Sharpe anualizado (ej: 2.5)

    Notas:
    - El Sharpe usa días de trading (TRADING_DAYS_YEAR = 252) para anualizar.
    - Si duration_years <= 0 o hay insuficientes retornos, retorna (0.0, 0.0).

    Ejemplo:
        >>> equity = [10000, 10100, 10200, 10400]
        >>> cagr, sharpe = calcular_retornos_anualizados(10400, 10000, 1.0, equity)
    """
    if duration_years <= 0: return 0.0, 0.0
    
    # 1. CAGR: Retorno compuesto anualizado (solo para mostrar retorno geométrico)
    cagr = (final_balance / initial_balance) ** (1 / duration_years) - 1 if final_balance > 0 else -1.0
    
    # Convertir a serie y asegurar que es datetime para filtrar
    equity_series = pd.Series(equity_curve)
    
    # 2. Calcular retornos diarios
    returns = equity_series.pct_change().dropna()
    
    if len(returns) < 2: return cagr, 0.0
    
    # --- CÁLCULO DE SHARPE RATIO ---
    # Si tu equity_curve incluye fines de semana (freq='D'), hay que filtrarlos
    # o si prefieres mantener 'D', debes usar sqrt(365). 
    # Lo estándar en finanzas es usar días de bolsa (Business Days = 252).
    
    # Calcular Sharpe sobre la media aritmética de retornos diarios
    mean_return_daily = returns.mean()
    std_return_daily = returns.std()
    
    # Anualizar usando media aritmética para el Sharpe
    ann_return_arithmetic = mean_return_daily * TRADING_DAYS_YEAR
    ann_vol = std_return_daily * np.sqrt(TRADING_DAYS_YEAR)
    
    # Fórmula Estándar del Sharpe Ratio
    sharpe = (ann_return_arithmetic - RISK_FREE_RATE) / ann_vol if ann_vol != 0 else 0.0
    
    return cagr, sharpe


# ═══════════════════════════════════════════════════════════════════════════════
#                         FUNCIONES DE BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════════
def calcular_buy_and_hold(df_bruto, start_date, end_date):
    """
    Calcula el desempeño de una estrategia Buy & Hold (comprar y mantener).

    Compra XAU/USD al precio de cierre del primer día y mantiene la posición
    hasta el final del período. No tiene costos de transacción ni comisiones.

    Args:
        df_bruto (pd.DataFrame): DataFrame con columnas 'Date' y 'Close'.
        start_date (datetime): Fecha inicial del período de evaluación.
        end_date (datetime): Fecha final del período de evaluación.

    Returns:
        dict: Diccionario con claves:
            - 'CAGR': Retorno anualizado compuesto
            - 'Sharpe': Ratio de Sharpe anualizado
            - 'MDD': Máximo drawdown
            - 'Equity': pd.Series con la curva de equity diaria

    Notas:
    - Si el período no tiene datos, retorna métricas en cero y una serie
      con balance inicial.
    - El precio inicial es el close del primer día disponible.

    Ejemplo:
        >>> result = calcular_buy_and_hold(df, date(2020,1,1), date(2021,12,31))
        >>> print(result['CAGR'])  # Ej: 0.05 para 5% anual
    """
    mask = (df_bruto['Date'] >= start_date) & (df_bruto['Date'] <= end_date)
    df_bh = df_bruto.loc[mask].copy().set_index('Date')
    if df_bh.empty: return {'CAGR': 0, 'Sharpe': 0, 'MDD': 0, 'Equity': pd.Series([INITIAL_BALANCE])}
    entry_price = df_bh['Close'].iloc[0]
    equity_series = INITIAL_BALANCE * (df_bh['Close'] / entry_price)
    mdd, _ = calcular_max_drawdown(equity_series.values)
    dur = (end_date - start_date).days / 365.25
    cagr, sharpe = calcular_retornos_anualizados(equity_series.iloc[-1], INITIAL_BALANCE, dur, equity_series)
    return {'CAGR': cagr, 'Sharpe': sharpe, 'MDD': mdd, 'Equity': equity_series}


def calcular_cruce_medias(df_bruto, start_date, end_date):
    """
    Implementa un benchmark de cruce de medias móviles (MA crossover).

    Estrategia simple:
    - Compra (long) cuando la media corta (SMA) > media larga (LMA)
    - Vende (close long) cuando la media corta <= media larga

    Parámetros (configurables en GLOBAL):
    - SMA_WINDOW: Ventana de media móvil corta (default: 50)
    - LMA_WINDOW: Ventana de media móvil larga (default: 200)

    Args:
        df_bruto (pd.DataFrame): DataFrame con columnas 'Date' y 'Close'.
        start_date (datetime): Fecha inicial del período de evaluación.
        end_date (datetime): Fecha final del período de evaluación.

    Returns:
        dict: Diccionario con claves:
            - 'CAGR': Retorno anualizado compuesto
            - 'Sharpe': Ratio de Sharpe anualizado
            - 'MDD': Máximo drawdown
            - 'Equity': pd.Series con la curva de equity diaria

    Notas:
    - Si el período no tiene datos suficientes, retorna métricas en cero.
    - Las señales se generan sobre datos históricos (sin anticipación).
    - El retorno de la estrategia es: daily_return * position_signal.shift(1)

    Ejemplo:
        >>> result = calcular_cruce_medias(df, date(2020,1,1), date(2021,12,31))
        >>> print(result['CAGR'])  # Ej: 0.08 para 8% anual
    """
    df_tf = df_bruto.copy()
    df_tf['SMA'] = df_tf['Close'].rolling(window=SMA_WINDOW).mean()
    df_tf['LMA'] = df_tf['Close'].rolling(window=LMA_WINDOW).mean()
    mask = (df_tf['Date'] >= start_date) & (df_tf['Date'] <= end_date)
    df_tf = df_tf.loc[mask].copy().set_index('Date')
    if df_tf.empty: return {'CAGR': 0, 'Sharpe': 0, 'MDD': 0, 'Equity': pd.Series([INITIAL_BALANCE])}
    
    df_tf['Signal'] = np.where(df_tf['SMA'] > df_tf['LMA'], 1, 0)
    df_tf['Strat_Ret'] = df_tf['Close'].pct_change() * df_tf['Signal'].shift(1)
    df_tf['Equity'] = INITIAL_BALANCE * (1 + df_tf['Strat_Ret'].fillna(0)).cumprod()
    
    mdd, _ = calcular_max_drawdown(df_tf['Equity'].values)
    dur = (end_date - start_date).days / 365.25
    cagr, sharpe = calcular_retornos_anualizados(df_tf['Equity'].iloc[-1], INITIAL_BALANCE, dur, df_tf['Equity'])
    return {'CAGR': cagr, 'Sharpe': sharpe, 'MDD': mdd, 'Equity': df_tf['Equity']}


# ═══════════════════════════════════════════════════════════════════════════════
#                        FUNCIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def analizar_resultados_completo():
    """
    Función principal que orquesta todo el análisis comparativo.

    Workflow:
    1. Carga de datos: Lee trades_rppo.csv y XAUUSD_D1_rl.csv
    2. Filtrado: Extrae el período de validación (no incluye entrenamiento)
    3. Reconstrucción: Genera curva de equity del agente RL desde PnL diarios
    4. Cálculo de benchmarks: Buy & Hold y Trend Follower
    5. Métricas: Calcula CAGR, Sharpe, MDD, Calmar Ratio, Profit Factor, Hit Rate
    6. Visualización: Genera gráficos comparativos
    7. Reporte: Imprime tabla en consola y guarda PNG

    Salidas:
    - Imprime en consola una tabla comparativa con todas las métricas
    - Guarda archivo 'board_validacion_rl_vs_benchmarks.png' con dos gráficos:
        * Gráfico 1 (arriba): Curvas de equity de las 3 estrategias
        * Gráfico 2 (abajo): Drawdowns diarios en porcentaje

    Manejo de errores:
    - Si algún archivo no existe, imprime un mensaje de error y retorna sin continuar
    - Si trades_df está vacío, retorna sin procesamiento

    Notas:
    - Usa business days ('B') para generar el índice de fechas de equity
    - Los benchmarks usan datos completos (incluyen período de entrenamiento)
    - El período evaluado es desde el primer trade hasta el último cierre

    Ejemplo:
        >>> analizar_resultados_completo()
        # Genera board_validacion_rl_vs_benchmarks.png
        # Imprime tabla de métricas en consola
    """
    print("Iniciando análisis comparativo completo (Excluyendo Entrenamiento)...")
    
    # --- Validación de archivos ---
    if not os.path.exists(TRADES_FILE) or not os.path.exists(BRUTE_DATA_FILE):
        print("Error: Archivos no encontrados.")
        print(f"  - TRADES_FILE: {TRADES_FILE}")
        print(f"  - BRUTE_DATA_FILE: {BRUTE_DATA_FILE}")
        return

    # --- Carga de datos ---
    trades_df = pd.read_csv(TRADES_FILE, parse_dates=['entry_date', 'exit_date'])
    df_bruto = pd.read_csv(BRUTE_DATA_FILE, parse_dates=['Date'])
    
    if trades_df.empty:
        print("Error: El archivo de trades está vacío.")
        return

    # --- 1. FILTRADO Y MÉTRICAS DE ACTIVIDAD ---
    # Ordena trades por fecha de entrada para análisis temporal
    trades_df = trades_df.sort_values(by='entry_date').reset_index(drop=True)
    start_eval = trades_df['entry_date'].min().normalize()
    end_eval = trades_df['exit_date'].max().normalize()
    
    # Calcula días de espera entre operaciones (inactividad)
    trades_df['wait_days'] = (trades_df['entry_date'] - trades_df['exit_date'].shift(1)).dt.days.clip(lower=0)
    avg_wait = trades_df['wait_days'].mean()
    num_trades = len(trades_df)

    # --- 2. RECONSTRUCCIÓN CURVA DE EQUITY RL ---
    # Crea una serie temporal diaria (solo días de trading, 'B'=Business Days)
    daily_index = pd.date_range(start=start_eval, end=end_eval, freq="B")
    daily_pnl = pd.Series(0.0, index=daily_index)
    
    # Suma PnL de todas las operaciones que cerraron en cada día
    for _, row in trades_df.iterrows():
        exit_day = row['exit_date'].normalize()
        if exit_day in daily_pnl.index:
            daily_pnl.loc[exit_day] += row['net_pnl']
    
    # Genera curva de equity acumulativa
    equity_rl = INITIAL_BALANCE + daily_pnl.cumsum().ffill()
    
    # Calcula métricas del agente RL
    mdd_rl, dd_curve_rl = calcular_max_drawdown(equity_rl.values)
    dur_years = (end_eval - start_eval).days / 365.25
    cagr_rl, sharpe_rl = calcular_retornos_anualizados(equity_rl.iloc[-1], INITIAL_BALANCE, dur_years, equity_rl)
    pf_rl = calcular_profit_factor(trades_df)
    hr_rl = (trades_df['net_pnl'] > 0).mean()

    # --- 3. CÁLCULO DE BENCHMARKS ---
    bh_res = calcular_buy_and_hold(df_bruto, start_eval, end_eval)
    tf_res = calcular_cruce_medias(df_bruto, start_eval, end_eval)
    
    # Extraer curvas y calcular drawdowns de benchmarks
    equity_bh = bh_res['Equity']
    equity_tf = tf_res['Equity']
    _, dd_curve_bh = calcular_max_drawdown(equity_bh.values)
    _, dd_curve_tf = calcular_max_drawdown(equity_tf.values)

    # --- 4. CÁLCULO DE CALMAR RATIO ---
    # Calmar Ratio = CAGR / |Max Drawdown|
    # Interpreta el retorno por unidad de riesgo (drawdown)
    calmar_rl = cagr_rl / abs(mdd_rl) if mdd_rl != 0 else 0
    calmar_bh = bh_res['CAGR'] / abs(bh_res['MDD']) if bh_res['MDD'] != 0 else 0
    calmar_tf = tf_res['CAGR'] / abs(tf_res['MDD']) if tf_res['MDD'] != 0 else 0

    # --- 5. GENERACIÓN DE TABLA COMPARATIVA ---
    data_comp = {
        'Métrica': ['CAGR', 'Sharpe', 'Max Drawdown', 'Calmar Ratio', 'Profit Factor', 'Hit Rate'],
        'Agente RL': [
            f'{cagr_rl:.2%}', 
            f'{sharpe_rl:.2f}', 
            f'{mdd_rl:.2%}', 
            f'{calmar_rl:.2f}',
            f'{pf_rl:.2f}', 
            f'{hr_rl:.2%}'
        ],
        'Buy & Hold': [
            f'{bh_res["CAGR"]:.2%}', 
            f'{bh_res["Sharpe"]:.2f}', 
            f'{bh_res["MDD"]:.2%}', 
            f'{calmar_bh:.2f}',
            '-', '-'
        ],
        'Trend Follow': [
            f'{tf_res["CAGR"]:.2%}', 
            f'{tf_res["Sharpe"]:.2f}', 
            f'{tf_res["MDD"]:.2%}', 
            f'{calmar_tf:.2f}',
            '-', '-'
        ]
    }
    
    df_comp = pd.DataFrame(data_comp)
    
    # --- 6. REPORTE EN CONSOLA ---
    print("\n" + "═"*70)
    print(f"REPORTES DE TRADES: {num_trades} | ESPERA MEDIA: {avg_wait:.2f} días")
    print("═"*70)
    print(df_comp.to_markdown(index=False))
    print("═"*70 + "\n")

    # --- 7. VISUALIZACIÓN DASHBOARD ---
    # Configura estilo visual de seaborn para los gráficos
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.4)

    # Gráfico 1 (arriba): Comparativa de Curvas de Equidad
    # Muestra el desempeño de las tres estrategias en una misma escala
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(equity_rl.index, equity_rl.values, label=f'RL (CAGR {cagr_rl:.1%})', color='#0ea5e9', lw=2.5)
    ax1.plot(equity_bh.index, equity_bh.values, label=f'B&H (CAGR {bh_res["CAGR"]:.1%})', color='#10b981', ls='--')
    ax1.plot(equity_tf.index, equity_tf.values, label=f'Trend Follow (CAGR {tf_res["CAGR"]:.1%})', color='#f59e0b', ls=':')
    ax1.set_title('Comparativa de Equidad: RL vs Benchmarks', fontsize=14, fontweight='bold')
    ax1.legend()

    # Gráfico 2 (abajo): Drawdowns Normalizados
    # Muestra el riesgo/caídas porcentuales de cada estrategia
    ax2 = fig.add_subplot(gs[1, :])
    ax2.fill_between(equity_rl.index, 0, dd_curve_rl * 100, color='#ef4444', alpha=0.2, label='RL')
    ax2.plot(equity_rl.index, dd_curve_rl * 100, color='#dc2626', lw=1.5)
    ax2.plot(equity_bh.index, dd_curve_bh * 100, color='#10b981', lw=1, alpha=0.5, label='B&H')
    ax2.set_title('Curvas de Drawdown (%)', fontsize=13, fontweight='bold')
    ax2.legend()

    # Título general del dashboard
    plt.suptitle(f'Board de Validación: Agente RL vs Benchmarks ({start_eval.year}-{end_eval.year})', fontsize=20, fontweight='bold')
    
    # Guarda la figura como PNG
    plt.savefig("board_validacion_rl_vs_benchmarks.png", bbox_inches='tight', dpi=150)
    print("Dashboard de validación guardado: board_validacion_rl_vs_benchmarks.png")


if __name__ == "__main__":
    """
    Punto de entrada del script.
    
    Ejecuta el análisis completo de validación walk-forward cuando se corre
    como script principal (python analyze_walk_forward_validation.py).
    """
    analizar_resultados_completo()