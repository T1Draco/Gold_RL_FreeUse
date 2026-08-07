"""
analyze_trades_walk_forward_validation.py

Genera un dashboard de análisis detallado de operaciones (trades) ejecutadas
por el agente RL durante la validación walk-forward.

Comportamiento:
- Lee el CSV de trades generado durante el entrenamiento walk-forward
  (`trades_rppo.csv`), que contiene un registro de cada operación.
- Lee el CSV de precios limpios (`XAUUSD_D1_rl.csv`) como referencia.
- Calcula estadísticas granulares por operación (duración, PnL, tamaño).
- Genera un dashboard visual (4 subgráficos) con distribuciones y métricas.
- Guarda la imagen como `board_trades.png`.

Entradas esperadas:
- `trades_rppo.csv`: CSV con columnas mínimas:
    * entry_date, exit_date (timestamps de entrada/salida)
    * entry_step, exit_step (índices de velas en la serie temporal)
    * net_pnl (ganancia/pérdida neta en USD de cada trade)
    * notional_size (valor nocional/tamaño de posición en USD)
- `XAUUSD_D1_rl.csv`: CSV de precios históricos con columna `Date` para
  indexación.

Salidas:
- Archivo `board_trades.png` con visualizaciones de:
    * Distribución de PnL neto (histograma)
    * Frecuencia de duraciones (días/velas)
    * Distribución de tamaños de posición (boxplot)
    * Win rate por duración de trade (barplot con etiquetas)

Supuestos:
- Los datos ya están limpios y en formato correcto.
- Las fechas en ambos CSVs están en formato ISO (YYYY-MM-DD).
- El script se ejecuta en el directorio de evaluación o con rutas absolutas.

Notas técnicas:
- Columna `duration_steps`: diferencia (exit_step - entry_step)
- Columna `is_win`: booleano derivado de net_pnl > 0
- Win rate: porcentaje de trades ganadores sobre el total
- Profit factor: ratio de ganancias/pérdidas (informativo, no utilizado aquí)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ═══════════════════════════════════════════════════════════════════════════════
#                         CONFIGURACIÓN Y RUTAS
# ═══════════════════════════════════════════════════════════════════════════════

# Configuración de estilo visual (seaborn)
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# Rutas de entrada (relativas al script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRADES_FILE = os.path.join(BASE_DIR, "..", "results", "walk_forward", "trades_rppo.csv")
BRUTE_DATA_FILE = os.path.join(BASE_DIR, "..", "clean_data", "XAUUSD_D1_rl.csv")


def generate_final_dashboard(trades_path, price_path):
    """
    Genera y guarda un dashboard gráfico con análisis detallado de trades.

    El dashboard incluye 4 subgráficos que visualizan diferentes aspectos
    de las operaciones ejecutadas por el agente RL:
    1. Distribución de PnL neto: Histograma con línea en cero (ganancias vs pérdidas)
    2. Duración de trades: Gráfico de frecuencia (cuántos trades duraron N velas)
    3. Distribución de tamaño de posición: Boxplot para identificar outliers
    4. Win rate por duración: Barplot con etiquetas de porcentaje

    Args:
        trades_path (str): Ruta absoluta o relativa al CSV de trades.
            Debe contener columnas:
            - entry_date, exit_date (datetime)
            - entry_step, exit_step (int)
            - net_pnl (float, en USD)
            - notional_size (float, tamaño nocional de posición)

        price_path (str): Ruta al CSV con precios históricos.
            Se usa principalmente para validación de fechas.
            Debe contener columna `Date`.

    Output:
        Archivo `board_trades.png` guardado en el directorio actual.

    Notas:
        - Si trades_path o price_path no existen, la función intentará
          procesarlos igualmente; pandas lanzará un error si hay fallos.
        - Las duraciones se calculan como (exit_step - entry_step).
        - Win rate = porcentaje de trades con net_pnl > 0.
        - Los gráficos usan la paleta de colores configurada en matplotlib.

    Ejemplo de uso:
        >>> generate_final_dashboard(
        ...     './results/walk_forward/trades_rppo.csv',
        ...     './clean_data/XAUUSD_D1_rl.csv'
        ... )
        # Genera ./board_trades.png
    """
    # 1. Cargar datos
    # Lee el CSV de trades del agente RL y el CSV de precios históricos
    df_trades = pd.read_csv(trades_path)
    df_price = pd.read_csv(price_path)
    
    # Convertir columnas de fecha a datetime para operaciones temporales
    df_trades['entry_date'] = pd.to_datetime(df_trades['entry_date'])
    df_trades['exit_date'] = pd.to_datetime(df_trades['exit_date'])
    df_price['Date'] = pd.to_datetime(df_price['Date'])
    df_price.set_index('Date', inplace=True)

    # 2. Cálculos y métricas derivadas
    # Duración de cada operación en número de velas/pasos
    df_trades['duration_steps'] = (df_trades['exit_step'] - df_trades['entry_step']).astype(int)
    
    # Indicador booleano para identificar operaciones ganadoras
    df_trades['is_win'] = df_trades['net_pnl'] > 0
    
    # Estadísticas generales (informativas para contexto)
    total_trades = len(df_trades)
    win_rate = df_trades['is_win'].mean() * 100
    total_pnl = df_trades['net_pnl'].sum()
    avg_trade_pnl = df_trades['net_pnl'].mean()
    profit_factor = df_trades[df_trades['net_pnl'] > 0]['net_pnl'].sum() / abs(df_trades[df_trades['net_pnl'] < 0]['net_pnl'].sum())

    # 3. Diseño del Dashboard (3 filas x 2 columnas = 4 subgráficos)
    # Se usa GridSpec para posicionar los gráficos con control fino del layout
    fig = plt.figure(constrained_layout=True, figsize=(16, 14))
    gs = fig.add_gridspec(3, 2)

    # SUBGRÁFICO 1: Distribución de PnL Neto (Histograma)
    # Visualiza la distribución de ganancias y pérdidas de todos los trades
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(df_trades['net_pnl'], kde=False, ax=ax1, color='#27ae60', bins=40)
    ax1.axvline(0, color='red', linestyle='--', linewidth=1.5)  # Línea de referencia en cero
    ax1.set_title('Distribución de PnL Neto (USD)', fontweight='bold')
    ax1.set_xlabel('Ganancia/Pérdida (USD)')
    ax1.set_ylabel('Frecuencia')

    # SUBGRÁFICO 2: Duración de Trades (Gráfico de Frecuencia)
    # Muestra cuántos trades duraron exactamente N velas
    # Ayuda a identificar patrones de duración típicos
    ax2 = fig.add_subplot(gs[0, 1])
    sns.countplot(data=df_trades, x='duration_steps', hue='duration_steps', palette='Blues_d', ax=ax2, legend=False)
    ax2.set_title('Frecuencia de Duración (Velas/Pasos)', fontweight='bold')
    ax2.set_xlabel('Duración (velas)')
    ax2.set_ylabel('Número de trades')

    # SUBGRÁFICO 3: Distribución de Position Size (Boxplot)
    # Visualiza el rango, mediana y outliers del tamaño de posiciones
    # Útil para entender la asignación dinámica de capital
    ax3 = fig.add_subplot(gs[1, 0])
    sns.boxplot(x=df_trades['notional_size'], color='#f1c40f', ax=ax3, width=0.5)
    ax3.set_title('Distribución de Position Size', fontweight='bold')
    ax3.set_xlabel('Tamaño de Posición (USD)')

    # SUBGRÁFICO 4: Win Rate por Duración (Barplot con Etiquetas)
    # Analiza si la duración del trade correlaciona con la probabilidad de ganancia
    # Cada barra representa el % de trades ganadores para esa duración
    ax4 = fig.add_subplot(gs[1, 1])
    wr_duration = df_trades.groupby('duration_steps')['is_win'].mean() * 100
    bars = sns.barplot(x=wr_duration.index, y=wr_duration.values, hue=wr_duration.index, palette='RdYlGn', ax=ax4, legend=False)
    ax4.set_title('Win Rate % por Duración', fontweight='bold')
    ax4.set_xlabel('Duración (velas)')
    ax4.set_ylabel('% de Acierto')
    ax4.set_ylim(0, 105)
    
    # Añadir etiquetas de porcentaje encima de cada barra
    for p in bars.patches:
        height = p.get_height()
        ax4.annotate(f'{height:.1f}%', 
                     xy=(p.get_x() + p.get_width() / 2., height),
                     xytext=(0, 3),  # Offset de 3 puntos hacia arriba
                     textcoords='offset points',
                     ha='center', va='bottom', fontsize=9)

    # Guardar la figura en alta calidad
    output_file = "board_trades.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    print(f"Dashboard de trades generado: {output_file}")


if __name__ == "__main__":
    """
    Punto de entrada del script.
    
    Cuando se ejecuta como script principal (python analyze_trades_walk_forward_validation.py),
    genera el dashboard de análisis de trades usando los archivos CSV especificados
    en las constantes TRADES_FILE y BRUTE_DATA_FILE.
    """
    generate_final_dashboard(TRADES_FILE, BRUTE_DATA_FILE)