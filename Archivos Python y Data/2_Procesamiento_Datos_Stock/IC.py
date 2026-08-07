import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr # Importación clave: Coeficiente de correlación de Spearman
import os

# =========================================================================
# FUNCIÓN: analizar_ic
# Descripción: Calcula y guarda el Information Coefficient (IC).
# =========================================================================

def analizar_ic(path: str = "./processed_data/XAUUSD_D1_processed.csv", 
                save_path: str = "./plt_ic/ic_top15_analysis.png"):
    """
    Calcula el Information Coefficient (IC) (Correlación de Spearman) entre
    variables de entrada y el retorno futuro ('y_target').

    Guarda el gráfico resultante a un archivo PNG para su inspección.

    Args:
        path (str): Ruta del archivo CSV de datos pre-procesados.
        save_path (str): Ruta completa donde se guardará el gráfico de resultados.
    
    Raises:
        ValueError: Si la columna 'y_target' no está presente en el DataFrame.
    """
    # --- Cargar datos ---
    df = pd.read_csv(path)
    
    # Verificación de la columna target
    if "y_target" not in df.columns:
        raise ValueError("El CSV debe contener la columna 'y_target'.")
    
    # Filtrar solo columnas numéricas, excluyendo el target
    df_num = df.select_dtypes(include=[np.number]).drop(
        columns=["y_target"], 
        errors="ignore"
    )
    resultados = []
    
    # --- Calcular Information Coefficient (IC) ---
    for col in df_num.columns:
        x = df[col] 
        y = df["y_target"] 
        
        # Máscara para manejo de NaN (listwise deletion)
        mask = x.notna() & y.notna()
        
        if mask.sum() < 5: # Requiere al menos 5 observaciones válidas
            continue
            
        # Correlación de Spearman como métrica IC (mide relación no lineal)
        ic_value, _ = spearmanr(x[mask], y[mask])
        
        resultados.append({"variable": col, "IC": ic_value})
    
    # Convertir resultados y ordenar por el valor absoluto de IC
    df_ic = pd.DataFrame(resultados)
    df_ic_sorted = df_ic.reindex(df_ic["IC"].abs().sort_values(ascending=False).index)
    top15 = df_ic_sorted.head(15)
    
    print("\nTOP 15 variables por mayor Information Coefficient (|IC|):\n")
    print(top15)
    
    # --- Generación y Guardado del Gráfico ---
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Colores por dirección de la correlación
    colors = ['#2ecc71' if ic > 0 else '#e74c3c' for ic in top15["IC"]]
    bars = ax.barh(top15["variable"], top15["IC"], 
                   color=colors, alpha=0.85, 
                   edgecolor='white', linewidth=1.5)
    
    # Línea de referencia en cero
    ax.axvline(0, color='#34495e', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Information Coefficient (Spearman)', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 Variables - Information Coefficient vs y_target', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    
    # Anotaciones corregidas (nunca se salen del borde)
    xlim = ax.get_xlim()
    margin = (xlim[1] - xlim[0]) * 0.02   # margen del 2%

    for i, (bar, ic) in enumerate(zip(bars, top15["IC"])):
        if ic > 0:
            x_pos = ic - margin
            ha = 'right'
        else:
            x_pos = ic + margin
            ha = 'left'
        
        ax.text(x_pos, i, f'{ic:.4f}',
                va='center', ha=ha,
                fontsize=9, fontweight='bold')

    
    plt.tight_layout()
    
    # Se guarda el gráfico en la ruta especificada.
    os.makedirs(os.path.dirname(save_path), exist_ok=True) # Asegurar que la carpeta exista
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"\n Gráfico del IC guardado correctamente en: {save_path}")
    plt.close(fig) # Liberar memoria de la figura

if __name__ == "__main__":
    analizar_ic()