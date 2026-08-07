import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import time
import os

"""
download_historical_stocks_price_MT5_D1_blocks.py

Script para descargar datos históricos desde MetaTrader5 en bloques.

Propósito:
- Descargar series OHLCV históricas en bloques (evitando timeouts) y guardar
    un CSV en `raw_data/` con el historial del símbolo.

Entradas/Salidas:
- Lee la configuración definida abajo (LOGIN, PASSWORD, SERVER, SYMBOL, TIMEFRAME)
- Guarda un archivo CSV en `raw_data/` con nombre `SYMBOL_TIMEFRAME.csv`.

Dependencias y requisitos:
- Requiere paquete `MetaTrader5` (pip install MetaTrader5) y `pandas`.
- MT5 debe estar instalado y configurado en la máquina donde se ejecuta.
- Uso de type-hints con `|` implica Python 3.10+.

Seguridad:
- Las credenciales no deberían estar hardcodeadas en el repositorio. Usar
    variables de entorno o un vault en entornos productivos.

Notas operativas:
- `TIMEFRAME` es un entero constante de MT5 (por ejemplo `mt5.TIMEFRAME_D1`).
    El nombre del archivo resultante incluirá ese entero, no necesariamente
    una etiqueta legible (p.ej. D1). Si se desea, mapping adicional puede
    transformarlo a una etiqueta más legible (no implementado aquí).
- Este script usa dos métodos de la API: `copy_rates_from` (descarga en bloques
    hacia atrás) y `copy_rates_range` (descarga entre dos fechas). Ambos están
    documentados en la API de MetaTrader5.
"""

# ============================================================
# CONFIGURACIÓN
# ============================================================
# NOTA: Estas credenciales deben ser gestionadas de forma segura (e.g., variables de entorno)
LOGIN = 104549224
PASSWORD = "..4OQxr/"
SERVER = "FBS-Demo"
SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_D1 # Marco temporal diario (D1)
CHUNK_SIZE = 5000           # Número de barras a descargar por llamada (eficiente para bloques)
MAX_BARRAS = 100000         # Límite total de barras históricas a intentar descargar
PAUSA_SEGUNDOS = 0.25       # Pausa entre llamadas para evitar saturar el servidor de MT5
# ============================================================

# Recomendación: no dejar `LOGIN`/`PASSWORD` en código. Usar variables de entorno:
#   LOGIN = int(os.getenv('MT5_LOGIN'))
#   PASSWORD = os.getenv('MT5_PASSWORD')
# CHUNK_SIZE y PAUSA_SEGUNDOS se eligen por compromiso entre velocidad y límites
# del servidor; ajustar para la cuenta / proveedor.

# === RUTAS RELATIVAS ===
# Define el directorio base como la ubicación del script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "raw_data")
# Asegurar que el directorio de salida exista
os.makedirs(RAW_DATA_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(RAW_DATA_DIR, f"{SYMBOL}_{TIMEFRAME}.csv")
# ============================================================

# Nota: `OUTPUT_FILE` incluirá el valor entero de `TIMEFRAME`. Ejemplo:
#  XAUUSD_86400.csv (dependiendo del valor numérico de mt5.TIMEFRAME_D1).
# Si prefieres nombres legibles (ej. XAUUSD_D1.csv), reemplazar el uso de
# `TIMEFRAME` por una etiqueta mapeada antes de construir el nombre de archivo.


def conectar_mt5(login: int, password: str, server: str) -> bool:
    """
    Inicializa y conecta la API de MetaTrader 5.

    Args:
        login (int): Número de cuenta MT5.
        password (str): Contraseña de la cuenta MT5.
        server (str): Nombre del servidor (e.g., "FBS-Demo").

    Returns:
        bool: True si la conexión y el login fueron exitosos, False en caso contrario.
    """
    # Detener cualquier conexión MT5 previa para asegurar un estado limpio
    mt5.shutdown()
    
    # Inicializar la conexión
    if not mt5.initialize():
        print(f"Error al inicializar MT5: {mt5.last_error()}")
        return False

    # Intentar el login con las credenciales
    if mt5.login(login, password=password, server=server):
        print(f" Conectado a la cuenta #{login}")
        return True
    else:
        # Fallo en el login, reportar error y cerrar la conexión
        print(f" Error de conexión: {mt5.last_error()}")
        mt5.shutdown()
        return False


def obtener_historico_hacia_atras(simbolo: str, timeframe: int, chunk_size: int, 
                                 max_barras: int, desde: datetime = None) -> pd.DataFrame | None:
    """
    Descarga datos históricos de MT5 en bloques, moviéndose hacia el pasado.

    Utiliza copy_rates_from para descargas masivas, optimizado para evitar
    problemas de timeout o límites de API al solicitar grandes rangos.

    Args:
        simbolo (str): Símbolo de trading (e.g., "XAUUSD").
        timeframe (int): Marco temporal MT5 (e.g., mt5.TIMEFRAME_D1).
        chunk_size (int): Número de barras solicitadas en cada llamada.
        max_barras (int): Límite total de barras a descargar.
        desde (datetime, opcional): Punto de inicio para la descarga.
            Si es None, comienza desde la barra más reciente.

    Returns:
        pd.DataFrame or None: DataFrame con datos OHLCV si la descarga es exitosa.
    """
    # Asegurar que el símbolo esté habilitado para la descarga
    simbolo_info = mt5.symbol_info(simbolo)
    if simbolo_info is None:
        if not mt5.symbol_select(simbolo, True):
            print(f"No se pudo habilitar {simbolo}")
            return None

    print(f"\nDescargando histórico de {simbolo} ({timeframe}) hacia atrás...")
    print("============================================================")

    all_data = []
    total_barras = 0

    # Determinar el punto de inicio de la descarga
    if desde is None:
        # Usar copy_rates_from_pos(0, 1) para obtener la última barra disponible
        current_pos = mt5.copy_rates_from_pos(simbolo, timeframe, 0, 1)
        if current_pos is None or len(current_pos) == 0:
            print("No se pudo obtener la última barra.")
            return None
        # Si 'desde' es None, la descarga comienza implícitamente desde el tiempo actual
        desde_time = datetime.fromtimestamp(current_pos[0]['time'])
    else:
        # Si se especifica una fecha, se usa como el punto inicial para retroceder
        desde_time = desde

    # Bucle de descarga por bloques
    while total_barras < max_barras:
        # Usar copy_rates_from, que descarga DESDE el punto en el tiempo
        # La API de MT5 lo interpreta como 'desde esta hora, obtén las N barras más antiguas'.
        rates = mt5.copy_rates_from(simbolo, timeframe, desde_time, chunk_size)
        if rates is None or len(rates) == 0:
            err = mt5.last_error()
            print(f"  Sin más datos o error: {err}. Deteniendo descarga.")
            break

        df_chunk = pd.DataFrame(rates)
        all_data.append(df_chunk)
        total_barras += len(df_chunk)
        
        # **Lógica de Desplazamiento Hacia Atrás:**
        # Actualizar 'desde_time' al timestamp de la barra MÁS ANTIGUA del bloque actual, 
        # y retroceder 1 segundo (timedelta(seconds=1)). Esto garantiza que la próxima
        # llamada copie datos ANTERIORES al bloque actual sin duplicar.
        desde_time = datetime.fromtimestamp(df_chunk['time'].min()) - timedelta(seconds=1)

        print(f"   → {len(df_chunk):5d} barras obtenidas | Total acumulado: {total_barras:,}")
        time.sleep(PAUSA_SEGUNDOS) # Pausa para cumplimiento de límite de API

        # Condición de finalización: si se descargó un bloque incompleto,
        # significa que se llegó al inicio del historial disponible.
        if len(df_chunk) < chunk_size:
            print("Fin del historial alcanzado (tamaño de bloque menor al CHUNK_SIZE).")
            break

    if not all_data:
        print("No se obtuvieron datos históricos. Retornando None.")
        return None

    # Post-procesamiento
    df = pd.concat(all_data).drop_duplicates(subset=["time"])
    # Convertir el timestamp UNIX (segundos) a objeto datetime
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.sort_values("time").reset_index(drop=True)

    # Columnas típicas devueltas por MT5: ['time','open','high','low','close','tick_volume','spread','real_volume']
    # Dependiendo del servidor estas columnas pueden variar; revisar siempre el CSV resultante.

    print("============================================================")
    print(f"DESCARGA COMPLETA: {len(df):,} barras")
    print(f"Desde: {df['time'].min()}")
    print(f"Hasta: {df['time'].max()}")
    print("============================================================")

    return df


def obtener_datos_en_rango(simbolo: str, timeframe: int, fecha_inicio: datetime, 
                          fecha_fin: datetime) -> pd.DataFrame | None:
    """
    Descarga datos de MT5 dentro de un rango de fechas específico (para actualización).

    Args:
        simbolo (str): Símbolo de trading.
        timeframe (int): Marco temporal MT5.
        fecha_inicio (datetime): Fecha y hora de inicio del rango.
        fecha_fin (datetime): Fecha y hora de fin del rango.

    Returns:
        pd.DataFrame or None: DataFrame con datos OHLCV nuevos.
    """
    print(f"\nBuscando nuevos datos desde {fecha_inicio} hasta {fecha_fin}...")
    # copy_rates_range es ideal para descargar datos entre dos puntos específicos
    rates = mt5.copy_rates_range(simbolo, timeframe, fecha_inicio, fecha_fin)
    
    if rates is None or len(rates) == 0:
        err = mt5.last_error()
        # El error 4014 a menudo significa "historial no disponible"
           # `mt5.last_error()` devuelve una tupla (codigo, descripcion). Algunos
           # códigos comunes: 10009 (no more data). Ajustar manejo según casos.
        if err[0] != 10009: # 10009 es "no more data" / no hay datos
               print(f"Error al obtener rango o sin datos: {err}")
        else:
             print("No se encontraron datos nuevos en el rango especificado.")
        return None
    
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    print(f"Se encontraron {len(df)} barras nuevas.")
    return df

def main():
    """
    Función principal que gestiona la conexión, la lógica de actualización
    y la descarga inicial de datos históricos de MT5.
    """
    # 1. Conexión a MT5
    if not conectar_mt5(LOGIN, PASSWORD, SERVER):
        return

    df_final = pd.DataFrame()

    # 2. Lógica de Actualización vs. Descarga Inicial
    if os.path.exists(OUTPUT_FILE):
        print(f"\nArchivo existente encontrado: {OUTPUT_FILE}")
        df_existente = pd.read_csv(OUTPUT_FILE)
        # Asegurarse de que la columna 'time' se interprete como datetime
        df_existente["time"] = pd.to_datetime(df_existente["time"])
        
        # Obtener la marca de tiempo más reciente del archivo local
        fecha_mas_reciente = df_existente["time"].max()
        print(f"Fecha más reciente en el archivo: {fecha_mas_reciente}")

        # Descargar datos nuevos desde la última barra + 1 segundo hasta ahora
        # Sumar timedelta(seconds=1) evita duplicar la última barra ya guardada.
        df_nuevos_recientes = obtener_datos_en_rango(
            SYMBOL, TIMEFRAME, 
            fecha_mas_reciente + timedelta(seconds=1), 
            datetime.now()
        )

        # Nota: `datetime.now()` es naive (sin timezone). Si trabajas con datos en
        # distintas zonas horarias o con servidores que devuelven timezone-aware,
        # considera normalizar a UTC y usar datetimes con tzinfo.
        
        # Combinar los datos existentes y los nuevos
        if df_nuevos_recientes is not None and not df_nuevos_recientes.empty:
            # Usar .values para evitar re-indexación compleja si las columnas no coinciden
            df_final = pd.concat([df_existente, df_nuevos_recientes], ignore_index=True)
            print(f"Datos combinados: {len(df_existente):,} (viejos) + {len(df_nuevos_recientes):,} (nuevos)")
        else:
            df_final = df_existente
            print("No se encontraron nuevos datos para actualizar.")

    else:
        print("\nNo se encontró archivo previo. Se realizará una descarga histórica completa hacia atrás.")
        # Primera ejecución: descarga masiva hacia atrás (máximo MAX_BARRAS)
        df_final = obtener_historico_hacia_atras(SYMBOL, TIMEFRAME, CHUNK_SIZE, MAX_BARRAS)

    # 3. Guardar el resultado final
    if df_final is not None and not df_final.empty:
        # Limpiar duplicados y ordenar cronológicamente
        df_final = (df_final
                    .drop_duplicates(subset=["time"])
                    .sort_values("time")
                    .reset_index(drop=True)
                   )
        
        # Guardar el archivo final en la carpeta 'raw_data'
        df_final.to_csv(OUTPUT_FILE, index=False)
        print(f"\nDatos totales guardados en: {OUTPUT_FILE}")
        print(f"Total de barras en archivo: {len(df_final):,}")
        print(f"Desde: {df_final['time'].min()}")
        print(f"Hasta: {df_final['time'].max()}")
    else:
        print("\nEl DataFrame final está vacío. No se guardó ningún archivo.")


    # 4. Desconexión
    mt5.shutdown()
    print("\nDesconectado de MT5")


if __name__ == "__main__":
    main()