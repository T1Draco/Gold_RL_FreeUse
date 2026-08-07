@echo off
REM ═══════════════════════════════════════════════════════════════════════════════
REM  SCRIPT BATCH: DESCARGADOR DE DATOS HISTÓRICOS (MT5)
REM ═══════════════════════════════════════════════════════════════════════════════
REM
REM Propósito:
REM   Automatiza la descarga de datos históricos de múltiples timeframes desde
REM   MetaTrader5 (MT5) para el activo XAU/USD (oro).
REM
REM Funcionamiento:
REM   1. Define rutas del proyecto y entorno virtual.
REM   2. Activa el entorno virtual (requirements.txt).
REM   3. Ejecuta tres scripts Python secuencialmente para descargar datos en:
REM      - H1 (horario) 
REM      - M15 (15 minutos)
REM      - D1 (diario)
REM
REM Requisitos previos:
REM   - Python instalado y en PATH.
REM   - Entorno virtual creado con: python -m venv .venv
REM   - MetaTrader5 activo y con cuenta conectada.
REM   - Credenciales MT5 configuradas (ver download_historical_stocks_price_MT5_*.py).
REM
REM Nota: Ajusta PROJECT_PATH según tu estructura local.
REM ═══════════════════════════════════════════════════════════════════════════════

title Descargador de Datos Históricos - MT5

echo === Iniciando proceso de descarga de datos historicos ===
echo.

REM ───────────────────────────────────────────────────────────────────────────────
REM PASO 1: DEFINIR RUTAS DEL PROYECTO Y ENTORNO VIRTUAL
REM ───────────────────────────────────────────────────────────────────────────────
REM PROJECT_PATH: Ruta raíz del proyecto (ajustar)
REM VENV_ACTIVATION_SCRIPT: Script para activar el entorno virtual Python
REM SCRIPTS_FOLDER: Carpeta que contiene los scripts de descarga
set "PROJECT_PATH=C:\Users\Admin\Documents\GitHub\Gold-RL-Austranet"
set "VENV_ACTIVATION_SCRIPT=%PROJECT_PATH%\.venv\Scripts\activate.bat"
set "SCRIPTS_FOLDER=%PROJECT_PATH%\Archivos Python y Data\1_Recoleccion_Datos\stock_data"

REM ───────────────────────────────────────────────────────────────────────────────
REM PASO 2: ACTIVAR ENTORNO VIRTUAL
REM ───────────────────────────────────────────────────────────────────────────────
REM El entorno virtual aísla las dependencias Python de este proyecto.
REM - Se activa el script activate.bat dentro de .venv\Scripts
REM - Si falla (errorlevel ≠ 0), muestra error y termina
REM - Si tiene éxito, todos los comandos "python" usan el intérprete del venv
echo Activando entorno virtual...
call "%VENV_ACTIVATION_SCRIPT%"
if %errorlevel% neq 0 (
    echo [ERROR] No se pudo activar el entorno virtual en: %VENV_ACTIVATION_SCRIPT%
    echo [ERROR] Asegurate de que la ruta es correcta.
    echo [ERROR] Si no existe, crea el venv con: python -m venv .venv
    goto end
)
echo Entorno virtual activado exitosamente.
echo.


REM ───────────────────────────────────────────────────────────────────────────────
REM PASO 3: CAMBIAR AL DIRECTORIO DE TRABAJO
REM ───────────────────────────────────────────────────────────────────────────────
REM - /d permite cambiar de unidad de disco si es necesario
REM - Necesario porque los scripts están en stock_data/
REM - Los datos se guardarán con rutas relativas a este directorio
echo Cambiando al directorio de trabajo...
cd /d "%SCRIPTS_FOLDER%"
if %errorlevel% neq 0 (
    echo [ERROR] No se pudo encontrar la carpeta de los scripts: %SCRIPTS_FOLDER%
    echo [ERROR] Verifica la estructura de tu proyecto.
    goto end
)
echo Directorio actual: %cd%
echo.


REM ───────────────────────────────────────────────────────────────────────────────
REM PASO 4: EJECUTAR SCRIPTS DE DESCARGA SECUENCIALMENTE
REM ───────────────────────────────────────────────────────────────────────────────
REM Cada script descarga datos de un timeframe diferente:
REM
REM - H1 (Horario): velas de 1 hora
REM   Archivo: download_historical_stocks_price_MT5_H1_blocks.py
REM   Salida: raw_data/XAUUSD_H1.csv
REM
REM - M15 (15 minutos): velas de 15 minutos
REM   Archivo: download_historical_stocks_price_MT5_M15_blocks.py
REM   Salida: raw_data/XAUUSD_M15.csv
REM
REM - D1 (Diario): velas de 1 día
REM   Archivo: download_historical_stocks_price_MT5_D1_blocks.py
REM   Salida: raw_data/XAUUSD_D1.csv
REM
REM Características:
REM - Descarga por bloques para evitar timeouts de MT5.
REM - Incluye denoising opcional con Filtro de Kalman.
REM - Crea backup automático si ya existe el archivo.
REM
REM Ejecución:
REM - Cada script se valida (if not exist) antes de ejecutar.
REM - Si hay error, se muestra el mensaje pero continúa con el siguiente.
REM - El proceso completo puede tomar varios minutos.
echo Ejecutando scripts de descarga de datos...
echo.


REM EJECUCIÓN DEL SCRIPT H1
REM Si el archivo no existe → error
REM Si existe → ejecuta con "python" (usa el intérprete del venv)
if not exist "download_historical_stocks_price_MT5_H1_blocks.py" (
    echo [ERROR] No se encuentra el script: download_historical_stocks_price_MT5_H1_blocks.py
) else (
    echo [INICIANDO] Descarga H1 (horario)...
    python download_historical_stocks_price_MT5_H1_blocks.py
    if %errorlevel% neq 0 (
        echo [ADVERTENCIA] Error durante descarga H1. Continuando con siguientes timeframes...
    ) else (
        echo [EXITOSO] Descarga H1 completada.
    )
)
echo.

REM EJECUCIÓN DEL SCRIPT M15
REM Mismo patrón: validación → ejecución → reporte
if not exist "download_historical_stocks_price_MT5_M15_blocks.py" (
    echo [ERROR] No se encuentra el script: download_historical_stocks_price_MT5_M15_blocks.py
) else (
    echo [INICIANDO] Descarga M15 (15 minutos)...
    python download_historical_stocks_price_MT5_M15_blocks.py
    if %errorlevel% neq 0 (
        echo [ADVERTENCIA] Error durante descarga M15. Continuando con siguientes timeframes...
    ) else (
        echo [EXITOSO] Descarga M15 completada.
    )
)
echo.

REM EJECUCIÓN DEL SCRIPT D1
REM Mismo patrón: validación → ejecución → reporte
if not exist "download_historical_stocks_price_MT5_D1_blocks.py" (
    echo [ERROR] No se encuentra el script: download_historical_stocks_price_MT5_D1_blocks.py
) else (
    echo [INICIANDO] Descarga D1 (diario)...
    python download_historical_stocks_price_MT5_D1_blocks.py
    if %errorlevel% neq 0 (
        echo [ADVERTENCIA] Error durante descarga D1.
    ) else (
        echo [EXITOSO] Descarga D1 completada.
    )
)
echo.

REM ───────────────────────────────────────────────────────────────────────────────
REM PASO 5: FINALIZACIÓN
REM ───────────────────────────────────────────────────────────────────────────────
REM - Muestra mensaje de finalización.
REM - Pausa para que el usuario pueda revisar los mensajes de éxito/error.
REM - Presionar cualquier tecla cierra la ventana.
echo === Proceso de descarga terminado ===
echo.
echo Revisa los mensajes arriba para verificar que todos los timeframes se descargaron correctamente.
echo Los archivos se guardarán en: raw_data/
echo.

:end
pause