@echo off
chcp 65001 >nul
title SAM Annotator

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                    SAM Annotator                            ║
echo ║           By A. Mauricio Devia Santoya                      ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

:: ── Verificar que la instalacion fue completada ───────────────────────────
if not exist venv (
    echo  [ES] ERROR: No se encontro el entorno virtual.
    echo  [EN] ERROR: Virtual environment not found.
    echo  [PT] ERRO: Ambiente virtual nao encontrado.
    echo.
    echo  [ES] Ejecuta primero: instalar.bat
    echo  [EN] Run first: instalar.bat
    echo  [PT] Execute primeiro: instalar.bat
    echo.
    pause
    exit /b 1
)

:: ── Activar venv ──────────────────────────────────────────────────────────
call venv\Scripts\activate.bat

:: ── Verificar modelo SAM ──────────────────────────────────────────────────
if not exist models\sam2.1_hiera_small.pt (
    echo  [ES] ADVERTENCIA: No se encontro el modelo SAM 2.1.
    echo  [EN] WARNING: SAM 2.1 model not found.
    echo  [PT] AVISO: Modelo SAM 2.1 nao encontrado.
    echo.
    echo  [ES] Ejecuta instalar.bat para descargarlo, o colocalo manualmente en models\
    echo  [EN] Run instalar.bat to download it, or place it manually in models\
    echo  [PT] Execute instalar.bat para baixa-lo, ou coloque manualmente em models\
    echo.
    pause
    exit /b 1
)

:: ── Abrir navegador automaticamente despues de 3 segundos ────────────────
echo  [ES] Iniciando SAM Annotator...
echo  [EN] Starting SAM Annotator...
echo  [PT] Iniciando SAM Annotator...
echo.
echo  [ES] La aplicacion abrira en tu navegador en unos segundos.
echo  [EN] The application will open in your browser in a few seconds.
echo  [PT] O aplicativo abrira no seu navegador em alguns segundos.
echo.
echo  URL: http://localhost:5000
echo.
echo  [ES] Para cerrar la app, cierra esta ventana o presiona Ctrl+C
echo  [EN] To close the app, close this window or press Ctrl+C
echo  [PT] Para fechar o app, feche esta janela ou pressione Ctrl+C
echo.
echo ──────────────────────────────────────────────────────────────

:: Abrir navegador con delay de 4 segundos (da tiempo a Flask para arrancar)
start /b cmd /c "timeout /t 4 /nobreak >nul && start http://localhost:5000"

:: ── Lanzar la aplicacion ──────────────────────────────────────────────────
python app.py

:: Si app.py termina (error o cierre), mostrar mensaje
echo.
echo  [ES] La aplicacion se ha cerrado.
echo  [EN] The application has been closed.
echo  [PT] O aplicativo foi encerrado.
echo.
pause
