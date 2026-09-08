@echo off
chcp 65001 >nul
title SAM Annotator - Instalador

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║              SAM Annotator - Instalador                     ║
echo ║         By A. Mauricio Devia Santoya                        ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo  [ES] Iniciando instalacion... esto puede tomar varios minutos.
echo  [EN] Starting installation... this may take several minutes.
echo  [PT] Iniciando instalacao... isso pode levar varios minutos.
echo.

:: ── 1. Verificar Python ────────────────────────────────────────────────────
echo [1/7] Verificando Python / Checking Python / Verificando Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  [ES] ERROR: Python no esta instalado o no esta en el PATH.
    echo  [EN] ERROR: Python is not installed or not in PATH.
    echo  [PT] ERRO: Python nao esta instalado ou nao esta no PATH.
    echo.
    echo  [ES] Descarga Python 3.11 desde: https://www.python.org/downloads/
    echo  [EN] Download Python 3.11 from:  https://www.python.org/downloads/
    echo  [PT] Baixe Python 3.11 em:       https://www.python.org/downloads/
    echo.
    echo  [ES] Asegurate de marcar "Add Python to PATH" durante la instalacion.
    echo  [EN] Make sure to check "Add Python to PATH" during installation.
    echo  [PT] Certifique-se de marcar "Add Python to PATH" durante a instalacao.
    echo.
    pause
    exit /b 1
)
python --version
echo  OK

:: ── 2. Crear entorno virtual ───────────────────────────────────────────────
echo.
echo [2/7] Creando entorno virtual / Creating virtual environment / Criando ambiente virtual...
if exist venv (
    echo  Ya existe, omitiendo / Already exists, skipping / Ja existe, ignorando.
) else (
    python -m venv venv
    if errorlevel 1 (
        echo  ERROR al crear venv.
        pause
        exit /b 1
    )
    echo  OK
)

:: ── 3. Activar venv ────────────────────────────────────────────────────────
echo.
echo [3/7] Activando entorno virtual / Activating virtual environment / Ativando ambiente virtual...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo  ERROR al activar venv.
    pause
    exit /b 1
)
echo  OK

:: ── 4. Actualizar pip ──────────────────────────────────────────────────────
echo.
echo [4/7] Actualizando pip / Upgrading pip / Atualizando pip...
python -m pip install --upgrade pip --quiet
echo  OK

:: ── 5. Detectar GPU y instalar PyTorch ────────────────────────────────────
echo.
echo [5/7] Detectando GPU / Detecting GPU / Detectando GPU...
echo.

:: Intentar detectar NVIDIA GPU via nvidia-smi
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo  [ES] No se detecto GPU NVIDIA. Instalando PyTorch version CPU.
    echo  [EN] No NVIDIA GPU detected. Installing PyTorch CPU version.
    echo  [PT] Nenhuma GPU NVIDIA detectada. Instalando PyTorch versao CPU.
    echo.
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
) else (
    echo  [ES] GPU NVIDIA detectada. Instalando PyTorch con soporte CUDA 12.4.
    echo  [EN] NVIDIA GPU detected. Installing PyTorch with CUDA 12.4 support.
    echo  [PT] GPU NVIDIA detectada. Instalando PyTorch com suporte CUDA 12.4.
    echo.
    pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124 --quiet
)
echo  OK

:: ── 6. Instalar dependencias del proyecto ─────────────────────────────────
echo.
echo [6/7] Instalando dependencias / Installing dependencies / Instalando dependencias...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo.
    echo  ERROR instalando dependencias. Revisa requirements.txt
    pause
    exit /b 1
)
echo  OK

:: ── 7. Descargar modelo SAM 2.1 ───────────────────────────────────────────
echo.
echo [7/7] Verificando modelo SAM 2.1 / Checking SAM 2.1 model / Verificando modelo SAM 2.1...
if exist models\sam2.1_hiera_small.pt (
    echo  Modelo ya existe / Model already exists / Modelo ja existe. OK
) else (
    echo.
    echo  [ES] Descargando modelo SAM 2.1 Small (~40MB)...
    echo  [EN] Downloading SAM 2.1 Small model (~40MB)...
    echo  [PT] Baixando modelo SAM 2.1 Small (~40MB)...
    mkdir models 2>nul
    python -c "import urllib.request; print('  Descargando...'); urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt', 'models/sam2.1_hiera_small.pt'); print('  Completado.')"
    if errorlevel 1 (
        echo.
        echo  [ES] No se pudo descargar automaticamente. Descargalo manualmente desde:
        echo  [EN] Could not download automatically. Download it manually from:
        echo  [PT] Nao foi possivel baixar automaticamente. Baixe manualmente em:
        echo  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
        echo  [ES] y colocalo en la carpeta: models\
        echo  [EN] and place it in the folder: models\
        echo  [PT] e coloque na pasta: models\
    )
)

:: ── Finalizar ──────────────────────────────────────────────────────────────
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                  Instalacion completada                     ║
echo ║               Installation complete                         ║
echo ║               Instalacao concluida                          ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo  [ES] Ahora puedes ejecutar: iniciar.bat
echo  [EN] You can now run: iniciar.bat
echo  [PT] Agora voce pode executar: iniciar.bat
echo.
pause
