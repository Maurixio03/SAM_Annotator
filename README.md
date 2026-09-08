# ▣ SAM Annotator

**By A. Mauricio Devia Santoya**  
📧 devia2001@outlook.com

---

## 🇪🇸 Español

### ¿Qué es SAM Annotator?

SAM Annotator es una aplicación web local para **anotación de imágenes y entrenamiento de modelos YOLO** de segmentación e instancias. 

**Características principales:**
- Anotación asistida por IA con SAM 2.1 (Segment Anything Model)
- Soporte multi-proyecto y multi-usuario con roles (superadmin, local_admin, annotator)
- Pipeline integrado: anotar → aumentar datos → exportar dataset → entrenar YOLO
- 30 modelos YOLO disponibles (YOLOv8, YOLO11, YOLO26 × n/s/m/l/x × seg/det)
- Publicación remota con Ngrok integrado
- Interfaz en Español, Português e English simultáneamente

### Requisitos del sistema

- **Windows 10/11** (64 bits)
- **Python 3.11** — [Descargar aquí](https://www.python.org/downloads/) *(marcar "Add Python to PATH")*
- **RAM:** mínimo 8 GB recomendado 16 GB
- **GPU NVIDIA** (opcional pero recomendado para SAM y entrenamiento YOLO)
- **Espacio en disco:** ~5 GB (PyTorch + modelos)
- Conexión a internet solo para la instalación inicial

### Instalación

1. Descarga o clona este repositorio
2. Doble clic en **`instalar.bat`**
3. Espera a que finalice (puede tardar 5-15 minutos dependiendo de tu conexión)
4. ¡Listo!

### Uso

1. Doble clic en **`iniciar.bat`**
2. La aplicación abrirá automáticamente en tu navegador en `http://localhost:5000`
3. Credenciales por defecto: usuario `admin` / contraseña `admin123`

> ⚠️ **Cambia la contraseña del administrador** después del primer inicio desde el panel de administración.

### Estructura de carpetas

```
SAM_Annotator/
├── instalar.bat          ← Instalador automático
├── iniciar.bat           ← Lanzador de la aplicación
├── requirements.txt      ← Dependencias Python
├── app.py                ← Aplicación principal (ofuscada)
├── templates/            ← Interfaz web
├── collab_data/          ← Proyectos y usuarios (se genera automáticamente)
└── models/               ← Modelos SAM (se descarga automáticamente)
```

---

## 🇧🇷 Português

### O que é o SAM Annotator?

SAM Annotator é uma aplicação web local para **anotação de imagens e treinamento de modelos YOLO** de segmentação e detecção de instâncias. 

**Principais funcionalidades:**
- Anotação assistida por IA com SAM 2.1 (Segment Anything Model)
- Suporte multi-projeto e multi-usuário com funções (superadmin, local_admin, annotator)
- Pipeline integrado: anotar → aumentar dados → exportar dataset → treinar YOLO
- 30 modelos YOLO disponíveis (YOLOv8, YOLO11, YOLO26 × n/s/m/l/x × seg/det)
- Publicação remota com Ngrok integrado
- Interface em Espanhol, Português e Inglês simultaneamente

### Requisitos do sistema

- **Windows 10/11** (64 bits)
- **Python 3.11** — [Baixar aqui](https://www.python.org/downloads/) *(marcar "Add Python to PATH")*
- **RAM:** mínimo 8 GB, recomendado 16 GB
- **GPU NVIDIA** (opcional, mas recomendado para SAM e treinamento YOLO)
- **Espaço em disco:** ~5 GB (PyTorch + modelos)
- Conexão à internet apenas para a instalação inicial

### Instalação

1. Baixe ou clone este repositório
2. Dê duplo clique em **`instalar.bat`**
3. Aguarde o término (pode levar 5 a 15 minutos dependendo da sua conexão)
4. Pronto!

### Uso

1. Dê duplo clique em **`iniciar.bat`**
2. O aplicativo abrirá automaticamente no navegador em `http://localhost:5000`
3. Credenciais padrão: usuário `admin` / senha `admin123`

> ⚠️ **Altere a senha do administrador** após o primeiro acesso no painel de administração.

---

## 🇬🇧 English

### What is SAM Annotator?

SAM Annotator is a local web application for **image annotation and YOLO model training** for instance segmentation and detection. 

**Key features:**
- AI-assisted annotation with SAM 2.1 (Segment Anything Model)
- Multi-project and multi-user support with roles (superadmin, local_admin, annotator)
- Integrated pipeline: annotate → augment data → export dataset → train YOLO
- 30 YOLO models available (YOLOv8, YOLO11, YOLO26 × n/s/m/l/x × seg/det)
- Remote publishing with integrated Ngrok
- Interface in Spanish, Portuguese, and English simultaneously

### System requirements

- **Windows 10/11** (64-bit)
- **Python 3.11** — [Download here](https://www.python.org/downloads/) *(check "Add Python to PATH")*
- **RAM:** minimum 8 GB, recommended 16 GB
- **NVIDIA GPU** (optional but recommended for SAM and YOLO training)
- **Disk space:** ~5 GB (PyTorch + models)
- Internet connection only required during initial installation

### Installation

1. Download or clone this repository
2. Double-click **`instalar.bat`**
3. Wait for it to finish (may take 5–15 minutes depending on your connection)
4. Done!

### Usage

1. Double-click **`iniciar.bat`**
2. The application will open automatically in your browser at `http://localhost:5000`
3. Default credentials: username `admin` / password `admin123`

> ⚠️ **Change the administrator password** after the first login from the administration panel.

---

## 📄 License / Licencia / Licença

This software is proprietary. The source code is protected.  
Este software es propietario. El código fuente está protegido.  
Este software é proprietário. O código-fonte está protegido.

© A. Mauricio Devia Santoya — Universidade de Brasília (UnB)
