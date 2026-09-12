# SAM Annotator v3

<p align="center">
  <b>🇧🇷 Português</b> | <b>🇪🇸 Español</b> | <b>🇺🇸 English</b>
</p>

---

## 🇧🇷 Português

### O que é o SAM Annotator?
Ferramenta web completa para anotação de imagens com inteligência artificial (SAM 2.1), treinamento de modelos YOLO e colaboração em equipe. Roda localmente no seu computador — sem internet, sem custos.

**Funcionalidades:**
- Anotação automática com SAM 2.1 (clique e segmenta)
- Treinamento de YOLOv8, YOLO11 e YOLO26 (detecção e segmentação)
- Múltiplos usuários e projetos
- Exportação em YOLO, COCO, Pascal VOC, LabelMe e mais
- Colaboração em rede local ou via ngrok

---

### ✅ Requisitos do sistema
- Windows 10 ou 11 (64 bits)
- Mínimo 8GB RAM (16GB recomendado)
- Mínimo 10GB de espaço livre em disco
- GPU NVIDIA com CUDA (recomendado) ou CPU
- Navegador web (Chrome, Firefox, Edge)

---

### 📥 Passo 1 — Baixe os arquivos do Google Drive

Acesse os links abaixo e baixe os dois arquivos:

> 📦 **[SAM_Annotator_app.zip](https://drive.google.com/file/d/1z3cjAZ4Wt-aSZIA50QLi7CwXkE6yttIc/view?usp=sharing)** (~4GB) — Aplicativo compilado
> 📦 **[checkpoints.zip](https://drive.google.com/file/d/1HIUwQaxFn2S3bMdtLgcXbEW9_CbPSuQI/view?usp=sharing)** (~534MB) — Modelos SAM

> ⚠️ **Atenção:** Não é necessário instalar Python, pip ou nenhuma dependência manualmente. Tudo já está incluído nos arquivos acima.

---

### 📂 Passo 2 — Organize as pastas

Crie uma pasta em qualquer lugar do seu computador (ex: `C:\SAM_Annotator\`) e extraia os dois ZIPs dentro dela. A estrutura final deve ficar **exatamente assim**:

```
C:\SAM_Annotator\
├── app.dist\
│   ├── SAM_Annotator.exe   ← executável principal
│   ├── torch\
│   ├── templates\
│   └── ... (outros arquivos, não mover nem apagar)
└── checkpoints\
    ├── sam2.1_hiera_small.pt
    └── sam_vit_b_01ec64.pth
```

> ⚠️ **Importante:** As pastas `app.dist` e `checkpoints` devem estar **na mesma pasta raíz**. Se ficarem em locais diferentes, o modelo SAM não será encontrado.

---

### ▶️ Passo 3 — Execute o aplicativo

1. Abra a pasta `app.dist`
2. Dê **dois cliques** em `SAM_Annotator.exe`
3. Uma janela preta pode aparecer brevemente — é normal, aguarde
4. Após 10–15 segundos, abra seu navegador e acesse:

```
http://localhost:5000
```

---

### 🔑 Passo 4 — Primeiro acesso

| Campo | Valor |
|-------|-------|
| Usuário | `admin` |
| Senha | `admin123` |

> ⚠️ Troque a senha imediatamente após o primeiro acesso em: **Admin → Usuários → Alterar senha**

---

### 👥 Colaboração em equipe (opcional)

Para que sua equipe acesse o SAM Annotator pelo navegador de outros computadores:

1. Crie uma conta gratuita em [ngrok.com](https://ngrok.com)
2. Copie seu **Authtoken** no painel do ngrok
3. No SAM Annotator: **Admin → aba "Publicar"**
4. Cole o token e clique em **Iniciar**
5. Compartilhe o link gerado com sua equipe

---

### ❓ Problemas comuns

**O aplicativo não abre ou fecha imediatamente**
- Verifique se a porta 5000 não está sendo usada por outro programa
- Tente executar o `SAM_Annotator.exe` como Administrador (clique direito → Executar como administrador)

**Erro ao carregar modelo SAM**
- Verifique se a pasta `checkpoints` está no lugar correto (veja estrutura no Passo 2)
- Os dois arquivos `.pt` e `.pth` devem estar presentes dentro de `checkpoints`

**Página em branco no navegador**
- Aguarde mais tempo (até 30 segundos) após abrir o `.exe`
- Verifique se o endereço é exatamente `http://localhost:5000` (não https)

**Antivírus bloqueia o .exe**
- Adicione a pasta `app.dist` às exceções do seu antivírus
- O arquivo é seguro — foi compilado com Nuitka a partir de código Python

---

---

## 🇪🇸 Español

### ¿Qué es SAM Annotator?
Herramienta web completa para anotación de imágenes con inteligencia artificial (SAM 2.1), entrenamiento de modelos YOLO y colaboración en equipo. Corre localmente en tu computador — sin internet, sin costos.

**Funcionalidades:**
- Anotación automática con SAM 2.1 (clic y segmenta)
- Entrenamiento de YOLOv8, YOLO11 y YOLO26 (detección y segmentación)
- Múltiples usuarios y proyectos
- Exportación en YOLO, COCO, Pascal VOC, LabelMe y más
- Colaboración en red local o via ngrok

---

### ✅ Requisitos del sistema
- Windows 10 u 11 (64 bits)
- Mínimo 8GB RAM (16GB recomendado)
- Mínimo 10GB de espacio libre en disco
- GPU NVIDIA con CUDA (recomendado) o CPU
- Navegador web (Chrome, Firefox, Edge)

---

### 📥 Paso 1 — Descarga los archivos de Google Drive

Accede a los links y descarga los dos archivos:

> 📦 **[SAM_Annotator_app.zip](https://drive.google.com/file/d/1z3cjAZ4Wt-aSZIA50QLi7CwXkE6yttIc/view?usp=sharing)** (~4GB) — Aplicación compilada
> 📦 **[checkpoints.zip](https://drive.google.com/file/d/1HIUwQaxFn2S3bMdtLgcXbEW9_CbPSuQI/view?usp=sharing)** (~534MB) — Modelos SAM

> ⚠️ **Atención:** No es necesario instalar Python, pip ni ninguna dependencia manualmente. Todo ya está incluido en los archivos de arriba.

---

### 📂 Paso 2 — Organiza las carpetas

Crea una carpeta en cualquier lugar de tu computador (ej: `C:\SAM_Annotator\`) y extrae los dos ZIPs dentro de ella. La estructura final debe quedar **exactamente así**:

```
C:\SAM_Annotator\
├── app.dist\
│   ├── SAM_Annotator.exe   ← ejecutable principal
│   ├── torch\
│   ├── templates\
│   └── ... (otros archivos, no mover ni borrar)
└── checkpoints\
    ├── sam2.1_hiera_small.pt
    └── sam_vit_b_01ec64.pth
```

> ⚠️ **Importante:** Las carpetas `app.dist` y `checkpoints` deben estar en la **misma carpeta raíz**. Si quedan en lugares diferentes, el modelo SAM no será encontrado.

---

### ▶️ Paso 3 — Ejecuta la aplicación

1. Abre la carpeta `app.dist`
2. Haz **doble clic** en `SAM_Annotator.exe`
3. Puede aparecer una ventana negra brevemente — es normal, espera
4. Después de 10–15 segundos, abre tu navegador y accede a:

```
http://localhost:5000
```

---

### 🔑 Paso 4 — Primer acceso

| Campo | Valor |
|-------|-------|
| Usuario | `admin` |
| Contraseña | `admin123` |

> ⚠️ Cambia la contraseña inmediatamente después del primer acceso en: **Admin → Usuarios → Cambiar contraseña**

---

### 👥 Colaboración en equipo (opcional)

Para que tu equipo acceda al SAM Annotator desde otros computadores:

1. Crea una cuenta gratuita en [ngrok.com](https://ngrok.com)
2. Copia tu **Authtoken** en el panel de ngrok
3. En SAM Annotator: **Admin → pestaña "Publicar"**
4. Pega el token y haz clic en **Iniciar**
5. Comparte el link generado con tu equipo

---

### ❓ Problemas comunes

**La aplicación no abre o se cierra inmediatamente**
- Verifica que el puerto 5000 no esté siendo usado por otro programa
- Intenta ejecutar `SAM_Annotator.exe` como Administrador (clic derecho → Ejecutar como administrador)

**Error al cargar modelo SAM**
- Verifica que la carpeta `checkpoints` esté en el lugar correcto (ver estructura en Paso 2)
- Los dos archivos `.pt` y `.pth` deben estar presentes dentro de `checkpoints`

**Página en blanco en el navegador**
- Espera más tiempo (hasta 30 segundos) después de abrir el `.exe`
- Verifica que la dirección sea exactamente `http://localhost:5000` (no https)

**El antivirus bloquea el .exe**
- Agrega la carpeta `app.dist` a las excepciones de tu antivirus
- El archivo es seguro — fue compilado con Nuitka desde código Python

---

---

## 🇺🇸 English

### What is SAM Annotator?
A complete web-based tool for AI-powered image annotation (SAM 2.1), YOLO model training, and team collaboration. Runs locally on your computer — no internet required, no costs.

**Features:**
- Automatic annotation with SAM 2.1 (click and segment)
- Training of YOLOv8, YOLO11 and YOLO26 (detection and segmentation)
- Multiple users and projects
- Export in YOLO, COCO, Pascal VOC, LabelMe and more
- Team collaboration via local network or ngrok

---

### ✅ System requirements
- Windows 10 or 11 (64-bit)
- Minimum 8GB RAM (16GB recommended)
- Minimum 10GB free disk space
- NVIDIA GPU with CUDA (recommended) or CPU
- Web browser (Chrome, Firefox, Edge)

---

### 📥 Step 1 — Download files from Google Drive

Access the links and download both files:

> 📦 **[SAM_Annotator_app.zip](https://drive.google.com/file/d/1z3cjAZ4Wt-aSZIA50QLi7CwXkE6yttIc/view?usp=sharing)** (~4GB) — Compiled application
> 📦 **[checkpoints.zip](https://drive.google.com/file/d/1HIUwQaxFn2S3bMdtLgcXbEW9_CbPSuQI/view?usp=sharing)** (~534MB) — SAM models

> ⚠️ **Note:** You do NOT need to install Python, pip, or any dependencies manually. Everything is already included in the files above.

---

### 📂 Step 2 — Organize the folders

Create a folder anywhere on your computer (e.g. `C:\SAM_Annotator\`) and extract both ZIPs inside it. The final structure must look **exactly like this**:

```
C:\SAM_Annotator\
├── app.dist\
│   ├── SAM_Annotator.exe   ← main executable
│   ├── torch\
│   ├── templates\
│   └── ... (other files, do not move or delete)
└── checkpoints\
    ├── sam2.1_hiera_small.pt
    └── sam_vit_b_01ec64.pth
```

> ⚠️ **Important:** The `app.dist` and `checkpoints` folders must be in the **same root folder**. If they are in different locations, the SAM model will not be found.

---

### ▶️ Step 3 — Run the application

1. Open the `app.dist` folder
2. **Double-click** `SAM_Annotator.exe`
3. A black window may appear briefly — this is normal, wait
4. After 10–15 seconds, open your browser and go to:

```
http://localhost:5000
```

---

### 🔑 Step 4 — First login

| Field | Value |
|-------|-------|
| Username | `admin` |
| Password | `admin123` |

> ⚠️ Change the password immediately after first login at: **Admin → Users → Change password**

---

### 👥 Team collaboration (optional)

To allow your team to access SAM Annotator from other computers:

1. Create a free account at [ngrok.com](https://ngrok.com)
2. Copy your **Authtoken** from the ngrok dashboard
3. In SAM Annotator: **Admin → "Publicar" tab**
4. Paste the token and click **Start**
5. Share the generated link with your team

---

### ❓ Common issues

**The application doesn't open or closes immediately**
- Check that port 5000 is not being used by another program
- Try running `SAM_Annotator.exe` as Administrator (right-click → Run as administrator)

**Error loading SAM model**
- Check that the `checkpoints` folder is in the correct location (see structure in Step 2)
- Both `.pt` and `.pth` files must be present inside `checkpoints`

**Blank page in browser**
- Wait longer (up to 30 seconds) after opening the `.exe`
- Make sure the address is exactly `http://localhost:5000` (not https)

**Antivirus blocks the .exe**
- Add the `app.dist` folder to your antivirus exceptions
- The file is safe — it was compiled with Nuitka from Python source code

---

*SAM Annotator v3 — 2026*
