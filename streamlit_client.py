"""
Cliente Streamlit para clasificación de Retinopatía Diabética
Se conecta a la API FastAPI para obtener predicciones
"""

import streamlit as st
from PIL import Image
import requests
import base64
from io import BytesIO

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
st.set_page_config(
    page_title="Detección de Retinopatía Diabética",
    page_icon="👁️",
    layout="wide"
)

# URL de la API (ajustar según tu configuración)
API_URL = "http://localhost:8000"
ESTUDIANTES = """
 - Bonilla Zarate, María Camila
 - Carranza Villarroel, Carlos Daniel 
 - Lincango Simbaña, Betsy Belén
 - Saguay Saguay, Bryan Alexander """

# =============================================================================
# CSS PERSONALIZADO
# =============================================================================
st.markdown("""
<style>
    #overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(255, 255, 255, 0.9);
        z-index: 9999;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .loader {
        border: 8px solid #f3f3f3;
        border-top: 8px solid #1f77b4;
        border-radius: 50%;
        width: 80px;
        height: 80px;
        animation: spin 1s linear infinite;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .result-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
    }
    .result-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
    }
    .result-danger {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================


def image_to_base64(image: Image.Image) -> str:
    """Convierte imagen PIL a base64"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def get_severity_color(prediction: str) -> str:
    """Retorna color según severidad"""
    colors = {
        'No_DR': '🟢',
        'Mild': '🟡',
        'Moderate': '🟠',
        'Severe': '🔴',
        'Proliferate_DR': '⚫'
    }
    return colors.get(prediction, '⚪')


def check_api_health() -> bool:
    """Verifica si la API está disponible"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.title("🏥 UIDE")
st.sidebar.markdown("---")

# Estado de la API
api_status = check_api_health()
if api_status:
    st.sidebar.success("✅ API Conectada")
else:
    st.sidebar.error("❌ API Desconectada")

st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "📋 Menú",
    ["Instrucciones", "Subir imágenes"]
)

st.sidebar.markdown("---")
st.sidebar.info(f"🔗 API: {API_URL}")

st.sidebar.markdown("---")
st.sidebar.info(f"🧑🏻‍💻👩🏻‍💻 Estudiantes: {ESTUDIANTES}")

# =============================================================================
# CONTENIDO PRINCIPAL
# =============================================================================
if menu == "Subir imágenes":
    st.title("👁️ Detección de Retinopatía Diabética")
    st.write(
        "Sube una imagen del fondo de ojo para obtener una clasificación automática.")

    # Verificar API
    if not api_status:
        st.error("⚠️ La API no está disponible. Asegúrate de que esté ejecutándose.")
        st.code("python api_retinopathy.py", language="bash")
        st.stop()

    # Subir imagen
    uploaded_file = st.file_uploader(
        "Arrastra o selecciona una imagen",
        type=["png", "jpg", "jpeg"],
        help="Formatos soportados: PNG, JPG, JPEG"
    )

    if uploaded_file:
        # Cargar imagen
        image = Image.open(uploaded_file)

        # Layout en columnas
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("🔍 Imagen cargada")
            st.image(image, caption=uploaded_file.name,
                     use_container_width=True)

        with col2:
            st.subheader("📊 Resultado")

            # Botón de clasificación
            if st.button("🔮 Clasificar imagen", type="primary", use_container_width=True):

                # Mostrar spinner
                with st.spinner("Analizando imagen..."):
                    try:
                        # Convertir imagen a base64
                        image_b64 = image_to_base64(image)

                        # Enviar a la API
                        response = requests.post(
                            f"{API_URL}/predict",
                            json={"image_base64": image_b64},
                            timeout=30
                        )

                        if response.status_code == 200:
                            result = response.json()

                            # Mostrar resultado principal
                            severity_icon = get_severity_color(
                                result['prediction'])

                            st.success("✅ Análisis completado")

                            st.markdown(f"""
                            ### {severity_icon} Diagnóstico: **{result['description']}**
                            
                            **Clase:** `{result['prediction']}`  
                            **Confianza:** `{result['confidence']*100:.2f}%`
                            """)

                            # Barra de progreso de confianza
                            st.progress(result['confidence'])

                            # Mostrar todas las probabilidades
                            st.markdown("---")
                            st.subheader("📈 Probabilidades por clase")

                            for clase, prob in sorted(
                                result['all_probabilities'].items(),
                                key=lambda x: x[1],
                                reverse=True
                            ):
                                icon = get_severity_color(clase)
                                st.write(f"{icon} **{clase}**")
                                st.progress(prob)
                                st.caption(f"{prob*100:.2f}%")

                        else:
                            st.error(f"Error en la API: {response.text}")

                    except requests.exceptions.Timeout:
                        st.error(
                            "⏱️ Tiempo de espera agotado. Intenta de nuevo.")
                    except requests.exceptions.ConnectionError:
                        st.error("🔌 No se pudo conectar con la API.")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

elif menu == "Instrucciones":
    st.title("ℹ️ Sistema de Detección de Retinopatía Diabética")

    st.markdown("""
    Esta aplicación utiliza **Inteligencia Artificial** para clasificar imágenes del fondo 
    de ojo y detectar signos de Retinopatía Diabética.
    """)

    st.subheader("🎯 Clases de clasificación")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        | Icono | Clase | Descripción |
        |:---:|:---|:---|
        | 🟢 | No_DR | Sin retinopatía |
        | 🟡 | Mild | Leve |
        | 🟠 | Moderate | Moderada |
        """)

    with col2:
        st.markdown("""
        | Icono | Clase | Descripción |
        |:---:|:---|:---|
        | 🔴 | Severe | Severa |
        | ⚫ | Proliferate_DR | Proliferativa |
        """)

    st.subheader("📋 Instrucciones de uso")
    st.markdown("""
    1. Selecciona **"Subir imágenes"** en el menú lateral
    2. Arrastra o selecciona una imagen del fondo de ojo
    3. Haz clic en **"Clasificar imagen"**
    4. Espera el resultado del análisis
    """)

    st.subheader("⚙️ Modelo utilizado")
    st.markdown("""
    - **Arquitectura:** VGG16 con Transfer Learning
    - **Entrada:** Imágenes 224x224 RGB
    - **Salida:** 5 clases de clasificación
    """)

    st.warning("""
    ⚠️ **Disclaimer:** Esta herramienta es solo para fines educativos y de investigación. 
    No reemplaza el diagnóstico médico profesional.
    """)
