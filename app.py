import streamlit as st
from PIL import Image
import time

# =========================
# CONFIGURACIÓN
# =========================
st.set_page_config(
    page_title="Subida de Imágenes",
    layout="wide"
)

# =========================
# CSS LOADING PANTALLA COMPLETA
# =========================
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
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("UIDE")

menu = st.sidebar.radio(
    "Menú",
    ["Instrucciones","Subir imágenes"]
)

# =========================
# CONTENIDO PRINCIPAL
# =========================
if menu == "Subir imágenes":

    st.title("👁️ Retinopatia Diabética")
    st.write("Arrastra y suelta una imagen o haz clic para seleccionarla.")

    uploaded_file = st.file_uploader(
        "Sube una imagen",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)

        st.subheader("🔍 Previsualización")
        st.image(image, width=300, caption=uploaded_file.name)

        st.markdown("---")

        if st.button("🔮 Clasificar"):

            # Mostrar overlay
            overlay = st.empty()
            overlay.markdown("""
            <div id="overlay">
                <div class="loader"></div>
            </div>
            """, unsafe_allow_html=True)

            # Simula procesamiento
            time.sleep(5)

            # Quitar overlay
            overlay.empty()

            st.success("Predicción completada ✅")

elif menu == "Instrucciones":

    st.title("ℹ️ Información")
    st.markdown("""
    Esta aplicación permite clasificar imagenes para la Detección Temprana de Retinopatía Diabética mediante el procesamiento de imágenes retinales.
    """)

    st.subheader("📋 Instrucciones")
    st.markdown("""
    1. Selecciona **Subir imágenes**
    2. Carga una imagen válida
    3. Presiona **Predecir**
    4. Espera el resultado
    """)

    st.subheader("⚙️ Características")
    st.markdown("""
    - Drag & Drop de imágenes
    - Previsualización controlada
    - Loading pantalla completa
    - Spinner animado
    """)
