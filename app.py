import streamlit as st
from PIL import Image

# Configuración de la página
st.set_page_config(
    page_title="Retinopatia Diabética",
    layout="wide"
)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("UIDE")

menu = st.sidebar.radio(
    "Menú",
    ["Instrucciones", "Subir imágenes"]
)

# =========================
# CONTENIDO PRINCIPAL
# =========================
if menu == "Subir imágenes":

    st.title("👁️ Retinopatia Diabética")
    st.write("Arrastra y suelta una imagen o haz clic para seleccionarla.")

    uploaded_file = st.file_uploader(
        "Sube una imagen",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)

            st.subheader("🔍 Previsualización")
            st.image(
                image,
                caption=f"Imagen cargada: {uploaded_file.name}",
                use_container_width=True
            )

            st.success("Imagen cargada correctamente ✅")

        except Exception:
            st.error("El archivo no es una imagen válida.")

elif menu == "Instrucciones":

    st.title("ℹ️ Información")
    st.write(
        "Esta aplicación permite clasificar imagenes para la Detección Temprana de Retinopatía Diabética mediante el procesamiento de imágenes retinales "
        "y posterior procesamiento."
    )

    st.subheader("📋 Instrucciones")
    st.markdown("""
    1. Selecciona la opción **Subir imágenes** en el menú lateral.
    2. Arrastra y suelta una imagen o haz clic en *Browse files*.
    3. Revisa la previsualización mostrada.
    """)

    st.subheader("⚙️ Características")
    st.markdown("""
    - Subida exclusiva de imágenes (PNG, JPG, JPEG)
    - Previsualización inmediata
    - Validación automática de archivos
    """)
