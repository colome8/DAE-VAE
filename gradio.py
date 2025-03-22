import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Cargar modelos guardados
encoder = load_model("vae_encoder.keras", compile=False)
decoder = load_model("vae_decoder.keras", compile=False)

# -------------------------------
# DAE demo (usando decoder como reconstructor)
# -------------------------------
def reconstruir_imagen(img):
    img_resized = tf.image.resize(img, (128, 128)) / 255.0
    img_input = tf.expand_dims(img_resized, axis=0)  # Agregar batch dim
    _, _, z = encoder.predict(img_input)
    reconstruida = decoder.predict(z)[0]
    return reconstruida

# -------------------------------
# VAE generador aleatorio
# -------------------------------
def generar_aleatoria():
    z_random = tf.random.normal(shape=(1, 16))
    imagen_generada = decoder.predict(z_random)[0]
    return imagen_generada

# -------------------------------
# Interfaz Gradio
# -------------------------------
demo = gr.Interface(
    title="Demo Interactivo DAE / VAE",
    description="Sube una imagen para reconstruirla (DAE) o genera una imagen aleatoria (VAE).",
    inputs=[gr.Image(label="Imagen de entrada", type="numpy", optional=True)],
    outputs=gr.Image(label="Resultado"),
    fn=reconstruir_imagen,
    live=False
)

boton_generar = gr.Interface(
    fn=generar_aleatoria,
    inputs=[],
    outputs=gr.Image(label="Imagen Generada"),
    title="Generación Aleatoria VAE"
)

# Lanzar ambos demos
gr.TabbedInterface(
    [demo, boton_generar],
    tab_names=["Reconstrucción (DAE)", "Generación Aleatoria (VAE)"]
).launch()
