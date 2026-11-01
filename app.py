import base64
import io
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import tensorflow as tf

app = FastAPI()

# Allow your frontend HTML page to talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load the exact trained model from your notebook
MODEL_PATH = "malaria_model.keras"
print(f"Loading trained weights from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# Hook the last Conv2D layer for Grad-CAM explainability
last_conv_layer = next(
    l for l in reversed(model.layers) if isinstance(l, tf.keras.layers.Conv2D)
)
grad_model = tf.keras.models.Model(
    inputs=model.inputs, outputs=[last_conv_layer.output, model.output]
)

CLASSES = ["Parasitized", "Uninfected"]  # 0 = Parasitized, 1 = Uninfected


def get_gradcam(img_tensor, orig_img_pil):
  with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_tensor)
    p_uninfected = predictions[:, 0]
    predicted_idx = int(p_uninfected[0] >= 0.5)
    target = p_uninfected if predicted_idx == 1 else (1.0 - p_uninfected)

  grads = tape.gradient(target, conv_outputs)
  pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

  heatmap = tf.reduce_sum(pooled_grads * conv_outputs[0], axis=-1)
  heatmap = np.maximum(heatmap.numpy(), 0)
  if heatmap.max() > 0:
    heatmap /= heatmap.max()

  heatmap = cv2.resize(heatmap, orig_img_pil.size)
  heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

  orig_np = np.array(orig_img_pil.convert("RGB"))
  orig_bgr = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)
  superimposed = cv2.addWeighted(orig_bgr, 0.6, heatmap_color, 0.4, 0)
  superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)

  cam_pil = Image.fromarray(superimposed_rgb)
  buf = io.BytesIO()
  cam_pil.save(buf, format="PNG")
  return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
  # Read image
  contents = await file.read()
  raw_img = Image.open(io.BytesIO(contents)).convert("RGB")

  # Preprocess: resize to 64x64 and rescale by 1/255.0 (identical to notebook)
  resized_img = raw_img.resize((64, 64))
  img_array = np.array(resized_img, dtype=np.float32) / 255.0
  img_tensor = tf.convert_to_tensor(np.expand_dims(img_array, axis=0))

  # Predict
  prob = float(model.predict(img_tensor, verbose=0)[0][0])

  # Notebook logic: >= 0.5 is Uninfected, < 0.5 is Parasitized
  if prob >= 0.5:
    label = "Uninfected"
    confidence = prob * 100.0
  else:
    label = "Parasitized"
    confidence = (1.0 - prob) * 100.0

  gradcam_base64 = get_gradcam(img_tensor, raw_img)

  return {
      "filename": file.filename,
      "prediction": label,
      "confidence": round(confidence, 2),
      "raw_prob": round(prob, 4),
      "is_parasitized": (label == "Parasitized"),
      "gradcam_image": f"data:image/png;base64,{gradcam_base64}",
  }