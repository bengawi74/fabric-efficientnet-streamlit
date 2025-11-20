import io
from pathlib import Path
from typing import List

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ---------------------------------------------------
# Config
# ---------------------------------------------------
CHECKPOINT_PATH = Path("models/fabric_efficientnet_b0_best.pt")

# Same class order as your dataset
CLASS_NAMES: List[str] = [
    "cotton",
    "denim",
    "leather",
    "linen",
    "polyester",
    "wool",
]


# ---------------------------------------------------
# Model loading
# ---------------------------------------------------
@st.cache_resource
def load_model(checkpoint_path: Path, num_classes: int) -> nn.Module:
    """
    Load EfficientNet-B0 model with trained weights.
    Cached so Streamlit does not reload on every interaction.
    """
    device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------
# Preprocessing
# ---------------------------------------------------
def get_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def preprocess_image(image: Image.Image) -> torch.Tensor:
    transform = get_transform()
    return transform(image).unsqueeze(0)


# ---------------------------------------------------
# Prediction
# ---------------------------------------------------
def predict_image(
    model: nn.Module,
    image: Image.Image,
    class_names: List[str],
):
    device = torch.device("cpu")
    input_tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred_class = class_names[pred_idx]

    return pred_class, probs


# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Fabric Classification Demo", layout="centered")

    st.title("🧵 Fabric Classification with EfficientNet-B0")
    st.write(
        """
        Upload a fabric image and the model will predict one of the six types:

        **cotton, denim, leather, linen, polyester, wool**.
        """
    )

    if not CHECKPOINT_PATH.exists():
        st.error(f"Checkpoint not found: `{CHECKPOINT_PATH}`")
        return

    model = load_model(CHECKPOINT_PATH, num_classes=len(CLASS_NAMES))

    uploaded_file = st.file_uploader(
        "Upload a fabric image (JPG or PNG)", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        pred_class, probs = predict_image(model, image, CLASS_NAMES)

        with col2:
            st.markdown("### Prediction")
            st.write(f"**Predicted class:** `{pred_class}`")

            st.markdown("### Class probabilities")
            st.dataframe({
                "class": CLASS_NAMES,
                "probability": [float(p) for p in probs],
            })
    else:
        st.info("👆 Upload an image to run the model.")


if __name__ == "__main__":
    main()