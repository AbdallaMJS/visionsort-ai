from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from visionsort.data import LABELS
from visionsort.model import build_mobilenet, build_transforms, load_weights

WEIGHTS = Path("artifacts/mobilenet.pt")

st.set_page_config(page_title="VisionSort AI", page_icon="♻️", layout="centered")
st.title("♻️ VisionSort AI")
st.caption("Explainable recyclable-material classification with transfer learning")

st.markdown(
    "Upload a clear photo of one object. The trained model predicts one of six TrashNet classes. "
    "This prototype is educational and should not be used as an automated recycling authority."
)

if not WEIGHTS.exists():
    st.info(
        "No trained weights are committed. Run `python scripts/train.py --architecture mobilenet`, "
        "review the real held-out results, and then launch this app locally."
    )
else:
    import torch

    model = load_weights(build_mobilenet(pretrained=False), WEIGHTS)
    transform = build_transforms(train=False)
    uploaded = st.file_uploader("Image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_container_width=True)
        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            probabilities = torch.softmax(model(tensor), dim=1).squeeze(0)
        top_id = int(probabilities.argmax().item())
        confidence = float(probabilities[top_id].item())
        st.metric("Prediction", LABELS[top_id])
        st.metric("Model confidence", f"{confidence:.1%}")
        st.bar_chart({label: float(probabilities[i]) for i, label in enumerate(LABELS)})
        st.caption(
            "Confidence is not the probability that an item is recyclable. It is the model's "
            "relative confidence among the six dataset classes."
        )
