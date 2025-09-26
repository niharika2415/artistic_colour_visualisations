import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

st.title("Artistic Style Colorization (Better Model)")

# Better CNN Model for Colorization
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
    def forward(self, x):
        return x + self.conv(x)

class ColorizerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU()
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1), nn.Sigmoid()
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.res_blocks(x)
        x = self.decoder(x)
        return x

model = ColorizerModel()

# GUI
uploaded_file = st.file_uploader("Upload Grayscale Image", type=["png","jpg","jpeg"])
style_option = st.selectbox("Select Style (simple artistic filter)", ["None", "Invert", "Edges"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("L")  # grayscale
    st.image(img, caption="Original Grayscale", use_column_width=True)

    transform = transforms.Compose([transforms.ToTensor()])
    input_tensor = transform(img).unsqueeze(0)  # 1x1xHxW

    with torch.no_grad():
        colorized = model(input_tensor)  # 1x3xHxW
    colorized_img = colorized.squeeze().permute(1,2,0).numpy()
    colorized_img = (colorized_img * 255).astype(np.uint8)
    colorized_img = Image.fromarray(colorized_img)

    # Simple Artistic Filters
    if style_option == "Invert":
        colorized_img = Image.fromarray(255 - np.array(colorized_img))
    elif style_option == "Edges":
        from PIL import ImageFilter
        colorized_img = colorized_img.filter(ImageFilter.FIND_EDGES)

    st.image(colorized_img, caption="Colorized + Styled", use_column_width=True)
