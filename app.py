import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import hf_hub_download

# 1. Reconstructed model architecture matching your state_dict keys
class PneumoniaCNN(nn.Module):
    def __init__(self):
        super(PneumoniaCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Matches your exact weight matrix size: 86528 inputs -> 128 outputs
        self.fc1 = nn.Linear(86528, 128)
        self.fc2 = nn.Linear(128, 2) 

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)  # Flatten dimensions to batch size x 86528
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. Load the trained PyTorch model safely using cache from Hugging Face
@st.cache_resource
def load_pytorch_model():
    model = PneumoniaCNN() 
    
    # Securely fetches the model weights from your Hugging Face space/repo
    # CRITICAL: Replace "your-username/your-repo-name" with your real HF repo ID!
    model_path = hf_hub_download(
        repo_id="faiz4320/cnn-model", 
        filename="chest_xray_cnn_model.pth",
        token=st.secrets["HF_TOKEN"]  # Reads the token you added in Streamlit Secrets
    )
    
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()  
    return model

model = load_pytorch_model()

# 3. Web application structure
st.title("Chest X-Ray Classification (Pneumonia Detection)")
st.write(
    "Upload a chest X-ray image, and the trained PyTorch model will classify it as Normal or Pneumonia."
)

# 4. Upload image file
uploaded_file = st.file_uploader(
    "Choose a Chest X-Ray image", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Display the uploaded file
    st.image(
        image,
        caption="Uploaded Chest X-Ray",
        use_container_width=True
    )

    # 5. Define PyTorch transform pipeline
    preprocess = transforms.Compose([
        transforms.Resize((104, 104)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0) 

    # 6. Make prediction
    with torch.no_grad():  
        output = model(input_batch)
        probabilities = F.softmax(output, dim=1)
        
    # Get the flat 1D list of probabilities for the first batch item
    prob_scores = probabilities[0].tolist() 
    
    classes = ["Normal", "Pneumonia"]
    predicted_class_idx = torch.argmax(probabilities, dim=1).item()
    predicted_label = classes[predicted_class_idx]
    
    # Safely compute percentage
    confidence = prob_scores[predicted_class_idx] * 100

    # 7. Display Results
    st.subheader("Prediction Result")
    
    if predicted_label == "Normal":
        st.success(f"Prediction: {predicted_label}")
    else:
        st.error(f"Prediction: {predicted_label}")

    st.write(f"Confidence: {confidence:.2f}%")
