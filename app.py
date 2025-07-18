from fastapi import FastAPI, File, UploadFile
from torchvision import transforms
from PIL import Image
import torch
import io
from fastapi.middleware.cors import CORSMiddleware
from model import EnglishCharacterClassifier

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = EnglishCharacterClassifier(numClasses=26)
model.load_state_dict(torch.load("english_char_model_state.pt", map_location="cpu"))   
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

@app.get("/")
def home():
    return {"message": "English Character Classifier API"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
    prediction = torch.argmax(output, dim=1).item()
    return {"prediction": prediction}