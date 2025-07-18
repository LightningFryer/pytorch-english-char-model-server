from fastapi import FastAPI, File, UploadFile
from torchvision import transforms
from PIL import Image
import torch
import io

app = FastAPI()
model = torch.load("english_char_model.pt", map_location=torch.device("cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
    prediction = torch.argmax(output, dim=1).item()
    return {"prediction": prediction}