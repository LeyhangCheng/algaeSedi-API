from PIL import Image
from fastapi import FastAPI, File, UploadFile
import torch
import torchvision
from skimage import io
import numpy as np

import io

app = FastAPI()

model = torchvision.models.mobilenet_v3_small(weights=True)
model.load_state_dict(torch.load('model_v2_2.pth'))
model.eval()

@app.post("/prediction/")
async def create_upload_file(img: bytes = File(...)):
    
    imgRead = np.array(Image.open(io.BytesIO(img)))
    img_arr = torch.from_numpy(imgRead).permute(2, 0, 1).unsqueeze(0)

    score = model(img_arr/255)
    probs = torch.nn.functional.softmax(score, dim=1)
    confidenceScore, resultTensor = probs.max(1)

    # algae = 0, sediment = 1
    return {"Results": int(resultTensor), 
    "ConfidenceScore": "{:.2f}%".format(float(confidenceScore)*100)}