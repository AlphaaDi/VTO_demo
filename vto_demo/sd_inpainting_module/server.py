from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from io import BytesIO
from PIL import Image
import uvicorn
import yaml

from stable_diffusion_wraper import StableDiffusionInpaintWrapper

app = FastAPI()

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

inpaint_wrapper = StableDiffusionInpaintWrapper(
    **config['inpainting_diffusion_args']
)

@app.post("/inpaint")
async def inpaint(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    ip_adapter_image: UploadFile = File(...),
    pos_prompt: str = Form(...)
):
    # Read the uploaded files
    image_data = await image.read()
    mask_data = await mask.read()
    ip_adapter_image_data = await ip_adapter_image.read()

    # Open images using PIL
    image_pil = Image.open(BytesIO(image_data)).convert("RGB")
    mask_pil = Image.open(BytesIO(mask_data)).convert("RGB")
    ip_adapter_image_pil = Image.open(BytesIO(ip_adapter_image_data)).convert("RGB")

    # Call the forward method
    result_image = inpaint_wrapper.forward(
        image=image_pil,
        mask=mask_pil,
        pos_prompt=pos_prompt,
        ip_adapter_image=ip_adapter_image_pil
    )

    # Save the result to bytes
    buf = BytesIO()
    result_image.save(buf, format='PNG')
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host=config['server']['host'], port=int(config['server']['port']))
