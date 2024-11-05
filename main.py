from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import logging
from preprocessing.text_preprocessor import TextPreprocessor
from augmentation.text_augmentor import TextAugmentor
from preprocessing.image_preprocessor import ImagePreprocessor
from augmentation.image_augmentor import ImageAugmentor
import cv2
import numpy as np
import base64

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize preprocessor and augmentor
preprocessor = TextPreprocessor()
augmentor = TextAugmentor()

# Initialize image processors
image_preprocessor = ImagePreprocessor()
image_augmentor = ImageAugmentor()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "preprocess_options": preprocessor.get_available_operations(),
            "augmentation_options": augmentor.get_available_operations(),
            "image_preprocess_options": image_preprocessor.get_available_operations(),
            "image_augmentation_options": image_augmentor.get_available_operations()
        }
    )

@app.post("/process")
async def process_text(
    request: Request,
    file: UploadFile = File(...),
    preprocess_ops: list[str] = Form(default=[]),
    augment_ops: list[str] = Form(default=[])
):
    try:
        logger.debug(f"Received file: {file.filename}")
        logger.debug(f"Preprocess ops: {preprocess_ops}")
        logger.debug(f"Augment ops: {augment_ops}")
        
        content = await file.read()
        text = content.decode('utf-8')
        
        # Apply preprocessing
        processed_text = text
        preprocess_results = {}
        for op in preprocess_ops:
            try:
                processed_text = preprocessor.apply_operation(op, processed_text)
                preprocess_results[op] = processed_text
                logger.debug(f"Applied preprocessing operation {op}")
            except Exception as e:
                logger.error(f"Error in preprocessing operation {op}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error in preprocessing operation {op}: {str(e)}")
        
        # Apply augmentation
        augmented_results = {}
        for op in augment_ops:
            try:
                augmented_text = augmentor.apply_operation(op, processed_text)
                augmented_results[op] = augmented_text
                logger.debug(f"Applied augmentation operation {op}")
            except Exception as e:
                logger.error(f"Error in augmentation operation {op}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error in augmentation operation {op}: {str(e)}")
        
        response_data = {
            "original_text": text,
            "preprocessed_results": preprocess_results,
            "augmented_results": augmented_results
        }
        
        logger.debug("Successfully processed text")
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_image")
async def process_image(
    request: Request,
    file: UploadFile = File(...),
    preprocess_ops: list[str] = Form(default=[]),
    augment_ops: list[str] = Form(default=[])
):
    try:
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Apply preprocessing
        preprocess_results = {}
        for op in preprocess_ops:
            # Apply each preprocessing operation on the original image
            processed_image = image_preprocessor.apply_operation(op, original_image.copy())
            # Convert to base64 for frontend display
            _, buffer = cv2.imencode('.png', processed_image)
            preprocess_results[op] = base64.b64encode(buffer).decode('utf-8')

        # Apply augmentation
        augmented_results = {}
        for op in augment_ops:
            # Apply each augmentation operation on the original image
            augmented_image = image_augmentor.apply_operation(op, original_image.copy())
            _, buffer = cv2.imencode('.png', augmented_image)
            augmented_results[op] = base64.b64encode(buffer).decode('utf-8')

        # Convert original image to base64
        _, buffer = cv2.imencode('.png', original_image)
        original_base64 = base64.b64encode(buffer).decode('utf-8')

        return JSONResponse(content={
            "original_image": original_base64,
            "preprocessed_results": preprocess_results,
            "augmented_results": augmented_results
        })

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 