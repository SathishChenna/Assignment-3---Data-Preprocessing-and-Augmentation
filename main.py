from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import logging
from preprocessing.text_preprocessor import TextPreprocessor
from augmentation.text_augmentor import TextAugmentor

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

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "preprocess_options": preprocessor.get_available_operations(),
            "augmentation_options": augmentor.get_available_operations()
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 