from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import logging
import torch
import torchaudio
import io
import base64
from preprocessing.text_preprocessor import TextPreprocessor
from augmentation.text_augmentor import TextAugmentor
from preprocessing.image_preprocessor import ImagePreprocessor
from augmentation.image_augmentor import ImageAugmentor
from preprocessing.audio_preprocessor import AudioPreprocessor
from augmentation.audio_augmentor import AudioAugmentor

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Device configuration - define once and use throughout
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Initialize all processors with device
preprocessor = TextPreprocessor()
augmentor = TextAugmentor()
image_preprocessor = ImagePreprocessor()
image_augmentor = ImageAugmentor()
audio_preprocessor = AudioPreprocessor()
audio_augmentor = AudioAugmentor()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "preprocess_options": preprocessor.get_available_operations(),
            "augmentation_options": augmentor.get_available_operations(),
            "image_preprocess_options": image_preprocessor.get_available_operations(),
            "image_augmentation_options": image_augmentor.get_available_operations(),
            "audio_preprocess_options": audio_preprocessor.get_available_operations(),
            "audio_augmentation_options": audio_augmentor.get_available_operations()
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
        content = await file.read()
        text = content.decode('utf-8')
        
        # Apply preprocessing
        processed_text = text
        preprocess_results = {}
        for op in preprocess_ops:
            processed_text = preprocessor.apply_operation(op, processed_text)
            preprocess_results[op] = processed_text
        
        # Apply augmentation
        augmented_results = {}
        for op in augment_ops:
            augmented_text = augmentor.apply_operation(op, processed_text)
            augmented_results[op] = augmented_text
        
        return JSONResponse(content={
            "original_text": text,
            "preprocessed_results": preprocess_results,
            "augmented_results": augmented_results
        })
        
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
        contents = await file.read()
        
        # Apply preprocessing
        preprocess_results = {}
        for op in preprocess_ops:
            processed_bytes = image_preprocessor.apply_operation(op, contents)
            preprocess_results[op] = base64.b64encode(processed_bytes).decode('utf-8')

        # Apply augmentation
        augmented_results = {}
        for op in augment_ops:
            augmented_bytes = image_augmentor.apply_operation(op, contents)
            augmented_results[op] = base64.b64encode(augmented_bytes).decode('utf-8')

        # Convert original image to base64
        original_base64 = base64.b64encode(contents).decode('utf-8')

        return JSONResponse(content={
            "original_image": original_base64,
            "preprocessed_results": preprocess_results,
            "augmented_results": augmented_results
        })

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_audio")
async def process_audio(
    request: Request,
    file: UploadFile = File(...),
    preprocess_ops: list[str] = Form(default=[]),
    augment_ops: list[str] = Form(default=[])
):
    try:
        contents = await file.read()
        audio_bytes = io.BytesIO(contents)
        
        try:
            # Load audio using torchaudio with better error handling
            audio_tensor, sr = torchaudio.load(audio_bytes)
            audio_tensor = audio_tensor.to(device)
            logger.info(f"Successfully loaded audio: shape={audio_tensor.shape}, sr={sr}")
        except Exception as e:
            logger.error(f"Error loading audio file: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid audio file format")

        # Generate original waveform using the shared utility
        from utils.audio_utils import generate_waveform
        original_waveform = generate_waveform(audio_tensor.cpu(), sr)
        if not original_waveform:
            logger.error("Failed to generate original waveform")
            raise HTTPException(status_code=500, detail="Failed to generate waveform")

        # Apply preprocessing with better error handling
        preprocess_results = {}
        preprocess_waveforms = {}
        for op in preprocess_ops:
            try:
                processed_audio, processed_sr, waveform = audio_preprocessor.apply_operation(op, audio_tensor, sr)
                if processed_audio is not None:
                    buffer = io.BytesIO()
                    torchaudio.save(buffer, processed_audio.cpu(), processed_sr, format='wav')
                    preprocess_results[op] = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    preprocess_waveforms[op] = waveform
            except Exception as e:
                logger.error(f"Error in preprocessing operation {op}: {str(e)}")
                continue

        # Apply augmentation with better error handling
        augmented_results = {}
        augment_waveforms = {}
        for op in augment_ops:
            try:
                augmented_audio, augmented_sr, waveform = audio_augmentor.apply_operation(op, audio_tensor, sr)
                if augmented_audio is not None:
                    buffer = io.BytesIO()
                    torchaudio.save(buffer, augmented_audio.cpu(), augmented_sr, format='wav')
                    augmented_results[op] = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    augment_waveforms[op] = waveform
            except Exception as e:
                logger.error(f"Error in augmentation operation {op}: {str(e)}")
                continue

        # Convert original audio to base64
        original_buffer = io.BytesIO()
        torchaudio.save(original_buffer, audio_tensor.cpu(), sr, format='wav')
        original_base64 = base64.b64encode(original_buffer.getvalue()).decode('utf-8')

        return JSONResponse(content={
            "original_audio": original_base64,
            "original_waveform": original_waveform,
            "preprocessed_results": preprocess_results,
            "preprocessed_waveforms": preprocess_waveforms,
            "augmented_results": augmented_results,
            "augmented_waveforms": augment_waveforms
        })

    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)