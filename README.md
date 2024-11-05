# Data Processing Studio

A web-based application for processing and augmenting different types of data files (text, images, and audio) with various preprocessing and augmentation techniques.

## Features

### Text Processing
- **Preprocessing Options:**
  - Lowercase conversion
  - Punctuation removal
  - Number removal
  - Extra whitespace removal
  - Stopwords removal

- **Augmentation Options:**
  - Synonym replacement
  - Random word swap
  - Spelling error injection
  - Word deletion

### Image Processing
- **Preprocessing Options:**
  - Grayscale conversion
  - Image resizing
  - Normalization
  - Blur effect
  - Image sharpening

- **Augmentation Options:**
  - Rotation
  - Flipping
  - Brightness adjustment
  - Contrast adjustment
  - Noise addition

### Audio Processing
- **Preprocessing Options:**
  - Audio normalization
  - Noise reduction
  - Silence trimming
  - Resampling
  - Speed adjustment

- **Augmentation Options:**
  - Pitch shifting
  - Time stretching
  - Noise addition
  - Audio reversal
  - Volume adjustment



## Installation

1. Clone the repository: 
bash
git clone https://github.com/SathishChenna/Assignment-3---Data-Preprocessing-and-Augmentation.git
cd Assignment-3---Data-Preprocessing-and-Augmentation


2. Create a virtual environment and activate it:
bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate


3. Install the required dependencies:
bash
pip install -r requirements.txt


## Usage

1. Start the application:
bash
python main.py


2. Open your web browser and navigate to:
http://localhost:8000


3. Use the interface to:
   - Select and upload your file (text, image, or audio)
   - Choose preprocessing or augmentation options
   - View the results in real-time

## Supported File Formats

- **Text Files:** .txt
- **Image Files:** .png, .jpg, .jpeg
- **Audio Files:** .mp3, .wav

## Project Structure
data-processing-studio/
├── main.py # FastAPI application entry point
├── requirements.txt # Project dependencies
├── README.md # Project documentation
├── static/
│ └── styles.css # Application styles
├── templates/
│ └── index.html # Main application template
├── preprocessing/
│ ├── text_preprocessor.py # Text preprocessing operations
│ ├── image_preprocessor.py # Image preprocessing operations
│ └── audio_preprocessor.py # Audio preprocessing operations
└── augmentation/
├── text_augmentor.py # Text augmentation operations
├── image_augmentor.py # Image augmentation operations
└── audio_augmentor.py # Audio augmentation operations



## Dependencies

Major dependencies include:
- FastAPI & Uvicorn for web server
- NLTK for text processing
- OpenCV & Pillow for image processing
- Librosa & SoundFile for audio processing
- PyTorch & NumPy for computations
- See requirements.txt for complete list

## Development

The application is built with:
- FastAPI for backend API
- Jinja2 templates for frontend
- Modern CSS for styling
- Vanilla JavaScript for interactivity

## License

This project is licensed under the MIT License.

This README provides:
1. Project overview
2. Features list
3. Installation steps
4. Usage instructions
5. File format support
6. Project structure
7. Dependencies list
8. Development information
9. License information
