# Data Processing Studio

A web-based application for processing and augmenting different types of data files (text, images, and audio) with various preprocessing and augmentation techniques. The application provides a user-friendly interface for uploading files and applying different operations in real-time.

###################### Features ######################

### Text Processing
- **Preprocessing Options:**
  - Lowercase conversion: Converts text to lowercase
  - Punctuation removal: Removes all punctuation marks
  - Number removal: Removes all numerical digits
  - Extra whitespace removal: Normalizes spacing between words
  - Stopwords removal: Removes common English stopwords
  - URL removal: Removes web URLs from text
  - Email address removal: Removes email addresses
  - HTML tags removal: Strips HTML markup

- **Augmentation Options:**
  - Synonym replacement: Replaces words with their synonyms
  - Random word swap: Randomly swaps positions of words
  - Spelling error injection: Introduces common spelling mistakes
  - Word deletion: Randomly removes words (10-20% of text)
  - Shuffle sentences: Randomly reorders sentences
  - Context-based substitution: Replaces words with contextually similar ones

### Image Processing
- **Preprocessing Options:**
  - Grayscale conversion: Converts color images to grayscale
  - Image resizing: Resizes images to 224x224 pixels
  - Normalization: Normalizes pixel values to 0-255 range
  - Blur effect: Applies Gaussian blur
  - Image sharpening: Enhances image edges
  - Histogram equalization: Improves image contrast
  - Noise removal: Removes image noise using Non-local Means Denoising
  - Edge detection: Detects edges using Canny algorithm

- **Augmentation Options:**
  - Rotation: Randomly rotates image (-30° to 30°)
  - Flipping: Randomly flips image (horizontal/vertical)
  - Brightness adjustment: Adjusts image brightness (0.5-1.5x)
  - Contrast adjustment: Modifies image contrast (0.5-1.5x)
  - Noise addition: Adds Gaussian noise
  - Random cropping: Crops random 80% portion of image
  - Random erasing: Erases random patches (2-40% of image)

### Audio Processing
- **Preprocessing Options:**
  - Audio normalization: Normalizes audio amplitude
  - Noise reduction: Reduces background noise
  - Silence trimming: Removes silent portions
  - Resampling: Changes sample rate to 22050Hz
  - Speed adjustment: Changes playback speed (1.5x)
  - Hum removal: Removes 50/60Hz power line hum

- **Augmentation Options:**
  - Pitch shifting: Changes audio pitch (-4 to 4 semitones)
  - Time stretching: Modifies audio duration (0.8-1.2x)
  - Noise addition: Adds random noise
  - Audio reversal: Reverses audio playback
  - Volume change: Adjusts volume (0.5-1.5x)
  - Room simulation: Adds room acoustics effect
  - Compression: Applies dynamic range compression
  - Chorus effect: Adds chorus effect
  - Echo addition: Adds echo effect (0.3s delay)

###################### Installation ######################

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


###################### Usage ######################

1. Start the application:
bash
python main.py

2. Open your web browser and navigate to:
http://localhost:8000

3. Using the interface:
   - Select file type (text/image/audio) using the radio buttons
   - Upload a file using the appropriate file card
   - View the original content in the first column
   - Select preprocessing operations from the middle column
   - Select augmentation operations from the right column
   - View results in real-time with visualizations

###################### Supported File Formats ######################

- **Text Files:** .txt
- **Image Files:** .png, .jpg, .jpeg
- **Audio Files:** .mp3, .wav

###################### Technical Implementation ######################

### Backend (Python/FastAPI)
- FastAPI for REST API endpoints
- File upload handling with multipart form data
- Asynchronous processing of operations
- Error handling and logging
- Real-time processing feedback

### Frontend (HTML/CSS/JavaScript)
- Responsive design with modern CSS
- Real-time UI updates
- File type switching
- Dynamic operation selection
- Result visualization
- Error handling and user feedback

### Processing Libraries
- NLTK and NLPAug for text processing
- OpenCV and Pillow for image processing
- Librosa and SoundFile for audio processing
- Matplotlib for audio visualization
- NumPy for numerical operations
- SciPy for signal processing

###################### Project Structure ######################

data-processing-studio/
├── main.py # FastAPI application
├── static/
│ └── styles.css # Application styles
├── templates/
│ └── index.html # Main UI template
├── preprocessing/
│ ├── text_preprocessor.py # Text preprocessing
│ ├── image_preprocessor.py # Image preprocessing
│ └── audio_preprocessor.py # Audio preprocessing
└── augmentation/
├── text_augmentor.py # Text augmentation
├── image_augmentor.py # Image augmentation
└── audio_augmentor.py # Audio augmentation operations


###################### Development ######################

The application is built with:
- FastAPI for robust API development
- Jinja2 for server-side templating
- Modern CSS for responsive design
- Vanilla JavaScript for client-side interactions
- Real-time processing and visualization
- Error handling and user feedback
- Modular code organization

###################### License ######################
This project is licensed under the MIT License.

###################### Dependencies ######################

### Core Web Framework
- **FastAPI** (>=0.68.0): Modern web framework for building APIs
- **Uvicorn** (>=0.15.0): ASGI server implementation
- **Jinja2** (>=3.0.1): Template engine for Python
- **python-multipart** (>=0.0.5): Multipart form data parser

### Text Processing
- **NLTK** (>=3.6.3): Natural Language Toolkit for text processing
  - Used for tokenization, stopwords removal
  - Provides WordNet for synonym operations
- **NLPAug** (>=1.1.11): Text augmentation library
  - Implements synonym replacement
  - Provides word and character-level augmentations
- **BeautifulSoup4** (>=4.11.1): HTML parsing library
  - Used for HTML tag removal

### Image Processing
- **OpenCV** (opencv-python>=4.8.0): Computer vision library
  - Implements image preprocessing operations
  - Provides image augmentation capabilities
- **Pillow** (>=10.0.0): Python Imaging Library
  - Handles image file operations
  - Supports various image formats

### Audio Processing
- **Librosa** (>=0.10.0): Music and audio analysis
  - Core audio processing functionality
  - Provides audio feature extraction
- **SoundFile** (>=0.12.1): Audio file I/O
  - Handles audio file reading/writing
- **AudioRead** (>=3.0.0): Audio file reading
  - Backend for audio file support
- **Numba** (>=0.58.1): JIT compiler
  - Required for Librosa performance
- **SciPy** (>=1.11.4): Scientific computing
  - Signal processing functions
  - Audio filtering operations

### Visualization
- **Matplotlib** (>=3.8.0): Plotting library
  - Generates audio waveform visualizations
  - Creates visual representations of audio data

### Core Dependencies
- **NumPy** (>=1.21.0): Numerical computing
  - Foundation for array operations
  - Used across all processing modules

### Development Tools
- **python-dotenv** (>=0.19.0): Environment variable management
- **tqdm** (>=4.65.0): Progress bar functionality
- **requests** (>=2.31.0): HTTP library for Python

All these dependencies can be installed using the provided requirements.txt file:
