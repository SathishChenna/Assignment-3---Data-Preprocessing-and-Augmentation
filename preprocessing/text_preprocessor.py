import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Download all required NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
except Exception as e:
    print(f"Warning: Error downloading NLTK data: {str(e)}")

class TextPreprocessor:
    def __init__(self):
        self.operations = {
            "apply_all": self.apply_all_preprocessing,
            "lowercase": self.to_lowercase,
            "remove_punctuation": self.remove_punctuation,
            "remove_numbers": self.remove_numbers,
            "remove_whitespace": self.remove_extra_whitespace,
            "remove_stopwords": self.remove_stopwords
        }
        
        # Ensure stopwords are loaded
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            print(f"Warning: Could not load stopwords: {str(e)}")
            self.stop_words = set()
    
    def get_available_operations(self):
        return list(op for op in self.operations.keys() if op != "apply_all")
    
    def apply_operation(self, operation: str, text: str) -> str:
        if operation in self.operations:
            return self.operations[operation](text)
        return text

    def apply_all_preprocessing(self, text: str) -> str:
        processed_text = text
        for op_name, op_func in self.operations.items():
            if op_name != "apply_all":
                processed_text = op_func(processed_text)
        return processed_text
    
    def to_lowercase(self, text: str) -> str:
        return text.lower()
    
    def remove_punctuation(self, text: str) -> str:
        return re.sub(r'[^\w\s]', '', text)
    
    def remove_numbers(self, text: str) -> str:
        return re.sub(r'\d+', '', text)
    
    def remove_extra_whitespace(self, text: str) -> str:
        return ' '.join(text.split())
    
    def remove_stopwords(self, text: str) -> str:
        try:
            words = word_tokenize(text)
            return ' '.join([word for word in words if word.lower() not in self.stop_words])
        except Exception as e:
            print(f"Warning: Error in remove_stopwords: {str(e)}")
            return text