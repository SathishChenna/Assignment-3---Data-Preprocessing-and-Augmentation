import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
from nltk.corpus import wordnet
import random
import nltk

# Download required NLTK data
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class TextAugmentor:
    def __init__(self):
        self.operations = {
            "apply_all": self.apply_all_augmentations,
            "synonym_replacement": self.synonym_replacement,
            "random_swap": self.random_swap,
            "spelling_error": self.spelling_error,
            "word_deletion": self.word_deletion
        }
        
        # Initialize augmenters
        self.synonym_aug = naw.SynonymAug(aug_src='wordnet')
        self.random_swap_aug = naw.RandomWordAug()
        self.spelling_aug = nac.KeyboardAug()
    
    def get_available_operations(self):
        return list(op for op in self.operations.keys() if op != "apply_all")
    
    def apply_operation(self, operation: str, text: str) -> str:
        if operation in self.operations:
            return self.operations[operation](text)
        return text

    def apply_all_augmentations(self, text: str) -> str:
        augmented_text = text
        for op_name, op_func in self.operations.items():
            if op_name != "apply_all":
                augmented_text = op_func(augmented_text)
        return augmented_text
    
    def synonym_replacement(self, text: str) -> str:
        try:
            return self.synonym_aug.augment(text)[0]
        except:
            return text
    
    def word_deletion(self, text: str) -> str:
        words = text.split()
        if len(words) <= 1:
            return text
        
        # Randomly delete 10-20% of words
        n_to_delete = max(1, round(len(words) * random.uniform(0.1, 0.2)))
        indices_to_delete = random.sample(range(len(words)), n_to_delete)
        return ' '.join([word for i, word in enumerate(words) if i not in indices_to_delete])
    
    def random_swap(self, text: str) -> str:
        try:
            return self.random_swap_aug.augment(text)[0]
        except:
            return text
    
    def spelling_error(self, text: str) -> str:
        try:
            return self.spelling_aug.augment(text)[0]
        except:
            return text