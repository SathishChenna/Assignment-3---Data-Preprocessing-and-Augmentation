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
            "word_deletion": self.word_deletion,
            "shuffle_sentences": self.shuffle_sentences,
            "context_substitution": self.context_substitution
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
    
    def word_deletion(self, text: str) -> str:
        words = text.split()
        if len(words) <= 1:
            return text
        
        n_to_delete = max(1, round(len(words) * random.uniform(0.1, 0.2)))
        indices_to_delete = random.sample(range(len(words)), n_to_delete)
        return ' '.join([word for i, word in enumerate(words) if i not in indices_to_delete])
    
    def shuffle_sentences(self, text: str) -> str:
        sentences = text.split('.')
        if len(sentences) <= 1:
            return text
        random.shuffle(sentences)
        return '. '.join(sentences).strip()
    
    def context_substitution(self, text: str) -> str:
        try:
            words = text.split()
            if len(words) <= 1:
                return text
            
            idx = random.randint(0, len(words) - 1)
            word = words[idx]
            
            synsets = wordnet.synsets(word)
            if synsets:
                lemmas = synsets[0].lemmas()
                if lemmas:
                    new_word = random.choice(lemmas).name()
                    words[idx] = new_word
            
            return ' '.join(words)
        except Exception as e:
            print(f"Context substitution error: {str(e)}")
            return text