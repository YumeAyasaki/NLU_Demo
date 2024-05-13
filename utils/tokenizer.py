import transformers
from tokenizers import BertWordPieceTokenizer

def BERT_tokenizer():
    # First load the real tokenizer
    tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-cased')
    # Save the loaded tokenizer locally
    tokenizer.save_pretrained('.')
    # Reload it with the huggingface tokenizers library
    fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=False)
    return fast_tokenizer