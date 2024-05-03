# python basic tokenization 

text = "Tokenization is one of the first step in any NLP pipeline. Tokenization is nothing but splitting the raw text into small chunks of words or sentences"

# word
print("Simple word tokenization")
print(text.split())

print("Simple sentence tokenization")
print(text.split(". "))

# Natural Language Toolkit (NLTK)
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

#nltk.download("punkt")

tokens = word_tokenize(text)
print("nltk word tokenization")
print(tokens)
print("nltk sentence tokenization")
print(sent_tokenize(text))

# spacy
#!python -m spacy download en_core_web_sm
from spacy.lang.en import English
nlp = English()
my_doc = nlp(text)
token_list = []
for token in my_doc:
    token_list.append(token.text)

print("spacy word tokenization")
print(token_list)

print("spacy sentence tokenization")

nlp = English()
nlp.add_pipe('sentencizer')
my_doc = nlp(text)
sentence_list =[]
for sentence in my_doc.sents:
    sentence_list.append(sentence.text)
print(sentence_list)


from torchtext.data import get_tokenizer
tokenizer = get_tokenizer("basic_english")
tokens = tokenizer(text)
print("torchtext word tokenization")
print(tokens)