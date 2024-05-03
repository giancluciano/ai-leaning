from nltk import pos_tag, download
from nltk.tokenize import word_tokenize

#nltk.download("punkt")
#https://www.nltk.org/_modules/nltk/tag/perceptron.html
# pre-trained English tagger
download("averaged_perceptron_tagger")

text = "POS tagging is important to get an idea that which parts of speech does tokens belongs to i.e whether it is noun, verb, adverb, conjunction, pronoun, adjective, preposition, interjection, if it is verb then which form and so on"
tokens = word_tokenize(text)
print(text)
print(pos_tag(tokens))
print("I left the room " + str(pos_tag(word_tokenize("I left the room"))))
print("Left of the room " + str(pos_tag(word_tokenize("Left of the room"))))
