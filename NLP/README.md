# Tokenization
```
Simple word tokenization
['Tokenization', 'is', 'one', 'of', 'the', 'first', 'step', 'in', 'any', 'NLP', 'pipeline.', 'Tokenization', 'is', 'nothing', 'but', 'splitting', 'the', 'raw', 'text', 'into', 'small', 'chunks', 'of', 'words', 'or', 'sentences']
Simple sentence tokenization
['Tokenization is one of the first step in any NLP pipeline', 'Tokenization is nothing but splitting the raw text into small chunks of words or sentences']
nltk word tokenization
['Tokenization', 'is', 'one', 'of', 'the', 'first', 'step', 'in', 'any', 'NLP', 'pipeline', '.', 'Tokenization', 'is', 'nothing', 'but', 'splitting', 'the', 'raw', 'text', 'into', 'small', 'chunks', 'of', 'words', 'or', 'sentences']
nltk sentence tokenization
['Tokenization is one of the first step in any NLP pipeline.', 'Tokenization is nothing but splitting the raw text into small chunks of words or sentences']
spacy word tokenization
['Tokenization', 'is', 'one', 'of', 'the', 'first', 'step', 'in', 'any', 'NLP', 'pipeline', '.', 'Tokenization', 'is', 'nothing', 'but', 'splitting', 'the', 'raw', 'text', 'into', 'small', 'chunks', 'of', 'words', 'or', 'sentences']
spacy sentence tokenization
['Tokenization is one of the first step in any NLP pipeline.', 'Tokenization is nothing but splitting the raw text into small chunks of words or sentences']
torchtext word tokenization
['tokenization', 'is', 'one', 'of', 'the', 'first', 'step', 'in', 'any', 'nlp', 'pipeline', '.', 'tokenization', 'is', 'nothing', 'but', 'splitting', 'the', 'raw', 'text', 'into', 'small', 'chunks', 'of', 'words', 'or', 'sentences']
```


# Part-of-Speech Tagging
POS tagging is important to get an idea that which parts of speech does tokens belongs to i.e whether it is noun, verb, adverb, conjunction, pronoun, adjective, preposition, interjection, if it is verb then which form and so on
[('POS', 'NNP'), ('tagging', 'NN'), ('is', 'VBZ'), ('important', 'JJ'), ('to', 'TO'), ('get', 'VB'), ('an', 'DT'), ('idea', 'NN'), ('that', 'WDT'), ('which', 'WDT'), ('parts', 'NNS'), ('of', 'IN'), ('speech', 'NN'), ('does', 'VBZ'), ('tokens', 'NNS'), ('belongs', 'NNS'), ('to', 'TO'), ('i.e', 'VB'), ('whether', 'IN'), ('it', 'PRP'), ('is', 'VBZ'), ('noun', 'JJ'), (',', ','), ('verb', 'NN'), (',', ','), ('adverb', 'NN'), (',', ','), ('conjunction', 'NN'), (',', ','), ('pronoun', 'NN'), (',', ','), ('adjective', 'JJ'), (',', ','), ('preposition', 'NN'), (',', ','), ('interjection', 'NN'), (',', ','), ('if', 'IN'), ('it', 'PRP'), ('is', 'VBZ'), ('verb', 'JJ'), ('then', 'RB'), ('which', 'WDT'), ('form', 'NN'), ('and', 'CC'), ('so', 'RB'), ('on', 'IN')]


I left the room [('I', 'PRP'), ('left', 'VBD'), ('the', 'DT'), ('room', 'NN')]
Left of the room [('Left', 'NNP'), ('of', 'IN'), ('the', 'DT'), ('room', 'NN')]