'''
Created on Apr 17, 2023

@author: yanyo
'''
import spacy
from spacy.tokens import Doc

nlp = spacy.blank("en")
# Import the Doc class


#Desired text: "spaCy is cool!"
words = ["spaCy", "is", "cool", "!"]
spaces = [True, True, False, False]
# Create a Doc from the words and spaces
doc1 = Doc(nlp.vocab, words=words, spaces=spaces)
print(doc1.text)

# Desired text: "Go, get started!"
words = ["Go", ",", "get", "started", "!"]
spaces = [False, True, True, False, False]
# Desired text: "Oh, really?!"
words = ["Oh", ",", "really", "?", "!"]
spaces = [False, True, False, False, False]

# Create a Doc from the words and spaces
doc = Doc(nlp.vocab, words=words, spaces=spaces)
print(doc.text)