'''
Created on Apr 17, 2023

@author: yanyo
'''
import spacy
nlp = spacy.blank("en")
# Import the Doc and Span classes
from spacy.tokens import Doc, Span
words = ["I", "like", "David", "Bowie"]
spaces = [True, True, True, False]
# Create a doc from the words and spaces
doc = Doc(nlp.vocab, words, spaces)
print(doc.text)
# Create a span for "David Bowie" from the doc and assign it the label "PERSON"
span = Span(doc, 2, 4, label="PERSON")
print(span.text, span.label_)
span1 = Span(doc, 0, 2, label="work")
print(span1.text, span1.label_)
# Add the span to the doc's entities
doc.ents = [span, span1]
# Print entities' text and labels
print([(ent.text, ent.label_) for ent in doc.ents])