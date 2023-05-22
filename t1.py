'''
Created on Apr 14, 2023

@author: yanyo
'''
# Import spaCy
import spacy
# Create the English nlp object
nlp = spacy.load("en_core_web_trf")

# Process a text
doc = nlp("MOTION TO PERMIT LATE PAYMENT OF JURY FEES.")

# Iterate over the tokens
for token in doc:
    print(token.text, token.pos_, token.dep_, token.head.text)
    
print("acl:", spacy.explain("acl"))
print("PART:", spacy.explain("PART"))
print("aux:", spacy.explain("aux"))
print("ADP:", spacy.explain("ADP"))
