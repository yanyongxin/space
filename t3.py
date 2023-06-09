'''
Created on Apr 17, 2023

@author: yanyo
'''
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)
doc = nlp(
    "i downloaded Fortnite on my laptop and can't open the game at all. Help? "
    "so when I was downloading Minecraft, I got the Windows version where it "
    "is the '.zip' folder and I used the default program to unpack it... do "
    "I also need to download Winzip?"
)
# Write a pattern that matches a form of "download" plus proper noun
pattern1 = [{"LEMMA": "download"}, {"POS": "PROPN"}]
pattern2 = [ {"POS": "PRON"}]
# Add the pattern to the matcher and apply the matcher to the doc
matcher.add("DOWNLOAD_THINGS_PATTERN", [pattern1, pattern2])
matches = matcher(doc)
print("Total matches found:", len(matches))
# Iterate over the matches and print the span text
for match_id, start, end in matches:
    print("Match found:", doc[start:end].text)
    
print(spacy.explain("PRON"))
print(spacy.explain("PROPN"))
