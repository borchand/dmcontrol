import re

with open('logs/state-sac_2374889_1.o', 'r') as file:
    contents = file.read()
contents = contents.replace('\r', '')
lines = contents.split('\n')
results = [l for l in lines if 'eval' in l]
results = [re.sub('\[[^\]\]]*\]','',l) for l in results]
results[:200]
