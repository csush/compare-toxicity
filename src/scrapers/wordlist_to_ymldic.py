import re
import yaml

dickey = {}
a = ['negative']
for line in open("badwords.txt"):
    word = re.split(', |; |\*|\n',line)
    dickey[word[0]] = ['negative']
with open('data.yml', 'w') as outfile:
    yaml.dump(dickey, outfile, default_flow_style=False)    
