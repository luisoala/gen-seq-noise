#!/bin/bash

#download zip file with the data
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1zo4RDJ6boLHKnHajWMgQ3ty7PsZuJz2U' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1zo4RDJ6boLHKnHajWMgQ3ty7PsZuJz2U" -O gen-seq-noise-data.zip && rm -rf /tmp/cookies.txt

#unzip
unzip gen-seq-noise-data.zip -d gen-seq-noise-data

#remove zip file
rm gen-seq-noise-data.zip

#alternatively you can also manually download the data from https://drive.google.com/open?id=1zo4RDJ6boLHKnHajWMgQ3ty7PsZuJz2U
