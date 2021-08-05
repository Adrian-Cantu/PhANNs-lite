#!/usr/bin/bash


FILE=model.tar
if [ -f "$FILE" ]; then
        echo "$FILE exists."
    else 
        wget https://edwards.sdsu.edu/phanns/download/model.tar
fi

if md5sum --status -c model.md5; then
        tar -xvf  model.tar
    else
        echo "$FILE is corrupt, please redownload"
        exit
fi



