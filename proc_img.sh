#!/bin/bash

for file in ./dirty_chords/*.jpg;
    do
        python3 detect_single_threaded.py -src $file
done
