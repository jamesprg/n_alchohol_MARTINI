#!/bin/bash
#prep for metada as it only take two values
for i in {0..39}; do awk '{print $1, $4}' COLVAR.$i > COLVAR$i; done

for i in {0..39}; do pwd COLVAR$i ; done > metadata1

for i in {0..39}; do echo /COLVAR$i; done > file

for i in `seq 0.0 0.1 3.9` ; do echo $i 1000; done > position
paste -d '\0' metadata1 file > metadata2
paste metadata2 position > metadata

#calling wham
/home/james/Downloads/wham/wham/wham 0 3.9 100 0.001 300 0 metadata pmf  



