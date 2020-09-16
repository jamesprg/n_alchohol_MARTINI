#!/bin/bash

for AT in `seq 0 1 39`;
do cat >header$AT.dat << EOF
# UMBRELLA      3.0
# Component selection: 0 0 1
# nSkip 1
# Ref. Group 'TestAtom'
# Nr. of pull groups 1
#  Group 1 'GR1'  Umb. Pos. $AT Umb. Cons. 1000.0
##### 
EOF
echo "bla$AT.pdo" >> wham.dat
done

for i in {0..39}; do awk '{print $1, $4}' COLVAR.$i > COLVAR$i; done
for i in {0..39}; do cat header$i.dat  COLVAR$i > what$i.pdo; done 
for i in {0..39}; do rm COLVAR$i; done

for ((i=0;i<=39;i++)); do shift=`echo "($i)/10." | bc -l`; echo $shift;  cat what$i.pdo | awk -v S=$shift '!/#/ {print $1, $2-S} /#/ {print;}' > bla$i.pdo; done

for i in {0..9}; do sed -i "s/Umb. Pos. $i/Umb. Pos. 0.$i/g" bla$i.pdo ; done
for i in {0..9}; do sed -i "s/Umb. Pos. 1$i/Umb. Pos. 1.$i/g" bla1$i.pdo ; done
for i in {0..9}; do sed -i "s/Umb. Pos. 2$i/Umb. Pos. 2.$i/g" bla2$i.pdo ; done
for i in {0..9}; do sed -i "s/Umb. Pos. 3$i/Umb. Pos. 3.$i/g" bla3$i.pdo ; done
