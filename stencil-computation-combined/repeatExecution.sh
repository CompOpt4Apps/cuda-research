#! /bin/bash

for i in {1..20}
do
	printf "$i\n----------------------------\n" >> out
	nvprof ./exec &>> out
	printf "\n" >> out
done
