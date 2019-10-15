#!/bin/bash

for i in {1..20}
do
	python3 capture.py -q -l RANDOM -n 50 -b monteCarlo.py > ./output/output$i.log
done
