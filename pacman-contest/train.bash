#!/bin/bash

for i in {1..20}
do
	python3 capture.py -q -l RANDOM -n 50 -r vi.py -b myTeam.py > ./output/output$i.log
done
