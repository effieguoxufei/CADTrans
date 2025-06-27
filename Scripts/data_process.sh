#!/bin/bash\

# PROCESS SOLID
python dataprocess/convert.py --input data/cad_raw --output data/solid --format solid --bit 6
python dataprocess/deduplicate.py --data_path data/solid --format solid

# PROCESS PROFILE
python dataprocess/convert.py --input data/cad_raw --output data/profile --format profile --bit 6
python dataprocess/deduplicate.py --data_path data/profile --format profile

# PROCESS LOOP
python dataprocess/convert.py --input data/cad_raw --output data/loop --format loop --bit 6
python dataprocess/deduplicate.py --data_path data/loop --format loop

# PROCESS FULL CAD MODEL
python dataprocess/convert.py --input data/cad_raw --output data/model --format model --bit 6
python dataprocess/deduplicate.py --data_path data/model --format model

