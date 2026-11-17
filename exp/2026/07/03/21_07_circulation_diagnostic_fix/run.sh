PYTHONPATH=$PWD python3 -u tst/substrate/probe_saha_spheromak.py --n-steps 4000 --report 1000 --out saha_spheromak_fixed.png
PYTHONPATH=$PWD python3 -u tst/substrate/probe_saha_cross.py --out saha_cross_fixed.png
