python training.py -s 512 512 1024 -e 100 -t data/accuracy_results/accuracy_512x2.1024.csv --accuracy_rate 50
python training.py -s 512 512 512 1024 -e 100 -t data/accuracy_results/accuracy_512x3.1024.csv --accuracy_rate 50
python training.py -s 256 256 -e 100 -t data/accuracy_results/accuracy_256x2.csv --accuracy_rate 50
python training.py -s 256 512 1024 -e 100 -t data/accuracy_results/accuracy_256.512.1024.csv --accuracy_rate 50
python training.py -s 1024 1024 512 256 -e 100 -t data/accuracy_results/accuracy_1024x2.512.256.csv --accuracy_rate 50
