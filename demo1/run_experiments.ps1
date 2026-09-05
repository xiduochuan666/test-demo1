$python = "D:\anaconda\envs\pytorch\python.exe"

# & $python .\train.py --learning_rate 2e-5 --batch_size 16 --dropout 0.1
# & $python .\train.py --learning_rate 1e-5 --batch_size 16 --dropout 0.1
# & $python .\train.py --learning_rate 3e-5 --batch_size 16 --dropout 0.1
& $python .\train.py --learning_rate 2e-5 --batch_size 8 --dropout 0.1
& $python .\train.py --learning_rate 2e-5 --batch_size 32 --dropout 0.1
& $python .\train.py --learning_rate 2e-5 --batch_size 16 --dropout 0.3
