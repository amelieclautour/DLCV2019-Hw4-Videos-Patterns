wget https://www.dropbox.com/s/0h3zgol9xomfhty/8.ckpt?dl=0 -O './rnnp3_model'

python3  predict_p3.py $1 $2 './rnnp3_model'