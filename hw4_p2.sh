wget https://www.dropbox.com/s/tbsnask28syt7xj/best_rnn.pth?dl=0 -O './rnnp2_model'

python3  predict_p2.py $1 $2 './rnnp2_model' $3