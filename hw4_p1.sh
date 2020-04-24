wget https://www.dropbox.com/s/z6r07x2okcjdncc/best_cnn.pth?dl=0 -O './cnn_model'
python3  predict_p1.py $1 $2 './cnn_model' $3