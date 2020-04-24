import numpy as np
import sys
import os

test_predict_path = sys.argv[1]
test_label_path = sys.argv[2]  #TrimmedVideos/label/gt_valid.csv

# read files
f = open(os.path.join(test_predict_path),'r')
predict_vals = f.read().splitlines()
g = open(os.path.join(test_label_path),'r')
label_vals = g.read().splitlines()
# evaluation ans
print("\nevaluation ans...")
predict_vals = np.array(predict_vals).astype(int)
label_vals = np.array(label_vals).astype(int)
accuracy = np.mean(predict_vals == label_vals)
print("accuracy:",accuracy)