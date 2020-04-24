import sys
import torch
from torchvision import transforms
import os
import numpy as np
from tqdm import tqdm
from models import Resnet50
from videos_read import Short_Vid, get_Vid_List

# data source
video_path = sys.argv[1]
test_label_path = sys.argv[2]
dict = get_Vid_List(test_label_path)
video_names = (dict['Video_name'])
video_categorys = (dict['Video_category'])
total_num = len(video_names)


# loading videos
test_videos = []
test_labels = []
print("\nloading videos...")
with tqdm(total=total_num) as pbar:
    for i,(video_category, video_name) in enumerate(zip(video_categorys,video_names)):
            train_video = Short_Vid(video_path, video_category, video_name)
            test_videos.append(train_video)
            pbar.update(1)
print("test_videos_len:",len(test_videos))


# extracting features
cnn_feature_extractor = Resnet50().cuda() # to 2048 dims
transform=transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Pad((0,40),fill=0,padding_mode='constant'),
                        transforms.Resize(224),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                        ])
cnn_feature_extractor.eval()
train_features = []
with torch.no_grad():
    print("\nextracting videos feature...")
    with tqdm(total=total_num) as pbar:
        for train_video in test_videos:
            local_batch = []
            for frame in train_video:
                frame = transform(frame)
                local_batch.append(frame)
            local_batch = torch.stack(local_batch)
            feature = cnn_feature_extractor(local_batch.cuda())
            train_features.append(torch.mean(feature,0))
            pbar.update(1)
train_features = torch.stack(train_features)



model_path = sys.argv[3]
pre_label_path = sys.argv[4]


# loading features extracted by pretrain model
print("\nloading features...")
test_features = train_features.view(-1,2048)

# load model
my_net = torch.load(model_path)
my_net = my_net.eval()
my_net = my_net.cuda()
predict_labels,_ = my_net(test_features.cuda())
predict_vals = torch.argmax(predict_labels,1).cpu().data




# output as csv file
with open(os.path.join(pre_label_path,"p1_valid.txt"),'w') as f:
    for i,predict_val in enumerate(predict_vals):
        f.write(str(int(predict_val)))
        if (i==len(predict_vals)-1): break
        f.write('\n')
print(" predicted file at",os.path.join(pre_label_path,"p1_valid.txt"))