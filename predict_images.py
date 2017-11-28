# -*- coding:utf-8 -*-
__author__ = 'Yi'

import torch
import torchvision.transforms as transforms
import sys

sys.path.append('~/workingspace/pytorch_space')
from pytorch_space.train_model_for_image_folder import FineTuneModel


def predict_one_image(imgpath, input_space='RGB', softmax=True):
    from PIL import Image
    import torch.nn as nn

    if softmax:
        softmax = nn.Softmax()

    img = Image.open(imgpath).convert(input_space)
    input_data = preprocess(img).unsqueeze(0)
    input_tensor = torch.autograd.Variable(input_data)
    output = model(input_tensor)

    if softmax:
        score = softmax(output).data.squeeze().cpu().numpy()
    else:
        score = output.data.squeeze().cpu().numpy()
    # score = output.numpy() #shape=(120,)
    # score = torch.sigmoid(output).numpy() #map score to [0,1]
    return score


if __name__ == '__main__':
    num_classes = 120
    file_id = "train"
    data_dir = "~/dog-breed/stanford/models/incres/%s" % file_id
    # data_dir = "/home/fan/dog-breed/models/resnet101_ft_crops/%s" % file_id
    softmax = True

    import os
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    trained_model_path = '%s/../best.model' % data_dir

    model = torch.load(trained_model_path)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    preprocess = transforms.Compose([
        transforms.Scale(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        normalize
    ])

    from tqdm import tqdm
    import glob

    data = {}
    imgpaths = glob.glob('~/dog-breed/%s/*.jpg' % file_id)
    # imgpaths = glob.glob('/home/fan/dog-breed/res101_tf_outs/newcrops/%s/*.jpg' % file_id)
    for x in tqdm(imgpaths):
        pred = predict_one_image(x, softmax=softmax)
        data[x] = pred

    with open('%s/imgpaths.txt' % data_dir, 'w') as f:
        for x in data.keys():
            f.write(x + '\n')

    scores = []
    for x in data.keys():
        scores.append(data[x])
    import numpy as np

    np.array(scores).dump('%s/feats' % data_dir)
