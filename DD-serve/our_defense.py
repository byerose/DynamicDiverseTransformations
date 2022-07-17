import sys

from pydantic import condecimal
sys.path.append('/home/ait/ws/DD-serve')

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import random
import numpy as np
from estimator_custom import PyTorchClassifier
from defense_cifar10.ensemble_pytorch import EnsembleClassifier
from defense_cifar10.dynamic_preprocessor import *

from utils.trans_pool_cifar10 import *
import gradio as gr

trans = Transforms()

def test(inp):
    k=5
    g=6
    model_dir="./cifar10_wideresnet_re.pth"
    image_size = [32,32]
    input_shape = [3,32,32]
    nb_classes = 10

    tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

    model=torch.load(model_dir)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),.1,momentum=0.9,weight_decay=1e-4)
    trans_pool = [Adjusment,Affine,Compression,Denoise,Filters,Geometric,Morphology,Rotate,Shift]

    classifiers = []
    trans_func = random.sample(trans_pool,g)
    for func in trans_func:
        for _ in range(k):
            classifiers.append(PyTorchClassifier(
                model=model,
                loss=criterion,
                clip_values=(0,1),
                optimizer=optimizer,
                input_shape=input_shape,
                nb_classes=nb_classes,
                preprocessing_defences=func(),
                preprocessing=((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ))
    ensemble_classifier = EnsembleClassifier(
        classifiers=classifiers,
        classifier_weights=None,
        clip_values=(0,1),
        channels_first=True
    )

    labels = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    inp = tf(inp)
    input = inp.numpy()[np.newaxis,:,:,:]
    predictions = ensemble_classifier.predict(input, mode='avg')[0]
    # print(predictions)
    output_sum = predictions.sum()
    predictions = (predictions/output_sum).tolist()
    # print(predictions)
    confidences = {labels[i]:list(predictions)[i] for i in range(10)}
    return dict(sorted(confidences.items(),key=lambda k:k[1]))
        

def main():
    gr.Interface(
        fn=test,
        inputs=gr.inputs.Image(type="pil"),
        outputs=gr.outputs.Label(num_top_classes=3),
        examples=["dog.png","frog.png"]
    ).launch()
if __name__ == '__main__':
    main()
