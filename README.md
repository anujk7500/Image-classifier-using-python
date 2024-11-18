here are requirement to train the mode and save the trained model in the same directory . after then you can browse your model using python language and imorting the important module like 
from flask import Flask, request, render_template
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
import torch.nn.functional as F  # For softmax
