import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torchvision.models as models
from PIL import Image
import PIL
import matplotlib.pyplot as plt



####################### load and show images
imsize = 200

loader = transforms.Compose([
            transforms.Scale(imsize),
            transforms.ToTensor()])
unloader = transforms.ToPILImage()

def image_loader(image_name):
    image = Image.open(image_name)
    image = Variable(loader(image))
    # fake batch dimension:
    image = image.unsqueeze(0)
    return image

def imshow(tensor):
    image = tensor.clone().cpu()
    image.resize_(3,imsize,imsize)
    image = unloader(image)
    plt.imshow(image)

####################### load neural network

network = models.resnet18(pretrained=True)
#network = models.alexnet(pretrained=True)
network.cuda()

# define target ouput
print network

###################### def content loss

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.test = Variable(torch.rand(1).fill_(target)).long().cuda()
        self.target = self.test.detach()
        self.loss = 0
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input):
        self.loss = self.criterion.forward(input, self.target)
        self.output = input
        return self.output

    def backward(self, retain_variables=True):
        self.loss.backward(retain_variables=retain_variables)
        return self.loss.data[0]

# recreate class 1
content_loss = ContentLoss(1)

###################### create a module that contains the input image as a parameter

# variable with data of the same dimensions than content or style
image =image_loader("content.jpg").cuda()
# fill it with a white noise
#image.data = torch.rand(image.data.size()).cuda()

from torch.nn.parameter import Parameter
class input_image(nn.Module):
    def __init__(self, image):
        super(input_image,self).__init__()
        self.image = Parameter(image.data)

input = input_image(image)

##################### define optimizer

optimizer = optim.Adam(input.parameters(), lr = 0.01)

##################### run descents

for run in range(500):

    input.image.data = (input.image.data - torch.min(input.image.data)).div_(torch.max(input.image.data)-torch.min(input.image.data))

    optimizer.zero_grad()
    output = network.forward(input.image)
    content_loss.forward(output)
    loss = content_loss.backward()
    optimizer.step()

    if run%50==0:
        print "run "+str(run)+":"
        print loss

# vew result

imshow(input.image.data)
plt.show()
#'''
