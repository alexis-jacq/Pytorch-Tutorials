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

style = image_loader("images/picasso.jpg").cuda()
content = image_loader("images/dancing.jpg").cuda()

style_weight = 500
content_weight = 5

print style.data.size()
print content.data.size()

fig = plt.figure()
plt.subplot(221)
imshow(style.data)
plt.subplot(223)
imshow(content.data)

####################### define content loss

class ContentLoss(nn.Module):
    def __init__(self, target, strength):
        super(ContentLoss, self).__init__()
        self.target = target.detach() * strength
        self.strength = strength
        self.loss = 0
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion.forward(input*self.strength, self.target)
        self.output = input
        return self.output

    def backward(self, retain_variables=True):
        self.loss.backward(retain_variables=retain_variables)
        return self.loss.data[0]

####################### define style loss

class GramMatrix(nn.Module):
    def forward(self, input):
        a,b,c,d = input.data.size()
        input.data.resize_(a*b,c*d)
        return torch.mm(input, input.t()).div(a*b*c*d)

class StyleLoss(nn.Module):
    def __init__(self, target, strength):
        super(StyleLoss, self).__init__()
        self.target = target.detach()*strength
        self.strength = strength
        self.loss = 0
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.output = input.clone()
        self.G = self.gram.forward(input)
        self.G.mul_(self.strength)
        self.loss = self.criterion.forward(self.G, self.target)
        return self.output

    def backward(self, retain_variables=True):
        self.loss.backward(retain_variables=retain_variables)
        return self.loss.data[0]

####################### load neural network

alexnet = models.alexnet(pretrained=True).cuda()
print alexnet.features

art_net = nn.Sequential().cuda()
gram = GramMatrix().cuda()

i = 0
for layer in list(alexnet.features):
    if isinstance(layer,nn.Conv2d):
        i+=1
        art_net.add_module("conv"+str(i),layer)

        # add content loss:
        target = art_net.forward(content.cuda()).clone()
        content_loss = ContentLoss(target, content_weight).cuda()
        art_net.add_module('content_loss'+str(i),content_loss)

        # add style loss:
        target_feature = art_net.forward(style.cuda()).clone()
        target_feature_gram = gram.forward(target_feature)
        style_loss = StyleLoss(target_feature_gram, style_weight).cuda()
        art_net.add_module('style_loss'+str(i),style_loss)

    if isinstance(layer,nn.ReLU):
        art_net.add_module("relu"+str(i),layer)

    if isinstance(layer,nn.MaxPool2d):
        avgpool = nn.AvgPool2d(kernel_size=layer.kernel_size, stride=layer.stride, padding = layer.padding)
        art_net.add_module("avgpool"+str(i),avgpool)
        #art_net.add_module("avgpool"+str(i),layer)

print art_net

##################### create a module containing trained image

# variable with data of the same dimensions than content or style
image = image_loader("images/dancing.jpg").cuda()
# fill it with a white noise
image.data = torch.randn(image.data.size()).cuda()

from torch.nn.parameter import Parameter
class input_image(nn.Module):
    def __init__(self, image):
        super(input_image,self).__init__()
        self.image = Parameter(image.data)

input = input_image(image)
for params in input.parameters():
    print params.size()

# view of the input image:
plt.subplot(222)
imshow(input.image.data)

###################### define optimizer (Adam)

optimizer = optim.Adam(input.parameters(), lr = 0.01)

###################### run gradient descent

for run in range(500):

    input.image.data = (input.image.data - torch.min(input.image.data)).div_(torch.max(input.image.data)-torch.min(input.image.data))
    optimizer.zero_grad()
    art_net.forward(input.image)

    loss1 = art_net.style_loss1.backward()
    #loss2 = art_net.content_loss1.backward()
    loss1 = art_net.style_loss2.backward()
    #loss2 += art_net.content_loss2.backward()
    loss1 += art_net.style_loss3.backward()
    #loss2 = art_net.content_loss3.backward()
    loss1 += art_net.style_loss4.backward()
    loss2 = art_net.content_loss4.backward()
    loss1 += art_net.style_loss5.backward()
    #loss2 += art_net.content_loss5.backward()

    optimizer.step()

    if run%50==0:
        print "run "+str(run)+":"
        print loss1
        print loss2

###################### show result
input.image.data = (input.image.data - torch.min(input.image.data)).div_(torch.max(input.image.data)-torch.min(input.image.data))

# absurd values for resulting image:
print torch.min(style.data)
print torch.min(input.image.data)
print torch.max(style.data)
print torch.max(input.image.data)

#torchvision.utils.save_image(input.image.data,'result')

plt.subplot(224)
imshow(input.image.data)
plt.show()
#'''
