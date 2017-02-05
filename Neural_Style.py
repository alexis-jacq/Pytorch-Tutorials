import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import PIL
from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models


######### images

imsize = 200 # desired size of the output image

loader = transforms.Compose([
            transforms.Scale(imsize), # scale imported image
            transforms.ToTensor()]) # transform it into a torch tensor

def image_loader(image_name):
    image = Image.open(image_name)
    image = Variable(loader(image))
    image = image.unsqueeze(0) # fake batch dimension required to fit network's input dimensions
    return image

style = image_loader("images/picasso.jpg").cuda()
content = image_loader("images/dancing.jpg").cuda()

assert style.data.size() == content.data.size(), "we need to import style and content images of the same size"


########## display

unloader = transforms.ToPILImage() # reconvert into PIL image

def imshow(tensor):
    image = tensor.clone().cpu() # we clone the tensor in order to not do changes on it
    image.resize_(3,imsize,imsize) # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)

fig = plt.figure()

plt.subplot(221)
imshow(style.data)
plt.subplot(222)
imshow(content.data)

########## content loss

class ContentLoss(nn.Module):
    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        self.target = target.detach() * weight # we 'detach' the target content from the tree used
                                               # to dynamically compute the gradient: this is a stated value,
                                               # not a variable. Otherwise the forward method of the criterion
                                               # will throw an error.
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion.forward(input*self.weight, self.target)
        self.output = input
        return self.output

    def backward(self, retain_variables=True):
        self.loss.backward(retain_variables=retain_variables)
        return self.loss.data[0]

######### style loss

class GramMatrix(nn.Module):
    def forward(self, input):
        a,b,c,d = input.data.size() # a=batch size(=1)
                                    # b=number of feature maps
                                    # (c,d)=dimensions of a f. map (N=c*d)

        input.data.resize_(a*b,c*d) # resise F_XL into \hat F_XL

        G = torch.mm(input, input.t()) # compute the gram product

        return G.div_(a*b*c*d) # we 'normalize' the values of the gram matrix
                           # by dividing by the number of element in each feature maps.

class StyleLoss(nn.Module):
   def __init__(self, target, strength):
       super(StyleLoss, self).__init__()
       self.target = target.detach()*strength
       self.strength = strength
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

####### load model

cnn = models.alexnet(pretrained=True).features.cuda()

# desired depth layers to compute style/content losses :
content_layers = ['conv_3','conv_4']
style_layers = ['conv_1','conv_2','conv_3','conv_4','conv_5']

# just in order to have an iterable access to or list of content/syle losses
content_losses = []
style_losses = []

art_net = nn.Sequential().cuda() # the new Sequential module network
gram = GramMatrix().cuda() # we need a gram module in order to compute style targets

# weigth associated with content and style losses
content_weight = 5
style_weight = 500

i = 1
for layer in list(cnn):
    if isinstance(layer,nn.Conv2d):
        name = "conv_"+str(i)
        art_net.add_module(name,layer)

        if name in content_layers:
            # add content loss:
            target = art_net.forward(content.cuda()).clone()
            content_loss = ContentLoss(target, content_weight).cuda()
            art_net.add_module("content_loss_"+str(i),content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = art_net.forward(style.cuda()).clone()
            target_feature_gram = gram.forward(target_feature)
            style_loss = StyleLoss(target_feature_gram, style_weight).cuda()
            art_net.add_module("style_loss_"+str(i),style_loss)
            style_losses.append(style_loss)

    if isinstance(layer,nn.ReLU):
        name = "relu_"+str(i)
        art_net.add_module(name,layer)

        if name in content_layers:
            # add content loss:
            target = art_net.forward(content.cuda()).clone()
            content_loss = ContentLoss(target, content_weight).cuda()
            art_net.add_module("content_loss_"+str(i),content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = art_net.forward(style.cuda()).clone()
            target_feature_gram = gram.forward(target_feature)
            style_loss = StyleLoss(target_feature_gram, style_weight).cuda()
            art_net.add_module("style_loss_"+str(i),style_loss)
            style_losses.append(style_loss)

        i+=1

    if isinstance(layer,nn.MaxPool2d):
        name = "pool_"+str(i)
        art_net.add_module(name,layer) # ***

print art_net

###### input image

input = image_loader("images/dancing.jpg").cuda()
# if we want to fill it with a white noise:
# input.data = torch.randn(input.data.size()).cuda()

# add the original input image to the figure:
plt.subplot(223)
imshow(input.data)

######## gradient descent

input = nn.Parameter(input.data) # this line to show that input is a parameter that requires a gradient
optimizer = optim.Adam([input], lr = 0.01)

for run in range(500):

    # correct the values of updated input image
    updated = input.data.cpu().clone()
    updated = updated.numpy()
    updated[updated<0] = 0
    updated[updated>1] = 1
    input.data = torch.from_numpy(updated).cuda()

    optimizer.zero_grad()
    art_net.forward(input)
    style_score = 0
    content_score = 0

    for sl in style_losses:
        style_score += sl.backward()
    for cl in content_losses:
        content_score += cl.backward()

    optimizer.step()

    if run%10==0:
        print "run "+str(run)+":"
        print style_score
        print content_score

# a last correction...
result = input.data.cpu().clone()
result = result.numpy()
result[result<0] = 0
result[result>1] = 1
result = torch.from_numpy(result)

# finally enjoy the result:
plt.subplot(224)
imshow(input.data)
plt.show()
