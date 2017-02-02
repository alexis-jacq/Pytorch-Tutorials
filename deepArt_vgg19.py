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


####################### just for vgg19 from lua:

from torch.utils.serialization import load_lua
lua_vgg = load_lua('vgg19')
print lua_vgg
print lua_vgg.get(0)

def conv_lua(lua_conv):
    conv = nn.Conv2d(lua_conv.nInputPlane, lua_conv.nOutputPlane, lua_conv.kW, lua_conv.dW, lua_conv.padW)
    conv.weight.data = lua_conv.weight
    conv.bias.data = lua_conv.bias
    return conv

def maxpool_lua(lua_maxpool):
    return nn.MaxPool2d(lua_maxpool.kW, lua_maxpool.dW, lua_maxpool.padW )

print conv_lua(lua_vgg.get(0))
print maxpool_lua(lua_vgg.get(4))

vgg = nn.Sequential()

vgg.add_module('0',conv_lua(lua_vgg.get(0)))
vgg.add_module('1',nn.ReLU())
vgg.add_module('2',conv_lua(lua_vgg.get(2)))
vgg.add_module('3',nn.ReLU())
vgg.add_module('4',maxpool_lua(lua_vgg.get(4)))

vgg.add_module('5',conv_lua(lua_vgg.get(5)))
vgg.add_module('6',nn.ReLU())
vgg.add_module('7',conv_lua(lua_vgg.get(7)))
vgg.add_module('8',nn.ReLU())
vgg.add_module('9',maxpool_lua(lua_vgg.get(9)))

vgg.add_module('10',conv_lua(lua_vgg.get(10)))
vgg.add_module('11',nn.ReLU())
vgg.add_module('12',conv_lua(lua_vgg.get(12)))
vgg.add_module('13',nn.ReLU())
vgg.add_module('14',conv_lua(lua_vgg.get(14)))
vgg.add_module('15',nn.ReLU())
vgg.add_module('16',conv_lua(lua_vgg.get(16)))
vgg.add_module('17',nn.ReLU())
vgg.add_module('18',maxpool_lua(lua_vgg.get(18)))

vgg.add_module('19',conv_lua(lua_vgg.get(19)))
vgg.add_module('20',nn.ReLU())
vgg.add_module('21',conv_lua(lua_vgg.get(21)))
vgg.add_module('22',nn.ReLU())
vgg.add_module('23',conv_lua(lua_vgg.get(23)))
vgg.add_module('24',nn.ReLU())
vgg.add_module('25',conv_lua(lua_vgg.get(25)))
vgg.add_module('26',nn.ReLU())
vgg.add_module('27',maxpool_lua(lua_vgg.get(27)))

vgg.add_module('28',conv_lua(lua_vgg.get(28)))
vgg.add_module('29',nn.ReLU())
vgg.add_module('30',conv_lua(lua_vgg.get(30)))
vgg.add_module('31',nn.ReLU())
vgg.add_module('32',conv_lua(lua_vgg.get(32)))
vgg.add_module('33',nn.ReLU())
vgg.add_module('34',conv_lua(lua_vgg.get(34)))
vgg.add_module('35',nn.ReLU())
vgg.add_module('36',maxpool_lua(lua_vgg.get(36)))

print vgg

####################### load and show images
imsize = 200

loader = transforms.Compose([
            transforms.Scale(imsize),
            transforms.ToTensor()])
unloader = transforms.ToPILImage()

def image_loader(image_name):
    image = Image.open(image_name)
    image = Variable(loader(image)*255)
    # fake batch dimension:
    image = image.unsqueeze(0)
    return image

def imshow(tensor):
    image = tensor.clone().cpu()
    image.resize_(3,imsize,imsize)
    image = unloader(image)
    plt.imshow(image)

style = image_loader("style.jpg").cuda()
content = image_loader("content.jpg").cuda()

style_weight = 500
content_weight = 5

print style.data.size()
print content.data.size()

fig = plt.figure()
plt.subplot(221)
imshow(style.data.clone().div_(255))
plt.subplot(223)
imshow(content.data.clone().div_(255))

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
'''
# alexnet
cnn = models.alexnet(pretrained=True).features.cuda()
print cnn
'''

cnn = vgg.cuda()

content_layers = ['conv4_2']
style_layers = ['conv1_1','conv2_1','conv3_1','conv4_1','conv5_1']

cont_losses = []
style_losses = []

art_net = nn.Sequential().cuda()
gram = GramMatrix().cuda()

i = 1
j = 1
for layer in list(cnn):
    if isinstance(layer,nn.Conv2d):
        name = "conv"+str(j)+"_"+str(i)
        art_net.add_module(name,layer)

        if name in content_layers:
            # add content loss:
            target = art_net.forward(content.cuda()).clone()
            content_loss = ContentLoss(target, content_weight).cuda()
            art_net.add_module('content_loss'+str(j)+"_"+str(i),content_loss)
            cont_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = art_net.forward(style.cuda()).clone()
            target_feature_gram = gram.forward(target_feature)
            style_loss = StyleLoss(target_feature_gram, style_weight).cuda()
            art_net.add_module('style_loss'+str(j)+"_"+str(i),style_loss)
            style_losses.append(style_loss)

        i+=1

    if isinstance(layer,nn.ReLU):
        name = "relu"+str(j)+"_"+str(i)
        art_net.add_module(name,layer)

        if name in content_layers:
            # add content loss:
            target = art_net.forward(content.cuda()).clone()
            content_loss = ContentLoss(target, content_weight).cuda()
            art_net.add_module('content_loss'+str(j)+"_"+str(i),content_loss)
            cont_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = art_net.forward(style.cuda()).clone()
            target_feature_gram = gram.forward(target_feature)
            style_loss = StyleLoss(target_feature_gram, style_weight).cuda()
            art_net.add_module('style_loss'+str(j)+"_"+str(i),style_loss)
            style_losses.append(style_loss)

    if isinstance(layer,nn.MaxPool2d):
        name = "pool"+str(j)+"_"+str(i)
        art_net.add_module(name,layer)
        #avgpool = nn.AvgPool2d(kernel_size=layer.kernel_size, stride=layer.stride, padding = layer.padding)
        #art_net.add_module(name,avgpool)
        i = 0
        j+=1

print art_net


##################### create a module containing trained image

# variable with data of the same dimensions than content or style
image = image_loader("content.jpg").cuda()
# fill it with a white noise
#image.data = torch.randn(image.data.size()).cuda()

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
imshow(input.image.data.clone().div_(255))

###################### define optimizer (Adam)

optimizer = optim.Adam(input.parameters(), lr = 10)

###################### run gradient descent

for run in range(300):

    #input.image.data = (input.image.data - torch.min(input.image.data)).div_(torch.max(input.image.data)-torch.min(input.image.data))

    result = input.image.data.cpu().clone()
    result = result.numpy()
    result[result<0] = 0
    result[result>255] = 255
    input.image.data = torch.from_numpy(result).cuda()

    optimizer.zero_grad()
    art_net.forward(input.image)

    style_score = 0
    content_score = 0

    #loss1 += art_net.style_loss1_1.backward()
    for sl in style_losses:
        style_score += sl.backward()
    for cl in cont_losses:
        content_score += cl.backward()

    optimizer.step()

    if run%50==0:
        print "run "+str(run)+":"
        print style_score
        print content_score

###################### show result
#input.image.data = (input.image.data - torch.min(input.image.data)).div_(torch.max(input.image.data)-torch.min(input.image.data))
result = input.image.data.cpu().clone()
result = result.numpy()
result[result<0] = 0
#result[result>1] = 1
result = result/np.max(result)
result = torch.from_numpy(result)

#torchvision.utils.save_image(input.image.data,'result')

plt.subplot(224)
imshow(result)
plt.show()
#'''
