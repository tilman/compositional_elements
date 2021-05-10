import torchvision.models as models
import torch
from PIL import Image
from torchvision import transforms

######################################################## config params
arch = "vgg19_bn"
eval_arch_name = "_imageNet_"+arch+"_features"

######################################################## model setup
vgg19 = models.vgg19_bn(pretrained=True)
model = vgg19.features.eval() # only take head of the network
# notes
# VGG 19-layer model (configuration ‘E’) with batch normalization “Very Deep Convolutional Networks For Large-Scale Image Recognition” <https://arxiv.org/pdf/1409.1556.pdf>.,  also city batch norm
# pretrained (bool) – If True, returns a model pre-trained on ImageNet

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
def precompute(filename):
    input_image = Image.open(filename).convert('RGB')
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    with torch.no_grad():
        output = model(input_batch)
    return output[0]