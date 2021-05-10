# call this script with `python -m playground.create_datastore`
from PIL import Image
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn

arch = "resnet50"
eval_arch_name = "_places365_"+arch+"_feature_noFC"
model = models.resnet50(num_classes=365)
checkpoint = torch.load(
        "/Users/tilman/Documents/Programme/Python/new_bachelor_thesis/evaluation/eval other image retrieval methods/placesCNN/ResNet152-places365/resnet50_places365.pth.tar",
        map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
#model = torch.nn.Sequential(*(list(model.children())[:8])) # noFC_noAVG
model = torch.nn.Sequential(*(list(model.children())[:9])) # noFC
model.eval()

centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def precompute(filename):
    img = Image.open(filename).convert('RGB')
    input_img = V(centre_crop(img).unsqueeze(0))
    output = model.forward(input_img)[0]
    return output # shape noFC torch.Size([2048, 7, 7]