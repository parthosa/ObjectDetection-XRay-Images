import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

# Load the pretrained model
# model = models.alexnet(pretrained=True)
# # Use the model object to select the desired layer
# layer = list(model.features.children())[12]

original_model = models.alexnet(pretrained=True)
print(original_model._modules)

# class AlexNetConv4(nn.Module):
#             def __init__(self):
#                 super(AlexNetConv4, self).__init__()
#                 self.features = nn.Sequential(
#                     # stop at conv4
#                     *list(original_model.features.children())
#                 )
#             def forward(self, x):
#                 x = self.features(x)
#                 return x

# model = AlexNetConv4()

# model.eval()

# scaler = transforms.Scale((227, 227))
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
# to_tensor = transforms.ToTensor()

# def get_vector(image_name):
#     # 1. Load the image with Pillow library
#     img = Image.open(image_name).convert('RGB')
#     # 2. Create a PyTorch Variable with the transformed image
#     t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
#     # 3. Create a vector of zeros that will hold our feature vector
#     #    The 'avgpool' layer has an output size of 512
#     my_embedding = torch.zeros(9216)
#     # 4. Define a function that will copy the output of a layer
#     def copy_data(m, i, o):
#         my_embedding.copy_(o.data)
#     # 5. Attach that function to our selected layer
#     # h = layer.register_forward_hook(copy_data)
#     # 6. Run the model on our transformed image
#     return model.forward(t_img)
#     # 7. Detach our copy function from the layer
#     # h.remove()
#     # 8. Return the feature vector
#     # return my_embedding


# pic_one_vector = get_vector('/data/train/blade/B0051_0001.png')
# print(pic_one_vector)

