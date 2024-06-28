import PIL
import torchvision
from PIL import Image
from torchvision import transforms
from model_resnet import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pokemon = PokemonClassifier()
print(pokemon)
pokemon = pokemon.to(device)

pokemon.load_state_dict(torch.load("pokemoncClassfier2.pth"))

path = "../test/mwzz.jpg"
image = Image.open(path)
transform = torchvision.transforms.Compose([
    transforms.Grayscale(3), # 将图片变为三通道的
    transforms.Resize((280,280)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
image = transform(image)
# print(image.size())
image = image.unsqueeze(0)
image = image.to(device)
with torch.no_grad():
    output = pokemon(image)
print(output)
print(output.argmax(1))


# 在验证集上测试结果良好  但单独用测试集外的图片测试结果不好 是否是因为数据处理的问题  还是数据集太小还是过拟合问题。。。未解决。。。
