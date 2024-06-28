import torch
import torchvision
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from pokemon_dataSet import *
from torch import nn, optim
from model_resnet import *

root = "../images"
batch_size = 64
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 224x224常用尺寸
train_dataset = Pokemon(root, 224, 'train')
test_dataset = Pokemon(root, 224, 'test')
val_dataset = Pokemon(root, 224, 'val')

# print(train_dataset[0])
train_loader = data.DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size, shuffle=False)
val_loader = data.DataLoader(val_dataset, batch_size, shuffle=False)

# for img, label in train_loader:
#     print(img)
#     print(label)
pokemon = PokemonClassifier()
pokemon = pokemon.to(device)

# 损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
# 随机梯度下降
optimizer = optim.SGD(pokemon.parameters(), lr=learning_rate, momentum=0.9)
# 训练模型
writer = SummaryWriter("logs")

for epoch in range(10):
    running_loss = 0.0
    for data in train_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = pokemon(inputs)
        loss = loss_fn(outputs, labels)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    writer.add_scalar("train_loss", running_loss, epoch)
    print(f'训练轮数:{epoch + 1}, loss: {running_loss}')

    # 在验证集上评估模型
    pokemon.eval()
    val_correct = 0
    val_total = len(val_dataset)
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = pokemon(images)

            # val_total += labels.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()
    val_correct_rate = 100 * val_correct / val_total
    writer.add_scalar("val_correct", val_correct_rate, epoch)
    print(f'验证集准确率: {val_correct_rate}%')

# 在测试集上评估模型
test_correct = 0
test_total = len(test_dataset)
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = pokemon(images)
        # _, predicted = torch.max(outputs.data, 1)
        # test_total += labels.size(0)
        test_correct += (outputs.argmax(1) == labels).sum().item()
    test_correct_rate = 100 * test_correct / test_total
    # writer.add_scalar("test_correct", test_correct_rate, epoch)
    print(f'测试集正确率: {test_correct_rate}%')

writer.close()
torch.save(pokemon.state_dict(), "pokemoncClassfier2.pth")
