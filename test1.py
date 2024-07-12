import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchsummary import summary
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.datasets import mnist

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

print("x = ", x_train.shape)

features_train, features_test, targets_train, targets_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

featuresTrain = torch.from_numpy(features_train)
targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor)

featuresTest = torch.from_numpy(features_test)
targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor)

train = torch.utils.data.TensorDataset(featuresTrain, targetsTrain)
test = torch.utils.data.TensorDataset(featuresTest, targetsTest)

LR = 0.01
batch_size = 100
n_inters = 10000
num_epochs = n_inters / (len(features_train) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)


class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        # Convolution 1 , input_shape=(1,28,28), output_shape=(16,24,24)
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        # activation
        self.relu1 = nn.ReLU()
        # Max pool 1, output_shape=(16,12,12)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # Convolution 2, output_shape=(32,8,8)
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        # activation
        self.relu2 = nn.ReLU()
        # Max pool 2, output_shape=(32,4,4)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # Fully connected 1, input_shape=(32*4*4)
        self.fc1 = nn.Linear(32 * 4 * 4, 10)

    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        # Max pool 1
        out = self.maxpool1(out)
        # Convolution 2
        out = self.cnn2(out)
        out = self.relu2(out)
        # Max pool 2
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        # Linear function (readout)
        out = self.fc1(out)
        return out

model = CNN_Model().to(device)                            # Create the CNN Model
optimizer = torch.optim.Adam(model.parameters(), lr = LR) # 選擇你想用的 optimizer(Adam)
summary(model, (1, 28, 28))                               # 利用 torchsummary 的 summary package 印出模型資訊，input size: (1 * 28 * 28)
loss_func = nn.CrossEntropyLoss()                         # 選擇想用的 loss function(CrossEntropy)
input_shape = (-1, 1, 28, 28)

def fit_model(model, loss_func, optimizer, input_shape, num_epochs, train_loader, test_loader):
    # Traning the Model
    # 儲存訓練資訊的 List
    training_loss, training_accuracy = [], []
    validation_loss, validation_accuracy = [], []
    for epoch in range(num_epochs):
        # ---------------------------
        # Training Stage
        # ---------------------------
        correct_train, total_train = 0, 0
        for i, (images, labels) in enumerate(train_loader):
            train, labels = images.view(input_shape).to(device), labels.to(device)  # 取出 training data 以及 labels(轉 device 的型態)
            optimizer.zero_grad()                                                   # 清空梯度
            outputs = model(train)                                                  # 將訓練資料輸入至模型進行訓練 (Forward propagation)
            train_loss = loss_func(outputs, labels)                                 # 計算 loss
            train_loss.backward()                                                   # 將 loss 反向傳播
            optimizer.step()

            # 計算訓練資料的準確度 (correct_train / total_train)
            predicted = torch.max(outputs.data, 1)[1]  # 取出預測的 maximum
            total_train += len(labels)  # 全部的 label 數 (Total number of labels)
            correct_train += (predicted == labels).float().sum()

        # 將 accuracy 和 loss 存入 list
        train_accuracy = 100 * correct_train / float(total_train)  # training accuracy (To cpu())
        training_accuracy.append(train_accuracy.cpu())
        training_loss.append(train_loss.data.cpu())  # training loss (To cpu())

        # --------------------------
        # Testing Stage
        # --------------------------
        correct_test, total_test = 0, 0
        for images, labels in test_loader:
            test, labels = images.view(input_shape).to(device), labels.to(device)  # 取出 testing data 以及 labels(轉 device 的型態)
            outputs = model(test)  # 將測試資料輸入至模型進行測試 (Forward propagation)
            val_loss = loss_func(outputs, labels)  # 計算 loss

            # 計算測試資料的準確度 (correct_test / total_test)
            predicted = torch.max(outputs.data, 1)[1]  # 取出預測的 maximum
            total_test += len(labels)  # 全部的 label 數 (Total number of labels)
            correct_test += (predicted == labels).float().sum()  # 全部猜中的個數 (Total correct predictions)

        # 將 accuracy 和 loss 存入 list
        val_accuracy = 100 * correct_test / float(total_test)  # testing accuracy (To cpu())
        validation_accuracy.append(val_accuracy.cpu())
        validation_loss.append(val_loss.data.cpu())  # testing loss (To cpu())

        # 顯現當前 Epoch 訓練情況
        print('Train Epoch: {}/{} Traing_Loss: {} Traing_acc: {:.6f}% Val_Loss: {} Val_accuracy: {:.6f}%'.format(epoch + 1, num_epochs, train_loss.data, train_accuracy, val_loss.data, val_accuracy))
    return training_loss, training_accuracy, validation_loss, validation_accuracy

# 訓練模型
training_loss, training_accuracy, validation_loss, validation_accuracy = fit_model(model, loss_func, optimizer, input_shape, num_epochs, train_loader, test_loader)

# Loss
plt.plot(range(num_epochs), training_loss, 'b-', label='Training_loss')
plt.plot(range(num_epochs), validation_loss, 'g-', label='validation_loss')
plt.title('Training & Validation loss')
plt.xlabel('Number of epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss.png')
plt.show()

# Accuracy
plt.plot(range(num_epochs), training_accuracy, 'b-', label='Training_accuracy')
plt.plot(range(num_epochs), validation_accuracy, 'g-', label='Validation_accuracy')
plt.title('Training & Validation accuracy')
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('acc.png')
plt.show()