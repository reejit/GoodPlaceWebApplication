import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(12800, 1024)
        self.fc2 = nn.Linear(1024, 13)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        #self.conv2_drop = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(12800, 1024)
        self.fc2 = nn.Linear(1024, 12)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        #self.conv2_drop = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(12800, 1024)
        self.fc2 = nn.Linear(1024, 4)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

device = torch.device('cpu')
kojaSezona = Net().to(device)
kojaSezona.load_state_dict(torch.load("sezone.pth",map_location=device))
kojaSezona.eval()

def transform_image(image_bytes):
    transform = transforms.Compose([
                                          transforms.Resize((96, 96)),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4883, 0.4551, 0.4170), (0.2208, 0.2161, 0.2163))])
    return transform(image_bytes).unsqueeze_(0)

def get_prediction(slika):
    slika = slika.to(device)
    outputs = kojaSezona(slika)
    rez = int(torch.argmax(outputs.data) + 1)
    if rez == 1 or rez == 4:
        model = Net1().to(device)
        if rez == 1:
            model.load_state_dict(torch.load("season1.pth",map_location=device))
        else:
            model.load_state_dict(torch.load("season4.pth",map_location=device))
    else:
        model = Net2().to(device)
        if rez == 2:
            model.load_state_dict(torch.load("season2.pth",map_location=device))
        else:
            model.load_state_dict(torch.load("season3.pth",map_location=device))

    model.eval()
    outputs = model(slika)
    ep = int(torch.argmax(outputs.data) + 1)
    return "S{}-ep{}.jpg".format(rez, ep)
