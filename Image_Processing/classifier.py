"""
  classifier.py         :   This file contains the CNN Classifier for digit recognition.
  File created by       :   Shashank Goyal
  Last commit done by   :   Shashank Goyal
  Last commit date      :   3rd September
"""

# import sqrt function from math
from math import sqrt

# import the pytorch module
import torch
# import neural-network module from pytorch
import torch.nn as nn
# import image tranfomations from torch-vision
import torchvision.transforms as T
# import random split function to split the dataset in test-train
from torch.utils.data import random_split
# import Dataloader to feed dataset in batches to the neural-network
from torch.utils.data.dataloader import DataLoader
# import ImageFolder to use a directory as source for image data
from torchvision.datasets import ImageFolder
# import progress bar
from tqdm import tqdm

# import assistive functions for device selection incase of GPU systems
from Image_Processing.pytorch_gpu_assist import get_default_device, to_device, DeviceDataLoader


def accuracy(outputs, labels):
    """determine accuracy for a set of labels and predicted outputs"""

    # get max prediction from each row
    _, preds = torch.max(outputs, dim=1)
    # get average of number of predictions that match the labels
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class Char74kModel(nn.Module):
    """Convolution neural network"""

    def __init__(self):
        """default initialization"""

        # call init method from super class
        super().__init__()

        # create sequence of hidden layers
        self.features = nn.Sequential(
            # in: 1 x 28 x 8
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # out: 32 x 28 x 28

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # out: 32 x 14 x 14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # out: 64 x 14 x 14

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            # out: 64 x 7 x 7
        )

        # create sequence of output layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(64 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 10),
        )

        # initialize network weights
        self.init_weights()

    def init_weights(self):
        """initialize weights for each network as follows"""

        # iterate through each feature layer
        for layer in self.features.children():
            # if layer is of type convolution
            if isinstance(layer, nn.Conv2d):
                # normalize the weights based on the kernel-size of the layer
                n = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels
                layer.weight.data.normal_(0, sqrt(2. / n))
            # if layer is of type Batch Normalization
            elif isinstance(layer, nn.BatchNorm2d):
                # initialize all weights as 1
                layer.weight.data.fill_(1)
                # initialize all biases as 0
                layer.bias.data.zero_()

        # iterate through each classifier layer
        for layer in self.classifier.children():
            # if layer is of type linear classifier
            if isinstance(layer, nn.Linear):
                # implement xavier uniform initialization
                nn.init.xavier_uniform_(layer.weight)
            # if layer is of type Batch Normalization    
            elif isinstance(layer, nn.BatchNorm1d):
                # initialize all weights as 1
                layer.weight.data.fill_(1)
                # initialize all biases as 0
                layer.bias.data.zero_()

    def forward(self, x):
        """method for forward pass"""

        # input to the hidden layer
        x = self.features(x)
        # flatten the output
        x = x.view(x.size(0), -1)
        # feed it to the classifier
        x = self.classifier(x)
        # return output from classifier
        return x

    def training_step(self, batch):
        """function for an iteration of training process"""

        # get images and associated labels for each batch
        images, labels = batch
        # generate predicted output for set of images
        out = self(images)
        # create loss function and move to the computation device
        loss_fn = to_device(nn.CrossEntropyLoss(), get_default_device())
        # calculate loss based on predicted output and actual labels
        loss = loss_fn(out, labels)
        # return the loss value
        return loss

    def validation_step(self, batch):
        """function for an iteration of validation process"""

        # get images and associated labels for each batch
        images, labels = batch
        # generate predicted output for set of images
        out = self(images)
        # create loss function and move to the computation device
        loss_fn = to_device(nn.CrossEntropyLoss(), get_default_device())
        # calculate loss based on predicted output and actual labels
        loss = loss_fn(out, labels)
        # calculate accuracy based on predicted output and actual labels
        acc = accuracy(out, labels)
        # return loss accuracy as dictionary
        return {'val_loss': loss, 'val_acc': acc}

    @staticmethod
    def validation_epoch_end(outputs):
        """generate net statistics fofr each batch"""

        # losses for batch
        batch_losses = [x['val_loss'] for x in outputs]
        # combined losses
        epoch_loss = torch.stack(batch_losses).mean()
        # accuracies for batch
        batch_accs = [x['val_acc'] for x in outputs]
        # combined accuracies
        epoch_acc = torch.stack(batch_accs).mean()
        # return combined data for a batch
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    @staticmethod
    def epoch_end(epoch, result):
        """print formatted string for an epoch"""
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))


def evaluate(model, val_loader):
    """return metrics of a model for a validation dataset"""

    # get output for each batch in validatation dataset
    outputs = [model.validation_step(batch) for batch in val_loader]
    # return mean metrics for the batch
    return model.validation_epoch_end(outputs)


def fit(epochs, model, train_loader, val_loader, lr=0.003):
    """fit the model to the train data and get loss and accuracy against the validation data"""

    # empty list
    history = []
    # Adma optimizer function
    optimizer = torch.optim.Adam(model.parameters(), lr)
    # get learning rate using a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # iterate through epochs
    for epoch in range(epochs):
        # Training Phase 
        # iterate through batches
        for batch in tqdm(train_loader):
            # calculate loss for training step
            loss = model.training_step(batch)
            # calculate gradients
            loss.backward()
            # backpropagate gardients
            optimizer.step()
            # reset gardients to zero
            optimizer.zero_grad()
        # Validation phase
        # get model metrics against validation data set
        result = evaluate(model, val_loader)
        # print metrics of the epoch
        model.epoch_end(epoch, result)
        # append result to history
        history.append(result)
        # modify lr for the optimizer
        lr_scheduler.step()
    # return list of results for each epoch
    return history


def train_model():
    """function to train the model"""

    # sequence of transformations for tensors
    tfms = T.Compose([
        # convert images to gray scale
        T.Grayscale(),
        # resize image to 28 x 28 pixels
        T.Resize((28, 28)),
        # convert images to tensors
        T.ToTensor()
    ])

    # directory path containing images
    data_dir = './data/'
    # read data from the directory, apply sequence of tranforms and associate it with the respective class
    dataset = ImageFolder(data_dir, transform=tfms)

    # batch size i.e., the number of images to be passed in a single training/validation step
    batch_size = 64
    # random seed to split the dataset into same test/train for every run of the program
    random_seed = 42
    torch.manual_seed(random_seed)

    # split the data into 20-80 ratio for test and train sub sets
    val_size = int(0.2 * len(dataset))
    train_size = int(0.8 * len(dataset))

    # get the default computation device
    device = get_default_device()
    # print the device detected
    print("Device Found:", device)

    # split the dataset into train and test datasets
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    # create dataloader from the train dataset
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # move the dataloader to the computation device
    train_loader = DeviceDataLoader(train_loader, device)

    # create dataloader from the validation dataset
    val_loader = DataLoader(val_ds, batch_size * 2, num_workers=4, pin_memory=True)
    # move the dataloader to the computation device
    val_loader = DeviceDataLoader(val_loader, device)

    # create a model object and move it to the computation device
    model = to_device(Char74kModel(), device)

    # start training
    print("Training -")
    # train the model for 20 epochs
    fit(20, model, train_loader, val_loader)
    # evaluate model against the validation set
    result = evaluate(model, val_loader)
    print("(Trained) Model Result = val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}".format(**result))

    model_file_name = 'char74k-cnn.pth'
    print("\nModel saved as: '{}'".format(model_file_name))
    # save the model weights
    torch.save(model.state_dict(), model_file_name)


def eval_loaded_model(model_file_name):
    """evaluate loaded model against the complete dataset"""

    # sequence of transformations for tensors
    tfms = T.Compose([
        # convert images to gray scale
        T.Grayscale(),
        # resize image to 28 x 28 pixels
        T.Resize((28, 28)),
        # convert images to tensors
        T.ToTensor()
    ])

    # directory path containing images
    data_dir = './data/'
    # read data from the directory, apply sequence of tranforms and associate it with the respective class
    dataset = ImageFolder(data_dir, transform=tfms)

    # get the default computation device
    device = get_default_device()
    # batch size i.e., the number of images to be passed in a single training/validation step
    batch_size = 64

    # split the data into 20-80 ratio for test and train sub sets
    val_size = int(0.2 * len(dataset))
    train_size = int(0.8 * len(dataset))

    # split the dataset into train and test datasets
    _, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    # create dataloader from the validation dataset
    val_loader = DataLoader(val_ds, batch_size * 2, num_workers=4, pin_memory=True)
    # move the dataloader to the computation device
    val_loader = DeviceDataLoader(val_loader, device)

    # load the model object from the file
    loaded_model = load_model(model_file_name)
    print("Loaded Model:", model_file_name)

    # evaluate the loaded model against validation dataset as sanity check
    result = evaluate(loaded_model, val_loader)
    print("(Loaded) Model Result = val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}".format(**result))


def load_model(model_file_name):
    """Load model from a file name"""

    # create a model object and move it to the computation device    
    model = to_device(Char74kModel(), get_default_device())
    # load the model file to the object and map it to the available computation device
    model.load_state_dict(torch.load(model_file_name, map_location=get_default_device()))
    # return the loaded model
    return model
