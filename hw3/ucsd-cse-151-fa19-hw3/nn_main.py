import torch
import numpy as np
from matplotlib import pyplot as plt
from nn_models import *

"""
TODO: You may want to change these parameters
"""
num_iter = 50
learning_rate = 0.001
batch_size = 25


# returns: predictions -- rank 1 tensor of predicted labels
def evaluate_kaggle(loader, net):
    predictions = []
    # use model to get predictions
    for X in loader:
        outputs = net(X[0])
        predictions.append(torch.argmax(outputs.data, 1))

    return torch.stack(predictions)


# returns: data loader for training and validation (images and labels)
# and a data loader for testing (no labels)
def load_kaggle_data(train_data, val_data, test_data):
    # train, validation, and test data loader
    data_loaders = []

    # read training, test, and validation data
    for (data, labels) in [train_data, val_data]:
        imgs = data.float()
        labels = labels.long()

        # divide each image by its maximum pixel value for numerical stability
        imgs = imgs / torch.max(imgs, dim=1).values[:, None]

        # [batch x num_channel x image width x image height]
        imgs = imgs.view(-1, 1, 28, 28)

        # create dataset and dataloader, a container to efficiently load data in batches
        dataset = utils.TensorDataset(imgs, labels)
        dataloader = utils.DataLoader(dataset, batch_size=32, shuffle=True)
        data_loaders.append(dataloader)

    test_data = test_data.float()
    test_data = test_data / torch.max(test_data, dim=1).values[:, None]
    test_dataset = utils.TensorDataset(test_data.view(-1, 1, 28, 28))
    test_loader = utils.DataLoader(test_dataset)

    return data_loaders[0], data_loaders[1], test_loader


"""
Read data from the specified training, validation and test data files.
We are using the whole image, not creating other features now
"""
def read_data(trainFile, valFile, testFile):
    # trian, validation, and test data loader
    data_loaders = []

    # read training, test, and validation data
    for file in [trainFile, valFile, testFile]:
        # read data
        data = np.loadtxt(file)
        # digit images
        imgs = torch.tensor(data[:,:-1]).float()
        # divide each image by its maximum pixel value for numerical stability
        imgs = imgs / torch.max(imgs,dim=1).values[:,None]

        # labels for each image
        labels = torch.tensor(data[:,-1]).long()

        # if using CNN model, reshape each image:
        # [batch x num_channel x image width x image height]
        if not modelNum == 0:
            imgs = imgs.view(-1,1,28,28)

        # create dataset and dataloader, a container to efficiently load data in batches
        dataset = utils.TensorDataset(imgs,labels)
        dataloader = utils.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        data_loaders.append(dataloader)
    
    return data_loaders[0], data_loaders[1], data_loaders[2]

"""
Train your Multilayer Perceptron (MLP)
Initialize your MLP model --> define loss function --> define optimizer
--> train your model with num_iter epochs --> pick the best model and return
    - Parameters:   train_loader --- the train dataloader
                    val_loader --- the validation dataloader
    - Return:       net --- the best trained MLP network with the lowest validation loss
                    avg_train_loss --- a list of averaged training loss of length num_iter
                    avg_val_loss --- a list of averaged validation loss of length num_iter
"""
def trainMLP(train_loader,val_loader, modelNum):
    # average training loss, one value per iteration (averaged over all batches in one iteration)
    avg_train_loss = []
    # average validation loss, one value per iteration (averaged over all batches in one iteration)
    avg_val_loss = []
    # record the lowest validation loss, used to determine early stopping (best model)
    best_val_score = float('inf')
    if modelNum == 0:
        net = BaselineMLP()
    elif modelNum == 1:
        net = BaselineCNN(in_dim=28, in_channels=1, n_classes=10)
    elif modelNum == 2:
        net = TheNameOfYourClass(in_dim=28, in_channels=1, n_classes=10)
    # TODO4: define loss function
    #       define optimizer
    #       for each iteration, iteratively train all batches
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    i = 0
    while i < num_iter:
        train_loss = 0
        net.train()
        for x,y in train_loader:
            yp = net(x)
            loss = loss_func(yp,y.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        # TODO5: implement your training and early stopping
        # TODO6: save the best model with lowest validation loss and load it to do testing
        torch.save(net.state_dict(), f"epoch{i}.pt")

        avg_train_loss.append(train_loss)
        te_loss = 0
        net.eval()
        for x,y in val_loader:
            yp = net(x)
            loss = loss_func(yp,y.cuda())
            te_loss += loss.item()
        te_loss /= len(val_loader)
        avg_val_loss.append(te_loss)
        print(f"Epoch : {i}, Train loss: {train_loss}, Valid/Test loss: {te_loss}")
        i += 1

    state_dict = torch.load(f"epoch{avg_val_loss.index(min(avg_val_loss))}.pt")
    net.load_state_dict(state_dict)
        
    return net, avg_train_loss, avg_val_loss

"""
Train your Baseline Convolutional Neural Network (CNN)
Initialize your CNN model --> define loss function --> define optimizer
--> train your model with num_iter epochs --> pick the best model and return
    - parameters:   train_loader --- the train dataloader
                    val_loader --- the validation dataloader
    - return:       net --- the best trained CNN network with the lowest validation loss
                    train_loss --- a list of training loss
"""
def trainCNN(train_loader,val_loader):
    # average training loss, one value per iteration (averaged over all batches in one iteration)
    avg_train_loss = []
    # average validation loss, one value per iteration (averaged over all batches in one iteration)
    avg_val_loss = []
    # record the lowest validation loss, used to determine early stopping (best model)
    best_val_score = float('inf')
    net = BaselineCNN()
    # TODO9: define loss function
    #       define optimizer
    #       for each epoch, iteratively train all batches
    i = 0
    while i < num_iter:
        # TODO10: implement your training and early stopping
        # TODO11: save the best model with lowest validation loss and load it to do testing
        raise NotImplementedError
    
    return net, avg_train_loss, avg_val_loss


"""
Evaluate the model, using unseen data features "X" and
corresponding labels "y".
Parameters: loader --- the test loader
            net --- the best trained network
Return: the accuracy on test set
"""
def evaluate(loader, net):
    total = 0
    correct = 0
    # use model to get predictions
    for X, y in loader:
        outputs = net(X)
        predictions = torch.argmax(outputs.data, 1)
        
        # total number of items in dataset
        total += y.shape[0]

        # number of correctly labeled items in dataset
        correct += torch.sum(predictions == y.cuda())

    # return fraction of correctly labeled items in dataset
    return float(correct) / float(total)

if __name__ == "__main__":

    # TODO: you'll need to change this to False if you want to 
    # test your CNN model
    modelNum = 2

    # load data from file
    train_loader, val_loader, test_loader = \
        read_data('hw0train.txt','hw0validate.txt', 'hw0test.txt')

    net, t_losses, v_losses = trainMLP(train_loader,val_loader, modelNum)

    # evaluate model on validation data
    accuracy = evaluate(test_loader, net)

    print("Test accuracy: {}".format(accuracy))

    # plot losses
    plt.plot(t_losses)
    plt.plot(v_losses)
    plt.legend(["training_loss","validation_loss"])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss plot")
    plt.show()
"""
    # load data into pytorch tensors
    data_x = torch.from_numpy(np.load('kaggle/train/x_train.npy'))
    data_y = torch.from_numpy(np.load('kaggle/train/y_train.npy'))
    test_data = torch.from_numpy(np.load('kaggle/test/x_test.npy'))

    # split into training & validation
    val_size = 10000
    val_data = (data_x[0:val_size], data_y[0:val_size])
    train_data = (data_x[val_size:], data_y[val_size:])

    # create DataLoaders
    train_loader, val_loader, test_loader = load_kaggle_data(train_data, val_data, test_data)
    net, t_losses, v_losses = trainMLP(train_loader, val_loader, modelNum)

    # evaluate model on test data
    predictions = evaluate_kaggle(test_loader, net)

    predictions = predictions.cpu()
    pred_numpy = predictions.numpy()[:, 0]  # convert to numpy array

    predictions_file = open("predictions.txt", 'w')
    predictions_file.write("ImageId,Class\n")
    curr_id = 0
    for prediction in pred_numpy:
        predictions_file.write(str(curr_id) + "," + str(prediction) + "\n")
        curr_id += 1
    predictions_file.close()
    import csv

    with open("predictions.txt", 'r') as infile, open("result.csv", 'w', newline='') as outfile:
        stripped = (line.strip() for line in infile)
        lines = (line.split(",") for line in stripped if line)
        writer = csv.writer(outfile)
        writer.writerows(lines)
"""