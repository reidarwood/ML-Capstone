import torch
import pandas as pd

def train(model, dataloader, loss_func, optimizer,correct_num_func=None,device=None):
    '''
    Trains model for one epoch

    :param model: Torch model.
    :param dataloader: DataLoader for training data
    :param loss_func: Torch loss function
    :param optimizer: An optimizer instance from torch.optim.
    :param correct_num_func: A function to calculate how many samples are correctly classified.
    :param device: a torch device to move our data to
    :return:
        :epoch loss
        :epoch accuracy
    '''
    epoch_loss_sum = 0
    epoch_correct_num = 0
    for X, Y in dataloader:
        X = X.to(device)
        Y = Y.to(device)
        output = model(X)
        optimizer.zero_grad()
        loss = loss_func(output,Y)
        loss.backward()
        optimizer.step()
        epoch_loss_sum += loss.item() * X.shape[0]
        if correct_num_func is not None:
            epoch_correct_num += correct_num_func(output, Y)
    loss = epoch_loss_sum / len(dataloader.dataset)
    acc = epoch_correct_num / len(dataloader.dataset)
    return loss, acc

def train_model(model, train_data, test_data, loss_func, optimizer, num_epoch, correct_num_func=None, print_info=True, device=None, model_name='transformer'):
    """
    Trains the model for `num_epoch` epochs.
    :param model: Torch model.
    :param train_data: DataLoader for training data
    :param test_data: DataLoader for testing data
    :param loss_func: Torch loss function
    :param optimizer: An optimizer instance from torch.optim.
    :param num_epoch: The number of epochs we train our network.
    :param correct_num_func: A function to calculate how many samples are correctly classified.
    :param print_info: If True, print the average loss (and accuracy, if applicable) after each epoch.
    :param device: a torch device to move our data to
    :param model_name: a string we use to save models intermettinly through training
    :return:
        pandas df with following columns: ['train_loss', 'train_acc', 'test_loss', 'test_acc']
    """

    epoch_stats = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
    }
    model.train()

    for epoch in range(num_epoch):
        train_loss, train_acc = train(model, train_data, loss_func, optimizer, correct_num_func=correct_num_func, device=device)
        epoch_stats['train_loss'].append(train_loss)
        epoch_stats['train_acc'].append(train_acc)

        test_loss, test_acc = test(model, test_data, loss_func, correct_num_func=correct_num_func, device=device)
        epoch_stats['test_loss'].append(test_loss)
        epoch_stats['test_acc'].append(test_acc)

        if print_info:
            print("EPOCH {}:".format(epoch))
            print("Train Loss:     {:.4f}  Test Loss:     {:.4f}".format(train_loss, test_loss))
            print("Train Accuracy: {:.4f}% Test Accuracy: {:.4f}%".format(train_acc*100, test_acc*100))
        if epoch % 5 == 4:
            torch.save(model, "models/{}{}.pt".format(model_name, epoch+1))
    
    return pd.DataFrame.from_dict(epoch_stats)


def test(model, dataloader, loss_func, correct_num_func=None, device=None):
    '''
    Tests model for one epoch

    :param model: Torch model.
    :param dataloader: DataLoader for training data
    :param loss_func: Torch loss function
    :param correct_num_func: A function to calculate how many samples are correctly classified.
    :param device: a torch device to move our data to
    :return:
        :epoch loss
        :epoch accuracy
    '''
    loss = 0
    correct_predictions = 0
    model.eval()
    with torch.no_grad():
        for X, Y in dataloader:
            X = X.to(device)
            Y = Y.to(device)

            output = model(X)
            batch_loss = loss_func(output, Y)
            loss += batch_loss.item() * X.shape[0]
            if correct_num_func is not None:
                correct_predictions += correct_num_func(output, Y)
        loss /= len(dataloader.dataset)
        if correct_num_func is not None:
            accuracy = correct_predictions / len(dataloader.dataset)
    if correct_num_func is not None:
        return loss, accuracy
    return loss


def correct_predict_num(logit, target):
    """
    Returns the number of correct predictions.
    :param logit: 2D torch tensor of shape [n, class_num], where
        n is the number of samples, and class_num is the number of classes (10 for MNIST).
        Represents the output of CNN model.
    :param target: 1D torch tensor of shape [n],  where n is the number of samples.
        Represents the ground truth categories of images.
    :return: A python scalar. The number of correct predictions.
    """
    return torch.sum(torch.argmax(logit, 1) == target).item()