import torch
from data_helpers import get_loaders
from helpers import correct_predict_num, train_model, test
from cnn import CNN
from transformer import VIT
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

import argparse

def plot(csv_file):
    '''
    Create plot for losses and accuracies.
    :param csv_file: path to csv file with following columns: ['train_loss', 'train_acc', 'test_loss', 'test_acc']
    '''

    fig, ax = plt.subplots(2)
    df = pd.read_csv(csv_file)
    df[['train_acc','test_acc']] *= 100
    
    ax[0].set_title('Model Performance: {}'.format(csv_file))
    ax[0].set_ylabel('Cross Entropy Loss')
    ax[1].set_ylabel('Accuracy (%)')
    ax[1].set_xlabel('EPOCH')

    df.plot.line(y=['train_loss','test_loss'],ax=ax[0])
    df.plot.line(y=['train_acc','test_acc'],ax=ax[1])
    ax[0].legend(loc="upper right")
    ax[1].legend(loc="upper left")
    plt.show()

def test_model(model_file):
    '''
    Tests pretrained model on test set
    :param model_file: name of the file that the model is stored
    '''
    device = torch.device('cuda:0' if torch.cuda.is_available else "cpu")

    
    batch_size = 300
    class_num=555
    correct_num_func = correct_predict_num
    loss_func = torch.nn.CrossEntropyLoss()
    
    _, test_loader = get_loaders(batch_size)

    model = torch.load(model_file)
    model.to(device)

    # Train the model
    loss, acc = test(model, test_loader, loss_func, correct_num_func=correct_num_func, device=device)
    print('Loss: {:.4f}\nAccuracy: {:.4f}'.format(loss, acc))


def main(is_transformer):
    device = torch.device('cuda:0' if torch.cuda.is_available else "cpu")

    
    batch_size = 300
    input_channels = 3
    class_num=555
    learning_rate = 0.001
    num_epoch = 30
    correct_num_func = correct_predict_num
    loss_func = torch.nn.CrossEntropyLoss()
    

    train_loader, test_loader = get_loaders(batch_size)

    model = None
    model_name='transformer'
    if is_transformer:
        model = VIT(class_num=class_num)
    else:
        model = CNN(class_num=class_num)
        model_name='cnn'
    model.to(device)

    
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    

    # Train the model
    df = train_model(model, train_loader, test_loader, loss_func, optimizer, num_epoch, correct_num_func=correct_num_func, print_info=True, device=device, model_name=model_name)

    
    print('Saving Losses and Accuracies')
    df.to_csv('{}.csv'.format(model_name), index=False)

    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--make', help='builds files needed for easier data processing', action='store_true')
    parser.add_argument('-p', '--plot', help='Plot csv file of losses and accuracies')
    parser.add_argument('-t', '--test', help='Test model on test dataset for model provided')
    parser.add_argument('--cnn', help='Type of model to use, default is Transformer. With flag is CNN', action='store_true')
    args = parser.parse_args()

    is_transformer = True
    if args.cnn:
        is_transformer = False
    
    if args.make:
        print('making')
    elif args.plot:
        plot(args.plot)
    elif args.test:
        test_model(args.test)
    else:
        main(is_transformer)
