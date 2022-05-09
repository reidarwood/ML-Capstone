import torch
from data_helpers import get_loaders
from helpers import train, test, correct_predict_num, train_model
from cnn import CNN
from transformer import VIT
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

def plot(csv_file, model_name):
    fig, ax = plt.subplots(2)
    df = pd.read_csv(csv_file)
    df[['train_acc','test_acc']] *= 100
    
    ax[0].set_title('Model Performance: {}'.format(model_name))
    ax[0].set_ylabel('Cross Entropy Loss')
    ax[1].set_ylabel('Accuracy (%)')
    ax[1].set_xlabel('EPOCH')

    df.plot.line(y=['train_loss','test_loss'],ax=ax[0])
    df.plot.line(y=['train_acc','test_acc'],ax=ax[1])
    ax[0].legend(loc="upper right")
    ax[1].legend(loc="upper left")
    plt.show()



def main():
    # plot('transformer_6_0.5_128.csv', 'Transformer')
    # return

    device = torch.device('cuda:0' if torch.cuda.is_available else "cpu")

    
    batch_size = 300
    input_channels = 3
    class_num=555
    learning_rate = 0.001
    num_epoch = 30
    correct_num_func = correct_predict_num
    loss_func = torch.nn.CrossEntropyLoss()
    

    train_loader, test_loader = get_loaders(batch_size)

    model = VIT(dropout=0.1)
    model.to(device)

    
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    model_name='transformer'

    print(datetime.now())
    df = train_model(model, train_loader, test_loader, loss_func, optimizer, num_epoch, correct_num_func=correct_num_func, print_info=True, device=device, model_name=model_name)
    print(datetime.now())

    
    print('Saving Losses and Accuracies')
    print(df)
    df.to_csv('{}.csv'.format(model_name), index=False)

    
if __name__=='__main__':
    main()