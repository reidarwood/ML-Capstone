import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import default_loader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

DATA_BASE_PATH="data/"
NA_BIRDS_PATH = DATA_BASE_PATH + "nabirds/" 
  
class NABirds(Dataset):
  def __init__(self, df, transform=None):
    """
    Initialize the NABirds dataset variables.
    :param df: pandas DataFrame with the following columns:
                img_id: unique identifier for image. Not needed but included for tracking if needed
                path: path to the data
                label: label for the image
    :param transform: transformation function for image preprocessing
    """
    self.transform = transform
    self.loader = default_loader
    self.data = df
    self.class_names = load_class_names(NA_BIRDS_PATH)
    self.class_hierarchy = load_hierarchy(NA_BIRDS_PATH)

  def __len__(self):
    """
    Returns the number of samples.
    :return: The number of samples.
    """
    return len(self.data)
      
  def __getitem__(self, idx):
    """
    Returns feature and label of the sample at the given index.
    :param index: Index of a sample.
    :return: Feature and label of the sample at the given index.
    """
    sample = self.data.iloc[idx]
    path = os.path.join(NA_BIRDS_PATH, 'images/', sample['path'])
    img = default_loader(path)
    if self.transform is not None:
      img = self.transform(img)
    
    label = sample['label']
    return img, torch.tensor(label)

def get_loaders(batch_size):
  train_path = os.path.join(NA_BIRDS_PATH,'train.csv')
  test_path = os.path.join(NA_BIRDS_PATH,'test.csv')
  train_csv = pd.read_csv(train_path)
  test_csv = pd.read_csv(test_path)

  train_transform=transforms.Compose([transforms.Resize((300, 300), Image.BILINEAR),
                                        transforms.RandomCrop((224, 224)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
  test_transform=transforms.Compose([transforms.Resize((300, 300), Image.BILINEAR),
                                        transforms.CenterCrop((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
  train = DataLoader(NABirds(train_csv, train_transform), batch_size=batch_size, shuffle=True)
  test = DataLoader(NABirds(test_csv, test_transform), batch_size=batch_size, shuffle=False)

  return train, test

def load_class_names(dataset_path=''):
  
  names = {}
  
  with open(os.path.join(dataset_path, 'classes.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      class_id = pieces[0]
      names[class_id] = ' '.join(pieces[1:])
  
  return names

def load_image_labels(dataset_path=''):
  labels = {}
  
  with open(os.path.join(dataset_path, 'image_class_labels.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      image_id = pieces[0]
      class_id = pieces[1]
      labels[image_id] = class_id
  
  return labels
        
def load_image_paths(dataset_path='', path_prefix=''):
  
  paths = {}
  
  with open(os.path.join(dataset_path, 'images.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      image_id = pieces[0]
      path = os.path.join(path_prefix, pieces[1])
      paths[image_id] = path
  
  return paths

def load_hierarchy(dataset_path=''):
  
  parents = {}
  
  with open(os.path.join(dataset_path, 'hierarchy.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      child_id, parent_id = pieces
      parents[child_id] = parent_id
  
  return parents

def make_nabirds():
    '''
    Make train.txt and test.txt representing the data split
    '''
    image_paths = pd.read_csv(os.path.join(NA_BIRDS_PATH, 'images.txt'),
                                       sep=' ', names=['img_id', 'path'])
    labels = pd.read_csv(os.path.join(NA_BIRDS_PATH, 'image_class_labels.txt'),
                                       sep=' ', names=['img_id', 'lbl'])
    train_test_split = pd.read_csv(os.path.join(NA_BIRDS_PATH, 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
    df = image_paths.merge(labels,on='img_id').merge(train_test_split,on='img_id')
    
    # labels are non-contiguous
    labels = set(df['lbl'])
    label_remap = pd.DataFrame({key: v for v, key in enumerate(labels)}.items(), columns=['lbl','label'])
    df = df.merge(label_remap,on='lbl')
    df = df.drop('lbl',axis=1)
    
    test,train = [x for _, x in df.groupby(df['is_training_img']==1)]
    test.to_csv(NA_BIRDS_PATH+'test.csv',index=False,columns=('img_id','path','label'))
    train.to_csv(NA_BIRDS_PATH+'train.csv',index=False,columns=('img_id','path','label'))

make_nabirds()