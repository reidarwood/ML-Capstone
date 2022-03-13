DATA_BASE_PATH="data/"
NA_BIRDS_PATH = DATA_BASE_PATH + "nabirds/" 
    
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

def load_image_sizes(dataset_path=''):
  
  sizes = {}
  
  with open(os.path.join(dataset_path, 'sizes.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      image_id = pieces[0]
      width, height = map(int, pieces[1:])
      sizes[image_id] = [width, height]
  
  return sizes

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
    x=12