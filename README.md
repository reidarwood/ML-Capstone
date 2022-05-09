# ML-Capstone
For my capstone, I did Fine Grained Visual Classification. To read about my experience read `Capstone.pdf`. This document pertains solely to running the code.

## Environment
To start install an environment with 
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```
This environment is very similar to the default CS142 environment with a few additions

## Data
Additionally data needs to be installed. NABirds dataset can be installed [here](https://dl.allaboutbirds.org/nabirds). This installs a zipped folder, and extract the folder into `data/` such that all of the data can be found at `data/nabirds`. Inside there are folders with images as well as files with labels for the images, a test-train split, as well as some other files that store data that is unused in this project (such as part-level annotations).

After this, we create csv files that for train and test images respectively `'data/nabirds/train.csv'` and `'data/nabirds/test.csv'` by running

```
python main.py -m
```

At this point we should be able to run both a VIT with `python main.py` or CNN with `python main.py --cnn`

## Running
At this point we should be able to run both a VIT with `python main.py` or CNN with `python main.py --cnn`

We can tune the parameters such as `epochs, learning_rate, etc` by modifying the main function or `transformer.py` and `cnn.py` respectively. 

When training, we will test the model each epoch, and every 5 epochs we will save a model in case anything goes wrong and the computer crashes (which happened more than once).

Our `train_model()` function returns a pandas dataframe which we save as a `csv` file that contains epoch-level accuracy and loss for both training and testing. I have left the csv files in the repo with format `[model_name]_[transformerLayers]_[dropout]_[transformerDimension].csv` which were the parameters that I spent time fine tuning

## Plotting
Additionally the `csv` files are hard to read, and we can visualize them with `python main.py -p [path to csv]`