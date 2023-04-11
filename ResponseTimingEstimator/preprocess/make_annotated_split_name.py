import os
import random

NAME_PATH = '/mnt/aoni04/yaguchi/code/ResponseTimingEstimator/data/ATR_Annotated/data_-500_2000/names'
SEED = 0
TRAIN = 0.8
VALID = 0.1
TEST = 0.1

def main():
    
    with open(os.path.join(NAME_PATH, 'M1_all.txt')) as f:
        lines = f.readlines()
    
    file_names = [line.replace('\n', '') for line in lines]
            
    random.seed(SEED)
    random.shuffle(file_names)
    
    LEN = len(file_names)
    
    train_names = file_names[:int(LEN*TRAIN)]
    valid_names = file_names[int(LEN*TRAIN):int(LEN*(TRAIN+VALID))]
    test_names = file_names[int(LEN*(TRAIN+VALID)):]
    
    with open(os.path.join(NAME_PATH, 'M1_train.txt'), mode='w') as f:
        for name in train_names:
            f.write(name + "\n")
    with open(os.path.join(NAME_PATH, 'M1_valid.txt'), mode='w') as f:
        for name in valid_names:
            f.write(name + "\n")
    with open(os.path.join(NAME_PATH, 'M1_test.txt'), mode='w') as f:
        for name in test_names:
            f.write(name + "\n")


if __name__ == '__main__':
    main()