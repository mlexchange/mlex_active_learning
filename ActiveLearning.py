import numpy as np
import os

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Input

from keras import backend as K
from keras.models import load_model
import tensorflow as tf
from keras import applications
import pickle
import argparse
from datetime import datetime
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from Query_Methods import *
from AL_Train import *


print("----------------------------------------------------")
print("DATE and TIME of starting the code =", datetime.now())
print("----------------------------------------------------")
#=========================================================================================================================================#

def parse_input():
    p=argparse.ArgumentParser()
    p.add_argument('experiment_index',type=int, help="index of current experiment")
    p.add_argument('batch_size', type=int, help="active learning batch size")
    p.add_argument('initial_size', type=int, help="initial sample size for active learning")
    p.add_argument('iterations', type=int, help="number of active learning batches to sample")
    p.add_argument('--output_path', '-op', type=str,default=None)
    p.add_argument('method', type=str,
                   choices={'Random','LC','Entropy','BALD'},
                   help="sampling method ('Random','LC','Entropy','BALD')")
    p.add_argument('--initial_idx_path', '-idx', type=str,default=None,help="path to a pickle file with the initial labeled set")
    p.add_argument('--weights_init_path', '-wip', type=str,default=None,help="path to a hdf5 file with the initial weights of the NN")
    p.add_argument('--data_path','-dp',type=str,default=None,help="path to a csv file with the complete data set")
    p.add_argument('--train_path','-tp',type=str,default=None,help="path to a csv file with the train data set")
    p.add_argument('--validation_path','-vp',type=str,default=None,help="path to a csv file with the validation data set")
    p.add_argument('--gpu', '-gpu', type=int, default=0)
    args = p.parse_args()
    return args


#=========================================================================================================================================#

if __name__ == '__main__':
    args=parse_input()
    print(args)
    gpus = tf.config.list_physical_devices('GPU')
    print('Using gpu', args.gpu)
    if gpus:
        try: 
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus[args.gpu], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
# Visible devices must be set before GPUs have been initialized
            print(e)

    results_path=args.output_path
    print("loading Data")
    csv_data_path = args.data_path
    csv_validation_path = args.validation_path
    csv_train_path = args.train_path
    IMG_DIRECTORY = "/global/cfs/cdirs/als/Bilvin/Train_ALS2"
    CLASSES = ['Box','FullSphere','FullSpheroid','Icosahedron','Prism3','Prism6','SawtoothRippleBox']
    RAW_IMG_SIZE = (256, 256)
    BATCH_SIZE = 256
    X_data_dataframe = pd.read_csv(csv_data_path)
    X_train_dataframe = pd.read_csv(csv_train_path)
    X_validation_dataframe = pd.read_csv(csv_validation_path)
    X_data = create_dataset_database(X_data_dataframe,IMG_DIRECTORY,RAW_IMG_SIZE,BATCH_SIZE,CLASSES, aug = False)
    X_train = create_dataset_database(X_train_dataframe,IMG_DIRECTORY,RAW_IMG_SIZE,BATCH_SIZE,CLASSES, aug = False)
    X_validation = create_dataset_database(X_validation_dataframe,IMG_DIRECTORY,RAW_IMG_SIZE,BATCH_SIZE,CLASSES, aug = False)
    print("data loaded")

   
    batch_size=args.batch_size
    initial_size=args.initial_size
    iterations=args.iterations
    
    T=10
    initial_idx_path=args.initial_idx_path
    weights_init_path=args.weights_init_path
    if initial_idx_path is not None:
        idx_path = args.initial_idx_path
        with open(idx_path, 'rb') as f:
            labeled_idx = pickle.load(f)
    else:
        print("Random initial set")
        labeled_idx=np.random.choice(X_train_dataframe.shape[0],initial_size, replace=False)
    checkpoint_path=os.path.join(results_path,'{method}_{exp}_{b_s}_{i_s}/'.format(method=args.method,exp=args.experiment_index,b_s=args.batch_size,i_s=args.iterations))
    os.makedirs(checkpoint_path,exist_ok=True)
    
    model=Sequential()
    model.add(applications.ResNet50(input_shape=(256,256,3),include_top=False,weights=None, input_tensor=None, pooling=None,classes=7))
    model.add(Dropout(0.5))
    model.add(Dense(7,activation="softmax"))
    
    ResNet50_model = model
    checkpoint_initial_path=args.weights_init_path
    if checkpoint_initial_path is not None:
        model=load_model(checkpoint_initial_path)
    else:
        model=ResNet50_model
        model.save(checkpoint_path+'AL_initial_checkpoint')
    accuracies=[]
    queries=[]
    queries.append(labeled_idx)
    results_path=checkpoint_path+'AL_results.pkl'
    labeled_idx_complete=[]
    labeled_idx_complete.append(labeled_idx)
    with open(results_path, 'wb') as f:
            pickle.dump([accuracies,initial_size,batch_size,queries,labeled_idx_complete], f)
    for i in range(iterations):
        print("Starting iterations" )
        if args.method == 'Random':
            method = query_random(X_train_dataframe,labeled_idx,batch_size)
        elif args.method == 'LC':
            method = query_LC(X_train_dataframe,labeled_idx,batch_size,model,IMG_DIRECTORY,RAW_IMG_SIZE,BATCH_SIZE,CLASSES)
        elif args.method == 'Entropy':
            method = query_uncertainityentropy(X_train_dataframe,labeled_idx,batch_size,model,IMG_DIRECTORY,RAW_IMG_SIZE,BATCH_SIZE,CLASSES)
        elif args.method == 'BALD':
            method = query_bald(X_train_dataframe,labeled_idx,batch_size)
        path='ResNet50_{idx}.hdf5'.format(idx=i+1)
        checkpoint_path_i=checkpoint_path+path
        old_labeled = np.copy(labeled_idx)
        labeled_idx=method
        labeled_dataframe = select_files_by_index(X_train_dataframe, labeled_idx,output_file='Trainlist_{method}_{idx}.csv'.format(method=args.method, idx=i+1))
        labeled_data = create_dataset_database(labeled_dataframe,IMG_DIRECTORY,RAW_IMG_SIZE,BATCH_SIZE,CLASSES, aug = False)
        print(np.shape(labeled_idx))
        labeled_idx_complete.append(labeled_idx)
        new_idx = labeled_idx[np.logical_not(np.isin(labeled_idx, old_labeled))]
        queries.append(new_idx)
        print("New training set shape : ", labeled_dataframe.shape )
        K.clear_session()
        ResNet50_model = applications.ResNet50(input_shape=(256,256,3),include_top=True,weights=None, input_tensor=None, pooling=None,classes=7)
        acc, model = evaluate_sample(ResNet50_model,labeled_data, X_validation,X_data, checkpoint_path_i)
        
        accuracies.append(acc)
        
        print("Test Accuracy Is " + str(acc))
        with open(results_path, 'wb') as f:
            pickle.dump([accuracies,initial_size,batch_size,queries,labeled_idx_complete], f)
        print("----------------------------------------------------")
        print("DATE and TIME of finishing the {ite} iteration =".format(ite = i), datetime.now())
        print("----------------------------------------------------")

