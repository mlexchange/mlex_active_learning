import numpy as np
import os

from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint,EarlyStopping, History
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Input, UpSampling2D
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
from keras import regularizers
from keras import backend as K
from keras.models import load_model
from keras.utils import to_categorical
import tensorflow as tf
from keras import applications
from keras.models import load_model
import pickle
import argparse
from datetime import datetime
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
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
    p.add_argument('--initial_idx_path', '-idx', type=str,default=None,help="path to a csv file with the initial labeled set")
    p.add_argument('--weights_init_path', '-wip', type=str,default=None,help="path to a hdf5 file with the initial weights of the NN")
    p.add_argument('--data_path','-dp',type=str,default=None)
    p.add_argument('--labels_path','-lp',type=str,default=None)
    p.add_argument('--gpu', '-gpu', type=int, default=0)
    args = p.parse_args()
    return args

#=========================================================================================================================================#

def select_files_by_index(df, indexes,output_file=None):
    selected_files = df.loc[indexes, 'File']
    selected_labels = df.loc[indexes, 'Label']
    selected_df = pd.DataFrame({'File': selected_files, 'Label': selected_labels})
    if output_file:
        # Save the new DataFrame to a CSV file
        selected_df.to_csv(output_file, index=False)

    return selected_df

def create_dataset_database(train_dataframe,IMG_DIRECTORY,RAW_IMG_SIZE,BATCH_SIZE,CLASSES, aug = False):

    train_image_count = train_dataframe.shape[0]
    if aug:
        train_data_generator = ImageDataGenerator( rescale=1. / 255,
                                                  fill_mode="constant",
                                                  shear_range=0.2,
                                                  zoom_range=(0.5, 1),
                                                  horizontal_flip=True,
                                                  rotation_range=360,
                                                  channel_shift_range=25,
                                                  brightness_range=(0.75, 1.25))
    else:
        train_data_generator = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_data_generator.flow_from_dataframe(train_dataframe,
                                                                    IMG_DIRECTORY,
                                                                    x_col='File',
                                                                    y_col='Label',
                                                                    target_size=RAW_IMG_SIZE,
                                                                    batch_size=BATCH_SIZE,
                                                                    has_ext=True,
                                                                    shuffle =False,
                                                                    classes=CLASSES,
                                                                    class_mode='categorical')
    return train_generator

#=========================================================================================================================================#

def get_unlabeled_idx(train_dataframe, labeled_idx):
    """
    Given the training set and the indices of the labeled examples, return the indices of the unlabeled examples.
    """
    train_image_count = train_dataframe.shape[0]
    return np.arange(train_image_count)[np.logical_not(np.in1d(np.arange(train_image_count),labeled_idx))]

def query_random(train_dataframe,labeled_idx, amount):
    """
    Returns randomly selected samples
    """
    unlabeled_idx = get_unlabeled_idx(train_dataframe, labeled_idx)
    
    return np.hstack((labeled_idx, np.random.choice(unlabeled_idx, amount, replace=False)))

def query_LC(train_dataframe,labeled_idx,amount,ResNet50_model,IMG_DIRECTORY,RAW_IMG_SIZE,BATCH_SIZE,CLASSES):
    """
    Returns samples based on Least Confidence
    """
 
    unlabeled_idx = get_unlabeled_idx(train_dataframe, labeled_idx)
    unlabeled_dataframe = select_files_by_index(train_dataframe, unlabeled_idx)
    unlabeled_data = create_dataset_database(unlabeled_dataframe,IMG_DIRECTORY,RAW_IMG_SIZE,BATCH_SIZE,CLASSES, aug = False)
    
    predictions = ResNet50_model.predict(unlabeled_data)
    print(len(np.shape(predictions)))
    if len(np.shape(predictions))==4:
        unlabeled_predictions =np.amax( np.amax( np.amax(predictions, axis=3) , axis=2) ,axis=1)
    elif len(np.shape(predictions))==2:
        unlabeled_predictions =np.amax(predictions, axis=1)
    selected_indices = np.argpartition(unlabeled_predictions, amount )[:amount]
    indices_1 = np.hstack((labeled_idx, unlabeled_idx[selected_indices]))
    del selected_indices, unlabeled_predictions,predictions,unlabeled_data,unlabeled_dataframe,unlabeled_idx,ResNet50_model 
    return indices_1

def query_uncertainityentropy(train_dataframe,labeled_idx,amount,ResNet50_model,IMG_DIRECTORY,RAW_IMG_SIZE,BATCH_SIZE,CLASSES):
    """
    Returns samples with maximum predictive entropy
    """
    unlabeled_idx = get_unlabeled_idx(train_dataframe, labeled_idx)
    unlabeled_dataframe = select_files_by_index(train_dataframe, unlabeled_idx)
    unlabeled_data = create_dataset_database(unlabeled_dataframe,IMG_DIRECTORY,RAW_IMG_SIZE,BATCH_SIZE,CLASSES, aug = False)
    
    predictions = ResNet50_model.predict(unlabeled_data)
    print(len(np.shape(predictions)))
    if len(np.shape(predictions))==4:
        unlabeled_predictions = np.sum(np.amax( np.amax(predictions, axis=1) , axis=1)* np.log(np.amax( np.amax(predictions, axis=1) , axis=1)+ 1e-10),axis=1)
    elif len(np.shape(predictions))==2:
        unlabeled_predictions = np.sum(predictions * np.log(predictions + 1e-10), axis=1)

    selected_indices = np.argpartition(unlabeled_predictions, amount)[:amount]
    indices_1 = np.hstack((labeled_idx, unlabeled_idx[selected_indices]))
    
    return indices_1
#=========================================================================================================================================#

def dropout_predict(data):

    f = K.function([model.layers[0].input, K.learning_phase()],
                   [model.layers[-1].output])
    predictions = np.zeros((T, data.shape[0], num_labels))
    for t in range(T):
        predictions[t,:,:] = f([data, 1])[0]
    expected_entropy = - np.mean(np.sum(predictions * np.log(predictions + 1e-10), axis=-1), axis=0)  # [batch size]
    expected_p = np.mean(predictions, axis=0)
    entropy_expected_p = - np.sum(expected_p * np.log(expected_p + 1e-10), axis=-1)  # [batch size]
    BALD_acq = entropy_expected_p - expected_entropy

    final_prediction = np.mean(predictions, axis=0)
    prediction_uncertainty = BALD_acq

    return final_prediction, prediction_uncertainty

def query_bald(image_dataset,labeled_idx, amount):
    """
    Returns samples based on BALD acquisition
    """

    unlabeled_idx = get_unlabeled_idx(dataset_size, labeled_idx)

    predictions = np.zeros((unlabeled_idx.shape[0], num_labels))
    uncertainties = np.zeros((unlabeled_idx.shape[0]))
    i = 0
    split = 128  # split into iterations of 128 due to memory constraints
    while i < unlabeled_idx.shape[0]:

        if i+split > unlabeled_idx.shape[0]:
            preds, unc = dropout_predict(image_dataset[unlabeled_idx[i:], :])
            predictions[i:] = preds
            uncertainties[i:] = unc
        else:
            preds, unc = dropout_predict(image_dataset[unlabeled_idx[i:i+split], :])
            predictions[i:i+split] = preds
            uncertainties[i:i+split] = unc
        i += split

    
    selected_indices = np.argpartition(-uncertainties, amount)[:amount]
    return np.hstack((labeled_idx, unlabeled_idx[selected_indices]))
    
    return np.hstack((labeled_idx, np.random.choice(unlabeled_idx, amount, replace=False)))

#=========================================================================================================================================#


def evaluate_sample(ResNet50_model, X_train,  X_validation,X_data,checkpoint_path):
    """
    A function that accepts a labeled-unlabeled data split and trains the relevant model on the labeled data, returning
    the model and it's accuracy on the test set.
    """
   
    # train and evaluate the model:
    model = train_classification_model(ResNet50_model,X_train, X_validation, checkpoint_path)
    acc = model.evaluate(X_data, verbose=0)

    return acc, model
def train_classification_model(ResNet50_model,X_train,  X_validation,  checkpoint_path):

    save_model = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=0, save_best_only=True, mode='auto')
    history = History()
    #ResNet50_model.load_weights(checkpoint_initial_path)
    ResNet50_model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=['acc'])
    ResNet50_model.fit(X_train,epochs=100,batch_size=16,shuffle=True,validation_data=X_validation,callbacks=[save_model,history],verbose=2)
    ResNet50_model.load_weights(checkpoint_path)
    return ResNet50_model

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
    csv_data_path = 'files.csv'
    csv_validation_path = 'validation.csv'
    csv_train_path = 'train.csv'
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

