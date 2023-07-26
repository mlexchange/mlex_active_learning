import numpy as np
import os

from keras.models import Sequential, Model
from keras.layers import Input
from keras import backend as K
import tensorflow as tf
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

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
