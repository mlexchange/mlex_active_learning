import numpy as np
import os

from keras.models import Sequential, Model
from keras.layers import Input
from keras import backend as K
import tensorflow as tf
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

#=========================================================================================================================================#

def preprocess_image(image_path, label):
    img = tf.io.read_file("/global/cfs/cdirs/als/Bilvin/Train_ALS2/"+image_path)
    img = tf.io.decode_image(img, channels=3, dtype=tf.float32)
    img = img / 255.0  # Normalize pixel values to the range [0, 1]
    return img, label


def select_files_by_index(df, indexes,output_file=None):
    selected_files = df.loc[indexes, 'File']
    selected_labels = df.loc[indexes, 'Label']
    selected_df = pd.DataFrame({'File': selected_files, 'Label': selected_labels})
    if output_file:
        # Save the new DataFrame to a CSV file
        selected_df.to_csv(output_file, index=False)

    return selected_df

def create_dataset_tf(data,batch_size_per_gpu,Strategy=None,num_gpus=None ):
    image_paths = data['File'].values
    labels = data['Label'].values
    label_encoder = LabelEncoder()
    integer_labels = label_encoder.fit_transform(labels) 
    labels_cat = to_categorical(integer_labels, num_classes=7)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels_cat))
    if strategy:
        with strategy.scope():
            dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)# Use map function within the strategy scope for parallel processing   
    if num_gpus:
        global_batch_size = batch_size_per_gpu * num_gpus
    else:
        global_batch_size = batch_size_per_gpu 

    dataset = dataset.batch(global_batch_size)
    return dataset

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
    unlabeled_idx = np.arange(train_image_count)[np.logical_not(np.in1d(np.arange(train_image_count),labeled_idx))]
    return unlabeled_idx

def get_unlabeled_data(train_dataframe, labeled_idx,strategy=None,num_gpus=None,batch_size_per_gpu=32):
    unlabeled_idx = get_unlabeled_idx(train_dataframe, labeled_idx)
    unlabeled_dataframe = select_files_by_index(train_dataframe, unlabeled_idx)
    unlabeled_data = create_dataset_tf(unlabeled_dataframe,strategy,num_gpus,batch_size_per_gpu )
    return unlabeled_data


def query_random(train_dataframe,labeled_idx, amount):
    """
    Returns randomly selected samples
    """
    unlabeled_idx = get_unlabeled_idx(train_dataframe, labeled_idx)
    
    return np.hstack((labeled_idx, np.random.choice(unlabeled_idx, amount, replace=False)))

def query_LC(train_dataframe,labeled_idx,amount,ResNet50_model,Strategy,N_GPU,Batch_GPU):
    """
    Returns samples based on Least Confidence
    """
 
    unlabeled_idx = get_unlabeled_idx(train_dataframe, labeled_idx)
    
    unlabeled_data = get_unlabeled_data(train_dataframe, labeled_idx,Strategy,N_GPU,Batch_GPU)
    
    predictions = ResNet50_model.predict(unlabeled_data)
    unlabeled_predictions =np.amax(predictions, axis=1)
    selected_indices = np.argpartition(unlabeled_predictions, amount )[:amount]
    indices_1 = np.hstack((labeled_idx, unlabeled_idx[selected_indices]))
    return indices_1

def query_uncertainityentropy(train_dataframe,labeled_idx,amount,ResNet50_model,Strategy,N_GPU,Batch_GPU):
    """
    Returns samples with maximum predictive entropy
    """
    unlabeled_idx = get_unlabeled_idx(train_dataframe, labeled_idx)
    
    unlabeled_data = get_unlabeled_data(train_dataframe, labeled_idx,Strategy,N_GPU,Batch_GPU)
    
    predictions = ResNet50_model.predict(unlabeled_data)
    unlabeled_predictions = np.sum(predictions * np.log(predictions + 1e-10), axis=1)

    selected_indices = np.argpartition(unlabeled_predictions, amount)[:amount]
    indices_1 = np.hstack((labeled_idx, unlabeled_idx[selected_indices]))
    
    return indices_1
#=========================================================================================================================================#
