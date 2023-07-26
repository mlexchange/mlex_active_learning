import numpy as np
import os

from keras.callbacks import ModelCheckpoint, History
from keras.models import  Model
import tensorflow as tf
import pandas as pd

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
