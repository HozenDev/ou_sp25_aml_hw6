'''
PFAM Data loader and data set constructor

Author: Andrew H. Fagg


Two different ways to load full data sets:

prepare_data_set(basedir = '/home/fagg/datasets/pfam', rotation = 0, nfolds = 5, ntrain_folds = 3)
    loads the raw CSV files, does the splitting and tokenization

OR

load_rotation(basedir = '/home/fagg/datasets/pfam', rotation=0)
    loads an already stored data set from a pickle file




'''
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import fnmatch
import matplotlib.pyplot as plt
import random
import pickle

def load_pfam_file(basedir:str,
                   fold:int,
                   version:str='')->pd.DataFrame:
    '''
    Load a CSV file into a DataFrame
    :param basedir: Directory containing input files
    :param fold: Fold to load
    :param version: String with pfam version

    :return: Dataframe containing a single PFAM database
    '''
    
    df = pd.read_csv('%s/pfam%s_fold_%d.csv'%(basedir, version, fold))
    return df

def load_pfam_dataset(basedir:str = '/home/fagg/datasets/pfam',
                      rotation:int = 0,
                      nfolds:int = 5,
                      ntrain_folds:int = 3,
                      version:str='')->dict:
    '''
    Load train/valid/test datasets into DataFrames

    :param basedir: Directory containing input files
    :param rotation: Rotation to load
    :param nfolds: Total number of folds
    :param ntrain_folds: Number of training folds to use
    :param version: String with pfam version

    :return: Dictionary containing the DataFrames
    '''

    train_folds = (np.arange(ntrain_folds) + rotation)  % nfolds
    valid_folds = (np.array([ntrain_folds]) + rotation) % nfolds
    test_folds = (np.array([ntrain_folds]) + 1 + rotation) % nfolds

    train_dfs = [load_pfam_file(basedir, f, version=version) for f in train_folds]
    valid_dfs = [load_pfam_file(basedir, f, version=version) for f in valid_folds]
    test_dfs = [load_pfam_file(basedir, f, version=version) for f in test_folds]

    train_df = pd.concat(train_dfs, ignore_index=True)
    valid_df = pd.concat(valid_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)

    return {'train': train_df, 'valid': valid_df, 'test': test_df}


def prepare_data_set(basedir:str = '/home/fagg/datasets/pfam',
                     rotation:int = 0,
                     nfolds:int = 5,
                     ntrain_folds:int = 3,
                     version:int='')->dict:
    '''
    Generate a full data set

    :param basedir: Directory containing input files
    :param rotation: Rotation to load
    :param nfolds: Total number of folds
    :param ntrain_folds: Number of training folds to use
    :param version: String with pfam version

    :return: Dictionary containing a full train/validation/test data set

    Dictionary format:
    ins_train: tokenized training inputs (examples x len_max).  Values are 0 ... n_tokens-1
    outs_train: tokenized training outputs (examples x 1).  Values are 0 ... n_classes-1
    ins_valid: tokenized validation inputs (examples x len_max)
    outs_valid: tokenized validation outputs (examples x 1)
    ins_test: tokenized test inputs (examples x len_max)
    outs_test: tokenized test outputs (examples x 1)
    len_max: maximum length of a string
    n_tokens: Maximum number of output tokens 
    out_index_word: dictionary containing index -> class name map (note index is 1... n_classes)
    out_word_index: dictionary containing class name -> index map (note index is 1... n_classes)
    '''

    
    # Load the data from the disk
    dat = load_pfam_dataset(basedir=basedir, rotation=rotation, nfolds=nfolds, ntrain_folds=ntrain_folds,
                            version=version)

    # Extract ins/outs
    dat_out = {}

    # Extract ins/outs for each dataset
    for k, df in dat.items():
        # Get the set of strings
        
        dat_out['ins_'+k] = df['string'].values
        dat_out['outs_'+k] = df['label'].values

    # Compute max length: only defined with respect to the training set
    len_max = np.max(np.array([len(s) for s in dat_out['ins_train']]))

    print('tokenize fit...')
    # Convert strings to lists of indices
    tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True,
                                                   filters='\t\n')
    tokenizer.fit_on_texts(dat_out['ins_train'])

    print('tokenize...')
    # Loop over all data sets
    for k in dat.keys():
        # Loop over all strings and tokenize
        seq = tokenizer.texts_to_sequences(dat_out['ins_'+k])
        dat_out['ins_'+k] = tf.keras.preprocessing.sequence(seq, maxlen=len_max)

    n_tokens = np.max(dat_out['ins_train']) + 2

    print('outputs...')
    # Loop over all data sets: create tokenizer for output
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='\t\n')
    tokenizer.fit_on_texts(dat_out['outs_train'])

    # Tokenize all of the outputs
    for k in dat.keys():
        dat_out['outs_'+k] = np.array(tokenizer.texts_to_sequences(dat_out['outs_'+k]))-1
        #np.expand_dims(dat_out['outs_'+k],  axis=-1)seq =
        
    #
    dat_out['len_max'] = len_max
    dat_out['n_tokens'] = n_tokens
    dat_out['out_index_word'] = tokenizer.index_word
    dat_out['out_word_index'] = tokenizer.word_index
    dat_out['rotation'] = rotation
    dat_out['n_classes'] = len(tokenizer.index_word.keys())
    
    return dat_out

    
def save_data_sets(basedir:str = '/home/fagg/datasets/pfam',
                   out_basedir:str = None,
                   nfolds:int = 5,
                   ntrain_folds:int = 3,
                   version:str=''):
    '''
    Generate pickle files for all rotations.

    :param basedir: Directory containing input files
    :param out_basedir: Directory for output files (None -> use the basedir)
    :param nfolds: Total number of folds
    :param ntrain_folds: Number of training folds to use
    :param rotation: Rotation to load
    :param version: String with pfam version
    '''

    if out_basedir is None:
        out_basedir = basedir
        
    # Loop over all rotations
    for r in range(nfolds):
        # Load the rotation
        dat=prepare_data_set(basedir=basedir, rotation=r, nfolds=nfolds, ntrain_folds=ntrain_folds, version=version)

        # Write rotation to pickle file
        fname = '%s/pfam%s_rotation_%d.pkl'%(basedir, version, r)

        with open(fname, 'wb') as fp:
            pickle.dump(dat, fp)
            
def load_rotation(basedir:str = '/home/fagg/datasets/pfam',
                  rotation:int=0,
                  version:str='')->dict:
    '''
    Load a single rotation from a pickle file.  These rotations are 5 folds, 3 training folds

    :param basedir: Directory containing files
    :param rotation: Rotation to load
    :param version: String with pfam version

    :return: Dictionary containing a full train/validation/test data set
    '''
    fname = '%s/pfam%s_rotation_%d.pkl'%(basedir, version, rotation)
    with open(fname, 'rb') as fp:
        dat_out = pickle.load(fp)
        return dat_out
    return None

def create_tf_datasets(dat:dict,
                       batch:int=8,
                       prefetch:bool=None,
                       shuffle:int=None,
                       repeat:bool=False,
                       cache:bool=False)->[tf.data.Dataset]:
    '''
    Translate the data structure from load_rotation() or prepare_data_set() into a proper TF DataSet object
    for each of training, validation and testing.  These act as configurable generators that can be used by
    model.fit(), .predict() and .evaluate()

    :param dat: Dictionary from load_rotation() or prepare_data_set()
    :param batch: Batch size (int)
    :param prefetch: Number of batches to prefetch.  (None = no prefetch)
    :param shuffle: Shufle the training dataset
    :param repeat: Repeat the training data set
    :param cache: Cache the datasets to RAM
    :return: 3 TF.Datasets: training, validation, testing
    
    '''
    print("CREATE_TF_DS")
    
    # Translate tensors into datasets
    dataset_train = tf.data.Dataset.from_tensor_slices((dat['ins_train'], dat['outs_train']))
    dataset_valid = tf.data.Dataset.from_tensor_slices((dat['ins_valid'], dat['outs_valid']))
    dataset_test = tf.data.Dataset.from_tensor_slices((dat['ins_test'], dat['outs_test']))

    # Cache to RAM
    if cache:
        dataset_train = dataset_train.cache()
        dataset_valid = dataset_valid.cache()
        dataset_test = dataset_test.cache()

    # Repeat training set?  Must be coupled with steps_per_epoch in model.fit()
    if repeat:
        print("Repeating training set")
        dataset_train = dataset_train.repeat()
        
    # Shuffle training set?
    if shuffle is not None:
        dataset_train = dataset_train.shuffle(shuffle)

    # Batch the data sets
    dataset_train = dataset_train.batch(batch)
    dataset_valid = dataset_valid.batch(batch)
    dataset_test = dataset_test.batch(batch)

    # Prefetch if specified
    if prefetch is not None:
        if prefetch == -1:
            dataset_train = dataset_train.prefetch(tf.data.AUTOTUNE)
            dataset_valid = dataset_valid.prefetch(tf.data.AUTOTUNE)
            dataset_test = dataset_test.prefetch(tf.data.AUTOTUNE)
        else:
            dataset_train = dataset_train.prefetch(prefetch)
            dataset_valid = dataset_valid.prefetch(prefetch)
            dataset_test = dataset_test.prefetch(prefetch)

    return dataset_train, dataset_valid, dataset_test
