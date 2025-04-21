"""
Advanced Machine Learning, 2025, HW4

Author: Andrew H. Fagg (andrewhfagg@gmail.com)
Editor: Enzo B. Durel (enzo.durel@gmail.com)

Semantic labeling of the Chesapeake Bay
"""

#################################################################
#                           Imports                             #
#################################################################

from pfam_loader import *
from positional_encoder import *
import tensorflow as tf

# Gpus initialization
gpus = tf.config.experimental.list_physical_devices('GPU')
n_visible_devices = len(gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Enabled memory growth for {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}")

# Set threading parallelism
import os
cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
if cpus_per_task > 1:
    tf.config.threading.set_intra_op_parallelism_threads(cpus_per_task // 2)
    tf.config.threading.set_inter_op_parallelism_threads(cpus_per_task // 2)

# Tensorflow
import tensorflow_probability as tfp

# Sub namespaces that are useful later
# Tensorflow Distributions
tfd = tfp.distributions
# Probability Layers 
tfpl = tfp.layers

# Keras
import tf_keras as keras
from tf_keras.utils import plot_model
from tf_keras.callbacks import EarlyStopping, TerminateOnNaN

# WandB
import wandb

# Other imports
import pickle
import socket
import matplotlib.pyplot as plt

# Local imports
from job_control import JobIterator
from parser import *
from tools import *
from model import *

#################################################################
#                 Default plotting parameters                   #
#################################################################

FIGURESIZE=(10,6)
FONTSIZE=18

plt.rcParams['figure.figsize'] = FIGURESIZE
plt.rcParams['font.size'] = FONTSIZE

plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE
    
#################################################################
#                         Experiment                            #
#################################################################

def execute_exp(args, multi_gpus:int=1):
    '''
    Perform the training and evaluation for a single model
    
    :param args: Argparse arguments
    :param multi_gpus: True if there are more than one GPU
    '''

    #################################
    #        Argument Parser        #
    #################################
    
    if args is None:
        # Case where no args are given (usually, because we are calling from within Jupyter)
        #  In this situation, we just use the default arguments
        parser = create_parser()
        args = parser.parse_args([])
        
    print(args.exp_index)
    
    # Override arguments if we are using exp_index
    args_str = augment_args(args)

    #################################
    #         Load Datasets         #
    #################################

    if args.verbose >= 3:
        print('Starting data flow')

    dat = load_rotation(basedir=args.dataset, rotation=args.rotation, version=args.version)

    dataset_train, dataset_valid, dataset_test = create_tf_datasets(dat,
                                                                    batch=args.batch,
                                                                    prefetch=args.prefetch,
                                                                    shuffle=args.shuffle,
                                                                    repeat=(args.steps_per_epoch is not None))

    #################################
    #       Model Configuration     #
    #################################

    # Scale the batch size with the number of GPUs
    if multi_gpus > 1:
        args.batch = args.batch * multi_gpus

    print('Batch size', args.batch)

    if args.verbose >= 3:
        print('Building network')

    # Build the model
    conv_layers = []
    for s, f, p in zip(args.conv_size, args.conv_nfilters, args.pool):
        conv_layer = dict()
        conv_layer['filters'] = f
        conv_layer['kernel_size'] = (s,s)
        conv_layer['pool_size'] = (p,p) if p > 1 else None
        conv_layer['strides'] = (p,p) if p > 1 else None
        conv_layer['batch_normalization'] = args.batch_normalization
        conv_layers.append(conv_layer)

    dense_layers = []
    for i in args.hidden:
        dense_layer = dict()
        dense_layer['units'] = i
        dense_layer['batch_normalization'] = args.batch_normalization
        dense_layers.append(dense_layer)

    rnn_layers = []
    for i, p in zip(args.rnn_units, args.rnn_pool):
        rnn_layer = dict()
        rnn_layer['units'] = i
        rnn_layer['pool_size'] = (p,p) if p > 1 else None
        rnn_layers.append(rnn_layer)

    attention_layers = []
    for h, kd, p in zip(args.attention_heads, args.key_dim, args.attention_pool):
        attention_layer = dict()
        attention_layer['attention_heads'] = h
        attention_layer['key_dim'] = kd
        attention_layer['pool_size'] = (p,p) if p > 1 else None
        attention_layers.append(attention_layer)
        
    print("Dense layers:", dense_layers)
    print("Conv layers:", conv_layers)
    print("RNN layers:", rnn_layers)
    print("Attention layers:", attention_layers)

    # Single GPU
    # Build network: you must provide your own implementation
    model = build_model(model_type=args.model_type,
                        input_length=dat['ins_train'].shape[1],
                        n_tokens=dat['n_tokens'],
                        n_classes=dat['n_classes'],
                        n_embeddings=args.n_embeddings,
                        conv_layers=conv_layers,
                        conv_activation=args.activation_conv,
                        reg=args.reg,
                        rnn_layers=rnn_layers,
                        rnn_activation=args.activation_rnn,
                        dense_layers=dense_layers,
                        dense_activation=args.activation_dense,
                        attention_layers=attention_layers)
            
    # Report model structure if verbosity is turned on
    if args.verbose >= 1:
        print(model.summary())

    # Output file base and pkl file
    fbase = generate_fname(args, args_str)
    fname_out = "%s_results.pkl"%fbase

    # Plot the model
    render_fname = '%s_outer_model_plot.png'%fbase
    if args.render:
        plot_model(model, to_file=render_fname, show_shapes=True, show_layer_names=True)

    # Perform the experiment?
    if args.nogo:
        print("NO GO")
        return

    # Check if output file already exists
    if not args.force and os.path.exists(fname_out):
        # Results file does exist: exit
        print("File %s already exists"%fname_out)
        return

    #################################
    #             WandB             #
    #################################
    
    run = wandb.init(project=args.project, name='%s_R%d'%(args.label,args.rotation), notes=fbase, config=vars(args))

    # Log hostname
    wandb.log({'hostname': socket.gethostname()})

    # Log model design image
    if args.render:
        wandb.log({'Model architecture': wandb.Image(render_fname)})
            
    #################################
    #            Callbacks          #
    #################################
    
    cbs = []
    early_stopping_cb = EarlyStopping(patience=args.patience,
                                      restore_best_weights=True,
                                      min_delta=args.min_delta,
                                      monitor=args.monitor)

    cbs.append(early_stopping_cb)
    cbs.append(TerminateOnNaN())

    # Weights and Biases
    wandb_metrics_cb = wandb.keras.WandbCallback()
    cbs.append(wandb_metrics_cb)

    if args.verbose >= 3:
        print('Fitting model')

    #################################
    #              Learn            #
    #################################
        
    #  steps_per_epoch: how many batches from the training set do we use for training in one epoch?
    #          Note that if you use this, then you must repeat the training set
    #  validation_steps=None means that ALL validation samples will be used

    history = model.fit(dataset_train,
                        validation_data=dataset_valid,
                        epochs=args.epochs,
                        batch_size=args.batch,
                        steps_per_epoch=args.steps_per_epoch,
                        verbose=args.verbose>=2,
                        validation_steps=None,
                        callbacks=cbs)

    #################################
    #            Results            #
    #################################

    # Generate results data
    results = {}

    # Validation set
    print('#################')
    print('Validation')
    results['args'] = args
    # results_predict_validation = model.predict(dataset_valid))
    results_predict_validation_eval = model.evaluate(dataset_valid)
    # results['predict_validation'] = model.predict(dataset_valid)
    # results['predict_validation_eval'] = model.evaluate(dataset_valid)
    wandb.log({'final_val_loss': results_predict_validation_eval[0]})
    wandb.log({'final_val_sparse_categorical_accuracy': results_predict_validation_eval[1]})

    # Test set
    if dataset_test is not None:
        print('#################')
        print('Testing')
        results['predict_testing'] = model.predict(dataset_test)
        results['predict_testing_eval'] = model.evaluate(dataset_test)
        wandb.log({'final_test_loss': results['predict_testing_eval'][0]})
        wandb.log({'final_test_sparse_categorical_accuracy': results['predict_testing_eval'][1]})

    # Training set
    print('#################')
    print('Training')
    results_predict_training_eval = model.evaluate(dataset_train)
    # results['predict_training'] = model.predict(dataset_train)
    # results['predict_training_eval'] = model.evaluate(dataset_train)

    wandb.log({'final_train_loss': results_predict_training_eval[0]})
    wandb.log({'final_train_sparse_categorical_accuracy': results_predict_training_eval[1]})

    # History
    # results['history'] = history.history

    ## NOTE: may want to add some additional logging of test data performance

    # Save results
    fbase = generate_fname(args, args_str)
    results['fname_base'] = fbase
    with open("%s_results.pkl"%(fbase), "wb") as fp:
        pickle.dump(results, fp)
    
    # Save model
    if args.save_model:
        model.save("%s_model.keras"%(fbase))

    wandb.finish()

    return model


def check_completeness(args):
    '''
    Check the completeness of a Cartesian product run.

    All other args should be the same as if you executed your batch, however, the '--check' flag has been set

    Prints a report of the missing runs, including both the exp_index and the name of the missing results file

    :param args: ArgumentParser

    '''
    
    # Get the corresponding hyperparameters
    p = exp_type_to_hyperparameters(args)

    # Create the iterator
    ji = JobIterator(p)

    print("Total jobs: %d"%ji.get_njobs())

    print("MISSING RUNS:")

    indices = []
    # Iterate over all possible jobs
    for i in range(ji.get_njobs()):
        params_str = ji.set_attributes_by_index(i, args)
        # Compute output file name base
        fbase = generate_fname(args, params_str)
    
        # Output pickle file name
        fname_out = "%s_results.pkl"%(fbase)

        if not os.path.exists(fname_out):
            # Results file does not exist: report it
            print("%3d\t%s"%(i, fname_out))
            indices.append(i)

    # Give the list of indices that can be inserted into the --array line of the batch file
    print("Missing indices (%d): %s"%(len(indices),','.join(str(x) for x in indices)))

    
#################################################################
#                            Main                               #
#################################################################


if __name__ == "__main__":
    # Parse and check incoming arguments
    parser = create_parser()
    args = parser.parse_args()
    check_args(args)

    if args.verbose >= 3:
        print('Arguments parsed')

    # Turn off GPU?
    if not args.gpu or "CUDA_VISIBLE_DEVICES" not in os.environ.keys():
        tf.config.set_visible_devices([], 'GPU')
        print('NO VISIBLE DEVICES!!!!')

    if args.check:
        # Just check to see if all experiments have been executed
        check_completeness(args)
    else:
        # Execute the experiment

        # Do the work
        execute_exp(args, multi_gpus=n_visible_devices)
