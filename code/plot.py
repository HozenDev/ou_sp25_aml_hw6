import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

# Tensorflow
from tensorflow import keras
from pfam_loader import *
from parser import check_args, create_parser

plt.style.use('seaborn-v0_8-muted')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 14

#######################################
#            Load Functions           #
#######################################

def load_trained_model(model_dir, substring_name):
    """
    Load a trained models
    """
    model_files = [f for f in os.listdir(model_dir) if substring_name in f and f.endswith(".keras")]

    if not model_files:
        raise ValueError(f"No model found in {model_dir} matching {substring_name}")

    model_path = os.path.join(model_dir, model_files[0])
    model = keras.models.load_model(model_path)

    return model

def load_results_iter(results_dir):
    """
    Generator to load model results from a directory.
    Reduce memory usage compared to load results method.
    """
    files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith(".pkl")]
    
    for filename in files:
        with open(filename, "rb") as fp:
            data = pickle.load(fp)
            yield data
            
def load_results(results_dir):
    """
    Load model results from a directory
    """
    results = []
    files = []
    for r_dir in results_dir:
        files.extend([os.path.join(r_dir, f) for f in os.listdir(r_dir) if f.endswith(".pkl")])

    for filename in files:
        with open(filename, "rb") as fp:
            data = pickle.load(fp)
            results.append(data)

    return results

#######################################
#          Plotting Functions         #
#######################################

# Figure 2
def plot_accuracy_curves(gru_results, mha_results, filename="figures/accuracy_vs_accuracy.png"):
    """
    Plot validation accuracy curves: GRU vs MHA model per rotation.
    """
    nb_rotation = len(gru_results)

    plt.figure()

    for i in range(nb_rotation):
        gru_acc = gru_results[i]['history']['val_sparse_categorical_accuracy']
        mha_acc = mha_results[i]['history']['val_sparse_categorical_accuracy']

        # Pad the shorter sequence with its final value
        max_len = max(len(gru_acc), len(mha_acc))
        if len(gru_acc) < max_len:
            gru_acc += [gru_acc[-1]] * (max_len - len(gru_acc))
        if len(mha_acc) < max_len:
            mha_acc += [mha_acc[-1]] * (max_len - len(mha_acc))

        plt.plot(gru_acc, mha_acc, label=f"Rotation {i}")

    plt.xlabel("GRU Validation Accuracy")
    plt.ylabel("Attention Validation Accuracy")
    plt.title("Validation Accuracy GRU vs Attention")
    plt.legend()
    plt.savefig(filename)

# Figure 3
def plot_accuracy_bars(rnn_results, gru_results, mha_results, filename="figures/accuracy_bar.png"):
    """
    Plot models accuracy bars
    """
    gru_test_accuracies = [r['predict_testing_eval'][1] for r in gru_results]
    mha_test_accuracies = [r['predict_testing_eval'][1] for r in mha_results]
    rnn_test_accuracies = [r['predict_testing_eval'][1] for r in rnn_results]

    nb_rotation = 5
    x = np.arange(nb_rotation) # Five rotation
    bar_width = 0.10
    width = 0.35
    
    plt.figure()
    plt.bar(x - width/2, rnn_test_accuracies, bar_width, label="RNN")
    plt.bar(x, gru_test_accuracies, bar_width, label="GRU")
    plt.bar(x + width/2, mha_test_accuracies, bar_width, label="MHA")
    plt.xticks(x, [f"R{r}" for r in range(nb_rotation)])
    plt.ylabel("Sparse Categorical Accuracy")
    plt.title("Model Accuracy by Rotation")
    plt.legend()
    plt.savefig(filename)

# Figure 4
def plot_combined_confusion_matrix(args, models, num_classes, class_names, title="Confusion Matrix", filename="figure_4.png"):
    """
    Computes and plots a combined confusion matrix from multiple model rotations.

    :params args: Command-line arguments
    :params models: List of trained models
    :params num_classes: Number of classes
    :params class_names: List of class names
    :params title: Title of the plot
    :params filename: Filename to save the plot
    """
    all_y_true = []
    all_y_pred = []

    for i, model in enumerate(models):

        dat = load_rotation(basedir=args.dataset, rotation=i, version="B")

        _, _, dataset_test = create_tf_datasets(dat,
                                                batch=args.batch,
                                                prefetch=args.prefetch)

        for x_batch, y_batch in dataset_test:
            preds = model.predict(x_batch, verbose=0)
            y_pred = np.argmax(preds, axis=-1)  
            y_true = y_batch.numpy()            

            all_y_pred.append(y_pred.flatten())
            all_y_true.append(y_true.flatten())

        print(f"Rotation {i} done")
            
    y_true_flat = np.concatenate(all_y_true)
    y_pred_flat = np.concatenate(all_y_pred)

    labels = list(range(num_classes))
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=labels)

    square_size = 0.5
    fig_width = int(square_size * num_classes)
    fig_height = int(square_size * num_classes)

    _, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(cm, cmap="Blues")

    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar(im, ax=ax)
    plt.xticks(np.arange(num_classes), class_names, rotation=90, fontsize=6)
    plt.yticks(np.arange(num_classes), class_names, fontsize=6)


    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = create_parser()
    args = parser.parse_args()
    check_args(args)

    # Dataset metadata
    nb_rotation = 5
    num_classes = 46
    class_names = [str(i) for i in range(num_classes)]
    rnn_dir = "./models/rnn_0/"
    gru_dir = "./models/gru_0/"
    mha_dir = "./models/mha_1/"

    #######################
    #     Load Results    #
    #######################
    
    gru_results = load_results([gru_dir])
    rnn_results = load_results([rnn_dir])
    mha_results = load_results([mha_dir])

    #######################
    #     Load Models     #
    #######################

    rnn_models = []
    gru_models = []
    mha_models = []

    for i in range(nb_rotation):
        try:
            rnn_model = load_trained_model(rnn_dir, f"rot_0{i}")
            rnn_models.append(rnn_model)
        except Exception as e:
            print(f"Error loading rnn model: {e}")
        
        try:
            gru_model = load_trained_model(gru_dir, f"rot_0{i}")
            gru_models.append(gru_model)
        except Exception as e:
            print(f"Error loading gru model: {e}")

        try:
            mha_model = load_trained_model(mha_dir, f"rot_0{i}")
            mha_models.append(mha_model)
        except Exception as e:
            print(f"Error loading mha model: {e}")   

    print("Plotting results...")

    # Figure 2
    plot_accuracy_curves(gru_results, mha_results, filename="figures/figure_2.png")
    
    # Figure 3
    plot_accuracy_bars(rnn_results, gru_results, mha_results, filename="figures/figure_3.png")
    
    # Figure 4
    plot_combined_confusion_matrix(args=args, models=mha_models, class_names=class_names, title="Contingency Table across Folds", filename="figures/figure_4.png", num_classes=num_classes)
    
    print("Done")
