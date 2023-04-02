# Collection of metrics utilities
#   cm_cr - display confusion matrix dataframes and classification report
#   plot_cm - plot unnormalized or normalized confusion matrix heatmap
#   plot_cm_unnorm_and_norm - one model's confusion matrix heatmaps without and with normalization
#   plot_conf_matrices - plot heatmaps for normalized or unnormalized confusion matrices for all models

# Import dependencies
import pandas as pd
import numpy as np
# Matplotlib for visualization
from matplotlib import pyplot as plt
# Seaborn for easier visualization
import seaborn as sns
sns.set_style('darkgrid')
# To display dataframes side by side
from IPython.display import display_html 
# Metrics
from sklearn.metrics import confusion_matrix, classification_report

# in matplotlib >3.5.2 dpi default value was changed from 72 to 100
# deafult figsize was also changed: 6.0, 4.0 --> 6.4, 4.8
# we will use dpi=72; None - for notebook default
dpi = 72

# ===============

# Function to display confusion matrix dataframes and classification report
def cm_cr(model_name, y_test, y_pred, target_names, cr=True):
    """ Display confusion matrix dataframes with and without normalization
        and print classification report if cr = True

    Args:
        model_name: name of the model
        y_test: test target variable
        y_pred: prediction
        target_names: list of class names
        cr: print classification report if True - default

    Returns:
        Display confusion matrix dataframes side by side
        and classification report if selected (default)
    """
    
    # Print header
    print(' '*22, model_name)
    print(' '*22, '='*len(model_name))
    
    # Create dataframe for confusion matrix for y_test and y_pred
    cm = confusion_matrix(y_test, y_pred)
    conf_df = pd.DataFrame(cm, columns=target_names, index=target_names)
    conf_df.index.name = 'TRUE'
    conf_df = conf_df.rename_axis('PREDICTED', axis='columns')
    
    # Dataframe for normalizwzed confusion matrix
    cm = np.around(cm / cm.sum(axis=1)[:, np.newaxis], 2)
    conf_dfn = pd.DataFrame(cm, columns=target_names, index=target_names)
    conf_dfn.index.name = 'TRUE'
    conf_dfn = conf_dfn.rename_axis('PREDICTED', axis='columns')
  
    # Display dataframes side by side
    conf_df_styler = conf_df.style.set_table_attributes("style='display:inline'").set_caption('Confusion Matrix')
    conf_dfn_styler = conf_dfn.style.set_table_attributes("style='display:inline'").set_caption('Normalized Confusion Matrix').format(precision=2)
    
    space1 = "\xa0" * 2
    space2 = "\xa0" * 15
    display_html(space1 + conf_df_styler._repr_html_() + space2 + conf_dfn_styler._repr_html_(), raw=True)
    
    if cr:
        # Display classification report
        print()
        print(classification_report(y_test, y_pred, target_names=target_names))
    print()
    
    
# ==============

# Function to plot unnormalized or normalized confusion matrix heatmap
def plot_cm(model_name, y_test, y_pred, target_names, color, norm=True):
    """ Plot confusion matrix heatmap without or with normalization

    Args:
        model_name: name of the model
        y_test: test target variable
        y_pred: prediction
        target_names: list of class names
        color: color palette
        norm: plot normalized matrix if True - default 
              plot unnormalized matrix if False

    Returns:
        Plot one model's confusion matrix,
         normalized (default) or unnormalized
    """

    f, ax1 = plt.subplots(figsize=(6, 4), dpi=dpi)
    f.suptitle(model_name, fontsize=14)
    if norm:
        ax1.set_title('Normalized Confusion Matrix')
        fmt = '.2f'
        vmin = 0
        vmax = 1
    else:
        ax1.set_title('Unnormalized Confusion Matrix')
        fmt = 'd'
        vmin = None
        vmax = None
    f.subplots_adjust(top=0.85, wspace=0.3)

    # Unnormalized onfusion matrix
    mat = confusion_matrix(y_test, y_pred)
    if norm:
        # Normalized confusion matrix
        mat = mat / mat.sum(axis=1)[:, np.newaxis]
    sns.heatmap(mat,
                annot=True,
                annot_kws=dict(fontsize=14),
                fmt=fmt,
                cbar=True,
                square=True,
                cmap=color,
                linecolor='red',
                linewidth=0.01,
                vmin = vmin,
                vmax = vmax,
                ax=ax1)

    ax1.set_xticklabels(labels=target_names)
    ax1.set_yticklabels(labels=target_names, va='center')
    ax1.set_xlabel('PREDICTED', size=12)
    ax1.set_ylabel('TRUE', size=12)

    plt.show()
    
    
# ======================


# Function to plot one model's confusion matrix heatmaps without and with normalization
def plot_cm_unnorm_and_norm(model_name, y_test, y_pred, target_names, color):
    """ Plot confusion matrix heatmaps without and with normalization

    Args:
        model_name: name of the model
        y_test: test target variable
        y_pred: prediction
        target_names: list of class names
        color: color palette

    Returns:
        Plot one model's confusion matrix heatmaps side by side
        left unnormalized and right normalized
    """

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=dpi)
    f.suptitle(model_name, fontsize=14)
    f.subplots_adjust(top=0.85, wspace=0.3)

    # confusion matrix without normalization
    mat = confusion_matrix(y_test, y_pred)
    sns.heatmap(mat,
                annot=True,
                annot_kws=dict(fontsize=14),
                fmt='d',
                cbar=True,
                square=True,
                cmap=color,
                linecolor='red',
                linewidth=0.01,
                ax=ax1)

    ax1.set_xticklabels(labels=target_names)
    ax1.set_yticklabels(labels=target_names, va='center')
    ax1.set_title('Unnormalized Confusion Matrix')
    ax1.set_xlabel('PREDICTED', size=12)
    ax1.set_ylabel('TRUE', size=12)

    # normalized confusion matrix
    matn = mat / mat.sum(axis=1)[:, np.newaxis]
    sns.heatmap(matn,
                annot=True,
                annot_kws=dict(fontsize=14),
                fmt='.2f',
                cbar=True,
                square=True,
                cmap=color,
                linecolor='red',
                linewidth=0.01,
                vmin = 0,
                vmax = 1,
                ax=ax2)

    ax2.set_xticklabels(labels=target_names)
    ax2.set_yticklabels(labels=target_names, va='center')
    ax2.set_title('Normalized Confusion Matrix')
    ax2.set_xlabel('PREDICTED', size=12)
    ax2.set_ylabel('TRUE', size=12)

    plt.show()
    
    
    # ===========
    
    # Function for ploting heatmaps for normalized or unnormalized confusion matrices for all models
def plot_conf_matrices(models_pred, y_test, target_names, color, norm=True):
    """ Plot confusion matrices heatmaps for all models, 
        normalized or unnormalized

    Args:
        models_pred: dictionary with model names as keys and predictions as values 
        y_test: test target variable
        target_names: list of class names
        color: color palette
        norm: plot normalized matrices if True - default 
              plot unnormalized matrices if False

    Returns:
        Plot all model's confusion matrix heatmaps, in 2 columns,
        normalized (default) or unnormalized
    """    
        
    # Prepare lists of coordinates for axes
    lt = []  # list for full subplots
    ltd = []  # list for empty subplots
    col = 2  # number of columns
    n_mod = len(models_pred)  # number of fitted models
    
    # Number of rows
    row = - (n_mod // -col)
    
    # Create lists of coordinates for full and empty subplots
    for r in range(row):
        for c in range(col):
            if n_mod >= (r + 1) * (c + 1):
                lt.append([r, c])
            else:
                ltd.append([r, c])
    
    # Create figure and subplots
    figs_y = row * 4  # y size
    f, axs = plt.subplots(row, col, figsize=(10, figs_y), dpi=dpi)
    
    if norm:
        f.suptitle('Normalized Confusion Matrices', fontsize=14)
        fmt = '.2f'
        vmin = 0
        vmax = 1
    else:
        f.suptitle('Unnormalized Confusion Matrices', fontsize=14)
        fmt = 'd'
        vmin = None
        vmax = None
        
    f.subplots_adjust(top=0.94, wspace=0.90, hspace=0.2)
    
    # Reshape axes; needed in case of only 1 row
    axs = axs.reshape(row,-col)

    # Loop to delete N last empty subplots (if any)
    for n in range(len(ltd)):
        r = ltd[n][0]
        c = ltd[n][1]
        f.delaxes(ax= axs[r, c])
        
    # Loop to plot all full subplots
    i = 0
    # Loop for each fitted model        
    for model, pred in models_pred.items():
        y_pred = pred
        name = model
        r = lt[i][0]
        c = lt[i][1]
        i += 1
     
        mat = confusion_matrix(y_test, y_pred)    
        # normalized confusion matrix
        if norm:
            mat = mat / mat.sum(axis=1)[:, np.newaxis]

        ax = axs[r, c]
        sns.heatmap(mat,
                    annot=True,
                    annot_kws=dict(fontsize=14),
                    fmt=fmt,
                    cbar=False,
                    square=True,
                    cmap=color,
                    linecolor='red',
                    linewidth=0.01,
                    vmin = vmin,
                    vmax = vmax,
                    #cbar_kws = {'shrink' : 0.85},
                    ax=ax)
    
        ax.set_xticklabels(labels=target_names)
        ax.set_yticklabels(labels=target_names, va='center')
        ax.set_title(name)
        ax.set_xlabel('PREDICTED', size=12)
        ax.set_ylabel('TRUE', size=12)

    plt.show()