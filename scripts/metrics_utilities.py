# Collection of metrics utilities:
# 1.  cm_cr - display confusion matrix dataframes and classification report
# 2.  plot_cm - plot unnormalized or normalized confusion matrix heatmap
# 3.  plot_cm_unnorm_and_norm - one model's confusion matrix heatmaps without and with normalization
# 4.  plot_conf_matrices - plot heatmaps for normalized or unnormalized confusion matrices for all models
# 5.  plot_cm_sankey - interactive confusion matrix using Sankey diagram

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

from plotly import graph_objects as go

# 1. ===============

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
    conf_df.index.name = 'ACTUAL'
    conf_df = conf_df.rename_axis('PREDICTED', axis='columns')
    
    # Dataframe for normalizwzed confusion matrix
    cm = np.around(cm / cm.sum(axis=1)[:, np.newaxis], 2)
    conf_dfn = pd.DataFrame(cm, columns=target_names, index=target_names)
    conf_dfn.index.name = 'ACTUAL'
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
    
    
# 2. ==============

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
    ax1.set_ylabel('ACTUAL', size=12)

    plt.show()
    
    
# 3. ======================


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
    ax1.set_ylabel('ACTUAL', size=12)

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
    ax2.set_ylabel('ACTUAL', size=12)

    plt.show()
    
    
# 4. ===============
    
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
        ax.set_ylabel('ACTUAL', size=12)

    plt.show()
    
    
# 5. ==================
    
# Function to display interactive confusion matrix using Sankey diagram
def plot_cm_sankey(model_name, y_test, y_pred, target_names=None):
    """ Plot confusion matrix with Sankey diagram 

    Args:
        model_name: name of the model
        y_test: test target variable
        y_pred: prediction
        target_names: list of class names

    Returns:
        Plot Sankey diagram of confusion matrix
    """ 
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # If class labels not passed, create dummy class labels
    if target_names == None: 
        target_names = []
    if not len(target_names):
        target_names = [f'class-{i+1}' for i in range(len(cm))]
    
    # Prepare dataframe with parameters for Sankey
    def prepare_df_for_sankey(cm, target_names):
        # create a dataframe
        df = pd.DataFrame(cm, columns=[f'PREDICTED {s}' for s in target_names], index=[f'ACTUAL {s}' for s in target_names])
        
        # Create list of node labels
        # target nodes = column labels (PREDICTED ...)
        cl = df.columns.values.tolist()
        # source nodes = row (index) labels (ACTUAL ...)
        rl = df.index.values.tolist()
        node_labels = rl + cl
        
        # Create dictionary with indices for node labels
        node_labels_inds = {label:ind for ind, label in enumerate(node_labels)}
        
        # Stack label from column to row, output is Series
        # Reset index to get DataFrame and rename columns
        df = df.stack().reset_index()
        df.rename(columns={0:'samples', 'level_0':'actual', 'level_1':'predicted'}, inplace=True)
        
        """
               actual	       predicted	  samples
        0	ACTUAL Stays	PREDICTED Stays	    1979
        1	ACTUAL Stays	PREDICTED Exits	     410
        2	ACTUAL Exits	PREDICTED Stays	     198
        3	ACTUAL Exits	PREDICTED Exits	     413
        """

        # Normalized confusion matrix
        cmn = np.around(cm / cm.sum(axis=1)[:, np.newaxis], 2)
        # Add a column with normalized values of samples
        df['norm_samples'] = cmn.ravel()
        
        # Helper function to add new columns: color and link_hover_text 
        # 'color' - link color based on classification result (correct or incorrect)        
        incorrect_red = "rgba(205, 92, 92, 0.8)"
        correct_green = "rgba(144, 238, 144, 0.8)"
        # # 'link_hover_text' - text for hovering on connecting links of sankey diagram
        
        def new_columns(row):
            source_1 = ''.join(row.actual.split()[1:])
            target_1 = ''.join(row.predicted.split()[1:])
            # Correct classification
            if source_1 == target_1:
                row['color'] = correct_green
                row['link_hover_text'] = f"{row.samples} ({row.norm_samples:.0%}) {source_1} samples correctly classified as {target_1}"
            # Incorrect classification
            else:
                row['color'] = incorrect_red
                row['link_hover_text'] = f"{row.samples} ({row.norm_samples:.0%}) {source_1} samples incorrectly classified as {target_1}"
            return row

        # Apply "new_columns" function
        df = df.apply(lambda x: new_columns(x), axis=1)
        
        # Sankey only takes integers for node and target values,
        #  so we need to map node label columns (actual, predicted) to numbers
        # Using replace for multiple columns
        df = df.replace({'actual':node_labels_inds, 'predicted':node_labels_inds})
               
        return df, node_labels
    
    
    # Plotting confusion matrix as Sankey diagram
    # Get dataframe and node labels
    df, node_labels = prepare_df_for_sankey(cm, target_names)
    
    # Prepare for bold printing of some words in Plotly
    node_labels = [f'{ls[0]} <b>{ls[1]}</b>' for ls in [l.split() for l in node_labels]]
    df['link_hover_text'] = [f'{" ".join(ls[0:2])} <b>{ls[2]}</b> {" ".join(ls[3:-1])} <b>{ls[-1]}</b>' for ls in [l.split() for l in df['link_hover_text']]]
    

    fig = go.Figure(data=[go.Sankey(    
        node = dict(
        pad = 50,
        thickness = 30,
        line = dict(color = "gray", width = 1.0),
        label = node_labels,
        hovertemplate = "%{label} has total %{value:d} samples<extra></extra>"
        ),
    link = dict(
        source = df.actual, 
        target = df.predicted,
        value = df.samples,
        color = df.color,
        customdata = df['link_hover_text'], 
        hovertemplate = "%{customdata}<extra></extra>"  
    ))])
    
    margins = {'l': 25, 'r': 25, 't': 70, 'b': 25}
    
    fig.update_layout(
        title = {
        'text': f'<b>{model_name}</b>',
        'x':0.5,
        },
        font_size = 15,
        width = 625,
        height = 500,
        # paper_bgcolor = '#d3d3d3',
        margin = margins,
    )
    
    return fig

