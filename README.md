# Confusion Matrix Visualization

**NOTE**:

Because Plotly does not render on GitHub, I have to use nbviewer. Plotly should render on nbviewer and in many cases is still interactive. Use these links to get properly rendered notebooks:

- [Confusion Matrix Visualization.ipynb](https://nbviewer.org/github/zunicd/Confusion-Matrix-Visualization/blob/main/Confusion%20Matrix%20Visualization.ipynb) - using a collection of helper functions, this notebook will present several ways of visualizing a confusion matrix
- [Confusion Matrix as Sankey Diagram.ipynb](https://nbviewer.org/github/zunicd/Confusion-Matrix-Visualization/blob/main/Confusion%20Matrix%20as%20Sankey%20Diagram.ipynb) - this notebook leads you step by step through creation of an interactive Sankey confusion matrix using Plotly



## Objective

In machine learning a confusion matrix is a kind of a table that is used to understand how well our classification model predictions perform, typically a supervised learning. It helps us a lot in understanding the model behavior and interpreting the results.  

In this project we will not discuss how to interpret a confusion matrix, that could be find [here](https://en.wikipedia.org/wiki/Confusion_matrix). Instead, we will present several ways of how to visualize a confusion matrix. For each of these visualization we will create a helper function. 



## Results

The helper functions listed below were developed and collected in the script [metrics_utilities.py](scripts/metrics_utilities.py). The execution of these functions is presented in the notebook [Confusion Matrix Visualization.ipynb](Confusion%20Matrix%20Visualization.ipynb)

- **cm_cr** - display unnormalized and normalized confusion matrix dataframes side by side, and classification report 
- **plot_cm** - plot unnormalized or normalized confusion matrix heatmap
- **plot_cm_unnorm_and_norm** - one model's confusion matrix heatmaps without and with normalization, side by side
- **plot_conf_matrices** - plot heatmaps for normalized (default) or unnormalized confusion matrices for multiple models
- **plot_cm_sankey** - interactive confusion matrix using Sankey diagram



## Axes Convention

In the literature, we can find two variants for representing the samples in a confusion matrix:
1. each row of the matrix represents samples in an *actual* class, and each column represents samples in a *predicted* class
2. in the other variant, this arrangement is reversed 

In this project we will use the first variant (Sklearn representation), where *actual* labels are on the horizontal axes and *predicted* labels on the vertical axes, with the default parameter `labels=[0,1]`, meaning TN (True Negatrive) is at the top left corner.

<img src="images/cm_convention_table.png"> 



## Normalized vs. Unnormalized Matrix

Because most of real-life data is imbalanced, using a confusion matrix without normalization might lead to improper conclusions. The *unnormalized confusion matrix* is shown at the top and the *normalized confusion matrix* at the bottom. The normalized matrix will show % prediction of each class made by the model for that specific true (actual) label

```
[[1979  410]
 [ 198  413]]

[[0.83 0.17]
 [0.32 0.68]]
```



## Confusion Matrix Arrays

The simplest and most common way to display a confusion matrix is as raw numbers in an array, see below.
```
[[1979  410]
 [ 198  413]]
```
This does not look very appealing, and it is a binary class example. Just imagine a confusion matrix for a multi-class classification. It would be really tough to interpret and draw conclusions from.



## Confusion Matrix Dataframes - Side by Side

To make our confusion matrix more appealing, we can display it as a Pandas DataFrame, and display both, unnormalized and normalized, matrices side by side (function **cm_cr**).  

<figure>
<img src="images/df_cm_side_by_side.png" width="400" />
<figcaption><i>One model - unnormalized AND normalized (dataframes)</i></figcaption>
</figure>


## Confusion Matrix as Seaborn Heatmap

I think we can still do better and make it easier to interpret by introducing color palette in our visualization. To plot our confusion matrix as a color-encoded matrix, we will use the Seaborn `heatmap()` function. 

Several functions were developed to plot confusion matrices for different number of models and different normalization (normalized or not).

### a. Single Confusion Matrix

**plot_cm** - this function plots one model's confusion matrix, unnormalized or normalized, with normalized as default.

<figure>
<img src="images/cm_heatmap_single.png" height="130" /> 
<figcaption><i>One model - unnormalized</i></figcaption>
</figure>

### b. Confusion Matrices for 1 Model - Unnormalized AND Normalized

**plot_cm_unnorm_and_norm** - plots one model's confusion matrix heatmaps without and with normalization, side by side.

This plot displays nicely the usefulness of normalized confusion matrix.

<figure>
<img src="images/2_cm_heatmaps_1_model.png" height="130" /> 
<figcaption><i>One model - unnormalized AND normalized</i></figcaption>
</figure>

### c. Confusion Matrices for Multiple Models - Unnormalized OR Normalized

**plot_conf_matrices** - this function plots confusion matrices for multiple models with normalized matrices as default. You can switch to plot unnormalized matrices. 

<figure>
<img src="images/2_cm_heatmaps_2_models.png" height="130" /> 
<figcaption><i>Two models - normalized</i></figcaption>
</figure>

&nbsp;

<figure>
<img src="images/cm_heatmap_multiple.png" width="300" /> 
<figcaption><i>Seven models - normalized</i></figcaption>
</figure>



## Confusion Matrix as Sankey Diagram

And finally, we can introduce another, more elegant and interactive way to visualize a confusion matrix - [Sankey Diagram](https://en.wikipedia.org/wiki/Sankey_diagram). To create our Sankey confusion matrix we used Python and Plotly. 

The notebook [Confusion Matrix as Sankey Diagram](Confusion%20Matrix%20as%20Sankey%20Diagram.ipynb) describes step-by-step creation of the function **plot_cm_sankey**, which we use to display our confusion matrix as Sankey.  

&nbsp;

<img src="images/Sankey_cm.gif" width="450" />   

&nbsp;

The **GIF** above displays the main features of our Sankey confusion matrix.

- source nodes (ACTUAL ...) are on the left and target nodes (PREDICTED ...) on the right
- the size of nodes is proportional to the number of samples that belongs to each node
- the width of the links between nodes is proportional to the flow (result of classification), ie. the number of samples classified correctly (green) or incorrectly (red)
- hovering over the nodes and links will display numerical and textual representation of our confusion matrix



## Tools /  Techniques Used

- Python
- Jupyter Lab
- Pandas
- Numpy
- Seaborn - Heatmap
- Plotly - Sankey Diagram



## References

- [Understanding the Confusion Matrix from Scikit learn](https://towardsdatascience.com/understanding-the-confusion-matrix-from-scikit-learn-c51d88929c79)
- [(Visually) Interpreting the confusion-matrix](https://medium.com/analytics-vidhya/visually-interpreting-the-confusion-matrix-787a70b65678)
- [Plotly - Displaying Figures using Plotly's Python graphing library](https://plotly.com/python/renderers/#displaying-figures)
- [Plotly - Sankey Diagram in Python](https://plotly.com/python/sankey-diagram/#basic-sankey-diagram)

## Acknowledgements

I would like to thank Avi Chawla for his post [Enrich Your Confusion Matrix With A Sankey Diagram](https://www.blog.dailydoseofds.com/p/enrich-your-confusion-matrix-with) that inspired me to add Sankey visualization of a confusion matrix to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

