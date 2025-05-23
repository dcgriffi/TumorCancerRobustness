# TumorCancerRobustness
A collection of notebooks used to measure the robustness of a deep learning model for head and neck tumor delineation

# Background #

To test the robustness of the model, we rotate the test images and feed them into the neural network model. We rotate the corresponding manual labels through the same process as the image. Thus, the image and manual label go through the same image augmentation pipeline. We obtain the Dice and sDice metric from the rotated model output with the rotated manual label. This allows us to compare the results before and after rotation. 

We conducted four experiments with centered rotations around the $z$-axis using a Monai setup. The experimental setup can be seen in the table below.


|  Experiment | Rot Range  | Rot Step | Padding Dim Size| Num Samples|
| :------------: |:---------------:| :-----:| :-----:|:-----:|
| 1 | (-90,90) | 10 | 722 (max) | 10 |
| 2 | (-180, 180) | 10 | 722 (max) | 2 |
| 3 | (-10,10) | 1 | 594 (max) | 3|
| 4 | (-30,30)| 5| 585 (min) | 15 |

In the table, "Rot Range" gives the range of rotations while "Rot Step" tells the step size. For example, Experiment 3 used rotations $(-10,-9,-8,...,8,9,10).$ The column "Padding Dim Size'' refers to the $x$ and $y$ dimension after padding and "max" or `"min" refers to the approach for choosing padding, as is explained in the next paragraph. Lastly, "Num Samples" gives the number of samples used in the experiment.

We want to ensure that no portion of the patient is lost during rotation. Thus, we pad the output of the affine rotation to fit the rotated image. We create a tutorial to find the bounding box size the maximal effect of rotation. For instance, in Experiment 2, the maximal degrees of interest are 40 and 50 degrees. In other words, these require the largest amount of padding within the rotation range. A demo of this process is included in the content list below. The minimal padding only padded to the extent we could visually detect the labels and data to be unaffected by rotation. It allowed for faster computation time, but associated predictions should be reviewed. 

# Content List #
- experiment_01_02: Contain notebooks for running experiment 1 and 2
- experiment_03: Contain notebooks for running experiment 3
- experiment_04: Contain notebooks for running experiment 4
- get_dim_about_z_plot_demo_01: Demo for getting the correct padding dimension given a max rotation and input size

Each of the experimental notebooks have the same layout and use methods. You should store your image labels and data in the data folder. If you want to run the experiment, you should run first the notebook transformed_rotations_01.ipynb (adjusting paths as necessary), run your nnUnet model as usual, then run transformed_metrics_01.ipynb to get the results. You can also get a better understanding of the datapipeline by running the control_rotations_01.ipynb and control_predictions_01.ipynb in the same way. Results will be stored in the transformed_results and control_results folders. 

You can adapt any of these experiments as needed to futher understand the effect of rotations on the model. You can also easily extend to other data augmentation strategies by updating the Monai Transformations in the transformation notebooks. Enjoy!
