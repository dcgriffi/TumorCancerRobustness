from ast import Break
import os
import SimpleITK as sitk
import numpy as np
import glob
import math
#import google3
from absl.testing import absltest
from absl.testing import parameterized
import surface_distance
from surface_distance import metrics

### Info
# This script is to calculate the DICE, sDICE, and the Hausdorff distance of two sets of labels 
# The sets of labels are calles (a) original for the manual labels and (b) predicted for the predicted labels
# Documentation of metrics can be found here: https://github.com/deepmind/surface-distance
### 

def generate_np_array_of_single_structure(np_matrix, number):
    single_struct_array = np.where(np_matrix != number, 0, np_matrix)
    return single_struct_array.astype('bool')    # sets all 0-entries to False, and all others to True


if __name__ == '__main__': 

    original_label_path = '/home/chloe/nnUNet_trained_models/nnUNet/3d_fullres/Task603_SettingUp/nnUNetTrainerV2__nnUNetPlansv2.1/original_labels_TCIA'
    predicted_label_path = '/home/chloe/nnUNet_trained_models/nnUNet/3d_fullres/Task603_SettingUp/nnUNetTrainerV2__nnUNetPlansv2.1/predictions_TCIA'
    num_structures = 1 + 1

    dice_matrix = np.zeros(num_structures)
    hd_matrix = np.zeros(num_structures)
    avg_surface_d_matrix = np.zeros(num_structures)
    surf_dice_matrix = np.zeros(num_structures)
    s_dice_tolerance = 2 #mm
    num_patients = 0

    all_numbers = []
    for i in range(num_structures):
        all_numbers.append(i)
    print("all_numbers: ", all_numbers)

    test_patient_list = ['4E6RGF0M.nii.gz']

    # for each patient // iterate over the file paths
    for filename in os.listdir(predicted_label_path):
        num_patients += 1
        dice_arr = np.zeros(num_structures)
        hd_arr = np.zeros(num_structures)
        avg_surface_d_arr = np.zeros(num_structures)
        surf_dice_arr = np.zeros(num_structures)

        # extract the example name
        # first split the path by '/' and take the last result to get the file name
        # then exclude the last 7 characters to exclude the '.nii.gz'
        name = filename.split('/')[-1][:-7]

        #print("filename in process: ", name)
        print()
        print(name)


        ############ ORIGINAL ############

        # read the label file and save it as nifti with correct naming
        original_data_path = original_label_path + "/" +  name + ".nii.gz"
        original_img = sitk.ReadImage(original_data_path)  # read the file (file is the path to the label file)
        original_arr = sitk.GetArrayFromImage(original_img)

        # count amount of pixels with different labels
        unique, counts = np.unique(original_arr, return_counts=True)
        #print("original_dict: ", dict(zip(unique, counts)))
        print("missing labels in original labels:", list(set(all_numbers) - set(unique)))


        ############ PREDICTIONS ############

        # do the same for the corresponding image file
        pred_data_path = predicted_label_path + "/" +  name + ".nii.gz"
        pred_img = sitk.ReadImage(pred_data_path)  # exchange the _labels with _predictions in the file name to get the path to the predictions and reads the predictions
        pred_arr = sitk.GetArrayFromImage(pred_img)                 

        # count amount of pixels with different labels
        unique2, counts2 = np.unique(pred_arr, return_counts=True)
        #print("predicted_dict: ", dict(zip(unique2, counts2)))
        print("missing labels in predicted labels:", list(set(all_numbers) - set(unique2)))

        #for each structure
        for number in range(num_structures):

            # no evalutation for background
            if(number == 0): continue

            if(number in unique2):
                #print("Analyzing structure: ", number)
                original_structure = generate_np_array_of_single_structure(original_arr, number)
                predicted_structure = generate_np_array_of_single_structure(pred_arr, number)

                # 3rd agument spacing_mm: resp. 3-element list-like structure. Voxel spacing in x0, x1 and x2 directions.
                surf_dist = metrics.compute_surface_distances(original_structure, predicted_structure, [3, 1, 1])

                dice = metrics.compute_dice_coefficient(original_structure, predicted_structure)
                # 2nd argument: percent: a float value between 0 and 100
                hd = metrics.compute_robust_hausdorff(surf_dist, 99)
                avg_sd = metrics.compute_average_surface_distance(surf_dist)
                s_dice = metrics.compute_surface_dice_at_tolerance(surf_dist, s_dice_tolerance)

                # print('dice = ', dice)
                # print('HD = ', hd)
                # print('avg_sd = ', avg_sd)
                # print('s_dice = ', s_dice)

                #print('Hausdorff distance = ', metrics.compute_robust_hausdorff(metrics.compute_surface_distances(labels, pred, 2), 95))

                dice_arr[number] = dice
                hd_arr[number] = hd
                #avg_surface_d_arr[number] = avg_sd   #is a tuple and cannot be written in array
                surf_dice_arr[number] = s_dice

            else: 
                # check if a predicted structure is missing
                print("missing structure in prediction: ", number)


        print("dice coeffcients are ")
        #print(dice_arr)
        print(np.array2string(dice_arr, separator=","))

        print("Hausdorff distances are ")
        #print(hd_arr)
        print(np.array2string(hd_arr, separator=","))

        print("Surface dice is ")
        #print(surf_dice_arr)
        print(np.array2string(surf_dice_arr, separator=","))
        

        dice_matrix = np.vstack((dice_matrix, dice_arr))
        hd_matrix = np.vstack((hd_matrix, hd_arr))
        #avg_surface_d_matrix = np.vstack(avg_surface_d_matrix, avg_surface_d_arr)
        surf_dice_matrix = np.vstack((surf_dice_matrix, surf_dice_arr))         


    # dice_arr = dice_arr[1:]
    # hd_arr = hd_arr[1:]
    # avg_surface_d_arr = avg_surface_d_arr[1:]
    # surf_dice_arr = surf_dice_arr[1:]

    dice_matrix = dice_matrix[1:]
    hd_matrix = hd_matrix[1:]
    #avg_surface_d_matrix = 
    surf_dice_matrix = surf_dice_matrix[1:]

    # calculate mean and standard deviation
    #print("dice coeffcients are: ", np.sort(dice_arr))
    dice_matrix_avg_struc = np.zeros((num_structures, num_patients) )
    hd_matrix_avg_struc = np.zeros((num_structures, num_patients))
    s_dice_matrix_avg_struc = np.zeros((num_structures, num_patients))

    for i in range(num_structures):
        for j in range(num_patients):
            dice_matrix_avg_struc[i][j] = dice_matrix[j][i]
            hd_matrix_avg_struc[i][j] = hd_matrix[j][i]
            s_dice_matrix_avg_struc[i][j] = surf_dice_matrix[j][i]

    mean_std_dice_dict = {}
    mean_std_hd_dict = {}
    mean_std_s_dice_dict = {}

    for i in range(num_structures):
        mean_std_dice_dict[i] = (np.mean(dice_matrix_avg_struc[i]), np.std(dice_matrix_avg_struc[i]))
        mean_std_hd_dict[i] = (np.mean(hd_matrix_avg_struc[i]), np.std(hd_matrix_avg_struc[i]))
        mean_std_s_dice_dict[i] = (np.mean(s_dice_matrix_avg_struc[i]), np.std(s_dice_matrix_avg_struc[i]))

    print("DICE (mean, std): ", mean_std_dice_dict)
    print("HD (mean, std): ", mean_std_hd_dict)
    print("s_DICE (mean, std): ", mean_std_s_dice_dict)