import os
import glob
import nibabel as nib
import numpy as np

def main():
    # Paths to dataset folders
    dataset_path = "C:\\Project1\\Dataset\\Raw_AKU\\"
    output_folder = "C:\\Project1\\Dataset\\Cropped_Volumes\\"
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Retrieve file paths
    flair_list = sorted(glob.glob(os.path.join(dataset_path, '*/*/*_Flair.nii')))
    t1ce_list = sorted(glob.glob(os.path.join(dataset_path, '*/*/*_T1CE.nii')))
    mask_list = sorted(glob.glob(os.path.join(dataset_path, '*/*/*seg.nii')))

    # Calculate bounding box dimensions
    min_indices, max_indices = calculate_bounding_box(mask_list)

    # Crop and save volumes
    crop_and_save_volumes(mask_list, flair_list, t1ce_list, min_indices, max_indices, output_folder)

    print("Cropped volumes saved to:", output_folder)



def calculate_bounding_box(mask_list):
    # Initialize min and max indices to large and small values respectively
    min_indices = np.array([float('inf'), float('inf'), float('inf')])
    max_indices = np.array([0, 0, 0])

    # Loop through the list of segmentation volumes
    for mask_path in mask_list:
        # Load the segmentation mask
        mask_volume = nib.load(mask_path).get_fdata()

        # Find the indices of non-zero values in the mask
        non_zero_indices = np.argwhere(mask_volume > 0)

        # Update the min and max indices
        min_indices = np.minimum(min_indices, np.min(non_zero_indices, axis=0))
        max_indices = np.maximum(max_indices, np.max(non_zero_indices, axis=0))

    # Convert indices to integers
    min_indices = min_indices.astype(int)
    max_indices = max_indices.astype(int)

    print("Bounding Box Min Indices:", min_indices)
    print("Bounding Box Max Indices:", max_indices)
    print("Bounding Box Dimensions:", max_indices - min_indices + 1)

    return min_indices, max_indices



def crop_and_save_volumes(mask_list, flair_list, t1ce_list, min_indices, max_indices, output_folder):
    # Loop through the volumes
    for mask_path, flair_path, t1ce_path in zip(mask_list, flair_list, t1ce_list):
        # Load the volumes
        mask_volume = nib.load(mask_path).get_fdata()
        flair_volume = nib.load(flair_path).get_fdata()
        t1ce_volume = nib.load(t1ce_path).get_fdata()

        # Crop the volumes using calculated bounding box dimensions
        cropped_mask = mask_volume[min_indices[0]:max_indices[0] + 1,
                                   min_indices[1]:max_indices[1] + 1,
                                   min_indices[2]:max_indices[2] + 1]

        cropped_flair = flair_volume[min_indices[0]:max_indices[0] + 1,
                                     min_indices[1]:max_indices[1] + 1,
                                     min_indices[2]:max_indices[2] + 1]

        cropped_t1ce = t1ce_volume[min_indices[0]:max_indices[0] + 1,
                                   min_indices[1]:max_indices[1] + 1,
                                   min_indices[2]:max_indices[2] + 1]

        # Get the original filename
        _, filename = os.path.split(mask_path)
        filename_without_extension = os.path.splitext(filename)[0].split('.')[0]

        # Save the cropped volumes to the output folder with the same names
        mask_output_path = os.path.join(output_folder, filename_without_extension + "_seg.nii")
        flair_output_path = os.path.join(output_folder, filename_without_extension + "_Flair.nii")
        t1ce_output_path = os.path.join(output_folder, filename_without_extension + "_T1ce.nii")

        nib.save(nib.Nifti1Image(cropped_mask, np.eye(4)), mask_output_path)
        nib.save(nib.Nifti1Image(cropped_flair, np.eye(4)), flair_output_path)
        nib.save(nib.Nifti1Image(cropped_t1ce, np.eye(4)), t1ce_output_path)



if __name__ == "__main__":
    main()
