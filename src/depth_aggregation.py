
import numpy as np

def get_column_percentile_and_downsample(depth_img, p):
    """
    Compute the p percentile of all non-zero values per column in the depth image and downsample the result.
    Input:
        depth_img: input depth image (2D array)
        p: percentile (between 0 and 100)
    Output:
        downsampled_percentiles: downsampled array of p percentiles for each column
    """

    # Initialize an array to store the percentiles
    percentiles = np.zeros(depth_img.shape[1])
    print("percentiles shape: ", percentiles.shape)
    print("Type: ", type(depth_img))

    # Iterate over each column
    for col in range(depth_img.shape[1]):
        # Get the non-zero values in the column
        non_zero_values = depth_img[:, col][depth_img[:, col] > 0]
        
        if non_zero_values.size > 0:
            # Compute the p percentile for non-zero values
            percentiles[col] = np.percentile(non_zero_values, p)
        else:
            # If there are no non-zero values, set the percentile to zero
            percentiles[col] = 0

    # Downsample the percentiles array to 1/16 of its original size
    downsampled_percentiles = percentiles[::16]
    print("downsampled_percentiles shape: ", downsampled_percentiles.shape)

    return downsampled_percentiles

# Example usage:
# aligned_depth_img = ... (your aligned depth image here)
# p = 90  # for example, the 90th percentile
# downsampled_percentiles = column_percentile_nonzero_and_downsample(aligned_depth_img, p)
