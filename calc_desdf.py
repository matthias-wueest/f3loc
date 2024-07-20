from utils.generate_desdf import raycast_desdf



import matplotlib.image as mpimg
import numpy as np

#floorplans_path = "C:/Users/matth/Documents/ETHZ/01_DS/03_CapstoneProject/03_Datasets/LaMAR/Floorplans/HGE"
#floorplan_name = "HG_E_VF_20240215_preparing_occupancy_windows.png" #"floorplan_test.png" #"screenshot_paper_edited.png" #"occupancy_test.png"#""screenshot_paper_small_edited.bmp"
#floorplan_path = os.path.join(floorplans_path, floorplan_name)
#floorplan = mpimg.imread(floorplan_path)

floorplan_name = "map_cropped.png" #"map.png" #  

floorplan = mpimg.imread(floorplan_name)

pixel_per_meter = 18.315046895211292

occ = floorplan*255 # occ: the map as occupancy
original_resolution = 1/pixel_per_meter # original_resolution: the resolution of occ input [m/pixel]
resolution = 1#0.1#10#original_resolution # resolution: output resolution of the desdf [m/pixel]
orn_slice = 360
max_dist = 10



### ORIGINAL
desdf = raycast_desdf(occ, orn_slice=orn_slice, max_dist=max_dist, original_resolution=original_resolution, resolution=resolution)

#desdf = raycast_desdf(occ, orn_slice=72, max_dist=10, original_resolution=original_resolution, resolution=resolution)
#desdf = raycast_desdf(occ, orn_slice=144, max_dist=10, original_resolution=original_resolution, resolution=resolution)
#desdf = raycast_desdf(occ, orn_slice=288, max_dist=10, original_resolution=original_resolution, resolution=resolution)
#desdf = raycast_desdf(occ, orn_slice=360, max_dist=10, original_resolution=original_resolution, resolution=resolution)

#filename = "desdf_complete_orn_slice_36_resolution_01.npy"
filename = "desdf_cropped_orn_slice_360_resolution_1_attempt.npy"
#filename = "desdf_cropped_orn_slice_72.npy"
#filename = "desdf_cropped_orn_slice_144_resolution_1.npy"
#filename = "desdf_cropped_orn_slice_288_resolution_1.npy"
#filename = "desdf_cropped_orn_slice_360_resolution_1.npy"
np.save(filename, desdf)

###
