import open3d as o3d
import numpy as np
import sys, os
print("Load a ply point cloud, print it, and render it")
ply_point_cloud = o3d.data.PLYPointCloud()
pcd = o3d.io.read_point_cloud(sys.argv[1])
print(pcd)
print(np.asarray(pcd.points))
o3d.visualization.draw_geometries([pcd])
