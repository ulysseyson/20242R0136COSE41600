target = "data/01_straight_walk/pcd"
import os
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import time

# pcd file find
file_paths = os.listdir(target)
# sort file paths
file_paths.sort()
print(file_paths)
# create a visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.run()
# load pcd file
start_time = time.time()
first_call = True
for file_path in file_paths:

    original_pcd = o3d.io.read_point_cloud(f"{target}/{file_path}")
    # 빠른 연산 및 전처리를 위한 Voxel downsampling
    voxel_size = 0.4  # 필요에 따라 voxel 크기를 조정하세요.
    voxel_downsample_pcd = original_pcd.voxel_down_sample(voxel_size=voxel_size)

    # Radius Outlier Removal (ROR) 적용
    cl, ind = voxel_downsample_pcd.remove_radius_outlier(nb_points=6, radius=1.2)
    ror_pcd = voxel_downsample_pcd.select_by_index(ind)

    # RANSAC을 사용하여 평면 추정
    plane_model, inliers = ror_pcd.segment_plane(
        distance_threshold=0.1, ransac_n=3, num_iterations=2000
    )

    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    # 도로에 속하는 포인트 (inliers)
    road_pcd = ror_pcd.select_by_index(inliers)

    # 도로에 속하지 않는 포인트 (outliers)
    non_road_pcd = ror_pcd.select_by_index(inliers, invert=True)

    # 도로 영역을 초록색으로 표시
    road_pcd.paint_uniform_color([1, 0, 0])  # 빨간색으로 표시
    # 도로가 아닌 포인트를 초록색으로 표시
    non_road_pcd.paint_uniform_color([0, 1, 0])  # 녹색으로 표시

    if first_call:
        vis.add_geometry(road_pcd)
        vis.add_geometry(non_road_pcd)
        first_call = False
    else:
        vis.update_geometry(road_pcd)
        vis.update_geometry(non_road_pcd)
    # capture the image to make a video
    vis.capture_screen_image(f"tmp/{target}/{file_path}.png")

    time.sleep(0.5)

    if not vis.poll_events():
        break
    vis.update_renderer()
    print("Rendered")

    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
vis.close()
vis.destroy_window()
