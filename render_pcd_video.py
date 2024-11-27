ps = [
    "01_straight_walk",
    "02_straight_duck_walk",
    "03_straight_crawl",
    "04_zigzag_walk",
    "05_straight_duck_walk",
    "06_straight_crawl",
    "07_straight_walk",
]
# load view setting
import json

for p in ps:
    with open(f"view_setting_{p}.json", "r") as f:
        view_setting = json.load(f)
        trajectory = view_setting["trajectory"][0]
    target = f"data/{p}/pcd"
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
    vis.create_window(width=2800, height=1920)
    # ctr = vis.get_view_control()
    # ctr.set_zoom(0.8)
    # # Adjust the field of view to exactly 30 degrees
    # current_fov = ctr.get_field_of_view()  # Get the current field of view
    # desired_fov = 30  # Set your desired field of view
    # ctr.change_field_of_view(step=desired_fov - current_fov)
    # # Set point size
    # render_option = vis.get_render_option()
    # render_option.point_size = 2.0  # Adjust point size
    # load pcd file
    start_time = time.time()
    os.makedirs(f"tmp/{target}", exist_ok=True)
    first_call = True
    from tqdm import tqdm

    # Initialize variables to store geometry references
    road_geometry = None
    non_road_geometry = None
    # ========================
    # Parameters for detection
    voxel_size = 0.4
    nb_points = 3
    radius = 1.5
    distance_threshold = 0.15
    ransac_n = 3
    num_iterations = 3000
    eps = 1.0
    min_points = 4
    # 필터링 기준 1. 클러스터 내 최대 최소 포인트 수
    min_points_in_cluster = 5  # 클러스터 내 최소 포인트 수
    max_points_in_cluster = 15  # 클러스터 내 최대 포인트 수

    # 필터링 기준 2. 클러스터 내 최소 최대 Z값
    min_z_value = -10.0  # 클러스터 내 최소 Z값
    max_z_value = 10.5  # 클러스터 내 최대 Z값

    # 필터링 기준 3. 클러스터 내 최소 최대 Z값 차이
    min_height = 0.1  # Z값 차이의 최소값
    max_height = 0.0  # Z값 차이의 최대값

    max_distance = 400.0  # 원점으로부터의 최대 거리
    # ========================
    bbox_geometries = []
    for file_path in tqdm(file_paths):
        original_pcd = o3d.io.read_point_cloud(f"{target}/{file_path}")

        voxel_downsample_pcd = original_pcd.voxel_down_sample(voxel_size=voxel_size)

        cl, ind = voxel_downsample_pcd.remove_radius_outlier(
            nb_points=nb_points, radius=radius
        )
        ror_pcd = voxel_downsample_pcd.select_by_index(ind)

        plane_model, inliers = ror_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )

        # 도로에 속하지 않는 포인트 (outliers) 추출
        final_point = ror_pcd.select_by_index(inliers, invert=True)

        # 포인트 클라우드를 NumPy 배열로 변환
        points = np.asarray(final_point.points)

        # DBSCAN 클러스터링 적용
        with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug
        ) as cm:
            labels = np.array(
                final_point.cluster_dbscan(eps=eps, min_points=min_points)
            )

        # 각 클러스터를 색으로 표시
        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")

        # 노이즈를 제거하고 각 클러스터에 색상 지정
        colors = plt.get_cmap("tab20")(labels / (max_label + 1 if max_label > 0 else 1))
        colors[labels < 0] = 0  # 노이즈는 검정색으로 표시
        final_point.colors = o3d.utility.Vector3dVector(colors[:, :3])
        # 필터링 기준 1. 클러스터 내 최대 최소 포인트 수
        min_points_in_cluster = 5  # 클러스터 내 최소 포인트 수
        max_points_in_cluster = 15  # 클러스터 내 최대 포인트 수

        # 필터링 기준 2. 클러스터 내 최소 최대 Z값
        min_z_value = -10  # 클러스터 내 최소 Z값
        max_z_value = 10.5  # 클러스터 내 최대 Z값

        # 필터링 기준 3. 클러스터 내 최소 최대 Z값 차이
        min_height = 0.1  # Z값 차이의 최소값
        max_height = 2.0  # Z값 차이의 최대값

        max_distance = 400.0  # 원점으로부터의 최대 거리
        bboxes_1234 = []
        for i in range(max_label + 1):
            cluster_indices = np.where(labels == i)[0]
            if min_points_in_cluster <= len(cluster_indices) <= max_points_in_cluster:
                cluster_pcd = final_point.select_by_index(cluster_indices)
                points = np.asarray(cluster_pcd.points)
                z_values = points[:, 2]  # Z값 추출
                z_min = z_values.min()
                z_max = z_values.max()
                if min_z_value <= z_min and z_max <= max_z_value:
                    height_diff = z_max - z_min
                    if min_height <= height_diff <= max_height:
                        distances = np.linalg.norm(points, axis=1)
                        if distances.max() <= max_distance:
                            bbox = cluster_pcd.get_axis_aligned_bounding_box()
                            bbox.color = (1, 0, 0)
                            bboxes_1234.append(bbox)

        for bbox_geometry in bbox_geometries:
            vis.remove_geometry(bbox_geometry)
        bbox_geometries.clear()
        print(f"Number of bounding boxes: {len(bboxes_1234)}")
        for bbox in bboxes_1234:
            bbox_geometry = bbox
            vis.add_geometry(bbox)
            bbox_geometries.append(bbox_geometry)
        # Add or update geometries in the visualizer
        if first_call:
            road_geometry = final_point
            vis.add_geometry(road_geometry)
            first_call = False
        else:
            road_geometry.points = final_point.points
            road_geometry.colors = final_point.colors
            vis.update_geometry(road_geometry)

        # Get the view control
        ctr = vis.get_view_control()

        # Extract camera parameters from the JSON
        lookat = trajectory["lookat"]
        front = trajectory["front"]
        up = trajectory["up"]
        field_of_view = trajectory["field_of_view"]
        zoom = trajectory["zoom"]

        # Set the camera parameters using the JSON data
        ctr.set_lookat(lookat)
        ctr.set_front(front)
        ctr.set_up(up)
        ctr.set_zoom(zoom)
        ctr.change_field_of_view(step=field_of_view - ctr.get_field_of_view())

        # Optionally, apply zoom (this can be done by adjusting the camera position)
        # If the zoom is a scaling factor for the camera's distance from the target
        camera_params = ctr.convert_to_pinhole_camera_parameters()
        extrinsic = np.array(camera_params.extrinsic)

        extrinsic[2, 3] *= zoom  # Adjust the camera distance by zoom factor
        camera_params.extrinsic = extrinsic
        ctr.convert_from_pinhole_camera_parameters(camera_params)

        # # Capture the updated image
        # while vis.poll_events():
        #     ctr = vis.get_view_control()
        #     vis.update_renderer()
        #     camera_params = ctr.convert_to_pinhole_camera_parameters()
        #     if time.time() - start_time > 30:
        #         break
        #     ctr.convert_from_pinhole_camera_parameters(
        #         camera_params, allow_arbitrary=True
        #     )
        vis.poll_events()
        vis.update_renderer()
        os.makedirs(f"tmp/try4/{p}", exist_ok=True)
        vis.capture_screen_image(f"tmp/try4/{p}/{file_path}.png")
        # time.sleep(10)
        # vis.run()
        # vis.close()
    vis.close()
    vis.destroy_window()
