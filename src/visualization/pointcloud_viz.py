# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/python/geometry/camera_trajectory.py

import numpy as np
import open3d as o3d

if __name__ == "__main__":

    print("Testing camera in open3d ...")
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    print(intrinsic.intrinsic_matrix)
    print(o3d.camera.PinholeCameraIntrinsic())
    x = o3d.camera.PinholeCameraIntrinsic(640, 480, 525, 525, 320, 240)
    print(x)
    print(x.intrinsic_matrix)

    print("Read a trajectory and combine all the RGB-D images.")
    pcds = []
    trajectory = o3d.io.read_pinhole_camera_trajectory(
        "/media/seagate4TB/deeplearning/datasets/scene-net/pose_bedroom/bedroom.log")

    # print(trajectory)
    # print(trajectory.parameters[0].extrinsic)
    # print(np.asarray(trajectory.parameters[0].extrinsic))
    for i in range(50):
        rgb = o3d.io.read_image(
            "/media/seagate4TB/deeplearning/datasets/scene-net/rgbd_bedroom/bedroom/image/{:06d}.jpg".format(i))
        depth = o3d.io.read_image(
            "/media/seagate4TB/deeplearning/datasets/scene-net/rgbd_bedroom/bedroom/depth/{:06d}.png".format(i))
        print('rgb shape: ', rgb.is_empty())
        im = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb, depth, 1000.0, 5.0, False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            im, trajectory.parameters[i].intrinsic,
            trajectory.parameters[i].extrinsic)
        pcds.append(pcd)
    o3d.visualization.draw_geometries(pcds)
    print("")