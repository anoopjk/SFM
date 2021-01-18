import open3d as o3d
import numpy as np
import copy

if __name__ == "__main__":
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    T = np.eye(4)
    T[:3, :3] = mesh.get_rotation_matrix_from_xyz((0, np.pi / 3, np.pi / 2))
    T[0, 3] = 1
    T[1, 3] = 1.3
    print(T)
    mesh_t = copy.deepcopy(mesh).transform(T)
    o3d.visualization.draw_geometries([mesh, mesh_t])

