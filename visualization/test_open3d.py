import open3d as o3d

mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
mesh.compute_vertex_normals()

o3d.visualization.draw_geometries([mesh])
