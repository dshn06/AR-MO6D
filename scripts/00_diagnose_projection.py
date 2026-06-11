import os
import json
import cv2
import numpy as np
import trimesh

def project_vertices(vertices, R, t, K):
    pts_cam = np.dot(R, vertices.T) + t
    X, Y, Z = pts_cam[0, :], pts_cam[1, :], pts_cam[2, :]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = (fx * X) / Z + cx
    v = (fy * Y) / Z + cy
    return np.stack((u, v), axis=-1)

def verify_gt():
    base_dir = "armo6d/data/bop/ycbv"
    scene_id = "000048"

    gt_json_path = os.path.join(base_dir, "test", scene_id, "scene_gt.json")
    cam_json_path = os.path.join(base_dir, "test", scene_id, "scene_camera.json")

    if not os.path.exists(gt_json_path):
        print(f"错误：找不到真值文件 {gt_json_path}")
        return

    with open(gt_json_path, 'r') as f:
        gt_data = json.load(f)

    frame_keys = sorted(list(gt_data.keys()), key=lambda x: int(x))
    frame_key = frame_keys[0]

    img_filename = f"{int(frame_key):06d}.png"
    rgb_path = os.path.join(base_dir, "test", scene_id, "rgb", img_filename)

    print(f"起始帧: {img_filename}")
    img = cv2.imread(rgb_path)
    if img is None:
        print(f"错误：无法读取图片 {rgb_path}")
        return
    print(f"图像尺寸: {img.shape[1]}x{img.shape[0]}")

    with open(cam_json_path, 'r') as f:
        cam_data = json.load(f)
    frame_cam = cam_data[frame_key]
    K = np.array(frame_cam["cam_K"]).reshape(3, 3)
    print(f"相机内参 K:\n{K}")
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    frame_gts = gt_data[frame_key]
    print(f"物体数: {len(frame_gts)}")

    model_paths = [
        ("models/ (简化)", os.path.join(base_dir, "models", "obj_{:06d}.ply")),
        ("models_fine/ (完整)", os.path.join(base_dir, "models", "ycbv_models", "models_fine", "obj_{:06d}.ply")),
    ]

    for label, mesh_pattern in model_paths:
        img_copy = img.copy()
        for gt in frame_gts:
            obj_id = gt["obj_id"]
            R = np.array(gt["cam_R_m2c"]).reshape(3, 3)
            t = np.array(gt["cam_t_m2c"]).reshape(3, 1)

            mesh_path = mesh_pattern.format(obj_id)
            if not os.path.exists(mesh_path):
                print(f"  [{label}] 跳过 obj_{obj_id} (文件不存在)")
                continue

            mesh = trimesh.load(mesh_path)
            vertices = mesh.vertices
            print(f"  [{label}] obj_{obj_id}: {len(vertices)} 顶点, "
                  f"范围 X:[{vertices[:,0].min():.1f},{vertices[:,0].max():.1f}] "
                  f"Y:[{vertices[:,1].min():.1f},{vertices[:,1].max():.1f}] "
                  f"Z:[{vertices[:,2].min():.1f},{vertices[:,2].max():.1f}]")

            pts_2d = project_vertices(vertices, R, t, K)
            for pt in pts_2d[::20]:
                if 0 <= pt[0] < img_copy.shape[1] and 0 <= pt[1] < img_copy.shape[0]:
                    cv2.circle(img_copy, (int(pt[0]), int(pt[1])), 1, (255, 0, 0), -1)

            center_cam = t.flatten()
            u_c = fx * center_cam[0] / center_cam[2] + cx
            v_c = fy * center_cam[1] / center_cam[2] + cy
            cv2.circle(img_copy, (int(u_c), int(v_c)), 5, (0, 0, 255), -1)
            cv2.putText(img_copy, f"center_{obj_id}", (int(u_c)+5, int(v_c)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            print(f"    obj_{obj_id} 中心 相机坐标: X={center_cam[0]:.1f} Y={center_cam[1]:.1f} Z={center_cam[2]:.1f}")
            print(f"    投影像素: u={u_c:.1f} v={v_c:.1f}")

        window_name = f"BOP GT - {label}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, img_copy)

    print("\n请对比两个窗口中的投影（蓝色点云 + 红色中心点）。")
    print("如果简化模型偏下而完整模型对齐，说明是模型精度问题。")
    print("如果两者都偏下，则可能是 GT pose 或相机内参的系统偏差。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    verify_gt()
