import os
import json
import cv2
import numpy as np
import trimesh

def verify_gt():
    base_dir = "armo6d/data/bop/ycbv"
    scene_id = "000048"

    gt_json_path = os.path.join(base_dir, "test", scene_id, "scene_gt.json")
    cam_json_path = os.path.join(base_dir, "test", scene_id, "scene_camera.json")

    if not os.path.exists(gt_json_path):
        print(f"错误：找不到真值文件 {gt_json_path}，请检查数据集路径！")
        return

    with open(gt_json_path, 'r') as f:
        gt_data = json.load(f)

    frame_keys = sorted(list(gt_data.keys()), key=lambda x: int(x))
    frame_key = frame_keys[0]

    img_filename = f"{int(frame_key):06d}.png"
    rgb_path = os.path.join(base_dir, "test", scene_id, "rgb", img_filename)

    print(f"探测到该子集起始帧为: {img_filename}")
    img = cv2.imread(rgb_path)
    if img is None:
        print(f"错误：无法读取图片 {rgb_path}")
        return

    with open(cam_json_path, 'r') as f:
        cam_data = json.load(f)
    frame_cam = cam_data[frame_key]
    K = np.array(frame_cam["cam_K"]).reshape(3, 3)
    print(f"成功加载相机内参 K:\n{K}")

    frame_gts = gt_data[frame_key]
    print(f"当前帧共检测到 {len(frame_gts)} 个可见物体，开始投影验证...")

    for gt in frame_gts:
        obj_id = gt["obj_id"]
        R = np.array(gt["cam_R_m2c"]).reshape(3, 3)
        t = np.array(gt["cam_t_m2c"]).reshape(3, 1)

        mesh_path = os.path.join(base_dir, "models", f"obj_{obj_id:06d}.ply")
        if not os.path.exists(mesh_path):
            print(f"警告：未找到 obj_{obj_id} 的模型文件，跳过该物体。")
            continue

        mesh = trimesh.load(mesh_path)
        vertices = mesh.vertices

        pts_cam = np.dot(R, vertices.T) + t
        X, Y, Z = pts_cam[0, :], pts_cam[1, :], pts_cam[2, :]

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        u = (fx * X) / Z + cx
        v = (fy * Y) / Z + cy
        pts_2d = np.stack((u, v), axis=-1).astype(np.int32)

        for pt in pts_2d[::20]:
            if 0 <= pt[0] < img.shape[1] and 0 <= pt[1] < img.shape[0]:
                cv2.circle(img, (pt[0], pt[1]), 1, (255, 0, 0), -1)

        print(f"物体 ID: {obj_id} 投影点云绘制完成")

    window_name = "BOP GT Verification"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img)
    print("\n成功：请在弹出的图片窗口上按任意键退出程序。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    verify_gt()
