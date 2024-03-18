import numpy as np
import cv2

# Anaconda로 돌리려고 만든 테스트 코드입니다. cv2를 blender에 설치해서 addon내에서 테스트하는 것도 가능하긴 합니다.
def validate_matrix(cam, extrinsic, intrinsic, points_2d, points_3d):
    point_3d_homo = np.concatenate((points_3d, np.ones((points_3d.shape[0], 1))), axis=1)
    
    intrinsic_44 = np.eye(4)
    intrinsic_44[:3, :3] = intrinsic
    
    cam_space = extrinsic @ point_3d_homo.T
    
    # Intrinsic 매트릭스를 이용하여 이미지 공간으로 변환
    image_space_homo = intrinsic_44 @ cam_space
    #print(image_space_homo)
    # z값으로 나누어 비동차 좌표계로 변환
    image_space = image_space_homo[:3, :] / image_space_homo[2, :]
    
    # 최종 2D 포인트
    points_2d_projected = image_space[:2, :].T
    
    error = np.linalg.norm(points_2d - points_2d_projected, axis=1)
    print("Total error: ", error.mean())


def validate_matrix_project_to_image(img, extrinsic, intrinsic, points_3d):
    point_3d_homo = np.concatenate((points_3d, np.ones((points_3d.shape[0], 1))), axis=1)
    
    intrinsic_44 = np.eye(4)
    intrinsic_44[:3, :3] = intrinsic
    
    cam_space = extrinsic @ point_3d_homo.T
    
    # Intrinsic 매트릭스를 이용하여 이미지 공간으로 변환
    image_space_homo = intrinsic_44 @ cam_space
    #print(image_space_homo)
    # z값으로 나누어 비동차 좌표계로 변환
    image_space = image_space_homo[:3, :] / image_space_homo[2, :]
    
    # 최종 2D 포인트
    points_2d_projected = image_space[:2, :].T
    print(points_2d_projected)

    for point in points_2d_projected:
        x, y = point.astype(np.int32)
        cv2.circle(img, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
    return img

#path = 'D:/python/xrhumanlab_blender_renderer/output/'
img1 = cv2.imread("image/_frame_0000_cam_001.png")
img2 = cv2.imread("image/_frame_0000_cam_002.png")

extrinsic1 = np.load("camera/001_extrinsic.npy")
extrinsic2 = np.load("camera/002_extrinsic.npy")

intrinsic1 = np.load("camera/001_intrinsic.npy")
intrinsic2 = np.load("camera/002_intrinsic.npy")

points_3d = []
with open("test.obj", "r") as f:
    while True:
        line = f.readline().split("\n")[0]
        if not line: break
        _, x, y, z, _, _, _ = line.split(" ")
        x, y, z = float(x), float(y), float(z)
        points_3d.append([x, y, z])

    f.close()

points_3d = np.array(points_3d)

print(extrinsic1, intrinsic1)
print(extrinsic2, intrinsic2)
img1 = validate_matrix_project_to_image(img1, extrinsic1, intrinsic1, points_3d)
img2 = validate_matrix_project_to_image(img2, extrinsic2, intrinsic2, points_3d)

cv2.imwrite("image/_frame_0000_cam_001_proj.png",img1)
cv2.imwrite("image/_frame_0000_cam_002_proj.png",img2)
