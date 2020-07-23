import numpy as np
import torch


def get_calib_from_file(calib_file):
    with open(calib_file) as f:
        lines = f.readlines()

    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)

    return {'P2': P2.reshape(3, 4),
            'P3': P3.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}


class Calibration(object):
    def __init__(self, calib_file):
        if isinstance(calib_file, str):
            calib = get_calib_from_file(calib_file)
        else:
            calib = calib_file

        self.P2 = calib['P2']  # 3 x 4
        self.R0 = calib['R0']  # 3 x 3
        self.V2C = calib['Tr_velo2cam']  # 3 x 4

        # Camera intrinsics and extrinsics
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)

    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def rect_to_lidar(self, pts_rect):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)  # (N, 4)
        R0_ext = np.hstack((self.R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
        R0_ext = np.vstack((R0_ext, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        R0_ext[3, 3] = 1
        V2C_ext = np.vstack((self.V2C, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        V2C_ext[3, 3] = 1

        pts_lidar = np.dot(pts_rect_hom, np.linalg.inv(np.dot(R0_ext, V2C_ext).T))
        return pts_lidar[:, 0:3]

    def lidar_to_rect(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))
        # pts_rect = reduce(np.dot, (pts_lidar_hom, self.V2C.T, self.R0.T))
        return pts_rect

    def rect_to_img(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[3, 2]  # depth in rect camera coord
        return pts_img, pts_rect_depth

    def lidar_to_img(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect)
        return pts_img, pts_depth

    def img_to_rect(self, u, v, depth_rect):
        """
        :param u: (N)
        :param v: (N)
        :param depth_rect: (N)
        :return:
        """
        x = ((u - self.cu) * depth_rect) / self.fu + self.tx
        y = ((v - self.cv) * depth_rect) / self.fv + self.ty
        pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
        return pts_rect

    def corners3d_to_img_boxes(self, corners3d):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

        img_pts = np.matmul(corners3d_hom, self.P2.T)  # (N, 8, 3)

        x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

        boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
        boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

        return boxes, boxes_corner

class Calibration_torch(object):
    def __init__(self, calib_file):
        if isinstance(calib_file, str):
            calib = get_calib_from_file(calib_file)
        else:
            calib = calib_file

        self.P2 = torch.Tensor(calib['P2'])  # 3 x 4
        self.R0 = torch.Tensor(calib['R0'])  # 3 x 3
        self.V2C = torch.Tensor(calib['Tr_velo2cam'])  # 3 x 4

        # Camera intrinsics and extrinsics
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)

    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = torch.cat((pts, torch.ones((pts.shape[0], 1), dtype=torch.float32)), dim=1)
        return pts_hom

    def rect_to_lidar(self, pts_rect):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)  # (N, 4)
        R0_ext = torch.cat((self.R0, torch.zeros((3, 1), dtype=torch.float32)), dim=1)  # (3, 4)
        R0_ext = torch.cat((R0_ext, torch.zeros((1, 4), dtype=torch.float32)), dim=0)  # (4, 4)
        R0_ext[3, 3] = 1
        V2C_ext = torch.cat((self.V2C, torch.zeros((1, 4), dtype=torch.float32)), dim=0)  # (4, 4)
        V2C_ext[3, 3] = 1

        pts_lidar = torch.matmul(pts_rect_hom, torch.inverse(torch.matmul(R0_ext, V2C_ext).T))
        return pts_lidar[:, 0:3]

    def lidar_to_rect(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        pts_rect = torch.matmul(pts_lidar_hom, torch.matmul(self.V2C.T, self.R0.T))
        return pts_rect

    def rect_to_img(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = torch.matmul(pts_rect_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[3, 2]  # depth in rect camera coord
        return pts_img, pts_rect_depth

    def lidar_to_img(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect)
        return pts_img, pts_depth

    def img_to_rect(self, u, v, depth_rect):
        """
        :param u: (N)
        :param v: (N)
        :param depth_rect: (N)
        :return:
        """
        x = ((u - self.cu) * depth_rect) / self.fu + self.tx
        y = ((v - self.cv) * depth_rect) / self.fv + self.ty
        pts_rect = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), dim=1)
        return pts_rect

    def corners3d_to_img_boxes(self, corners3d):
        """
        We so far have no need for this in the torch version.
        """
        raise NotImplementedError


if __name__ == "__main__":

    # # get sample calib matrices and sample lidar points
    # # calib_path = "/Volumes/CK/data/kitti/kitti_obj_det/data/training/calib/000000.txt"
    # # lidar_path = "/Volumes/CK/data/kitti/kitti_obj_det/data/training/velodyne/000000.bin"
    # calib_path = "/Users/ck/data_local/argo/argoverse-tracking-kitti-format/sample/calib/000000155.txt"
    # lidar_path = "/Users/ck/data_local/argo/argoverse-tracking-kitti-format/sample/velodyne/000000155.bin"
    # calib_np = Calibration(calib_path)
    # lidar = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    # print(repr(calib_np.P2))
    # print(repr(calib_np.R0))
    # print(repr(calib_np.V2C))
    # print(repr(lidar[:10, :]))
    # import pdb;pdb.set_trace()

    ############################## test using the kitti dataset
    # test by hardcoded calib and pts (we obtained this using the first method, and so we no longer keep these
    # files in disk
    calib = {
        "P2": np.array([[7.070493e+02, 0.000000e+00, 6.040814e+02, 4.575831e+01],
                        [0.000000e+00, 7.070493e+02, 1.805066e+02, -3.454157e-01],
                        [0.000000e+00, 0.000000e+00, 1.000000e+00, 4.981016e-03]]),
        "R0": np.array([[0.9999128, 0.01009263, -0.00851193],
                        [-0.01012729, 0.9999406, -0.00403767],
                        [0.00847067, 0.00412352, 0.9999556]]),
        "Tr_velo2cam": np.array([[0.00692796, -0.9999722, -0.00275783, -0.02457729],
                                 [-0.00116298, 0.00274984, -0.9999955, -0.06127237],
                                 [0.9999753, 0.00693114, -0.0011439, -0.3321029]]),
    }
    points_np = np.array([[1.8324e+01, 4.9000e-02, 8.2900e-01, 0.0000e+00],
                          [1.8344e+01, 1.0600e-01, 8.2900e-01, 0.0000e+00],
                          [5.1299e+01, 5.0500e-01, 1.9440e+00, 0.0000e+00],
                          [1.8317e+01, 2.2100e-01, 8.2900e-01, 0.0000e+00],
                          [1.8352e+01, 2.5100e-01, 8.3000e-01, 9.0000e-02],
                          [1.5005e+01, 2.9400e-01, 7.1700e-01, 2.0000e-01],
                          [1.4954e+01, 3.4000e-01, 7.1500e-01, 5.8000e-01],
                          [1.5179e+01, 3.9400e-01, 7.2300e-01, 0.0000e+00],
                          [1.8312e+01, 5.3800e-01, 8.2900e-01, 0.0000e+00],
                          [1.8300e+01, 5.9500e-01, 8.2800e-01, 0.0000e+00]], dtype=np.float32)

    ############################## or test using the argoverse dataset

    calib = {
        "P2": np.array([[1.402768e+03, 0.000000e+00, 9.817422e+02, 0.000000e+00],
                        [0.000000e+00, 1.402768e+03, 6.107618e+02, 0.000000e+00],
                        [0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00]],
                       dtype=np.float32),
        "R0": np.array([[1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.]], dtype=np.float32),
        "Tr_velo2cam": np.array([[-6.4694546e-03, -9.9997908e-01, 1.6122404e-04, 7.2263610e-03],
                                 [-6.4202384e-03, -1.1968778e-04, -9.9997938e-01, 1.3716986e+00],
                                 [9.9995846e-01, -6.4703566e-03, -6.4193299e-03, -1.6153628e+00]],
                                dtype=np.float32)
    }
    points_np = np.array([[0.9066696, -4.154523, -0.35889602, 0.],
                          [1.1218058, -18.891792, 1.2602975, 0.],
                          [0.9580487, -6.608958, -0.263404, 0.],
                          [0.42476445, -8.463129, -0.11291444, 0.],
                          [-1.4754575, -45.643753, 1.5897664, 0.],
                          [-1.588751, -18.313736, 1.3740683, 0.],
                          [0.67852163, -11.053506, -0.13283014, 0.],
                          [-0.14888406, -13.569483, -0.14784884, 0.],
                          [0.35047734, -16.23028, -0.16158926, 0.],
                          [-2.2267098, -57.16784, 2.9225237, 0.]],
                         dtype=np.float32)

    points_np = points_np[:, :3]  # remove the intensity channel for later coordinate transformations
    points_torch = torch.Tensor(points_np)
    print((points_np == points_torch.numpy()).all())
    calib_np = Calibration(calib)
    calib_torch = Calibration_torch(calib)

    # check going from lidar to (image coordinates + rect depth)
    rect_np_forward = calib_np.lidar_to_rect(points_np)
    rect_torch_forward = calib_torch.lidar_to_rect(points_torch)
    print(np.isclose(rect_np_forward, rect_torch_forward.numpy()).all())

    img_coords_np, rect_depths_np = calib_np.lidar_to_img(points_np)
    img_coords_torch, rect_depths_torch = calib_torch.lidar_to_img(points_torch)
    print(np.isclose(img_coords_np, img_coords_torch.numpy()).all())
    print(np.isclose(rect_depths_np, rect_depths_torch.numpy()).all())

    # check going from (img coordinates + depth) to lidar
    rect_np = calib_np.img_to_rect(img_coords_np[:, 0], img_coords_np[:, 1], rect_depths_np)
    rect_torch = calib_torch.img_to_rect(img_coords_torch[:, 0], img_coords_torch[:, 1], rect_depths_torch)
    print(np.isclose(rect_np, rect_torch.numpy()).all())

    lidar_np = calib_np.rect_to_lidar(rect_np)
    lidar_torch = calib_torch.rect_to_lidar(rect_torch)
    print(np.isclose(points_np, lidar_np).all())
    print(np.isclose(points_torch, lidar_torch).all())

"""
python ~/BEVSEG/PCDet2/pcdet/utils/calibration.py
"""
