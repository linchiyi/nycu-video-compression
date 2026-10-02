import cv2
import numpy as np
import os

# 建立輸出資料夾存放八張結果圖片
os.makedirs("outputs", exist_ok=True)

# 讀取圖片
img = cv2.imread("lena.png").astype(np.float32)

# 分離 R, G, B
B = img[:, :, 0]
G = img[:, :, 1]
R = img[:, :, 2]

# 儲存灰階 R, G, B
cv2.imwrite("outputs/R.png", R)
cv2.imwrite("outputs/G.png", G)
cv2.imwrite("outputs/B.png", B)

# -----------------
# RGB → YUV
# -----------------
Y = 0.299 * R + 0.587 * G + 0.114 * B
U = -0.169 * R - 0.331 * G + 0.5 * B + 128
V = 0.5 * R - 0.419 * G - 0.081 * B + 128

# 限制範圍[0,255]
Y = np.clip(Y, 0, 255).astype(np.uint8)
U = np.clip(U, 0, 255).astype(np.uint8)
V = np.clip(V, 0, 255).astype(np.uint8)

cv2.imwrite("outputs/Y.png", Y)
cv2.imwrite("outputs/U.png", U)
cv2.imwrite("outputs/V.png", V)

# -----------------
# RGB → YCbCr
# -----------------
CbCr_matrix = np.array([[0.257, 0.504, 0.098],
                        [-0.148, -0.291, 0.439],
                        [0.439, -0.368, -0.071]], dtype=np.float32)

offset = np.array([16, 128, 128], dtype=np.float32)

# 展平成 (N,3)
h, w = R.shape
rgb = np.stack([R.flatten(), G.flatten(), B.flatten()], axis=1)

ycbcr = np.dot(rgb, CbCr_matrix.T) + offset

Y2 = ycbcr[:, 0].reshape(h, w)
Cb = ycbcr[:, 1].reshape(h, w)
Cr = ycbcr[:, 2].reshape(h, w)

Y2 = np.clip(Y2, 0, 255).astype(np.uint8)
Cb = np.clip(Cb, 0, 255).astype(np.uint8)
Cr = np.clip(Cr, 0, 255).astype(np.uint8)

cv2.imwrite("outputs/Cb.png", Cb)
cv2.imwrite("outputs/Cr.png", Cr)

print("成功輸出八張結果圖片")