import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from tqdm import tqdm

def get_dct_coeffs(N):
    """預先計算 DCT 的 C(k) 縮放係數"""
    c = np.ones(N) * np.sqrt(2 / N)
    c[0] = np.sqrt(1 / N)
    return c

# ---------------------------------------------------------
# 2D DCT 與 IDCT 
# ---------------------------------------------------------

def dct_2d(image):
    """公式實現 2D-DCT"""
    N, M = image.shape
    dct_matrix = np.zeros((N, M))
    c_u = get_dct_coeffs(N)
    c_v = get_dct_coeffs(M)

    # 預先計算餘弦項以加速
    cos_u = np.zeros((N, N))
    cos_v = np.zeros((M, M))
    for i in range(N):
        for u in range(N):
            cos_u[i, u] = np.cos((2 * i + 1) * u * np.pi / (2 * N))
    for j in range(M):
        for v in range(M):
            cos_v[j, v] = np.cos((2 * j + 1) * v * np.pi / (2 * M))

    print(f"\n計算 {N}x{M} 的直接 2D-DCT")
    for u in tqdm(range(N), desc="DCT - 處理 u", ncols=100):
        for v in range(M):
            # 使用 NumPy 的向量化操作加速內層迴圈
            sum_val = np.sum(image * cos_u[:, u].reshape(-1, 1) * cos_v[:, v])
            dct_matrix[u, v] = c_u[u] * c_v[v] * sum_val
    return dct_matrix

def idct_2d(dct_matrix):
    """公式實現 2D-IDCT"""
    N, M = dct_matrix.shape
    image = np.zeros((N, M))
    c_u = get_dct_coeffs(N)
    c_v = get_dct_coeffs(M)

    # 預先計算餘弦項
    cos_u = np.zeros((N, N))
    cos_v = np.zeros((M, M))
    for i in range(N):
        for u in range(N):
            cos_u[i, u] = np.cos((2 * i + 1) * u * np.pi / (2 * N))
    for j in range(M):
        for v in range(M):
            cos_v[j, v] = np.cos((2 * j + 1) * v * np.pi / (2 * M))

    print("\n執行 2D-IDCT 重建影像...")
    for i in tqdm(range(N), desc="IDCT - 重建 i", ncols=100):
        for j in range(M):
            # 準備係數矩陣
            coeffs_matrix = c_u.reshape(-1, 1) * c_v * dct_matrix
            # 向量化計算
            sum_val = np.sum(coeffs_matrix * cos_u[i, :].reshape(-1, 1) * cos_v[j, :])
            image[i, j] = sum_val
    return image


# ---------------------------------------------------------
# 1D DCT 與 Two 1D-DCT 
# ---------------------------------------------------------

def dct_1d(vector):
    """實現 1D-DCT"""
    N = len(vector)
    result = np.zeros(N)
    c = get_dct_coeffs(N)
    for k in range(N):
        n = np.arange(N)
        cos_terms = np.cos((2 * n + 1) * k * np.pi / (2 * N))
        sum_val = np.sum(vector * cos_terms)
        result[k] = c[k] * sum_val
    return result

def two_1d_dct(image):
    """使用Two 1D-DCT 實現 2D-DCT"""
    N, M = image.shape
    intermediate = np.zeros((N, M))

    # 對每一列進行 1D-DCT
    for i in tqdm(range(N), desc="Two 1D-DCT (each row)", ncols=100):
        intermediate[i, :] = dct_1d(image[i, :])

    final_dct = np.zeros((N, M))
    # 對每一行進行 1D-DCT
    for j in tqdm(range(M), desc="Two 1D-DCT (each column)", ncols=100):
        final_dct[:, j] = dct_1d(intermediate[:, j])

    return final_dct

# ---------------------------------------------------------
# Evaluate the PSNR
# ---------------------------------------------------------

def calculate_psnr(img1, img2):
    """Evaluate the PSNR"""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(255.0 / math.sqrt(mse))
    return psnr

# ---------------------------------------------------------
# Main Function
# ---------------------------------------------------------

def main():
    # 讀取圖片
    img = cv2.imread('lena.png')
    img_gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray_orig = img_gray_orig.astype(np.float32)

    orig_N, orig_M = img_gray_orig.shape
    print(f"原始影像大小：{orig_N}x{orig_M}")

    # --- User決定要分析的尺寸 ---
    try:
        size_input = input(f"自行輸入要分析的影像大小，按 Enter 使用原始大小 {orig_N}）：").strip()
        if size_input == "":
            analysis_size = orig_N
        else:
            analysis_size = int(size_input)
        
        if analysis_size <= 0:
            analysis_size = orig_N
            print(f"輸入無效，使用原始大小 {orig_N}x{orig_M}。")

    except ValueError:
        analysis_size = orig_N
        print(f"輸入錯誤，使用原始大小 {orig_N}x{orig_M}。")

    # --- 根據User選擇調整圖片 ---
    if analysis_size == orig_N:
        img_gray = img_gray_orig.copy()
    else:
        print(f"影像已縮放至 {analysis_size}x{analysis_size} 分析。")
        img_gray = cv2.resize(img_gray_orig, (analysis_size, analysis_size))

    N, M = img_gray.shape

    # --- 判斷是否執行2D-DCT ---
    run_slow = True
    if N > 128:
        choice = input(f"分析尺寸 {N}x{M} 較大，2D-DCT 可能需數分鐘。是否要執行？(y/n): ").strip().lower()
        if choice != 'y':
            run_slow = False

    # ---------------------------------------------------------
    # Part 1：2D-DCT
    # ---------------------------------------------------------
    if run_slow:
        print("\n=== Part 1：2D-DCT ===")
        start_time_2d = time.time()
        dct_coeffs_2d = dct_2d(img_gray)
        end_time_2d = time.time()
        runtime_2d = end_time_2d - start_time_2d

        reconstructed_img = idct_2d(dct_coeffs_2d)
        psnr_val = calculate_psnr(img_gray, reconstructed_img)
    else:
        print("\n已跳過2D-DCT。")
        runtime_2d = 0
        psnr_val = float('inf') # 假設未執行就是完美重建
        reconstructed_img = img_gray.copy() # 讓圖像能正常顯示

    # ---------------------------------------------------------
    # Part 2：Two 1D-DCT
    # ---------------------------------------------------------
    print("\n=== Part 2：Two 1D-DCT ===")
    start_time_fast = time.time()
    dct_coeffs_fast = two_1d_dct(img_gray)
    end_time_fast = time.time()
    runtime_fast = end_time_fast - start_time_fast

    # ---------------------------------------------------------
    # Results Summary and Visualization
    # ---------------------------------------------------------
    print("\n" + "="*20 + " Results " + "="*20)
    print(f"分析影像尺寸：{N}x{M}")
    if run_slow:
        print(f"2D-DCT 執行時間：{runtime_2d:.4f} 秒")
        print(f"重建圖像的 PSNR：{psnr_val:.4f} dB")
    else:
        print("2D-DCT：未執行")

    print(f"Two 1D-DCT 執行時間：{runtime_fast:.4f} 秒")
    print("="*52)

    
    plt.figure(figsize=(15, 5))
    plt.suptitle(f'DCT Analysis (Image Size): {N}x{M})', fontsize=16)

    plt.subplot(1, 3, 1)
    plt.imshow(img_gray, cmap='gray')
    plt.title('Gray Scale Image for Analysis')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    dct_log_scaled = np.log(1 + np.abs(dct_coeffs_fast))
    plt.imshow(dct_log_scaled, cmap='gray')
    plt.title('DCT Coefficients (Log Scale)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(np.clip(reconstructed_img, 0, 255), cmap='gray')
    plt.title(f'Reconstructed Image (PSNR: {psnr_val:.2f} dB)')
    plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    main()