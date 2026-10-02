import cv2
import numpy as np
import time
import math

# --- 1. 輔助函式 (Helper Functions) ---

def load_images():
    """載入兩個灰階影像"""
    try:
        ref_frame = cv2.imread('one_gray.png', cv2.IMREAD_GRAYSCALE).astype(float)
        curr_frame = cv2.imread('two_gray.png', cv2.IMREAD_GRAYSCALE).astype(float)
        if ref_frame is None or curr_frame is None:
            raise FileNotFoundError
        return ref_frame, curr_frame
    except FileNotFoundError:
        print("錯誤")
        exit()

def calculate_sad(block1, block2):
    """計算兩個區塊的 Sum of Absolute Differences (SAD)"""
    return np.sum(np.abs(block1 - block2))

def calculate_psnr(img1, img2):
    """計算兩個影像的 Peak Signal-to-Noise Ratio (PSNR)"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def motion_compensation(ref_frame, curr_frame, mv_map, block_size):
    """
    根據運動向量(MV)進行運動補償, 產生重建影格與殘差影格
    """
    height, width = ref_frame.shape
    reconstructed_frame = np.zeros_like(ref_frame)
    residual_frame = np.zeros_like(ref_frame)

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            # 取得目前區塊的運動向量
            dy, dx = mv_map[y // block_size, x // block_size]
            
            # 計算參考區塊的座標
            ref_y, ref_x = y + dy, x + dx
            
            # 取得當前區塊
            curr_block = curr_frame[y:y + block_size, x:x + block_size]
            
            # 取得參考區塊 (注意邊界)
            ref_block = ref_frame[ref_y:ref_y + block_size, ref_x:ref_x + block_size]
            
            # 填入重建影格
            reconstructed_frame[y:y + block_size, x:x + block_size] = ref_block
            
            # 計算殘差
            residual_frame[y:y + block_size, x:x + block_size] = curr_block - ref_block

    return reconstructed_frame, residual_frame

# --- 2. 運動估計演算法 (Motion Estimation Algorithms) ---

def full_search(ref_frame, curr_frame, block_size, search_range):
    """
    全域搜尋演算法 (Full Search Block Matching, FSBM)
    search_range (p): 搜尋範圍, 例如 8 代表 [+-8]
    """
    height, width = curr_frame.shape
    # MV Map 儲存 (dy, dx)
    mv_map = np.zeros((height // block_size, width // block_size, 2), dtype=int)

    for y in range(0, height - block_size + 1, block_size):
        for x in range(0, width - block_size + 1, block_size):
            
            curr_block = curr_frame[y:y + block_size, x:x + block_size]
            min_sad = float('inf')
            best_mv = (0, 0)

            # 在 search_range 內進行暴力搜尋
            for dy in range(-search_range, search_range + 1):
                for dx in range(-search_range, search_range + 1):
                    
                    ref_y, ref_x = y + dy, x + dx
                    
                    # 檢查邊界 (確保參考區塊在影像內)
                    if (ref_y < 0 or ref_y + block_size > height or
                        ref_x < 0 or ref_x + block_size > width):
                        continue
                        
                    ref_block = ref_frame[ref_y:ref_y + block_size, ref_x:ref_x + block_size]
                    
                    sad = calculate_sad(curr_block, ref_block)
                    
                    if sad < min_sad:
                        min_sad = sad
                        best_mv = (dy, dx)
            
            mv_map[y // block_size, x // block_size] = best_mv
            
    return mv_map

def three_step_search(ref_frame, curr_frame, block_size, search_range):
    """
    三步搜尋演算法 (Three-Step Search, TSS)
    search_range (p): 搜尋範圍, 例如 8. 初始 step 設為 p/2.
    """
    height, width = curr_frame.shape
    mv_map = np.zeros((height // block_size, width // block_size, 2), dtype=int)
    
    # 初始 step size (e.g., p=8, step=4)
    step = search_range // 2 

    for y in range(0, height - block_size + 1, block_size):
        for x in range(0, width - block_size + 1, block_size):
            
            curr_block = curr_frame[y:y + block_size, x:x + block_size]
            
            # (cy, cx) 是搜尋中心, (best_y, best_x) 是 MV
            cy, cx = y, x 
            best_mv = (0, 0)
            min_sad = float('inf')

            # 先計算 (0, 0) 的 SAD
            ref_block = ref_frame[y:y + block_size, x:x + block_size]
            min_sad = calculate_sad(curr_block, ref_block)

            current_step = step
            while current_step >= 1:
                # 測試 9 個點 (中心點已在上一輪算過)
                for dy in range(-current_step, current_step + 1, current_step):
                    for dx in range(-current_step, current_step + 1, current_step):
                        
                        if dy == 0 and dx == 0:
                            continue # 中心點不用重算
                        
                        # (ref_y, ref_x) 是 "搜尋中心" + "位移"
                        ref_y, ref_x = cy + dy, cx + dx
                        
                        # 檢查邊界
                        if (ref_y < 0 or ref_y + block_size > height or
                            ref_x < 0 or ref_x + block_size > width):
                            continue
                            
                        ref_block = ref_frame[ref_y:ref_y + block_size, ref_x:ref_x + block_size]
                        sad = calculate_sad(curr_block, ref_block)
                        
                        if sad < min_sad:
                            min_sad = sad
                            # 更新最佳 MV (相對於 (x,y) 的位移)
                            best_mv = (ref_y - y, ref_x - x)
                
                # 更新下一輪的搜尋中心
                cy = y + best_mv[0]
                cx = x + best_mv[1]
                
                # 縮小 step
                current_step //= 2
            
            mv_map[y // block_size, x // block_size] = best_mv
            
    return mv_map


# --- 3. Main  ---

if __name__ == "__main__":
    
    BLOCK_SIZE = 8
    
    print("載入影像...")
    ref_frame, curr_frame = load_images()
    print(f"影像尺寸: {ref_frame.shape}")
    print("-" * 30)

    # --- 作業要求 1 & 2: FSBM [+-8], MC, 儲存影像 ---
    print("執行 [FSBM, Search Range=8]...")
    p_8 = 8
    start_time = time.time()
    mv_map_fsbm_8 = full_search(ref_frame, curr_frame, BLOCK_SIZE, p_8)
    end_time = time.time()
    runtime_fsbm_8 = end_time - start_time
    
    recon_frame_8, residual_frame_8 = motion_compensation(ref_frame, curr_frame, mv_map_fsbm_8, BLOCK_SIZE)
    psnr_fsbm_8 = calculate_psnr(curr_frame, recon_frame_8)
    
    print(f"  執行時間: {runtime_fsbm_8:.4f} 秒")
    print(f"  PSNR: {psnr_fsbm_8:.4f} dB")
    
    # 儲存影像
    # 殘差影像範圍在 [-255, 255], 為了可視化, 加上 128
    residual_visual = cv2.normalize(residual_frame_8, None, 0, 255, cv2.NORM_MINMAX)
    
    cv2.imwrite("reconstructed_frame.png", recon_frame_8.astype(np.uint8))
    cv2.imwrite("residual_frame.png", residual_visual.astype(np.uint8))
    print("  已儲存 'reconstructed_frame.png' 和 'residual_frame.png'")
    print("-" * 30)

    # --- 作業要求 3: 比較不同 Search Range ---
    
    # FSBM [+-16]
    print("執行 [FSBM, Search Range=16]...")
    p_16 = 16
    start_time = time.time()
    mv_map_fsbm_16 = full_search(ref_frame, curr_frame, BLOCK_SIZE, p_16)
    end_time = time.time()
    runtime_fsbm_16 = end_time - start_time
    
    recon_frame_16, _ = motion_compensation(ref_frame, curr_frame, mv_map_fsbm_16, BLOCK_SIZE)
    psnr_fsbm_16 = calculate_psnr(curr_frame, recon_frame_16)
    
    print(f"  執行時間: {runtime_fsbm_16:.4f} 秒")
    print(f"  PSNR: {psnr_fsbm_16:.4f} dB")
    print("-" * 30)
    
    # FSBM [+-32]
    print("執行 [FSBM, Search Range=32]...")
    p_32 = 32
    start_time = time.time()
    mv_map_fsbm_32 = full_search(ref_frame, curr_frame, BLOCK_SIZE, p_32)
    end_time = time.time()
    runtime_fsbm_32 = end_time - start_time
    
    recon_frame_32, _ = motion_compensation(ref_frame, curr_frame, mv_map_fsbm_32, BLOCK_SIZE)
    psnr_fsbm_32 = calculate_psnr(curr_frame, recon_frame_32)
    
    print(f"  執行時間: {runtime_fsbm_32:.4f} 秒")
    print(f"  PSNR: {psnr_fsbm_32:.4f} dB")
    print("-" * 30)

    # --- 作業要求 4: 比較 TSS (使用 [+-8] 範圍) ---
    print("執行 [TSS, Search Range=8]...")
    start_time = time.time()
    # TSS 的 search_range 參數決定了初始 step (p/2)
    mv_map_tss_8 = three_step_search(ref_frame, curr_frame, BLOCK_SIZE, p_8)
    end_time = time.time()
    runtime_tss_8 = end_time - start_time
    
    recon_frame_tss, _ = motion_compensation(ref_frame, curr_frame, mv_map_tss_8, BLOCK_SIZE)
    psnr_tss_8 = calculate_psnr(curr_frame, recon_frame_tss)
    
    print(f"  執行時間: {runtime_tss_8:.4f} 秒")
    print(f"  PSNR: {psnr_tss_8:.4f} dB")
    print("-" * 30)

    # --- 總結報告數據 ---
    print("\n" + "=" * 60)
    print("                  實驗數據總結")
    print("=" * 60)
    
    print("\n表一：搜尋範圍比較 (Full Search Block Matching)")
    print("-" * 60)
    print(f"| {'Range':<10} | {'Runtime (s)':>15} | {'PSNR (dB)':>15} |")
    print("-" * 60)
    print(f"| {'[+-8]':<10} | {runtime_fsbm_8:15.4f} | {psnr_fsbm_8:15.4f} |")
    print(f"| {'[+-16]':<10} | {runtime_fsbm_16:15.4f} | {psnr_fsbm_16:15.4f} |")
    print(f"| {'[+-32]':<10} | {runtime_fsbm_32:15.4f} | {psnr_fsbm_32:15.4f} |")
    print("-" * 60)

    print("\n表二：演算法比較 (Search Range = [+-8])")
    print("-" * 60)
    print(f"| {'Algorithm':<10} | {'Runtime (s)':>15} | {'PSNR (dB)':>15} |")
    print("-" * 60)
    print(f"| {'FSBM':<10} | {runtime_fsbm_8:15.4f} | {psnr_fsbm_8:15.4f} |")
    print(f"| {'TSS':<10} | {runtime_tss_8:15.4f} | {psnr_tss_8:15.4f} |")
    print("-" * 60)
    
    # 計算加速比
    speedup = runtime_fsbm_8 / runtime_tss_8 if runtime_tss_8 > 0 else 0
    print(f"\n加速比 (FSBM/TSS): {speedup:.2f}x")
    print("=" * 60)