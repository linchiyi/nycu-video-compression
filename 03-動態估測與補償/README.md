# HW3 — 運動估計與補償 (Motion Estimation & Compensation)

本專案為影片壓縮 (Video Compression) 課程之作業實作，提供兩種區塊基底的運動估計（block matching）方法，並展示運動補償後的重建結果與品質評估：

- 全域搜尋 (Full Search Block Matching, FSBM)
- 三步搜尋 (Three-Step Search, TSS)

主要程式：`motion_estimation.py`。此程式讀取兩張灰階影像（參考影格與當前影格），執行運動估計、做運動補償，並輸出 PSNR 與執行時間統計。

## 快速開始

先安裝必要套件（推薦在虛擬環境中執行）：

```bash
pip install opencv-python-headless numpy
```

執行：

```bash
python3 motion_estimation.py
```

程式會在目前目錄尋找：`one_gray.png`（reference frame）和 `two_gray.png`（current frame）。執行後會輸出：

- `reconstructed_frame.png`：由參考影格與估到的運動向量重建的影格（FSBM, p=8 範例）
- `residual_frame.png`：殘差影像（為了可視化會做範圍正規化）

程式也會在終端印出每組實驗的執行時間與 PSNR 值（預設實驗包含：FSBM p=[8,16,32]；TSS p=8）。

## 檔案說明

- `motion_estimation.py` — 主程式與所有核心實作：載入影像、SAD、FSBM、TSS、運動補償（MC）、PSNR 計算與結果輸出。
- `one_gray.png`, `two_gray.png` — 範例輸入影格（放在同一目錄）。
- `reconstructed_frame.png`, `residual_frame.png` — 程式執行後會生成的輸出檔案。

## 程式內可調參數（程式中常數）

- `BLOCK_SIZE`：預設為 8（區塊大小）。在 `motion_estimation.py` 中可調整以測試不同粒度。
- Search Range p：在程式中範例用了 8、16、32（FSBM）以及 p=8（TSS）。FSBM 的計算量會隨 p^2 上升，TSS 為快速近似法。

若要修改參數（例如使用不同影像或 block size），直接在 `motion_estimation.py` 中修改常數或擴充為命令列參數。

## 實作重點

- 影像載入：`load_images()` 使用 `cv2.imread(..., cv2.IMREAD_GRAYSCALE)`，回傳 float 陣列。若讀不到檔案會印錯誤並退出。
- SAD：`calculate_sad(block1, block2)` 以 `np.sum(np.abs(...))` 實作。
- MV 結構：`mv_map` 為 (H//B, W//B, 2) 的整數陣列，儲存每個區塊的 (dy, dx)。
- 邊界處理：在搜尋時會跳過導致參考區塊超出影像邊界的位移。

FSBM 為暴力搜尋（所有 dy,dx ∈ [-p, p]），TSS 則以 p/2 為初始 step，逐步減半搜尋候選點。
