# 視訊壓縮課程作業

本 repository 整理視訊壓縮課程中的程式作業與報告，內容涵蓋影像色彩表示、DCT、動態估測與熵編碼。

## 作業總覽

| 作業 | 主題 | 主要內容 |
| --- | --- | --- |
| [HW1｜色彩空間轉換](./01-色彩空間轉換/) | RGB、YUV、YCbCr | 以公式完成色彩轉換，輸出八張灰階影像 |
| [HW2｜二維離散餘弦轉換](./02-二維離散餘弦轉換/) | 2D-DCT / IDCT | 比較直接二維公式與可分離的兩次一維 DCT |
| [HW3｜動態估測與補償](./03-動態估測與補償/) | Motion Estimation | 實作 FSBM 與 TSS，完成運動補償與 PSNR 分析 |
| [HW4｜熵編碼](./04-熵編碼/) | JPEG-like pipeline | 串接 DCT、量化、Zigzag、RLE、反量化與 IDCT |

## 報告

| 報告 | 檔案 |
| --- | --- |
| HW1 | [HW1_Report.pdf](./報告/HW1_Report.pdf) |
| HW2 | [HW2_Report.pdf](./報告/HW2_Report.pdf) |
| HW3 | [HW3_Report.pdf](./報告/HW3_Report.pdf) |
| HW4 | [HW4_Report.pdf](./報告/HW4_Report.pdf) |

## 技術

- Python
- NumPy、OpenCV、Pillow、SciPy、Matplotlib
- DCT / IDCT、量化、Zigzag scan、Run-Length Encoding
- Full Search Block Matching、Three-Step Search
- PSNR、SSIM、LPIPS、Rate–Distortion analysis

## 公開版本說明

本 repository 目前整理 HW1–HW4 的程式碼、範例輸入輸出與報告 PDF；原本的期末專案已移除。影片、模型權重、虛擬環境、壓縮 bitstream 與大型產出檔未放入 repository。
