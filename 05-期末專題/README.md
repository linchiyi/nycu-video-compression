# 期末專題｜影像壓縮模型比較

## 專題目標

比較傳統 JPEG 與多種學習式影像壓縮方法在不同位元率下的表現，並以統一流程整理實驗結果。

目前整理的比較方法包括：

- JPEG
- Cheng2020 Anchor（CompressAI）
- MLIC++
- HiFiC

## 評估指標

- PSNR
- SSIM
- LPIPS
- Bits per Pixel（BPP）
- Rate–Distortion curve

統計摘要位於 analysis/analysis_results.txt；完整表格與繪圖程式位於 analysis/。

## 主要檔案

- jpeg.py：JPEG 基線壓縮與重建流程
- Cheng2020/Cheng2020.py：Cheng2020 Anchor 的壓縮與評估
- evaluation.py：統一計算影像品質指標
- analysis/plot.py：整理 CSV 並繪製比較圖
- MLIC/MLIC++/：MLIC++ 模型相關程式
- high-fidelity-generative-compression/：HiFiC 相關程式與原專案授權

## 注意事項

模型權重、原始資料集、壓縮 bitstream、影片與大型實驗輸出沒有放入公開版本；若要重現完整實驗，需要依各方法的 README 另外準備環境與資料。

