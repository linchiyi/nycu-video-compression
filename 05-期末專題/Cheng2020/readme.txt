Cheng2020 Image Compression Evaluation Script

專案目的

本程式用於評估 CompressAI 中 Cheng2020 Anchor 模型在不同 quality 設定下的影像壓縮效能。
-----

流程涵蓋：

單張影像的壓縮與解壓

Bitstream 與重建影像儲存

多種影像品質指標計算

輸出與 metrics.csv 相容的 long-table 格式結果
-----

功能總覽

1. 影像前處理

支援 .png / .jpg / .jpeg / .bmp

自動 padding 至 64 的倍數（符合 Cheng2020 架構需求）

解壓後自動 unpadding，恢復原始解析度

2. 壓縮與解壓

使用 compressai.zoo.cheng2020_anchor

依 quality 逐一建立模型並評估

儲存完整 bitstream（strings + shape + padding 資訊）

3. 評估指標

對每張重建影像計算：

PSNR（skimage）

SSIM（skimage）

MS-SSIM（pytorch-msssim）

LPIPS（AlexNet）

所有指標皆包在「安全計算」邏輯中：

若輸入含 NaN / Inf，會先修正

若仍計算失敗，該指標回傳 NaN，但不影響其他結果
-----

輸出內容

1. 重建影像
recon/
  imageName_cheng2020_anchor_qX.png

2. Bitstream
bitstreams/
  imageName_cheng2020_anchor_qX.pth


3. 評估結果 CSV（long table）
格式與常見 metrics.csv 相容，方便直接用於分析或報告。
-----

使用方式
python Cheng2020.py \
  --image_dir ./images \
  --out_csv metrics.csv \
  --qualities 1 2 3 4 5 6

重要參數

--image_dir：輸入影像資料夾（預期 15 張）

--qualities：Cheng2020 quality 設定

--expect_n：檢查圖片數量（預設 15）

環境需求

Python 3.x

PyTorch

compressai

lpips

pytorch-msssim

scikit-image

PIL, torchvision, numpy
