import os
import math
import csv
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from compressai.zoo import cheng2020_anchor
import lpips
from pytorch_msssim import ms_ssim
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_RECON_DIR = "recon"
DEFAULT_BITSTREAM_DIR = "bitstreams"


# ================================================================
# 工具：讀圖 / padding / unpadding
# ================================================================
def load_image(path: str):
    img = Image.open(path).convert("RGB")
    tensor = transforms.ToTensor()(img).unsqueeze(0)
    return img, tensor


def pad_to_64(x: torch.Tensor):
    _, _, H, W = x.size()
    pad_h = (64 - (H % 64)) % 64
    pad_w = (64 - (W % 64)) % 64
    padding = (0, pad_w, 0, pad_h)  # (left,right,top,bottom)
    x_padded = F.pad(x, padding, "constant", 0)
    return x_padded, padding


def unpad(x: torch.Tensor, padding):
    left, right, top, bottom = padding
    if bottom > 0:
        x = x[:, :, :-bottom, :]
    if right > 0:
        x = x[:, :, :, :-right]
    return x


# ================================================================
# 安全計算：盡量保住 PSNR；其他壞掉就 NaN
# ================================================================
def safe_calc_metrics(x: torch.Tensor, x_hat: torch.Tensor, lpips_fn):
    # 預設全部 NaN
    psnr = float("nan")
    ssim_val = float("nan")
    ms_val = float("nan")
    lp_val = float("nan")

    # 先確保張量有限值，避免整體指標爆炸
    if not torch.isfinite(x_hat).all():
        x_hat = torch.nan_to_num(x_hat, nan=0.0, posinf=1.0, neginf=0.0)

    # clamp 到 [0,1]，避免指標函式在 range 外不穩
    x_hat = x_hat.clamp(0.0, 1.0)

    # PSNR（優先保住）
    try:
        x_np = x.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        xhat_np = x_hat.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

        # 如果 numpy 裡還是有 NaN/Inf，PSNR 也會壞；再防一次
        if not np.isfinite(xhat_np).all():
            xhat_np = np.nan_to_num(xhat_np, nan=0.0, posinf=1.0, neginf=0.0)

        psnr = peak_signal_noise_ratio(x_np, xhat_np, data_range=1.0)
    except Exception:
        psnr = float("nan")

    # SSIM
    try:
        ssim_val = structural_similarity(
            x_np, xhat_np, data_range=1.0, channel_axis=2
        )
    except Exception:
        ssim_val = float("nan")

    # MS-SSIM
    try:
        ms_val = ms_ssim(x, x_hat, data_range=1.0).item()
    except Exception:
        ms_val = float("nan")

    # LPIPS
    try:
        lp_val = lpips_fn(x * 2 - 1, x_hat * 2 - 1).item()
    except Exception:
        lp_val = float("nan")

    return psnr, ssim_val, ms_val, lp_val


# ================================================================
# 單張：壓縮/解壓 + 儲存 bitstream + 儲存重建圖 + 計算指標
# ================================================================
def compress_and_evaluate_single(
    model,
    img_path: str,
    quality: int,
    lpips_fn,
    recon_dir: str,
    bitstream_dir: str,
    model_tag: str = "cheng2020_anchor",
):
    basename = os.path.splitext(os.path.basename(img_path))[0]

    _, x = load_image(img_path)
    x = x.to(DEVICE)
    _, _, H, W = x.size()

    x_padded, pad = pad_to_64(x)

    with torch.no_grad():
        comp = model.compress(x_padded)
        decomp = model.decompress(comp["strings"], comp["shape"])
        x_hat = decomp["x_hat"]

    x_hat = unpad(x_hat, pad)

    # compressed size (bytes)
    compressed_bytes = sum(len(s) for s in comp["strings"][0])
    compressed_bits = compressed_bytes * 8

    # bpp（分母用原圖像素數）
    num_pixels = H * W
    bpp = (compressed_bits / num_pixels) if num_pixels > 0 else float("nan")

    # save bitstream
    os.makedirs(bitstream_dir, exist_ok=True)
    bit_path = os.path.join(bitstream_dir, f"{basename}_{model_tag}_q{quality}.pth")
    torch.save(
        {
            "strings": comp["strings"],
            "shape": comp["shape"],
            "pad": pad,
            "orig_size_hw": (H, W),
            "quality": quality,
            "model": model_tag,
        },
        bit_path,
    )

    # save recon image（即使指標 NaN 也照存）
    os.makedirs(recon_dir, exist_ok=True)
    recon_img = transforms.ToPILImage()(x_hat.squeeze(0).detach().cpu().clamp(0, 1))
    recon_path = os.path.join(recon_dir, f"{basename}_{model_tag}_q{quality}.png")
    recon_img.save(recon_path)

    # metrics（盡量算 PSNR）
    psnr, ssim_val, ms_val, lp_val = safe_calc_metrics(x, x_hat, lpips_fn)

    original_size = W * H * 3
    compression_ratio = (original_size / compressed_bytes) if compressed_bytes > 0 else float("inf")

    row = {
        "image_name": basename,
        "reconstructed_name": os.path.splitext(os.path.basename(recon_path))[0],
        "method": f"{model_tag}_q{quality}",
        "width": W,
        "height": H,
        "psnr": psnr,
        "ssim": ssim_val,
        "ms_ssim": ms_val,
        "lpips": lp_val,
        "bpp": bpp,
        "original_size": original_size,
        "compressed_size": compressed_bytes,
        "compression_ratio": compression_ratio,
    }
    return row


# ================================================================
# 收集資料夾圖片
# ================================================================
def collect_images_from_folder(folder, exts=(".png", ".jpg", ".jpeg", ".bmp")):
    files = []
    for fn in sorted(os.listdir(folder)):
        if fn.lower().endswith(exts):
            files.append(os.path.join(folder, fn))
    return files


# ================================================================
# 寫出與 metrics.csv 相同格式（long table）
# ================================================================
def write_metrics_csv_like_metrics_csv(rows, out_csv_path: str):
    fieldnames = [
        "image_name",
        "reconstructed_name",
        "method",
        "width",
        "height",
        "psnr",
        "ssim",
        "ms_ssim",
        "lpips",
        "bpp",
        "original_size",
        "compressed_size",
        "compression_ratio",
    ]

    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[OK] Saved metrics.csv-format file: {out_csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="Folder containing the 15 input photos")
    parser.add_argument("--out_csv", type=str, default="metrics.csv", help="Output csv path (metrics.csv format)")
    parser.add_argument("--recon_dir", type=str, default=DEFAULT_RECON_DIR, help="Reconstructed image output folder")
    parser.add_argument("--bitstream_dir", type=str, default=DEFAULT_BITSTREAM_DIR, help="Bitstream output folder")
    parser.add_argument("--qualities", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6], help="Quality list")
    parser.add_argument("--expect_n", type=int, default=15, help="Expected number of images (default 15)")
    args = parser.parse_args()

    img_list = collect_images_from_folder(args.image_dir)
    print(f"[INFO] Found {len(img_list)} images in: {args.image_dir}")

    if args.expect_n is not None:
        assert len(img_list) == args.expect_n, f"Expected {args.expect_n} images, got {len(img_list)}"

    lpips_fn = lpips.LPIPS(net="alex").to(DEVICE).eval()

    all_rows = []
    model_tag = "cheng2020_anchor"

    for q in args.qualities:
        print(f"\n=== Evaluating {model_tag} at quality={q} ===")
        model = cheng2020_anchor(quality=q, metric="mse", pretrained=True)
        model = model.to(DEVICE).eval()
        if hasattr(model, "update"):
            model.update()

        for img_path in img_list:
            row = compress_and_evaluate_single(
                model=model,
                img_path=img_path,
                quality=q,
                lpips_fn=lpips_fn,
                recon_dir=args.recon_dir,
                bitstream_dir=args.bitstream_dir,
                model_tag=model_tag,
            )

            # 不再 skip：NaN 也照寫，方便後處理統一處理
            all_rows.append(row)

            # 額外印一下：PSNR 是否成功（你要「盡量算 psnr」）
            if math.isnan(row["psnr"]):
                print(f"[WARN] PSNR NaN: {row['method']} | {row['image_name']}")

    write_metrics_csv_like_metrics_csv(all_rows, args.out_csv)


if __name__ == "__main__":
    main()
