import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
import math
import json
import zlib
import matplotlib.pyplot as plt
import os

# ---------------------------
# 量化表（依題目圖做近似轉寫）
# 若你有精確數值，請直接修改下面的 q_table1, q_table2
# ---------------------------
q_table1 = np.array([
    [10, 7,  6, 10, 14, 24, 31, 37],
    [7,  7,  8, 11, 16, 35, 36, 33],
    [8,  8, 10, 14, 24, 34, 41, 34],
    [8, 10, 13, 17, 31, 52, 48, 37],
    [11,13, 22, 34, 41, 65, 62, 46],
    [14,21, 33, 38, 49, 62, 68, 55],
    [29,38, 47, 52, 62, 73, 72, 61],
    [43,55, 57, 59, 67, 60, 62, 59]
], dtype=np.float32)

q_table2 = np.array([
    [10,11,14,28,59,59,59,59],
    [11,13,16,40,59,59,59,59],
    [14,16,34,59,59,59,59,59],
    [28,40,59,59,59,59,59,59],
    [59,59,59,59,59,59,59,59],
    [59,59,59,59,59,59,59,59],
    [59,59,59,59,59,59,59,59],
    [59,59,59,59,59,59,59,59]
], dtype=np.float32)

# ---------------------------
# zigzag 與逆 zigzag indices
# ---------------------------
def zigzag_indices(n=8):
    # return list of (r,c) in zigzag order
    indices = []
    for s in range(2*n - 1):
        if s % 2 == 0:
            # down-left diagonal
            r_start = min(s, n-1)
            c_start = s - r_start
            r = r_start
            c = c_start
            while r >= 0 and c < n:
                indices.append((r,c))
                r -= 1
                c += 1
        else:
            # up-right diagonal
            c_start = min(s, n-1)
            r_start = s - c_start
            r = r_start
            c = c_start
            while c >= 0 and r < n:
                indices.append((r,c))
                r += 1
                c -= 1
    return indices

ZIGZAG_IDX = zigzag_indices(8)
# map (r,c) -> linear pos
RC_TO_POS = {rc: i for i, rc in enumerate(ZIGZAG_IDX)}

def block_to_zigzag(block):
    # block: 8x8
    flat = np.zeros(64, dtype=np.float32)
    for i, (r,c) in enumerate(ZIGZAG_IDX):
        flat[i] = block[r,c]
    return flat

def zigzag_to_block(flat):
    block = np.zeros((8,8), dtype=np.float32)
    for i, (r,c) in enumerate(ZIGZAG_IDX):
        block[r,c] = flat[i]
    return block

# ---------------------------
# 2D DCT / IDCT (separable) - 使用 type=2 DCT 正規化 'ortho'
# ---------------------------
def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(coeff):
    return idct(idct(coeff.T, norm='ortho').T, norm='ortho')

# ---------------------------
# Run-length encoding / decoding for one 8x8 block (after quantization)
# - We store DC then AC as RLE tuples
# - Format per block: { 'dc': int, 'ac': [(run, val), ..., ('EOB')] }
# - EOB represented as ('EOB',)
# ---------------------------
def rle_encode_block(qblock_flat):
    # qblock_flat: 64-length array in zigzag order (integers)
    out = []
    # DC
    dc = int(qblock_flat[0])
    # AC
    ac = []
    run = 0
    for k in range(1, 64):
        v = int(qblock_flat[k])
        if v == 0:
            run += 1
            if run == 16:
                # JPEG uses special code for run=16 of zeros (ZRL). We simply record it.
                ac.append((15, 0))  # represent as (15,0)
                run = 0
        else:
            ac.append((run, v))
            run = 0
    # end of block
    ac.append(('EOB',))
    return {'dc': dc, 'ac': ac}

def rle_decode_block(rle):
    flat = np.zeros(64, dtype=np.int32)
    flat[0] = int(rle['dc'])
    idx = 1
    for item in rle['ac']:
        if item[0] == 'EOB':
            break
        run, val = item
        # move idx by run zeros
        idx += run
        if idx < 64:
            flat[idx] = int(val)
            idx += 1
        else:
            break
    return flat.astype(np.float32)

# ---------------------------
# 影像處理主流程
# ---------------------------
def pad_to_multiple(img, block_size=8):
    h, w = img.shape
    ph = (block_size - (h % block_size)) % block_size
    pw = (block_size - (w % block_size)) % block_size
    if ph == 0 and pw == 0:
        return img, (0,0)
    new = np.pad(img, ((0, ph), (0, pw)), mode='constant', constant_values=0)
    return new, (ph,pw)

def process_with_quant_table(img_gray, qtable):
    h0, w0 = img_gray.shape
    img, pad = pad_to_multiple(img_gray, 8)
    H, W = img.shape
    blocks_vert = H // 8
    blocks_horz = W // 8

    rle_stream = []
    # store DCs for differential encoding optionally - here we store absolute
    # For reconstruction, we also keep block order information implicitly by stream order
    for by in range(blocks_vert):
        for bx in range(blocks_horz):
            y0 = by*8
            x0 = bx*8
            block = img[y0:y0+8, x0:x0+8].astype(np.float32) - 128.0  # level shift
            C = dct2(block)
            # quantize
            Q = np.round(C / qtable).astype(np.int32)
            # zigzag flatten
            flat = block_to_zigzag(Q)
            # RLE encode
            rle = rle_encode_block(flat)
            rle_stream.append(rle)

    # decode
    recon = np.zeros_like(img, dtype=np.float32)
    idx = 0
    for by in range(blocks_vert):
        for bx in range(blocks_horz):
            rle = rle_stream[idx]
            idx += 1
            flat_q = rle_decode_block(rle)
            qblock = zigzag_to_block(flat_q)
            # dequantize
            Cq = qblock.astype(np.float32) * qtable
            # inverse DCT
            block_rec = idct2(Cq) + 128.0
            y0 = by*8
            x0 = bx*8
            recon[y0:y0+8, x0:x0+8] = block_rec
    # crop padding
    ph, pw = pad
    if ph != 0:
        recon = recon[:-ph, :]
    if pw != 0:
        recon = recon[:, :-pw]
    # clip to [0,255]
    recon = np.clip(np.round(recon), 0, 255).astype(np.uint8)
    return recon, rle_stream

# ---------------------------
# 序列化 RLE 並取壓縮後大小（模擬編碼大小）
# ---------------------------
def rle_stream_size_bytes(rle_stream):
    # convert to JSON-friendly structure
    # to keep size small, store as list of [dc, [(run,val), ...]] with EOB as [-1]
    compact = []
    for r in rle_stream:
        ac_comp = []
        for item in r['ac']:
            if item[0] == 'EOB':
                ac_comp.append([-1])
            else:
                ac_comp.append([int(item[0]), int(item[1])])
        compact.append([int(r['dc']), ac_comp])
    s = json.dumps(compact, separators=(',',':')).encode('utf-8')
    compressed = zlib.compress(s, level=9)
    return len(compressed), len(s)

# ---------------------------
# main 
# ---------------------------
def run_pipeline(image_path):
    im = Image.open(image_path).convert('L')  # grayscale
    arr = np.array(im)
    print(f"原始影像大小: {arr.shape}")

    # Table 1
    recon1, stream1 = process_with_quant_table(arr, q_table1)
    bytes_z1, bytes_raw1 = rle_stream_size_bytes(stream1)
    print(f"Qtable1: RLE 種類數 (符號數) = {sum(1 + len(r['ac']) for r in stream1)} (block 數 = {len(stream1)})")
    print(f"Qtable1: RLE JSON 原始 bytes = {bytes_raw1}, zlib 壓縮後 bytes = {bytes_z1}")

    # Table 2
    recon2, stream2 = process_with_quant_table(arr, q_table2)
    bytes_z2, bytes_raw2 = rle_stream_size_bytes(stream2)
    print(f"Qtable2: RLE 種類數 (符號數) = {sum(1 + len(r['ac']) for r in stream2)} (block 數 = {len(stream2)})")
    print(f"Qtable2: RLE JSON 原始 bytes = {bytes_raw2}, zlib 壓縮後 bytes = {bytes_z2}")

    # 存檔比較
    Image.fromarray(recon1).save('recon_q1.png')
    Image.fromarray(recon2).save('recon_q2.png')
    print("重建影像已儲存： recon_q1.png, recon_q2.png")

    # 顯示原圖與兩張重建圖
    fig, axs = plt.subplots(1,3, figsize=(12,4))
    axs[0].imshow(arr, cmap='gray'); axs[0].set_title('Original Image')
    axs[1].imshow(recon1, cmap='gray'); axs[1].set_title(f'Reconstructed Q1 (zlib {bytes_z1} bytes)')
    axs[2].imshow(recon2, cmap='gray'); axs[2].set_title(f'Reconstructed Q2 (zlib {bytes_z2} bytes)')
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('comparison.png', dpi=150)
    plt.show()

    return {
        'q1': {'compressed_bytes': bytes_z1, 'raw_bytes': bytes_raw1, 'rle_symbols': sum(1+len(r['ac']) for r in stream1)},
        'q2': {'compressed_bytes': bytes_z2, 'raw_bytes': bytes_raw2, 'rle_symbols': sum(1+len(r['ac']) for r in stream2)}
    }

if __name__ == "__main__":
    img_path = 'lena.png'
    stats = run_pipeline(img_path)
    print("比較結果（供報告使用）：")
    print(stats)
