import cv2
import numpy as np
import torch
from einops import rearrange

from models.video_mae_vqa import CellRunningMaskAgent

if __name__ == '__main__':
    video_path = "/data/ly/VQA_ODV/Group1/G1AbandonedKingdom_ERP_7680x3840_fps30_qp27_45406k.mp4"
    videoCapture = cv2.VideoCapture(video_path)

    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fNUMS = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)

    # 读帧
    success, frame = videoCapture.read()
    videoCapture.release()

    # GMS框生成
    fragments_h = 7
    fragments_w = 7
    fsize_h = 32
    fsize_w = 32
    # 采样图片的高
    size_h = fragments_h * fsize_h
    # 采样图片的长
    size_w = fragments_w * fsize_w
    img = frame
    img = rearrange(img, 'h w c -> c h w ')
    res_h, res_w = img.shape[-2:]
    size = size_h, size_w

    ## make sure that sampling will not run out of the picture
    hgrids = torch.LongTensor(
        [min(res_h // fragments_h * i, res_h - fsize_h) for i in range(fragments_h)]
    )
    wgrids = torch.LongTensor(
        [min(res_w // fragments_w * i, res_w - fsize_w) for i in range(fragments_w)]
    )
    hlength, wlength = res_h // fragments_h, res_w // fragments_w
    if hlength > fsize_h:
        rnd_h = torch.randint(
            hlength - fsize_h, (len(hgrids), len(wgrids), 8)
        )
    else:
        rnd_h = torch.zeros((len(hgrids), len(wgrids)).int())
    if wlength > fsize_w:
        rnd_w = torch.randint(
            wlength - fsize_w, (len(hgrids), len(wgrids), 8)
        )
    else:
        rnd_w = torch.zeros((len(hgrids), len(wgrids)).int())

    target_img = torch.zeros((3, 224, 224))
    img = torch.tensor(img)

    for i, hs in enumerate(hgrids):
        for j, ws in enumerate(wgrids):
            h_s, h_e = i * fsize_h, (i + 1) * fsize_h
            w_s, w_e = j * fsize_w, (j + 1) * fsize_w
            h_so, h_eo = hs + rnd_h[i][j][0], hs + rnd_h[i][j][0] + fsize_h
            w_so, w_eo = ws + rnd_w[i][j][0], ws + rnd_w[i][j][0] + fsize_w
            target_img[:, h_s:h_e, w_s:w_e] = img[:, h_so:h_eo, w_so:w_eo]

    point_color = (0, 0, 255)  # BGR
    thickness = 10
    lineType = 4
    img = rearrange(img, 'c h w -> h w c ')
    img = img.numpy()
    for i, hs in enumerate(hgrids):
        for j, ws in enumerate(wgrids):
            h_s, h_e = i * fsize_h, (i + 1) * fsize_h
            w_s, w_e = j * fsize_w, (j + 1) * fsize_w
            h_so, h_eo = hs + rnd_h[i][j][0], hs + rnd_h[i][j][0] + fsize_h
            w_so, w_eo = ws + rnd_w[i][j][0], ws + rnd_w[i][j][0] + fsize_w
            ptLeftTop = (int(w_so), int(res_h - h_eo))
            ptRightBottom = (int(w_eo), int(res_h - h_so))
            print(ptLeftTop, ptRightBottom)
            cv2.rectangle(img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)

    target_img = rearrange(target_img, 'c h w -> h w c ')
    target_img = target_img.numpy()

    cv2.imwrite('1.png', img.astype('uint8'))
    cv2.imwrite('2.png', target_img.astype('uint8'))

    # MAE
    mask = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ]
    mask = torch.tensor(mask)
    img_mask = mask.repeat_interleave(16,dim=0).repeat_interleave(16,dim=1)
    img = torch.tensor(target_img)

    overlay = img.clone()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img_mask[i][j] == 0:
                overlay[i][j] = torch.zeros_like(img[i][j])

    img = img.numpy()
    overlay = overlay.numpy()
    print(img.shape)
    print(overlay.shape)
    img = cv2.addWeighted(overlay, 0.7, img, 0.3,
                    0)
    cv2.imwrite('3.png', img.astype('uint8'))
    img_mask = img_mask
    img = img[img_mask==1].reshape(112,112,3)
    print(img.shape)
    cv2.imwrite('4.png', img.astype('uint8'))
