import torch
import cv2
import numpy as np

def apply_fusion(
    fusion_mode: str,
    unfused_tensor:
    torch.Tensor, dim: int
) -> torch.Tensor:
    # for donut
    # decoder num attention-head = 16, decoder num layers = 4
    # len(decoder_output.cross_attentions[0]) = 4
    # decoder_output.cross_attentions[0][0].shape = torch.Size([1, 16, 1, 1200])
    if fusion_mode == "mean":
        fused_tensor = torch.mean(unfused_tensor, dim=dim)
    elif fusion_mode == "max":
        fused_tensor = torch.max(unfused_tensor, dim=dim)[0]
    elif fusion_mode == "min":
        fused_tensor = torch.min(unfused_tensor, dim=dim)[0]
    else:
        raise NotImplementedError(f"{fusion_mode} fusion not supported")
    return fused_tensor

def attn_heatmap(tkn_indices, decoder_cross_attentions, final_h=2560, final_w=1920, heatmap_h=80, heatmap_w=60, discard_ratio=0.99, return_thres_agg_heatmap=True):
    head_fusion_type = ["mean", "max", "min"][1]
    layer_fusion_type = ["mean", "max", "min"][1]
    agg_heatmap = np.zeros([final_h, final_w], dtype=np.uint8)
    
    for tidx in tkn_indices:
        hmaps = torch.stack(decoder_cross_attentions[tidx], dim=0)
        # shape [4, 1, 16, 1, 1200]->[4, 16, 1200]
        hmaps = hmaps.permute(1, 3, 0, 2, 4).squeeze(0)
        hmaps = hmaps[-1]
        # change shape [4, 16, 1200]->[4, 16, 40, 30] assuming (heatmap_h, heatmap_w) = (40, 30)
        hmaps = hmaps.view(4, 16, heatmap_h, heatmap_w)
        # fusing 16 decoder attention heads i.e. [4, 16, 40, 30]-> [16, 40, 30]
        hmaps = apply_fusion(head_fusion_type, hmaps, dim=1)
        # fusing 4 decoder layers from BART i.e. [16, 40, 30]-> [40, 30]
        hmap = apply_fusion(layer_fusion_type, hmaps, dim=0)

        # dropping discard ratio activations
        flat = hmap.view(heatmap_h * heatmap_w)
        _, indices = flat.topk(int(flat.size(-1) * discard_ratio), largest=False)
        flat[indices] = 0
        hmap = flat.view(heatmap_h, heatmap_w)

        hmap = hmap.unsqueeze(dim=-1).cpu().numpy()
        hmap = (hmap * 255.).astype(np.uint8)  # (40, 30, 1) uint8
        # fuse heatmaps for different tokens by taking the max
        agg_heatmap = np.max(np.asarray([agg_heatmap, cv2.resize(hmap, (final_w, final_h))]), axis=0).astype(np.uint8)

        # threshold to remove small attention pockets
        thres_heatmap = cv2.threshold(agg_heatmap, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # Find contours
        contours = cv2.findContours(thres_heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        bboxes = [cv2.boundingRect(ctr) for ctr in contours]
        # return box with max area
        x, y, w, h = max(bboxes, key=lambda box: box[2] * box[3])
        max_area_box = [x, y, x + w, y + h]
        if return_thres_agg_heatmap:
            return max_area_box, thres_heatmap, agg_heatmap
        return max_area_box
