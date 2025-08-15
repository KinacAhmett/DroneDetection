# Codes/tracking_wrapper/temporal_attention.py
from __future__ import annotations
import math, time
from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class TAState:
    # Pixel uzayında ham kutu (x,y,w,h) ve normalize edilmiş feature vektörü
    box_px: Tuple[float, float, float, float]
    feat: torch.Tensor  # [d_in]
    t: float

def _pos_encoding(L: int, d_model: int, device):
    pos = torch.arange(L, device=device).unsqueeze(1)  # [L,1]
    div = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(L, d_model, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe  # [L, d_model]

class TemporalAttention(nn.Module):
    """
    Hafif zaman boyutu dikkat katmanı.
    - Geçmiş N kutuyu (merkez, boyut, hız, alan, en-boy oranı, optik akış, skor) vektöre çevirir
    - Çok başlı dikkat ile geçmişten bağlam alır
    - Çıktı: mevcut kutuyu geçmişle karıştırılmış "rafine" box
    """
    def __init__(self, buffer_len: int = 30, d_in: int = 11, d_model: int = 64, heads: int = 4, device: Optional[torch.device] = None):
        super().__init__()
        self.buffer_len = buffer_len
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.enc = nn.Linear(d_in, d_model)
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.mix = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 5)  # 4 delta + 1 alpha
        )
        self.register_buffer("_dummy", torch.zeros(1))  # cihaz izlemek için
        self.hist: List[TAState] = []

    def _to_feat(self, box_px, img_shape, prev_box_px=None, oflow=0.0, score=0.5):
        # img_shape: (H, W, C) veya (H, W)
        H, W = img_shape[0], img_shape[1]
        x,y,w,h = box_px
        # normalize: [0,1] aralığı
        cx = (x + w/2) / max(W,1)
        cy = (y + h/2) / max(H,1)
        nw = w / max(W,1)
        nh = h / max(H,1)
        area = (w*h) / float(max(W*H,1))
        ar = (w+1e-6) / (h+1e-6)

        if prev_box_px is None:
            vx=vy=vw=vh=0.0
        else:
            px,py,pw,ph = prev_box_px
            pcx = (px+pw/2)/max(W,1); pcy = (py+ph/2)/max(H,1)
            pnw = pw/max(W,1); pnh = ph/max(H,1)
            vx = cx - pcx
            vy = cy - pcy
            vw = nw - pnw
            vh = nh - pnh

        feat = torch.tensor([cx,cy,nw,nh,area,ar,vx,vy,vw,vh,oflow], dtype=torch.float32, device=self._dummy.device)
        # skoru istersen d_in'e ekleyebilirsin; şimdilik oflow var, skor ağırlığı blend'e giriyor.
        return feat

    def push(self, box_px, img_shape, oflow: float = 0.0, score: float = 0.5):
        prev = self.hist[-1].box_px if self.hist else None
        feat = self._to_feat(box_px, img_shape, prev, oflow, score)
        self.hist.append(TAState(box_px=tuple(float(v) for v in box_px), feat=feat, t=time.time()))
        if len(self.hist) > self.buffer_len:
            self.hist = self.hist[-self.buffer_len:]

    def reset(self):
        self.hist = []

    @torch.no_grad()
    def refine(self, cur_box_px, img_shape, oflow: float = 0.0, score: float = 0.5):
        """
        cur_box_px: (x,y,w,h) piksel uzayında
        return: refined_box_px (x,y,w,h), alpha_used (0-1), attn_conf
        """
        if len(self.hist) < 2:
            # yumuşatma için yeter veri yok
            return cur_box_px, 0.0, 0.0

        prev_box_px = self.hist[-1].box_px
        cur_feat = self._to_feat(cur_box_px, img_shape, prev_box_px, oflow, score).unsqueeze(0).unsqueeze(0)  # [1,1,d_in]
        hist_feats = torch.stack([h.feat for h in self.hist], dim=0).unsqueeze(0)  # [1, L, d_in]

        # encode + pozisyon
        Q = self.enc(cur_feat)           # [1,1,d_model]
        K = self.enc(hist_feats)         # [1,L,d_model]
        pe = _pos_encoding(K.shape[1], K.shape[2], K.device).unsqueeze(0)
        K = K + pe                       # anahtara pos. ekle
        V = K.clone()

        # MHA
        ctx, attn = self.mha(Q, K, V, need_weights=True)  # ctx: [1,1,d_model], attn: [1,1,L]
        out = self.mix(ctx.squeeze(0).squeeze(0))         # [5]
        dx, dy, dw, dh, alpha = out[0].item(), out[1].item(), out[2].item(), out[3].item(), torch.sigmoid(out[4]).item()

        # tarihsel box'ların dikkat ağırlıklı ortalaması (normalize değil, piksel)
        weights = F.softmax(attn.squeeze(0).squeeze(0), dim=-1)  # [L]
        H, W = img_shape[0], img_shape[1]
        hx = sum(weights[i].item() * self.hist[i].box_px[0] for i in range(len(self.hist)))
        hy = sum(weights[i].item() * self.hist[i].box_px[1] for i in range(len(self.hist)))
        hw = sum(weights[i].item() * self.hist[i].box_px[2] for i in range(len(self.hist)))
        hh = sum(weights[i].item() * self.hist[i].box_px[3] for i in range(len(self.hist)))

        # delta'yı makul skala ile sınırla (gürültü patlamasını engelle)
        # burada delta normalize varsayıldığı için görüntü boyutuyla ölçekleyelim
        scale_x, scale_y = max(W,1)*0.05, max(H,1)*0.05
        scale_w, scale_h = max(W,1)*0.05, max(H,1)*0.05
        x,y,w,h = cur_box_px
        cand_from_delta = (
            x + max(min(dx*scale_x, +0.25*W), -0.25*W),
            y + max(min(dy*scale_y, +0.25*H), -0.25*H),
            max(2.0, w + max(min(dw*scale_w, +0.25*W), -0.25*W)),
            max(2.0, h + max(min(dh*scale_h, +0.25*H), -0.25*H)),
        )

        # iki aday: (1) delta ile güncellenmiş kutu, (2) geçmiş ağırlıklı ortalama
        # alpha ile karıştıralım; optik akış düşükse alpha'yı azalt
        motion_gate = 1.0 / (1.0 + math.exp(- (oflow - 0.7)))  # ~0-1
        alpha_eff = float(alpha) * float(motion_gate)

        mixed = (
            alpha_eff * cand_from_delta[0] + (1-alpha_eff) * hx,
            alpha_eff * cand_from_delta[1] + (1-alpha_eff) * hy,
            alpha_eff * cand_from_delta[2] + (1-alpha_eff) * hw,
            alpha_eff * cand_from_delta[3] + (1-alpha_eff) * hh,
        )

        # box'ı ekrana taşırma
        mx = max(0.0, min(mixed[0], W-2))
        my = max(0.0, min(mixed[1], H-2))
        mw = max(2.0, min(mixed[2], W - mx))
        mh = max(2.0, min(mixed[3], H - my))
        attn_conf = float(weights.max().item())
        return (int(mx), int(my), int(mw), int(mh)), alpha_eff, attn_conf
