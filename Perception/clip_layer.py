import torch
import clip
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

# The script is responsible for the clip model and matching functions

def encode_and_match(cropped_images, task_objects, device, return_scores=False):
    """
    For each object in task_objects, compute cosine similarity against each cropped image,
    then perform a one-to-one matching that maximizes total similarity.

    Args:
      cropped_images: list of H×W×3 NumPy arrays (RGB crops).
      task_objects:   list of object names (strings) to match.
      device:         torch.device on which to run CLIP.
      return_scores:  bool; if True, also return a list of per-object confidences.

    Returns:
      - If return_scores=False: a list of indices (one per object), where each index
        is the matched crop index or None if no valid match.
      - If return_scores=True: (best_indices, confidences), where confidences are
        the maximum cosine similarities in [0, 1].
    """
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    N = len(task_objects)
    M = len(cropped_images)

    # 1) Encode and normalize all text features
    text_feats = []
    for obj in task_objects:
        prompt = f"Pick up the {obj.strip()}"
        text_tokens = clip.tokenize([prompt]).to(device)
        with torch.no_grad():
            txt_emb = clip_model.encode_text(text_tokens)  # [1, D]
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        text_feats.append(txt_emb)

    # 2) Encode and normalize all image features
    image_feats = []
    for img in cropped_images:
        pil_image = Image.fromarray(img)
        inp = preprocess(pil_image).unsqueeze(0).to(device)  # [1, 3, H, W]
        with torch.no_grad():
            img_emb = clip_model.encode_image(inp)  # [1, D]
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        image_feats.append(img_emb)

    # 3) Build similarity matrix S (N x M)
    S = np.zeros((N, M), dtype=np.float32)
    for i, txt_emb in enumerate(text_feats):
        if M == 0:
            S[i, :] = 0.0
            continue

        sims = []
        for img_emb in image_feats:
            sim = (txt_emb @ img_emb.T).detach().squeeze().cpu().item()
            sims.append(sim)
        sims = np.array(sims, dtype=np.float32)

        # clamp -inf/NaN to 0.0
        sims[np.isneginf(sims)] = 0.0
        sims[np.isnan(sims)] = 0.0

        S[i, :] = sims

    best_match_indices = [None] * N
    confidences = [0.0] * N

    if M == 0:
        # No masks: leave all matches as None and confidences as 0.0
        pass
    else:
        # 4a) If Hungarian is available, run it to maximize total similarity
        if linear_sum_assignment is not None:
            cost = -S.copy()
            row_ind, col_ind = linear_sum_assignment(cost)
            for i, j in zip(row_ind, col_ind):
                sim = S[i, j]
                if sim > 0.0:
                    best_match_indices[i] = int(j)
                    confidences[i] = float(sim)
                else:
                    best_match_indices[i] = None
                    confidences[i] = 0.0
        else:
            # 4b) Fallback: global greedy sort of all (i, j, S[i,j])
            triples = []
            for i in range(N):
                for j in range(M):
                    triples.append((i, j, S[i, j]))
            triples.sort(key=lambda x: x[2], reverse=True)  # sort by sim desc

            used_obj = set()
            used_mask = set()
            for i, j, sim in triples:
                if i in used_obj or j in used_mask:
                    continue
                if sim <= 0.0:
                    break
                best_match_indices[i] = j
                confidences[i] = float(sim)
                used_obj.add(i)
                used_mask.add(j)
                if len(used_obj) == N:
                    break

    if return_scores:
        return best_match_indices, confidences
    else:
        return best_match_indices
