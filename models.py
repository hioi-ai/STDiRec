
import os
import pickle
import torch
import torch.nn.functional as F
import torch.nn as nn
import faiss
from modules import *
from utils import extract
from torch.fft import fft, ifft

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
# import networkx as nx
import numpy as np
import torch.nn.functional as F


#傅里叶变化
class FourierIntentMix(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, x):
        # x: [B, S, D]
        freq = fft(x, dim=1)
        S = x.size(1)
        mid = S // 2

        low = freq.clone(); low[:, mid:] = 0
        high = freq.clone(); high[:, :mid] = 0

        mixed = self.alpha * low + (1 - self.alpha) * high
        x_mixed = ifft(mixed, dim=1).real
        return x_mixed

class Diffusion():
    def __init__(self, args):
        self.timesteps = args.timesteps
        self.beta_start = args.beta_start
        self.beta_end = args.beta_end
        self.args = args

        if self.args.beta_sche == 'linear':
            self.betas = self.linear_beta_schedule(self.timesteps, self.beta_start, self.beta_end)
        elif self.args.beta_sche == 'exp':
            self.betas = self.exp_beta_schedule(self.timesteps)
        elif self.args.beta_sche =='cosine':
            self.betas = self.cosine_beta_schedule(self.timesteps)
        elif self.args.beta_sche =='sqrt':
            self.betas = torch.tensor(self.betas_for_alpha_bar(self.timesteps, lambda t: 1-np.sqrt(t + 0.0001),)).float()

        # define alphas（timesteps, ）
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)


        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def linear_beta_schedule(self, timesteps, beta_start, beta_end):
        beta_start = beta_start
        beta_end = beta_end
        return torch.linspace(beta_start, beta_end, timesteps)

    def cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def exp_beta_schedule(self, timesteps, beta_min=0.1, beta_max=10):
        x = torch.linspace(1, 2 * timesteps + 1, timesteps)
        betas = 1 - torch.exp(- beta_min / timesteps - x * 0.5 * (beta_max - beta_min) / (timesteps * timesteps))
        return betas

    def betas_for_alpha_bar(self, num_diffusion_timesteps, alpha_bar, max_beta=0.999):
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas)


def hybrid_select_two_centers(embeddings, indices):
    local_indices = np.arange(len(indices))
    if len(local_indices) < 2:
        return (0, 1)

    # 尝试最大距离选择
    max_dist = -1
    best_pair = None
    for i in local_indices:
        for j in local_indices[i + 1:]:
            dist = np.linalg.norm(embeddings[i] - embeddings[j])  # 使用局部索引
            if dist > max_dist:
                max_dist = dist
                best_pair = (i, j)
    if best_pair and max_dist > 1e-6:
        return best_pair

    # 退回到随机选择
    return tuple(np.random.choice(local_indices, 2, replace=False))



def assign_to_centers(embeddings, center_a, center_b):
    """
    将数据点分配到两个中心节点
    :param embeddings: 数据点的嵌入表示 (N, D)
    :param center_a: 第一个中心点 (D,)
    :param center_b: 第二个中心点 (D,)
    :return: group_a 和 group_b：两个分配的组（数据点的索引）
    """
    # 计算每个点到两个中心的距离
    dist_a = np.linalg.norm(embeddings - center_a, axis=1)
    dist_b = np.linalg.norm(embeddings - center_b, axis=1)

    group_a = np.where(dist_a < dist_b)[0]
    group_b = np.where(dist_b <= dist_a)[0]

    return group_a, group_b

class GranularBallClustering:
    def __init__(self, min_sample=None, quality_threshold=None, max_depth=None, save_path=None, item_embeddings=None, seq_model=None):#beauty sport toys 最小样本：2 质量阈值;0.4 最大深度15 viedo:最小样本：10 质量阈值;0.5 最大深度15
        self.min_sample = min_sample
        self.quality_threshold = quality_threshold
        self.max_depth = max_depth
        self.seq2cluster = None
        self.centroids = None
        self.granular_balls = None
        self.embeddings = None
        self.is_trained = False  # 训练状态标志
        self.locked = False  # 添加锁定标志，防止外部重置\
        self.save_path = save_path

        self.item_embeddings = item_embeddings
        self.seq_model = seq_model  # 确保传入的 seq_model 是有效的
        self.device = item_embeddings.weight.device if item_embeddings is not None else torch.device('cpu')



    def qity_embedding(self, cluster_embeddings, all_embeddings, seq2cluster, indices=None):
        if len(cluster_embeddings) == 0 or self.locked:
            return float('-inf')
        if all_embeddings is None or seq2cluster is None:
            return float('-inf')

        centroid = np.mean(cluster_embeddings, axis=0)
        intra_distance = np.mean([np.linalg.norm(e - centroid) for e in cluster_embeddings])
        if intra_distance == 0:
            intra_distance = 1e-6

        # 使用局部 indices 确定当前簇的标签
        current_cluster_label = None
        if indices is not None and len(indices) > 0:
            current_cluster_label = seq2cluster[indices[0]] if 0 <= indices[0] < len(seq2cluster) else -1
        else:
            # 回退到 FAISS 搜索（仅在初始调用时）
            index = faiss.IndexFlatL2(all_embeddings.shape[1])
            index.add(all_embeddings.astype(np.float32))
            distances, indices = index.search(cluster_embeddings[:1].astype(np.float32), k=1)
            current_cluster_label = seq2cluster[indices[0][0]] if len(indices[0]) > 0 and indices[0][0] < len(
                seq2cluster) else -1

        # 计算其他簇的质心，仅限于当前簇外的点
        other_cluster_labels = np.setdiff1d(np.unique(seq2cluster),[current_cluster_label] if current_cluster_label != -1 else [])
        other_centroids = []
        for label in other_cluster_labels:
            if label != -1:
                other_indices = np.where(seq2cluster == label)[0]
                if len(other_indices) > 0:
                    other_centroid = np.mean(all_embeddings[other_indices], axis=0)
                    other_centroids.append(other_centroid)
        inter_distance = float('inf') if not other_centroids else np.min(
            [np.linalg.norm(centroid - c) for c in other_centroids])
        if inter_distance == float('inf'):
            inter_distance = max(1e-6, intra_distance * 1.1)  # 确保 inter > intra 的最小值

        quality = inter_distance / intra_distance if intra_distance > 0 else 0.0
        quality = max(quality, 0)  # 确保非负
        print(f"qity_embedding: intra={intra_distance:.4f}, inter={inter_distance:.4f}, ratio={quality:.4f}")
        return quality

    def split_ball_embedding(self, embeddings, indices, split_GB_list, all_embeddings, seq2cluster, depth=0,
                             max_retry=10):
        print(
            f"进入split_ball_embedding, 嵌入形状: {embeddings.shape}, 最大索引: {np.max(indices) if indices.size > 0 else -1}")
        if depth >= self.max_depth or len(indices) <= self.min_sample or self.locked:
            split_GB_list.append(indices)
            print(f"Depth {depth}, Final cluster size: {len(indices)}, embeddings shape: {embeddings.shape}")
            return

        if np.max(indices) >= embeddings.shape[0]:
            print(f"Error: Index {np.max(indices)} out of bounds for embeddings shape {embeddings.shape}")
            split_GB_list.append(indices)
            return

        subset_embeddings = embeddings[indices]
        print(f"Depth {depth}, Subset embeddings shape: {subset_embeddings.shape}, Max index: {np.max(indices)}")

        for attempt in range(max_retry):
            center_a_idx, center_b_idx = hybrid_select_two_centers(subset_embeddings, np.arange(len(indices)))
            center_a = subset_embeddings[center_a_idx]
            center_b = subset_embeddings[center_b_idx]
            group_a, group_b = assign_to_centers(subset_embeddings, center_a, center_b)
            print(f"Attempt {attempt + 1}/{max_retry}, Group A size: {len(group_a)}, Group B size: {len(group_b)}")
            if len(group_a) > 0 and len(group_b) > 0:
                group_a_global = indices[group_a]
                group_b_global = indices[group_b]
                break
        else:
            print(f"Depth {depth}, Failed to split after {max_retry} attempts, cluster size: {len(indices)}")
            split_GB_list.append(indices)
            return

        q_father = self.qity_embedding(subset_embeddings, all_embeddings, seq2cluster, indices)
        q_son = self.qity_embedding(subset_embeddings[group_a], all_embeddings, seq2cluster, indices[group_a]) + \
                self.qity_embedding(subset_embeddings[group_b], all_embeddings, seq2cluster, indices[group_b])

        print(f"Depth {depth}, Father quality: {q_father:.4f}, Son quality: {q_son:.4f}, "
              f"Sizes: {len(indices)}, {len(group_a_global)}, {len(group_b_global)}")
        if q_son > q_father - 0.01 and q_son / 2 < self.quality_threshold:  # 放宽条件
            self.split_ball_embedding(embeddings, group_a_global, split_GB_list, all_embeddings, seq2cluster, depth + 1)
            self.split_ball_embedding(embeddings, group_b_global, split_GB_list, all_embeddings, seq2cluster, depth + 1)
        else:
            split_GB_list.append(indices)
            print(f"Depth {depth}, Stop splitting, cluster size: {len(indices)}")


    def train(self, seq_repr: torch.Tensor):
        if self.is_trained or self.locked:
            print("GranularBallClustering已训练或锁定，跳过重新训练。")
            return
        print("训练GranularBallClustering，形状:", seq_repr.shape)
        x_np = seq_repr.detach().cpu().numpy()
        self.embeddings = x_np
        print(f"设置嵌入形状: {self.embeddings.shape}")

        kmeans = KMeans(n_clusters=10, init='k-means++', random_state=42)#除了ml-1m都是10
        kmeans.fit(x_np)
        self.seq2cluster = kmeans.labels_
        self.centroids = kmeans.cluster_centers_

        overall_quality = self.qity_embedding(x_np, x_np, self.seq2cluster)
        print(f"Initial KMeans clusters: {len(np.unique(self.seq2cluster))}, Overall quality: {overall_quality}")

        split_GB_list = []
        unique_labels = np.unique(self.seq2cluster)
        for label in unique_labels:
            indices = np.where(self.seq2cluster == label)[0]
            if len(indices) >= self.min_sample:
                cluster_embeddings = self.embeddings[indices]
                q_value = self.qity_embedding(cluster_embeddings, x_np, self.seq2cluster, indices)
                print(f"Cluster {label} with quality {q_value}, Size: {len(indices)}, Max index: {np.max(indices)}")
                if q_value < self.quality_threshold:
                    print(f"Cluster {label} with quality {q_value} is split further.")
                    self.split_ball_embedding(self.embeddings, indices, split_GB_list, x_np, self.seq2cluster)
                else:
                    split_GB_list.append(indices)

        self.granular_balls = split_GB_list
        self.update_cluster_centroids()
        self.is_trained = True
        self.locked = True
        print(f"Final number of granular balls: {len(self.granular_balls)} with depths up to {self.max_depth}")

        # 保存训练状态
        if self.save_path:
            self.save_model()
            print(f"模型状态保存至 {self.save_path}")

    def update_cluster_centroids(self):
        if self.locked:
            return
        centers = []
        for ball in self.granular_balls:
            if len(ball) > 0:
                vectors = self.embeddings[ball]
                center = np.mean(vectors, axis=0)
                centers.append(center)
        self.centroids = np.array(centers)

    def query(self, seq_repr: torch.Tensor):
        """
        统一使用“余弦最相似”来返回：
        1) ball_id
        2) 对应球中心向量
        """
        if self.centroids is None:
            raise ValueError("Clusterer is not trained, centroids not initialized.")

        x_np = seq_repr.detach().cpu().numpy().astype(np.float32)
        all_centroids = self.centroids.astype(np.float32)

        similarities = cosine_similarity(x_np, all_centroids)  # [B, num_balls]
        max_sim_indices = np.argmax(similarities, axis=1)  # [B]

        selected_centroids = all_centroids[max_sim_indices].astype(np.float32)

        cluster_ids = torch.from_numpy(max_sim_indices).long().to(seq_repr.device)
        cluster_centers = torch.from_numpy(selected_centroids).to(seq_repr.device)

        return cluster_ids, cluster_centers

    def save_model(self):
        """将聚类模型状态保存到文件。"""
        # if not self.save_path:
        #     print("未指定保存路径，使用默认路径。")
        #     self.save_path = "/home/hp/home/zyy/InDiRec-main/granular_balls_toysq2.pkl"
        if not self.save_path:
            raise ValueError("保存路径未指定，请确保在初始化时传入有效的路径！")
        state = {
            'min_sample': self.min_sample,
            'quality_threshold': self.quality_threshold,
            'max_depth': self.max_depth,
            'seq2cluster': self.seq2cluster,
            'centroids': self.centroids,
            'granular_balls': self.granular_balls,
            'embeddings': self.embeddings,
            'is_trained': self.is_trained,
            'locked': self.locked
        }
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, 'wb') as f:
            pickle.dump(state, f)
        print(f"聚类模型保存至 {self.save_path}")

    def load_model(self):
        if not self.save_path:
            raise ValueError("保存路径未指定，请确保在初始化时传入有效的路径！")
        # if not self.save_path:
        #     print("未指定加载路径，使用默认路径: /home/hp/home/zyy/InDiRec-main/granular_balls_toysq2.pkl")
        #     self.save_path = "/home/hp/home/zyy/InDiRec-main/granular_balls_toysq2.pkl"

        """从文件中加载聚类模型状态。"""
        if not self.save_path or not os.path.exists(self.save_path):
            print(f"在 {self.save_path} 未找到模型，无法加载。")
            return False
        try:
            with open(self.save_path, 'rb') as f:
                state = pickle.load(f)
            self.min_sample = state['min_sample']
            self.quality_threshold = state['quality_threshold']
            self.max_depth = state['max_depth']
            self.seq2cluster = state['seq2cluster']
            self.centroids = state['centroids']
            self.granular_balls = state['granular_balls']
            self.embeddings = state['embeddings']
            self.is_trained = state['is_trained']
            self.locked = state['locked']

            # 如果 granular_balls 为 None，返回加载失败
            if self.granular_balls is None:
                print("加载失败：粒球数据为空！")
                return False

            print(f"从 {self.save_path} 加载聚类模型")
            return True
        except Exception as e:
            print(f"从 {self.save_path} 加载模型时出错: {e}")
            return False

    def try_load_or_train(self, seq_repr: torch.Tensor):
        """尝试加载模型；若无法加载，则训练并保存。"""
        if self.load_model():
            print("已加载预训练聚类模型，跳过训练。")
        else:
            print("未找到预训练模型或加载失败，继续训练。")
            self.train(seq_repr)

class InDiRec(nn.Module):

    def __init__(self, device, args, num_heads=1):
        super(InDiRec, self).__init__()
        self.state_size = args.max_seq_length
        self.hidden_size = args.hidden_size
        self.item_num = args.item_size
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.diffuser_type = args.diffuser_type
        self.device = device
        self.w = args.w
        # 嵌入和序列模型
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.seq_model = SASRecModel(self.item_embeddings, args=args)

        self.clusterer = GranularBallClustering(min_sample=args.min_sample, quality_threshold=args.quality_threshold, max_depth=15, save_path=args.cluster_save_path, seq_model=self.seq_model, item_embeddings=self.item_embeddings)  # 创建粒球聚类对象#
        # self.clusterer = GranularBallClustering(min_sample=100, quality_threshold=0.5, max_depth=15, save_path=args.cluster_save_path, seq_model=self.seq_model, item_embeddings=self.item_embeddings)#video数据集

        self.diff = Diffusion(args)

        self.diff_meta_rank = args.diff_meta_rank

        # ==================== 双扩散低秩元网络 ====================
        self.diff_meta_shared = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.diff_meta_rank),
            nn.LayerNorm(self.diff_meta_rank),
            nn.GELU()
        )

        self.diff_meta_w1 = nn.Linear(self.diff_meta_rank, self.hidden_size)
        self.diff_meta_w2 = nn.Linear(self.diff_meta_rank, self.hidden_size)

        self.diff_final_proj = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

        # ==================== LLM bank / 单路可训练投影 ====================
        self.llm_user_bank = None
        self.llm_user_mask = None
        self.llm_align_proj = None

        # 无条件嵌入
        self.none_embedding = nn.Embedding(
            num_embeddings=1,
            embedding_dim=self.hidden_size,
        )
        nn.init.normal_(self.none_embedding.weight, 0, 1)

        # 时间步嵌入
        self.step_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size*2),
            nn.GELU(),
            nn.Linear(self.hidden_size*2, self.hidden_size),
        )

        self.emb_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size*2)
        )

        self.diff_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size*2),
            nn.GELU(),
            nn.Linear(self.hidden_size*2, self.hidden_size),
        )

        if self.diffuser_type =='mlp1':
            self.diffuser = nn.Sequential(
                nn.Linear(self.hidden_size*3, self.hidden_size))

        elif self.diffuser_type =='mlp2':
            self.diffuser = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size*2),
            nn.GELU(),
            nn.Linear(self.hidden_size*2, self.hidden_size))



    def forward(self, x, s, step):
        #  x:(batch, dim) , h:(batch, dim), t:(batch, )
        t = self.step_mlp(step)  # t:(batch, dim)

        if self.diffuser_type == 'mlp1':
            res = self.diffuser(torch.cat((x, s, t), dim=1))  # (batch, 3 x dim) -->diffuser-->(batch, dim)
        elif self.diffuser_type == 'mlp2':
            res = self.diffuser(torch.cat((x, s, t), dim=1))
        return res  # (batch, dim)


    def forward_uncon(self, x, step):
        h = self.none_embedding(torch.tensor([0]).to(self.device))
        h = torch.cat([h.view(1, 64)]*x.shape[0], dim=0)

        t = self.step_mlp(step)

        if self.diffuser_type == 'mlp1':
            res = self.diffuser(torch.cat((x, h, t), dim=1))
        elif self.diffuser_type == 'mlp2':
            res = self.diffuser(torch.cat((x, h, t), dim=1))

        return res



    def calculate_x0(self, x0):
        return self.item_embeddings(x0)

    def diff_soft_ortho_loss(self, feat1, feat2):
        feat1 = F.normalize(feat1, dim=-1)
        feat2 = F.normalize(feat2, dim=-1)
        return ((feat1 * feat2).sum(dim=-1) ** 2).mean()

    def fuse_diff_views(self, center_vec, aug_vec):
        """
        center_vec: [B, D]
        aug_vec:    [B, D]
        返回:
            fused_diff: [B, D]
            shared:     [B, D]
            spec1:      [B, D]
            spec2:      [B, D]
            ortho_loss: scalar
        """
        meta = self.diff_meta_shared(torch.cat([center_vec, aug_vec], dim=-1))   # [B, D]
        shared1 = self.diff_meta_w1(meta)   # [B, D]
        shared2 = self.diff_meta_w2(meta)   # [B, D]

        shared = 0.5 * (shared1 + shared2)
        spec1 = center_vec - shared1
        spec2 = aug_vec - shared2

        ortho_loss = self.diff_soft_ortho_loss(spec1, spec2)
        fused_diff = self.diff_final_proj(torch.cat([shared, spec1, spec2], dim=-1))

        return fused_diff, shared, spec1, spec2, ortho_loss

    def load_llm_user_bank(self, pkl_path):
        """
        读取:
            seq_idx -> {"orig_user_id": ..., "item_seq": [...], "user_embedding": np.ndarray[4096]}
        构造成:
            self.llm_user_bank: [N, 4096]
            self.llm_user_mask: [N]
        并创建单路可训练投影:
            4096 -> 512 -> 64
        """
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        if len(data) == 0:
            raise ValueError(f"LLM pkl is empty: {pkl_path}")

        max_idx = max(int(k) for k in data.keys())
        first_key = next(iter(data.keys()))
        llm_dim = int(np.asarray(data[first_key]["user_embedding"]).shape[-1])

        bank = np.zeros((max_idx + 1, llm_dim), dtype=np.float32)
        mask = np.zeros((max_idx + 1,), dtype=np.float32)

        for k, v in data.items():
            idx = int(k)
            bank[idx] = np.asarray(v["user_embedding"], dtype=np.float32)
            mask[idx] = 1.0

        self.llm_user_bank = torch.tensor(bank, dtype=torch.float32, device=self.device)
        self.llm_user_mask = torch.tensor(mask, dtype=torch.float32, device=self.device)

        self.llm_align_proj = nn.Sequential(
            nn.Linear(llm_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, self.hidden_size)
        ).to(self.device)

        print(f"[LLM Bank] loaded from {pkl_path}, shape={self.llm_user_bank.shape}")

    def get_llm_user_vec_with_mask(self, seq_idx_tensor):
        """
        seq_idx_tensor: [B]，twodiff 训练 batch 里的子序列索引
        返回:
            llm_vec:  [B, llm_dim]
            has_llm:  [B] bool
        """
        if self.llm_user_bank is None or self.llm_user_mask is None:
            raise RuntimeError("LLM user bank is not loaded. Call load_llm_user_bank() first.")

        seq_idx_tensor = seq_idx_tensor.long()
        B = seq_idx_tensor.size(0)
        llm_dim = self.llm_user_bank.size(1)

        llm_vec = torch.zeros(B, llm_dim, device=self.device, dtype=torch.float32)
        has_llm = torch.zeros(B, device=self.device, dtype=torch.bool)

        valid = (seq_idx_tensor >= 0) & (seq_idx_tensor < self.llm_user_bank.size(0))
        if valid.any():
            valid_idx = seq_idx_tensor[valid]
            has_bank = self.llm_user_mask[valid_idx] > 0
            if has_bank.any():
                take_pos = valid.nonzero(as_tuple=False).flatten()[has_bank]
                take_idx = valid_idx[has_bank]
                llm_vec[take_pos] = self.llm_user_bank[take_idx]
                has_llm[take_pos] = True

        return llm_vec, has_llm

    def calculate_s(self, input_seq, p):
        """
        从序列中提取表示，并根据粒球中心进行意图建模。支持 dropout 掩码扰动。
        :param input_seq: 输入序列 (B, S)
        :param p: dropout 概率
        :return: 意图向量 (B, D)
        """
        seq_out = self.seq_model(input_seq)
        s = seq_out[:, -1, :]  # B x D

        # 查询最相似的意图中心
        _, cluster_centers = self.clusterer.query(s)

        # 掩码扰动
        B, D = cluster_centers.shape
        mask1d = (torch.sign(torch.rand(B, device=self.device) - p) + 1) / 2  # (B,)
        mask = mask1d.view(B, 1).expand(-1, D)  # (B, D)
        none_vec = self.none_embedding(torch.tensor([0], device=self.device)).expand(B, -1)

        s = cluster_centers * mask + none_vec * (1 - mask)
        return s  # 意图向量，扩散的条件

        # 移除动态训练，改为查询预训练结果
    def encode_and_cluster(self, input_seq):
        """
        编码序列并查询预训练的聚类结果，不执行动态训练。
        :param input_seq: 输入序列 (B, S)
        :return: cluster_ids, cluster_centers, pooled_output
        """
        encoder_output = self.seq_model(input_seq)  # [B, S, D]
        pooled_output = encoder_output[:, -1, :]  # [B, D]
        # 尝试加载预训练模型或训练新模型
        self.clusterer.try_load_or_train(pooled_output)
        cluster_ids, cluster_centers = self.clusterer.query(pooled_output)
        return cluster_ids, cluster_centers, pooled_output

    # 聚类中心计算用于指导
    # 将输入序列传入模型（通常是一个类似于 SASRec 这样的序列模型），得到 编码后的输出。

    def forward_process(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(self.diff.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.diff.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    @torch.no_grad()
    def sample_from_reverse_process(self, s):
        x = torch.randn_like(s)
        for n in reversed(range(0, self.diff.timesteps)):
            t = torch.full((s.shape[0], ), n, device=self.device, dtype=torch.long)
            # x:(batch, dim), s:(batch, dim), t:(batch, ), n:1~timesteps
            x = self.p_sample_with_guidance(self.forward, self.forward_uncon, x, s, t, n)
        
        return x
    
    @torch.no_grad()
    def p_sample_with_guidance(self, model_forward, model_forward_uncon, x, s, t, t_index):

        # guidance, (batch, dim)
        x_start = (1 + self.w) * model_forward(x, s, t) - self.w * model_forward_uncon(x, t)
        x_t = x

        # model_mean = (batch,1) x (batch, dim) + (batch,1) x (batch, dim) = (batch, dim)
        model_mean = (
            extract(self.diff.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.diff.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.diff.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)

            return model_mean + torch.sqrt(posterior_variance_t) * noise




class SASRecModel(nn.Module):
    def __init__(self, item_embeddings, args):
        super(SASRecModel, self).__init__()
        self.item_embeddings = item_embeddings
        # self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)#固定长度写法
        # 改为初始为最大值，同时支持后续动态扩展
        self.max_position_embeddings = 50  # 初始化为支持长度 100，可自行设大些
        self.position_embeddings = nn.Embedding(self.max_position_embeddings, args.hidden_size)
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        self.use_fourier_mix = True#想要用傅里叶再开这个
        self.fourier_mixer = FourierIntentMix(alpha=0.7) if self.use_fourier_mix else None

        self.criterion = nn.BCELoss(reduction="none")
        self.apply(self.init_weights)

    def item_embedding(self, sequence):
        return self.item_embeddings(sequence)



    #动态支持
    def add_position_embedding(self, sequence):
        seq_length = sequence.size(1)

        # 如有必要，动态扩展 position_embeddings
        if seq_length > self.max_position_embeddings:
            # 重新初始化一个更长的 position embedding（保留原参数并迁移至新 embedding）
            new_max_len = seq_length * 2  # 扩大一倍以减少多次扩展
            new_position_embeddings = nn.Embedding(new_max_len, self.args.hidden_size).to(sequence.device)

            # 拷贝已有位置嵌入的参数
            with torch.no_grad():
                new_position_embeddings.weight[:self.max_position_embeddings] = self.position_embeddings.weight

            self.position_embeddings = new_position_embeddings
            self.max_position_embeddings = new_max_len

        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)

        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        # Fourier Intent Mix enhancement（保持不变）
        if self.fourier_mixer is not None:
            sequence_emb = self.fourier_mixer(sequence_emb)

        return sequence_emb

    # def forward(self, input_ids):
    #     attention_mask = (input_ids > 0).long()
    #     extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
    #     max_len = attention_mask.size(-1)
    #     attn_shape = (1, max_len, max_len)
    #     subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
    #     subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
    #     subsequent_mask = subsequent_mask.long()
    #
    #     if self.args.cuda == 0:
    #         subsequent_mask = subsequent_mask.cuda()
    #
    #     extended_attention_mask = extended_attention_mask * subsequent_mask
    #     extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
    #     extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    #
    #     # 调用修改后的 add_position_embedding 方法，确保传入所需的参数
    #     sequence_emb = self.add_position_embedding(input_ids)
    #
    #     item_encoded_layers = self.item_encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True)
    #
    #     sequence_output = item_encoded_layers[-1]
    #     return sequence_output
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()

        # 1. attention mask: [B, 1, 1, L]
        attention_mask = (input_ids > 0).unsqueeze(1).unsqueeze(2)

        # 2. subsequent mask: [1, 1, L, L]
        subsequent_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=input_ids.device),
            diagonal=1
        ).bool()
        subsequent_mask = ~subsequent_mask  # 下三角为 True
        subsequent_mask = subsequent_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]

        # 3. broadcast combine: [B, 1, L, L]
        combined_mask = attention_mask & subsequent_mask  # 逻辑与
        combined_mask = combined_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - combined_mask) * -10000.0  # masked为 -10000

        # 4. 获取位置信息 + Fourier混合
        sequence_emb = self.add_position_embedding(input_ids)

        # 5. 编码器 forward
        item_encoded_layers = self.item_encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True)
        sequence_output = item_encoded_layers[-1]
        return sequence_output

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
