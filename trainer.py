import os
import pickle
import numpy as np
from torch import pdist
from tqdm import tqdm
import random
import gc
# import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
# from models import KMeans
from utils import recall_at_k, ndcg_k, generate_padded_sequences_tensor
# from models_twoctro import build_cooccurrence_matrix, load_cooccurrence_matrix
#保存sas的嵌入

class Trainer:
    def __init__(self, model, train_dataloader,cluster_dataloader, eval_dataloader, test_dataloader, device, args):

        self.args = args
        self.device = device
        self.model = model
        self.seq_model = self.model.seq_model

        self.batch_size = self.args.batch_size
        self.sim=self.args.sim


        self.train_dataloader = train_dataloader
        self.cluster_dataloader=cluster_dataloader

        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        self.sequences_list = None  # 初始化 sequences_list，防止 AttributeError

        self.seq_model.cuda()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr,weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    
    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader,self.cluster_dataloader)

    def valid(self, epoch):
        return self.iteration(epoch, self.eval_dataloader, train=False)

    def test(self, epoch):
        return self.iteration(epoch, self.test_dataloader, train=False)

    # def iteration(self, epoch, dataloader, train=True):
    #     raise NotImplementedError

    def iteration(self, epoch, dataloader, cluster_dataloader=None, train=True):
        raise NotImplementedError
    
    def predict_full(self, seq_out):
        test_item_emb = self.model.item_embeddings.weight
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred
    
    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HR@5": "{:.4f}".format(recall[0]),
            "NDCG@5": "{:.4f}".format(ndcg[0]),
            "HR@10": "{:.4f}".format(recall[1]),
            "NDCG@10": "{:.4f}".format(ndcg[1]),
            "HR@20": "{:.4f}".format(recall[3]),
            "NDCG@20": "{:.4f}".format(ndcg[3]),
        }
        print(post_fix)
        with open(self.args.log_file, "a") as f:
            f.write(str(post_fix) + "\n")
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)


    @torch.no_grad()
    def get_sequences_in_same_cluster(self, input_sequence=None, seq_repr=None):
        """
        返回与当前 batch 序列位于相同簇（或最近粒球中心）的中心向量（B×D）。
        - 优先使用传入的 seq_repr（形状 B×D），避免重复跑 self.seq_model。
        - 若未提供 seq_repr，则兼容旧接口，从 input_sequence 计算得到。
        """
        if seq_repr is None:
            if input_sequence is None:
                raise ValueError("Either seq_repr or input_sequence must be provided.")
            input_sequence = input_sequence.to(self.device)
            sequence_output = self.seq_model(input_sequence)
            seq_repr = sequence_output[:, -1, :]  # B×D

        # 确保在 GPU 上查询
        seq_repr = seq_repr.to(self.device)

        # 使用模型内的聚类器进行查询
        cluster_ids, cluster_centers = self.model.clusterer.query(seq_repr)
        if cluster_centers is None:
            raise ValueError("Cluster centers are not returned properly by the query method.")

        return cluster_centers  # (B×D)
    @torch.no_grad()
    def get_cluster_ids_and_centers(self, seq_repr=None, input_sequence=None):
        """
        一次查询，直接拿到：
        1) cluster_ids
        2) cluster_centers
        统一使用 clusterer.query() 的余弦最相似结果
        """
        if seq_repr is None:
            if input_sequence is None:
                raise ValueError("Either seq_repr or input_sequence must be provided.")
            input_sequence = input_sequence.to(self.device)
            seq_repr = self.seq_model(input_sequence)[:, -1, :]

        seq_repr = seq_repr.to(self.device)
        cluster_ids, cluster_centers = self.model.clusterer.query(seq_repr)

        if cluster_centers is None:
            raise ValueError("Cluster centers are not returned properly by the query method.")

        return cluster_ids, cluster_centers

    def apply_center_condition_dropout(self, center_s, p=None):
        """
        只对第一路球中心条件做扰动：
        - 以概率 p 用 none_embedding 替换该样本的中心条件
        - 第二路 aug_s 不做这个扰动
        """
        if p is None:
            p = self.args.p

        if p <= 0:
            return center_s

        B, D = center_s.shape
        device = center_s.device

        # 1 表示保留 center_s；0 表示替换为 none_embedding
        keep_mask_1d = (torch.rand(B, device=device) > p).float()
        keep_mask = keep_mask_1d.view(B, 1).expand(-1, D)

        none_vec = self.model.none_embedding(
            torch.tensor([0], device=device)
        ).expand(B, -1)

        center_s_drop = center_s * keep_mask + none_vec * (1.0 - keep_mask)
        return center_s_drop

    def _left_pad_input_seq(self, prefix_items):
        """
        将一个未padding的 prefix 序列，按 DatasetForInDiRec(train) 的口径左侧补零到 max_seq_length
        """
        prefix_items = [int(x) for x in prefix_items if int(x) > 0]
        prefix_items = prefix_items[-self.args.max_seq_length:]
        pad_len = self.args.max_seq_length - len(prefix_items)
        return [0] * pad_len + prefix_items

    def _replace_with_directed_cooccurrence(self, full_segment_seq):
        """
        对一条“完整分段子序列”做：
        - 只在其训练可见前缀 full_segment_seq[:-3] 上做替换
        - 随机选 replace_ratio 比例的位置
        - 用有向窗口共现里该 item 的最强后继 item 做替换
        - 若该 item 没有有向后继候选，则保持原 item 不变
        最终返回：左补零后的 input_seq（长度=max_seq_length）
        """
        if self.cooccur_matrix is None:
            raise RuntimeError("trainer.cooccur_matrix 还没有在 main_twoctro.py 中注入。")

        if self.train_seq is None:
            raise RuntimeError("trainer.train_seq 不存在，无法通过球内 seq_idx 回查训练子序列。")

        co_mat = self.cooccur_matrix
        if torch.is_tensor(co_mat):
            co_mat = co_mat.detach().cpu().numpy()
        else:
            co_mat = np.asarray(co_mat)

        full_segment_seq = [int(x) for x in full_segment_seq]
        prefix_items = full_segment_seq[:-3] if len(full_segment_seq) > 3 else []

        valid_positions = [idx for idx, item in enumerate(prefix_items) if item > 0]
        if len(valid_positions) == 0:
            return self._left_pad_input_seq(prefix_items)

        n_replace = max(1, int(np.ceil(len(valid_positions) * self.args.cooccur_replace_ratio)))
        n_replace = min(n_replace, len(valid_positions))

        chosen_positions = np.random.choice(valid_positions, size=n_replace, replace=False)

        for pos in chosen_positions:
            src_item = int(prefix_items[pos])
            if src_item <= 0 or src_item >= co_mat.shape[0]:
                continue

            row = np.asarray(co_mat[src_item]).copy()
            row[0] = 0.0
            row[src_item] = 0.0

            # 没有有向后继候选：保持原 item 不变
            if np.all(row <= 0):
                continue

            new_item = int(np.argmax(row))
            if new_item > 0:
                prefix_items[pos] = new_item

        return self._left_pad_input_seq(prefix_items)


    @torch.no_grad()
    def get_augmented_cluster_conditions(self, cluster_ids):
        """
        第二路条件：
        - 从同一个余弦最相似球里随机抽一条 train_seq
        - 对其前缀（[:-3]）按有向共现做 item 替换
        - 再送入 seq_model 得到条件向量
        """
        if self.train_seq is None:
            raise RuntimeError("self.train_seq 为空，无法按球内 seq_idx 回查训练子序列。")

        augmented_input_seqs = []
        cluster_ids_list = cluster_ids.detach().cpu().tolist()

        for ball_id in cluster_ids_list:
            member_seq_idxs = self.model.clusterer.granular_balls[ball_id]
            sampled_seq_idx = int(np.random.choice(member_seq_idxs))

            sampled_full_seq = self.train_seq[sampled_seq_idx]
            augmented_input_seq = self._replace_with_directed_cooccurrence(sampled_full_seq)
            augmented_input_seqs.append(augmented_input_seq)

        augmented_input_seqs = torch.tensor(
            np.array(augmented_input_seqs, dtype=np.int64),
            dtype=torch.long,
            device=self.device,
        )

        augmented_repr = self.seq_model(augmented_input_seqs)[:, -1, :]
        return augmented_repr


class InDiRecTrainer(Trainer):
    def __init__(self, model, train_dataloader, train_dataset, cluster_dataloader, eval_dataloader, test_dataloader, device, args):
        super(InDiRecTrainer, self).__init__(model, train_dataloader, cluster_dataloader, eval_dataloader,
                                             test_dataloader, device, args)

        # 新增：embedding 缓存字典
        self.item_emb_cache = {}

        # 训练子序列（完整分段序列，后面第二路分支要按 seq_idx 回查）
        self.train_seq = None
        if train_dataset is not None and hasattr(train_dataset, "user_seq"):
            self.train_seq = [list(map(int, seq)) for seq in train_dataset.user_seq]

        # 主程序里再注入：trainer.cooccur_matrix = ...
        self.cooccur_matrix = None
        # 初始化
        self.best_valid_ndcg = -float('inf')  # 记录验证集上最好的 NDCG
        self.best_epoch = -1  # 记录最好的 epoch

        # 预训练粒球聚类（仅在此处执行一次）
        print("Pre-training Granular Ball Clustering...")
        print(f"Clusterer instance id: {id(self.model.clusterer)}")
        self.model.eval()
        sequences_list = []
        sequence_counter = 0

        # 验证 cluster_dataloader 是否有效
        if cluster_dataloader is None or len(cluster_dataloader) == 0:
            raise ValueError("Cluster dataloader is empty or None. Check data loading process.")
        print(f"Cluster dataloader length: {len(cluster_dataloader)}")

        rec_t_data_iter = tqdm(enumerate(self.cluster_dataloader), total=len(self.cluster_dataloader))
        for i, (rec_batch) in rec_t_data_iter:
            try:
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                user_id, subsequence, _, _, _ = rec_batch
                batch_size = subsequence.size(0)
                with torch.no_grad():  # 避免梯度计算，节省内存
                    sequence_output_a = self.model.seq_model(subsequence)
                    sequence_repr = sequence_output_a[:, -1, :]
                    sequences_list.append(sequence_repr.detach().cpu().numpy())
                    sequence_counter += batch_size
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                continue

        # 验证收集的数据
        if not sequences_list:
            raise ValueError("No sequence representations collected. Check model or dataloader.")
        sequences_list = np.concatenate(sequences_list, axis=0)
        tensor_repr = torch.tensor(sequences_list, dtype=torch.float32).to(self.device)

        # 尝试加载或训练聚类模型
        print(f"使用 {sequence_counter} 个序列训练/加载聚类器...")
        self.model.clusterer.try_load_or_train(tensor_repr)

        #==========================================================================
        print(f"训练后状态: is_trained={self.model.clusterer.is_trained}, locked={self.model.clusterer.locked}, "
              f"粒度球数量: {len(self.model.clusterer.granular_balls) if self.model.clusterer.granular_balls else 0}")
        # 存储序列列表（可选，视需求）
        self.sequences_list = sequences_list

        # 清理内存
        del sequences_list
        del tensor_repr
        gc.collect()

        self.model.train()  # 切换回训练模式

    def iteration(self, epoch, dataloader, cluster_dataloader=None, train=True, user_ids=None):
        if train:
            self.model.train()
            avg_loss = 0.0
            avg_loss = 0.0
            avg_rec_loss = 0.0
            avg_diff_loss = 0.0
            avg_cl_loss = 0.0
            avg_ortho_loss = 0.0
            avg_mse_loss = 0.0

            # 展示预训练的最终粒球数并监控状态
            num_granular_balls = len(self.model.clusterer.granular_balls)
            print(
                f"Epoch {epoch}, Starting iteration, Number of granular balls: {num_granular_balls}, Locked: {self.model.clusterer.locked}")

            # 训练阶段，使用 dataloader
            rec_t_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))
            for i, (rec_batch) in rec_t_data_iter:
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                seq_idx, input_seq, input_seq_len, target_pos, target = rec_batch

                # 获取相同簇的序列表示（使用预训练的意图原型）
                # 原始输入序列的末位表示
                x0_out = self.seq_model(input_seq)  # [B, L, D]
                x0_last = x0_out[:, -1, :].detach()  # [B, D]

                # 一次查询，同时拿到：
                # 1) 同一个“余弦最相似球”的 ball_id
                # 2) 对应球中心向量（第一路扩散条件）
                cluster_ids, center_s = self.get_cluster_ids_and_centers(seq_repr=x0_last)
                center_s = center_s.to(self.device)

                # 第一条分支：球中心条件做扰动
                center_s_drop = self.apply_center_condition_dropout(center_s, p=self.args.p)

                # 第二路扩散条件：
                # 从同一个球里随机抽 train_seq，再做有向共现替换，编码成条件向量
                aug_s = self.get_augmented_cluster_conditions(cluster_ids).to(self.device)

                x0_gt = self.model.calculate_x0(target)  # [B, D]

                batch_size = input_seq.size(0)
                t = torch.randint(0, self.args.timesteps, (batch_size,), device=self.device).long()

                # ==================== 双路 diffusion loss ====================
                diff_loss_center, _ = self.calculate_diff_loss(self.model, x0_gt, center_s_drop, t)
                diff_loss_aug, _ = self.calculate_diff_loss(self.model, x0_gt, aug_s, t)

                diff_loss = (
                        self.args.center_branch_weight * diff_loss_center +
                        self.args.aug_branch_weight * diff_loss_aug
                )

                # ==================== 双路 reverse sample ====================
                if self.args.diff_weight == 0:
                    sequences_tensor = generate_padded_sequences_tensor(
                        self.args.batch_size,
                        self.args.max_seq_length,
                        self.args.item_size,
                        input_seq_len
                    ).to(self.device)
                    x0_hat_center = self.seq_model(sequences_tensor)[:, -1, :]
                    x0_hat_aug = x0_hat_center
                else:
                    x0_hat_center = self.model.sample_from_reverse_process(center_s)
                    x0_hat_aug = self.model.sample_from_reverse_process(aug_s)

                # ==================== 双路 contrastive loss ====================
                # 原始序列表示（主空间，64维）
                # ==================== 融合后的双扩散表示 vs 原序列表示 ====================
                seq_repr = x0_out[:, -1, :]  # [B, D]

                fused_diff_repr, shared_diff, spec_center, spec_aug, ortho_loss = \
                    self.model.fuse_diff_views(x0_hat_center, x0_hat_aug)

                # 只对融合后的扩散表示与序列表示做 CL
                cl_loss = self.calculate_cl_loss(fused_diff_repr, seq_repr, target_pos[:, -1])

                # ==================== LLM 对齐：单边可训练投影 + MSE ====================
                llm_mse_loss = torch.tensor(0.0, device=self.device)

                if getattr(self.model, "llm_user_bank", None) is not None:
                    llm_vec, has_llm = self.model.get_llm_user_vec_with_mask(seq_idx)  # [B, 4096], [B]
                    if has_llm.any():
                        llm_proj = self.model.llm_align_proj(llm_vec[has_llm])  # [B_valid, 64]
                        seq_for_mse = seq_repr[has_llm]  # [B_valid, 64]
                        llm_mse_loss = F.mse_loss(llm_proj, seq_for_mse)

                # ==================== 推荐损失 ====================
                logits = self.predict_full(seq_repr)
                rec_loss = nn.CrossEntropyLoss()(logits, target_pos[:, -1])

                # ==================== 总损失 ====================
                multi_task_loss = (
                        self.args.rec_weight * rec_loss +
                        self.args.diff_weight * diff_loss +
                        self.args.cl_weight * cl_loss +
                        self.args.ortho_weight * ortho_loss +
                        self.args.mse_weight * llm_mse_loss
                )
                
                self.optimizer.zero_grad()
                multi_task_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                avg_loss += multi_task_loss.item()
                avg_rec_loss += rec_loss.item()
                avg_diff_loss += diff_loss.item()
                avg_cl_loss += cl_loss.item()
                avg_ortho_loss += ortho_loss.item()
                avg_mse_loss += llm_mse_loss.item()

            log_record = {
                "epoch": epoch,
                "avg_total_loss": "{:.4f}".format(avg_loss / len(rec_t_data_iter)),
                "avg_rec_loss": "{:.4f}".format(avg_rec_loss / len(rec_t_data_iter)),
                "avg_diff_loss": "{:.4f}".format(avg_diff_loss / len(rec_t_data_iter)),
                "avg_cl_loss": "{:.4f}".format(avg_cl_loss / len(rec_t_data_iter)),
                "avg_ortho_loss": "{:.4f}".format(avg_ortho_loss / len(rec_t_data_iter)),
                "avg_llm_mse_loss": "{:.4f}".format(avg_mse_loss / len(rec_t_data_iter)),
            }

            if (epoch + 1) % self.args.print_log_freq == 0:
                print(str(log_record))

            with open(self.args.log_file, "a") as f:
                f.write(str(log_record) + "\n")

        else:  # Val or Test
            rec_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))
            self.model.eval()
            pred_list, answer_list = None, None

            for i, batch in rec_data_iter:
                batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids, input_seq_len, target_pos, answers = batch
                rec_output = self.seq_model(input_ids)[:, -1, :]
                rating_pred = self.predict_full(rec_output)
                rating_pred = rating_pred.cpu().data.numpy().copy()
                batch_user_index = user_ids.cpu().numpy()
                rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                ind = np.argpartition(rating_pred, -20)[:, -20:]
                arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]
                if i == 0:
                    pred_list = batch_pred_list
                    answer_list = answers.cpu().data.numpy()
                else:
                    pred_list = np.append(pred_list, batch_pred_list, axis=0)
                    answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)



            metrices_list, log_info = self.get_full_sort_score(epoch, answer_list, pred_list)
            return metrices_list, log_info


    def calculate_diff_loss(self, model, x_start, s, t, noise=None, loss_type="l2"):
        """
                计算扩散损失
                :param model: 模型
                :param x_start: 输入序列 (batch, dim)
                :param s: 意图表示 (batch, dim)
                :param t: 扩散步 (batch,)
                :param noise: 噪声 (batch, dim)
                :param loss_type: 损失类型
                :return: 扩散损失，预测结果
                """
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = model.forward_process(x_start=x_start, t=t, noise=noise)
        predicted_x = model(x_noisy, s, t)
        if loss_type == 'l1':
            loss = F.l1_loss(x_start, predicted_x)
        elif loss_type == 'l2':
            loss = F.mse_loss(x_start, predicted_x)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(x_start, predicted_x)
        else:
            raise NotImplementedError()
        return loss, predicted_x

    # def calculate_cl_loss(self, x0_hat_rep, x0_rep, fused_repr, target=None):
    #     """
    #     两对正样本的 NT-Xent:
    #         1) (x0_hat_rep, x0_rep)
    #         2) (x0_rep, fused_repr)
    #     期望输入:
    #         x0_hat_rep: [B, D] 扩散得到的表示
    #         x0_rep    : [B, D] 原序列最后一步表示
    #         fused_repr: [B, D] 频率感知门控融合后的表示
    #         target    : (可选, 未用)
    #     """
    #     # 形状防御
    #     for name, z in [('x0_hat_rep', x0_hat_rep), ('x0_rep', x0_rep), ('fused_repr', fused_repr)]:
    #         assert z.dim() == 2, f"{name} must be [B, D], got {z.shape}"
    #
    #     loss1 = self._nt_xent_loss(x0_hat_rep, x0_rep, temperature=self.args.temperature, sim=self.sim)
    #     loss2 = self._nt_xent_loss(x0_rep, fused_repr, temperature=self.args.temperature, sim=self.sim)
    #     return 0.5 * (loss1 + loss2)
    #
    #
    # def _nt_xent_loss(self, z_i, z_j, temperature=0.07, sim='cos'):
    #     """
    #     标准 NT-Xent / SimCLR 损失
    #     输入:
    #         z_i, z_j: [B, D] 的两视图向量表示（同一样本的两种增强/视图）
    #     返回:
    #         标量 loss
    #     说明:
    #         - 将两视图拼接为 [2B, D]，构造 [2B, 2B] 相似度矩阵
    #         - 去掉对角（自身相似度）
    #         - 每个样本的正样本索引：i <-> i+B（上半区对下半区）
    #         - 用 CrossEntropy 在整行上指向正样本位置
    #     """
    #     assert z_i.dim() == 2 and z_j.dim() == 2 and z_i.shape == z_j.shape, \
    #         f"z_i and z_j must be [B, D] with same shape, got {z_i.shape} and {z_j.shape}"
    #     B, D = z_i.shape
    #
    #     # (1) 归一化（用 dot 就等价 cos；更稳）
    #     z_i = F.normalize(z_i, dim=-1)
    #     z_j = F.normalize(z_j, dim=-1)
    #
    #     # (2) 拼接 [2B, D]
    #     z = torch.cat([z_i, z_j], dim=0)  # [2B, D]
    #
    #     # (3) 相似度 [2B, 2B]
    #     # 你原来的 sim='dot' 和 'cos' 在归一化后等价，这里统一用 dot
    #     sim_matrix = torch.matmul(z, z.t()) / temperature  # [2B, 2B]
    #
    #     # (4) 去掉对角（自身）
    #     self_mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    #     sim_matrix = sim_matrix.masked_fill(self_mask, float('-inf'))
    #
    #     # (5) 构造每个 anchor 的正样本索引:
    #     # 前 B 个的正样本在 i+B，后 B 个的正样本在 i-B
    #     pos_index = torch.arange(B, device=z.device)
    #     labels = torch.cat([pos_index + B, pos_index], dim=0)  # [2B]
    #
    #     # (6) CrossEntropy: 在每行 2B logits（去掉自身）中，指向正样本那一列
    #     loss = F.cross_entropy(sim_matrix, labels)
    #     return loss
    def calculate_cl_loss(self, x0_hat_rep, x0_rep, target):
        batch_size = x0_rep.shape[0]
        sem_nce_logits, sem_nce_labels = self.info_nce(x0_hat_rep, x0_rep, self.args.temperature, batch_size, self.sim,
                                                       target)
        cl_loss = nn.CrossEntropyLoss()(sem_nce_logits, sem_nce_labels)
        return cl_loss

    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot', intent_id=None):

        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)
        if sim == 'cos':
            sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.t()) / temp

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        mask = self.mask_correlated_samples(intent_id)
        negative_samples = sim
        negative_samples[mask == 0] = float("-inf")

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def mask_correlated_samples(self, label):
        label = label.view(1, -1)
        label = label.expand((2, label.shape[-1])).reshape(1, -1)
        label = label.contiguous().view(-1, 1)
        mask = torch.eq(label, label.t())
        return mask == 0

    def save_item_embeddings(self, path=None):
        """保存每个用户交互的所有物品嵌入。"""
        if not hasattr(self, "item_emb_cache") or len(self.item_emb_cache) == 0:
            print("[Cache] Warning: item_emb_cache is empty, nothing to save.")
            return

        # 默认路径：在当前项目目录下保存
        if path is None:
            save_dir = "../item_emb_cache"
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"{self.args.data_name}_best_user_item_embeddings.pkl")

        with open(path, "wb") as f:
            pickle.dump(self.item_emb_cache, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[Cache] ✅ Saved best embeddings for {len(self.item_emb_cache)} users → {path}")

        # 打印前 5 个用户预览
        preview = list(self.item_emb_cache.items())[:5]
        print("\n📦 [Embedding Cache Preview] 前 5 个用户交互嵌入:")
        for uid, records in preview:
            print(f"\n👤 User {uid}: 共 {len(records)} 次交互")
            for idx, rec in enumerate(records):
                item_id = rec["item_id"]
                emb = rec["embedding"]
                emb_str = " ".join([f"{v:.4f}" for v in emb[:6]])
                print(f"  [{idx + 1}] item={item_id:<6} emb[:6]=[{emb_str}]")