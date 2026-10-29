import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from collections import OrderedDict
from tqdm import tqdm

from seqllm4rec_sasco import llm4rec
import sys
import importlib.util
from types import SimpleNamespace

class llmrec_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # 设备
        if hasattr(args, "device"):
            self.device = torch.device(args.device)
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        # 初始化 LLM 封装
        self.llm = llm4rec(device=self.device, llm_model=args.llm, args=args)
        
        self.llm.to(self.device)
        self.llm.eval()

        # 读取共现矩阵（pkl）
        self.cooccurrence_matrix = self._load_cooccurrence_matrix(
            args.cooccur_matrix_path
        )

        # 固定投影层权重路径（必须传）
        self.item_proj_path = getattr(args, "item_proj_path", None)
        if self.item_proj_path is None:
            raise ValueError("args.item_proj_path 不能为空，请传入训练好的 item_proj.pt 路径。")

        # 运行时根据 sas_dim 再初始化
        self.item_emb_proj = None
        self._item_proj_loaded = False
        self.random_proj_seed = getattr(args, "random_proj_seed", 2022)

     # ============== 工具函数 ==============

    def _load_cooccurrence_matrix(self, file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def _init_item_emb_proj(self, sas_dim, llm_hidden_dim):
        """
        单层随机线性投影 + LayerNorm
        只初始化一次，后续固定使用
        """
        self.item_emb_proj = nn.Sequential(
            nn.Linear(sas_dim, llm_hidden_dim),
            nn.LayerNorm(llm_hidden_dim),
        ).to(self.device)

    # 固定随机初始化
        torch.manual_seed(self.random_proj_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_proj_seed)

        nn.init.xavier_normal_(self.item_emb_proj[0].weight)
        if self.item_emb_proj[0].bias is not None:
            nn.init.zeros_(self.item_emb_proj[0].bias)


    def _load_item_proj_if_needed(self, sas_dim):
        """
        有固定随机投影文件就加载；
        没有就初始化一份并保存，之后一直复用。
        """
        if self._item_proj_loaded and self.item_emb_proj is not None:
            return

        llm_hidden_dim = int(self.llm.llm_model.config.hidden_size)
        self._init_item_emb_proj(sas_dim, llm_hidden_dim)

        if os.path.exists(self.item_proj_path):
            state = torch.load(self.item_proj_path, map_location=self.device)

            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]

            self.item_emb_proj.load_state_dict(state, strict=True)
            print(f"[Info] Loaded fixed random item_emb_proj from: {self.item_proj_path}")
        else:
            save_dir = os.path.dirname(self.item_proj_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)

            torch.save(self.item_emb_proj.state_dict(), self.item_proj_path)
            print(f"[Info] Random item_emb_proj not found. Initialized once and saved to: {self.item_proj_path}")

        self.item_emb_proj.eval()
        for p in self.item_emb_proj.parameters():
            p.requires_grad = False

        self._item_proj_loaded = True

    def get_top_k_neighbors(self, last_item_id, top_k=3):
        """
        last_item_id 用的是真实 item_id（1-based），共现矩阵按同样索引。
        返回 top-k 的 neighbor item_id（仍是 1-based）。
        """
        scores = np.asarray(self.cooccurrence_matrix[last_item_id]).copy()

        # 不要 padding / 自己
        if scores.shape[0] > 0:
            scores[0] = -np.inf
        if last_item_id < scores.shape[0]:
            scores[last_item_id] = -np.inf

        valid = np.where(scores > 0)[0]
        if len(valid) == 0:
            return []

        valid_scores = scores[valid]
        order = np.argsort(valid_scores)[::-1]
        top_idx = valid[order[:top_k]]
        return top_idx.tolist()

    def build_prompt(self, user_history_titles, top_k_neighbor_titles):
        """
        user_history_titles: list[str]
        top_k_neighbor_titles: list[str]
        """
        history_lines = []
        for i, title in enumerate(user_history_titles, start=1):
            history_lines.append(f"{i}. Title: {title} [HistoryEmb]")
        history_text = "\n".join(history_lines)

        if len(top_k_neighbor_titles) == 0:
            co_text = "None"
        else:
            co_lines = []
            for i, title in enumerate(top_k_neighbor_titles, start=1):
                co_lines.append(f"{i}. {title}")
            co_text = "\n".join(co_lines)

        prompt = (
            "This user has made a series of purchases in the following order:\n"
            "(History Item List\n"
            f"{history_text}\n"
            ")\n\n"
            "Based on the most recent item, the following items show strong behavioral co-occurrence "
            "under sequential transition patterns:\n"
            f"{co_text}\n\n"
            "Please infer the user's current intent and stable preference, then output a user representation token: [UserOut]"
        )
        return prompt

    def _load_module_from_path(self, module_name, file_path):
        module_dir = os.path.dirname(file_path)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _ensure_segmented_file(self, data_file, max_seq_length):
        """
        严格复用 InDiRec 的 D() 逻辑生成 segmented_file。
        """
        datasets_py = os.path.join(self.args.indisas_dir, "datasets.py")
        datasets_mod = self._load_module_from_path("indisas_datasets_for_llm", datasets_py)

        segmented_file = os.path.splitext(data_file)[0] + "_s.txt"
        if not os.path.exists(segmented_file):
            print(f"[Info] segmented file not found, generating with D(): {segmented_file}")
            datasets_mod.D(data_file, segmented_file, max_seq_length)
        else:
            print(f"[Info] segmented file exists, reuse: {segmented_file}")
        return segmented_file

    def _get_max_item_from_data_file(self, data_file):
        max_item = 0
        with open(data_file, "r") as fr:
            for line in fr:
                parts = line.strip().split()
                if len(parts) <= 1:
                    continue
                items = list(map(int, parts[1:]))
                if len(items) > 0:
                    max_item = max(max_item, max(items))
        return max_item

    def _read_segmented_training_histories(self, segmented_file, debug_print_n=10):
        """
        从 D() 生成的 segmented_file 中读取训练子序列。
        对每条序列，只保留真实历史部分 seg_items[:-3] 进入 pt。
        seq_idx 严格按文件行顺序编号。
        """
        seq_records = []
        seq_idx = 0

        with open(segmented_file, "r") as fr:
            for line in fr:
                parts = line.strip().split()
                if len(parts) <= 4:
                    # 至少: user + 4 items，train 中 history 才至少有 1 个 item
                    continue

                orig_user_id = int(parts[0])
                seg_items = list(map(int, parts[1:]))

                # train 模式下真正进入 SAS 的历史
                hist_items = seg_items[:-3]
                held_out_tail = seg_items[-3:]

                if len(hist_items) == 0:
                    continue

                rec = {
                    "seq_idx": seq_idx,
                    "orig_user_id": orig_user_id,
                    "seg_items": seg_items,
                    "hist_items": hist_items,
                    "held_out_tail": held_out_tail,
                }
                seq_records.append(rec)

                if seq_idx < debug_print_n:
                    print(
                        f"[Debug-before-LLM] seq_idx={seq_idx}, "
                        f"orig_user_id={orig_user_id}, "
                        f"history_for_pt={hist_items}, "
                        f"held_out_tail(not enter pt)={held_out_tail}"
                    )

                seq_idx += 1

        print(f"[Info] total training subsequences used for SAS->LLM: {len(seq_records)}")
        return seq_records

    def _load_sas_model_from_pt(self, sas_pt_path, data_file, max_seq_length):
        """
        加载训练好的 SASOnly pt，并返回模型。
        """
        models_py = os.path.join(self.args.indisas_dir, "models.py")
        models_mod = self._load_module_from_path("indisas_models_for_llm", models_py)

        max_item = self._get_max_item_from_data_file(data_file)

        sas_args = SimpleNamespace(
            hidden_size=self.args.sas_hidden_size,
            num_hidden_layers=self.args.sas_num_hidden_layers,
            num_attention_heads=self.args.sas_num_attention_heads,
            hidden_act=self.args.sas_hidden_act,
            attention_probs_dropout_prob=self.args.sas_attention_probs_dropout_prob,
            hidden_dropout_prob=self.args.sas_hidden_dropout_prob,
            initializer_range=self.args.sas_initializer_range,
            max_seq_length=max_seq_length,
            item_size=max_item + 2,
            mask_id=max_item + 1,
            cuda=0,
        )

        model = models_mod.SASOnly(self.device, sas_args).to(self.device)
        state_dict = torch.load(sas_pt_path, map_location=self.device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        print(f"[Info] loaded SAS pt from: {sas_pt_path}")
        return model

    @torch.no_grad()
    def _build_seq_dict_from_sas_pt(
        self,
        sas_pt_path,
        data_file,
        max_seq_length=50,
        sas_batch_size=256,
        debug_print_n=10,
    ):
        """
        返回:
        seq_dict[seq_idx] = {
            "orig_user_id": ...,
            "item_seq": [真实历史 item_id 列表],
            "embs": [np.ndarray, ...]   # 与 item_seq 一一对应的上下文化位置表示
        }
        """
        segmented_file = self._ensure_segmented_file(data_file, max_seq_length)
        seq_records = self._read_segmented_training_histories(segmented_file, debug_print_n=debug_print_n)
        sas_model = self._load_sas_model_from_pt(sas_pt_path, data_file, max_seq_length)

        seq_dict = OrderedDict()

        for start in tqdm(range(0, len(seq_records), sas_batch_size), desc="Running SAS pt on training subsequences"):
            batch_records = seq_records[start:start + sas_batch_size]

            batch_input_ids = []
            batch_input_lens = []

            for rec in batch_records:
                hist_items = rec["hist_items"]
                input_len = len(hist_items)

                padded = [0] * (max_seq_length - input_len) + hist_items
                padded = padded[-max_seq_length:]

                batch_input_ids.append(padded)
                batch_input_lens.append(min(input_len, max_seq_length))

            batch_input_ids = torch.tensor(batch_input_ids, dtype=torch.long, device=self.device)
            batch_outputs = sas_model.seq_model(batch_input_ids)   # [B, L, D]

            for i, rec in enumerate(batch_records):
                seq_idx = rec["seq_idx"]
                orig_user_id = rec["orig_user_id"]
                hist_items = rec["hist_items"]
                input_len = batch_input_lens[i]

                # 只取真实历史 item 的非 padding 位置表示
                ctx_emb = batch_outputs[i, -input_len:, :].detach().cpu().numpy().astype(np.float32)

                seq_dict[seq_idx] = {
                    "orig_user_id": orig_user_id,
                    "item_seq": hist_items,
                    "embs": [ctx_emb[j] for j in range(ctx_emb.shape[0])],
                }

        return seq_dict

    @torch.no_grad()
    def generate_user_repr_from_sas_pt(
        self,
        sas_pt_path,
        data_file,
        meta_csv_path,
        out_path,
        max_seq_length=50,
        sas_batch_size=256,
        llm_batch_size=2,
        max_input_length=2048,
        debug_print_n=10,
    ):
        """
        严格复用 InDiRec 的 D() 子序列划分；
        只处理训练子序列；
        对每条子序列的真实历史 hist_items=seg_items[:-3] 过 SAS pt，
        提取所有非 padding 位置的上下文化表示，替换 prompt 中的 [HistoryEmb]。
        最终保存 seq_idx -> {orig_user_id, item_seq, user_embedding}
        """

        # 1) 从 SAS pt 构建 seq_dict
        seq_dict = self._build_seq_dict_from_sas_pt(
            sas_pt_path=sas_pt_path,
            data_file=data_file,
            max_seq_length=max_seq_length,
            sas_batch_size=sas_batch_size,
            debug_print_n=debug_print_n,
        )

        # 2) 读取 meta（title 映射：meta 的 itemID 是 0-based）
        meta_df = pd.read_csv(meta_csv_path)
        if "itemID" not in meta_df.columns or "title" not in meta_df.columns:
            raise ValueError("meta_csv 中必须包含列：itemID, title")

        meta_df["title"] = meta_df["title"].fillna("Unknown Title")
        id2title = meta_df.set_index("itemID")["title"].to_dict()

        # 3) 初始化/加载随机投影层
        first_seq = next(iter(seq_dict.values()))
        first_emb = first_seq["embs"][0]
        sas_dim = int(first_emb.shape[0])
        self._load_item_proj_if_needed(sas_dim)
        print(f"[Info] item_emb_proj ready: rec_dim={sas_dim}, llm_hidden={self.llm.llm_model.config.hidden_size}")

        # 4) 获取 [UserOut] token id
        userout_token_id = self.llm.llm_tokenizer(
            "[UserOut]", return_tensors="pt", add_special_tokens=False
        ).input_ids.item()

        seq_ids = list(seq_dict.keys())
        seq_user_repr = {}

        # 5) 分 batch 构造 prompt + 跑 LLM
        for start_idx in tqdm(range(0, len(seq_ids), llm_batch_size), desc="Building prompts & running LLM"):
            batch_seq_ids = seq_ids[start_idx:start_idx + llm_batch_size]

            batch_prompts = []
            batch_hist_embs = []

            for seq_idx in batch_seq_ids:
                seq_info = seq_dict[seq_idx]
                item_seq = seq_info["item_seq"]   # 真实历史 item ids（1-based）
                ctx_embs = seq_info["embs"]       # 对应位置的上下文化表示

                history_titles = []
                for item_id in item_seq:
                    meta_id = item_id - 1
                    title = id2title.get(meta_id, "Unknown Title")
                    history_titles.append(title)

                # 共现只基于真实历史最后一个 item
                last_item_id = int(item_seq[-1])
                top_k_neighbors = self.get_top_k_neighbors(last_item_id, top_k=3)

                top_k_neighbor_titles = []
                for neighbor in top_k_neighbors:
                    meta_id = int(neighbor) - 1
                    title = id2title.get(meta_id, "Unknown Title")
                    top_k_neighbor_titles.append(title)

                prompt = self.build_prompt(
                    user_history_titles=history_titles,
                    top_k_neighbor_titles=top_k_neighbor_titles,
                )

                ctx_emb_mat = np.stack(ctx_embs, axis=0)  # [L, D_sas]
                ctx_emb_tensor = torch.tensor(ctx_emb_mat, dtype=torch.float32, device=self.device)
                hist_emb = self.item_emb_proj(ctx_emb_tensor)  # [L, H]

                batch_prompts.append(prompt)
                batch_hist_embs.append(hist_emb)

            # tokenize
            llm_tokens = self.llm.llm_tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=max_input_length,
            ).to(self.device)

            input_ids = llm_tokens["input_ids"]
            inputs_embeds = self.llm.llm_model.get_input_embeddings()(input_ids)

            # 用投影后的历史 embedding 替换 [HistoryEmb]
            inputs_embeds = self.llm.replace_out_token_all(
                llm_tokens,
                inputs_embeds,
                token=["[UserOut]", "[HistoryEmb]"],
                embs={"[HistoryEmb]": batch_hist_embs},
            )

            # 运行 LLM
            outputs = self.llm.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=llm_tokens.get("attention_mask", None),
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            last_hidden = outputs.hidden_states[-1]

            # 提取 [UserOut] hidden
            B = input_ids.size(0)
            user_hidden_list = []
            for b in range(B):
                positions = (input_ids[b] == userout_token_id).nonzero(as_tuple=False).flatten()

                if len(positions) == 0:
                    pos = input_ids.size(1) - 1
                elif len(positions) == 1:
                    pos = positions.item()
                else:
                    pos = int(positions.float().mean().item())

                user_hidden_list.append(last_hidden[b, pos])

            user_hidden = torch.stack(user_hidden_list, dim=0)

            # 保存：seq_idx -> {orig_user_id, item_seq, user_embedding}
            for i, seq_idx in enumerate(batch_seq_ids):
                seq_user_repr[seq_idx] = {
                    "orig_user_id": seq_dict[seq_idx]["orig_user_id"],
                    "item_seq": seq_dict[seq_idx]["item_seq"],
                    "user_embedding": user_hidden[i].detach().cpu().numpy().astype(np.float32),
                }

        print(f"[Info] Saving raw LLM hidden representations, dim = {user_hidden.shape[-1]}")
        with open(out_path, "wb") as f:
            pickle.dump(seq_user_repr, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"✅ Saved {len(seq_user_repr)} user representations to {out_path}")

    # ============== 主功能：从 pkl 生成用户表示并保存 ==============
    @torch.no_grad()
    def generate_user_repr_from_pkl(self, sas_pkl_path, meta_csv_path, out_path, batch_size=2, max_input_length=2048):
        """
        从 SAS item embedding pkl + meta CSV + 有向共现矩阵，生成每个 seq_idx 的用户向量并保存。
        """
        # 1) 读 item embedding pkl
        with open(sas_pkl_path, "rb") as f:
            records = pickle.load(f)

        # 2) 按 seq_idx 聚合
        seq_dict = OrderedDict()
        for rec in records:
            seq_idx = int(rec["seq_idx"])
            orig_uid = int(rec.get("orig_user_id", -1))
            item_id = int(rec["item_id"])   # 1-based
            emb = np.asarray(rec["embedding"], dtype=np.float32)

            if seq_idx not in seq_dict:
                seq_dict[seq_idx] = {"orig_user_id": orig_uid, "items": []}
            seq_dict[seq_idx]["items"].append({"item_id": item_id, "emb": emb})

        # 3) 读取 meta（title 映射：meta 的 itemID 是 0-based）
        meta_df = pd.read_csv(meta_csv_path)
        if "itemID" not in meta_df.columns or "title" not in meta_df.columns:
            raise ValueError("meta_csv 中必须包含列：itemID, title")

        meta_df["title"] = meta_df["title"].fillna("Unknown Title")
        id2title = meta_df.set_index("itemID")["title"].to_dict()

        # 4) 初始化并加载固定 item_emb_proj
        first_seq = next(iter(seq_dict.values()))
        first_emb = first_seq["items"][0]["emb"]
        sas_dim = int(first_emb.shape[0])
        self._load_item_proj_if_needed(sas_dim)
        print(f"[Info] item_emb_proj ready: rec_dim={sas_dim}, llm_hidden={self.llm.llm_model.config.hidden_size}")

        # 5) 获取 [UserOut] token 的 ID
        userout_token_id = self.llm.llm_tokenizer(
            "[UserOut]", return_tensors="pt", add_special_tokens=False
        ).input_ids.item()

        seq_ids = list(seq_dict.keys())
        seq_user_repr = {}

        # 6) 分 batch 处理
        for start_idx in tqdm(range(0, len(seq_ids), batch_size), desc="Building prompts & running LLM"):
            batch_seq_ids = seq_ids[start_idx:start_idx + batch_size]

            batch_prompts = []
            batch_hist_embs = []

            for seq_idx in batch_seq_ids:
                seq_info = seq_dict[seq_idx]
                items = seq_info["items"]

                history_titles = []
                sas_embs = []

                for itm in items:
                    item_id = int(itm["item_id"])   # 1-based
                    meta_id = item_id - 1           # 映射到 title 的 0-based
                    title = id2title.get(meta_id, "Unknown Title")
                    history_titles.append(title)
                    sas_embs.append(itm["emb"])

                # 基于最后一个 item 查有向共现 top-k
                last_item_id = int(items[-1]["item_id"])
                top_k_neighbors = self.get_top_k_neighbors(last_item_id, top_k=3)

                top_k_neighbor_titles = []
                for neighbor in top_k_neighbors:
                    meta_id = int(neighbor) - 1
                    title = id2title.get(meta_id, "Unknown Title")
                    top_k_neighbor_titles.append(title)

                prompt = self.build_prompt(
                    user_history_titles=history_titles,
                    top_k_neighbor_titles=top_k_neighbor_titles,
                )

                # 历史 item embedding 升维到 LLM hidden
                sas_emb_mat = np.stack(sas_embs, axis=0)  # [L, D_sas]
                sas_emb_tensor = torch.tensor(sas_emb_mat, dtype=torch.float32, device=self.device)
                self._load_item_proj_if_needed(sas_dim)
                hist_emb = self.item_emb_proj(sas_emb_tensor)  # [L, H]

                batch_prompts.append(prompt)
                batch_hist_embs.append(hist_emb)

            # 7) tokenization
            llm_tokens = self.llm.llm_tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=max_input_length,
            ).to(self.device)

            input_ids = llm_tokens["input_ids"]
            inputs_embeds = self.llm.llm_model.get_input_embeddings()(input_ids)

            # 8) 用投影后的历史 embedding 替换 [HistoryEmb]
            inputs_embeds = self.llm.replace_out_token_all(
                llm_tokens,
                inputs_embeds,
                token=["[UserOut]", "[HistoryEmb]"],
                embs={"[HistoryEmb]": batch_hist_embs},
            )

            # 9) 运行 LLM
            outputs = self.llm.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=llm_tokens.get("attention_mask", None),
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            last_hidden = outputs.hidden_states[-1]  # [B, T, H]

            # 10) 提取 [UserOut] 位置 hidden -> pred_user
            B = input_ids.size(0)
            user_hidden_list = []
            for b in range(B):
                positions = (input_ids[b] == userout_token_id).nonzero(as_tuple=False).flatten()

                if len(positions) == 0:
                    pos = input_ids.size(1) - 1
                elif len(positions) == 1:
                    pos = positions.item()
                else:
                    pos = int(positions.float().mean().item())

                user_hidden_list.append(last_hidden[b, pos])

            user_hidden = torch.stack(user_hidden_list, dim=0)  # [B, H]

# 保存 LLM 原始 hidden 维度表示
            for i, seq_idx in enumerate(batch_seq_ids):
                seq_user_repr[seq_idx] = {
                    "orig_user_id": seq_dict[seq_idx]["orig_user_id"],
                    "user_embedding": user_hidden[i].detach().cpu().numpy().astype(np.float32),
                }

        # 12) dump
        print(f"[Info] Saving raw LLM hidden representations, dim = {user_hidden.shape[-1]}")
        with open(out_path, "wb") as f:
            pickle.dump(seq_user_repr, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"✅ Saved {len(seq_user_repr)} user representations to {out_path}")
