import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, OPTForCausalLM, AutoModelForCausalLM
from peft import (
    prepare_model_for_kbit_training,
)
class llm4rec(nn.Module):
    def __init__(
        self,
        device,
        llm_model="llama",
        max_output_txt_len=256,
        args= None
    ):
        super().__init__()
        self.device = device
        self.args = args

        
        if llm_model == 'llama':
            model_id = "/home/bigdata17/Meta-Llama-3-8B-Instruct"
        elif llm_model == 'llama-3b':
            model_id="meta-llama/Llama-3.2-3B-Instruct"
        else:
            raise Exception(f'{llm_model} is not supported')
        print()
        print("=========")

        if self.args.nn_parameter:
            self.llm_model = AutoModelForCausalLM.from_pretrained(model_id, device_map=self.device, torch_dtype=torch.float16)
        else:
            self.llm_model = AutoModelForCausalLM.from_pretrained(model_id, device_map=self.device, torch_dtype=torch.float16,load_in_8bit=True,)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
            
        
            
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'additional_special_tokens': ['[UserRep]','[HistoryEmb]', '[UserOut]', '[ItemOut]']})
        self.llm_tokenizer.add_special_tokens({'cls_token': "[CLS]"})
        
        
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
        self.llm_model = prepare_model_for_kbit_training(self.llm_model)
        
        for _, param in self.llm_model.named_parameters():
            if args.token:
                if 'token' in _:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = False

        if not args.token:
            if args.nn_parameter:
                self.CLS = nn.Parameter(torch.normal(0,1, size = (1,self.llm_model.config.hidden_size))).to(device)
                self.CLS_item = nn.Parameter(torch.normal(0,1, size = (1,self.llm_model.config.hidden_size))).to(device)
            else:
                self.CLS = nn.Embedding(1,self.llm_model.config.hidden_size).to(device)
                nn.init.normal_(self.CLS.weight, mean = self.llm_model.model.embed_tokens.weight.mean(), std = self.llm_model.model.embed_tokens.weight.std())
                self.CLS_item = nn.Embedding(1,self.llm_model.config.hidden_size).to(device)
                nn.init.normal_(self.CLS_item.weight, mean = self.llm_model.model.embed_tokens.weight.mean(), std = self.llm_model.model.embed_tokens.weight.std())
        
        self.pred_user = nn.Sequential(
                nn.Linear(self.llm_model.config.hidden_size, 2048),
                nn.LayerNorm(2048),
                nn.LeakyReLU(),
                nn.Linear(2048, 128)
            )
        nn.init.xavier_normal_(self.pred_user[0].weight)
        nn.init.xavier_normal_(self.pred_user[3].weight)

    # def get_top_k_neighbors(self, last_item_id, top_k=5):
    #     """
    #     获取与给定 item_id 共现最强的 Top-K neighbors。
    #     """
    #     cooccurrence_scores = self.cooccurrence_matrix[last_item_id]  # 取出与 last_item_id 共现的所有分数
    #     top_k_indices = np.argsort(cooccurrence_scores)[-top_k:][::-1]  # 获取 Top-K 的索引
    #     return top_k_indices

    #===================batch=1=================================
    # def replace_out_token_all(self, llm_tokens, inputs_embeds, token = [], embs= None,):
    #     for t in token:
    #         token_id = self.llm_tokenizer(t, return_tensors="pt", add_special_tokens=False).input_ids.item()
    #         vectors = []
    #         for inx in range(len(llm_tokens["input_ids"])):
    #             idx_tensor=(llm_tokens["input_ids"][inx]==token_id).nonzero().view(-1)
    #             user_vector = inputs_embeds[inx]
    #             if 'Emb' in t:
    #                 ee = embs[t][inx]
    #                 for idx, item_emb in zip(idx_tensor, ee):
    #                     user_vector = torch.cat((user_vector[:idx], item_emb.unsqueeze(0), user_vector[idx+1:]), dim=0)
    #             elif 'Rep' in t:
    #                 for idx in idx_tensor:
    #                     user_emb = embs[t][inx]
    #                     user_vector = torch.cat((user_vector[:idx], user_emb.unsqueeze(0), user_vector[idx+1:]), dim=0)
    #             else:
    #                 if not self.args.token:
    #                     for idx in idx_tensor:
    #                         if 'UserOut' in t:
    #                             if self.args.nn_parameter:
    #                                 user_vector = torch.cat((user_vector [:idx], self.CLS[torch.tensor([0]).to(self.device)], user_vector [idx+1:]), dim=0)
    #                             else:
    #                                 user_vector = torch.cat((user_vector [:idx], self.CLS(torch.tensor([0]).to(self.device)), user_vector [idx+1:]), dim=0)
    #                         elif 'ItemOut' in t:
    #                             if self.args.nn_parameter:
    #                                 user_vector = torch.cat((user_vector [:idx], self.CLS_item[torch.tensor([0]).to(self.device)], user_vector [idx+1:]), dim=0)
    #                             else:
    #                                 user_vector = torch.cat((user_vector [:idx], self.CLS_item(torch.tensor([0]).to(self.device)), user_vector [idx+1:]), dim=0)
    #
    #             vectors.append(user_vector.unsqueeze(0))
    #         inputs_embeds = torch.cat(vectors)
    #     return inputs_embeds
    #================batch=2=======================
    def replace_out_token_all(self, llm_tokens, inputs_embeds, token=[], embs=None):
        for t in token:
            token_id = self.llm_tokenizer(t, return_tensors="pt", add_special_tokens=False).input_ids.item()
            vectors = []
            for inx in range(len(llm_tokens["input_ids"])):
                idx_tensor = (llm_tokens["input_ids"][inx] == token_id).nonzero().view(-1)
                user_vector = inputs_embeds[inx]
                if 'Emb' in t:
                    ee = embs[t][inx]
                    # 扩展 item_emb 以确保与 user_vector 的维度匹配
                    ee = ee.unsqueeze(0) if ee.dim() == 1 else ee  # 确保 ee 变为 [1, H]
                    for idx, item_emb in zip(idx_tensor, ee):
                        user_vector = torch.cat((user_vector[:idx], item_emb.unsqueeze(0), user_vector[idx + 1:]),
                                                dim=0)
                elif 'Rep' in t:
                    for idx in idx_tensor:
                        user_emb = embs[t][inx]
                        user_vector = torch.cat((user_vector[:idx], user_emb.unsqueeze(0), user_vector[idx + 1:]),
                                                dim=0)
                else:
                    if not self.args.token:
                        for idx in idx_tensor:
                            if 'UserOut' in t:
                                if self.args.nn_parameter:
                                    user_vector = torch.cat((user_vector[:idx],
                                                             self.CLS[torch.tensor([0]).to(self.device)],
                                                             user_vector[idx + 1:]), dim=0)
                                else:
                                    user_vector = torch.cat((user_vector[:idx],
                                                             self.CLS(torch.tensor([0]).to(self.device)),
                                                             user_vector[idx + 1:]), dim=0)
                            elif 'ItemOut' in t:
                                if self.args.nn_parameter:
                                    user_vector = torch.cat((user_vector[:idx],
                                                             self.CLS_item[torch.tensor([0]).to(self.device)],
                                                             user_vector[idx + 1:]), dim=0)
                                else:
                                    user_vector = torch.cat((user_vector[:idx],
                                                             self.CLS_item(torch.tensor([0]).to(self.device)),
                                                             user_vector[idx + 1:]), dim=0)

                vectors.append(user_vector.unsqueeze(0))
            inputs_embeds = torch.cat(vectors)
        return inputs_embeds

    def get_embeddings(self, llm_tokens, token):
        token_idx = []
        token_id = self.llm_tokenizer(token, return_tensors="pt", add_special_tokens=False).input_ids.item()
        for inx in range(len(llm_tokens['input_ids'])):
            idx_tensor = (llm_tokens['input_ids'][inx] == token_id).nonzero().view(-1)
            token_idx.append(idx_tensor)
        return token_idx


    
    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "llm4rec 现在只用于特征抽取，不再实现训练 forward。"
            "请直接在上层调用 llm_model、replace_out_token_all、get_embeddings 和 pred_user。"
        )
    
    