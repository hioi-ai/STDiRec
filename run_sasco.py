import argparse
import os
import torch

from seqllm_model_sasco import llmrec_model


def get_args():
    parser = argparse.ArgumentParser()

    # ===== LLM 基本参数 =====
    parser.add_argument("--llm", type=str, default="llama")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--nn_parameter", action="store_true")
    parser.add_argument("--token", action="store_true")

    # ===== 数据与共现 =====
    parser.add_argument("--data_file", type=str,
                        default="/home/bigdata17/zyy/InDiRec-main/datasets/Beauty.txt")
    parser.add_argument("--meta_csv_path", type=str,
                        default="/home/bigdata17/zyy/InDiRec-main/coLLM/datasets/new-meta-beauty.csv")
    parser.add_argument("--cooccur_matrix_path", type=str,
                        default="/home/bigdata17/zyy/InDiRec-main/coo/beauty_directed_diff.pkl")

    # ===== SAS pt 与投影 =====
    parser.add_argument("--sas_pt_path", type=str,
                        default="/home/bigdata17/zyy/InDiRec-main/coLLM/SASOnly-Beauty-0.pt")
    parser.add_argument("--item_proj_path", type=str,
                        default="/home/bigdata17/zyy/InDiRec-main/coLLM/item_proj_random_beauty.pt")
    parser.add_argument("--random_proj_seed", type=int, default=2022)

    # ===== 输出 =====
    parser.add_argument("--out_user_path", type=str,
                        default="/home/bigdata17/zyy/InDiRec-main/coLLM/LLM_co_em/beauty_sasllm.pkl")

    # ===== 复用 InDiRec 的 D() / SASOnly =====
    parser.add_argument("--indisas_dir", type=str,
                        default="/home/bigdata17/zyy/InDiRec-main/coLLM/indaiSAS")

    # ===== 序列长度 / batch =====
    parser.add_argument("--max_seq_length", type=int, default=50)
    parser.add_argument("--llm_batch_size", type=int, default=2)
    parser.add_argument("--sas_batch_size", type=int, default=256)
    parser.add_argument("--max_input_length", type=int, default=2048)
    parser.add_argument("--debug_print_n", type=int, default=10)

    # ===== SAS 模型结构参数（必须和训练 pt 时一致） =====
    parser.add_argument("--sas_hidden_size", type=int, default=64)
    parser.add_argument("--sas_num_hidden_layers", type=int, default=2)
    parser.add_argument("--sas_num_attention_heads", type=int, default=2)
    parser.add_argument("--sas_hidden_act", type=str, default="gelu")
    parser.add_argument("--sas_attention_probs_dropout_prob", type=float, default=0.5)
    parser.add_argument("--sas_hidden_dropout_prob", type=float, default=0.2)
    parser.add_argument("--sas_initializer_range", type=float, default=0.02)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    print(f"[Info] Using device = {args.device}, llm = {args.llm}")
    model = llmrec_model(args)

    out_dir = os.path.dirname(args.out_user_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    model.generate_user_repr_from_sas_pt(
        sas_pt_path=args.sas_pt_path,
        data_file=args.data_file,
        meta_csv_path=args.meta_csv_path,
        out_path=args.out_user_path,
        max_seq_length=args.max_seq_length,
        sas_batch_size=args.sas_batch_size,
        llm_batch_size=args.llm_batch_size,
        max_input_length=args.max_input_length,
        debug_print_n=args.debug_print_n,
    )

    print("\n[Done] LLM user representations have been generated and saved.")