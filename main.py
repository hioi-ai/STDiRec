import gc
import os
import pickle
import numpy as np
import torch
import argparse
from tqdm import tqdm

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from datetime import datetime

from models import InDiRec
from build_diff import build_or_load_directed_cooccurrence_matrix
from utils import EarlyStopping, set_seed, check_path, show_args_info
from datasets import D,D_random, get_seqs_and_matrixes, DatasetForInDiRec
from trainer import InDiRecTrainer
#傅里叶加大模型

def get_args():
    parser = argparse.ArgumentParser()
    # system args
    parser.add_argument("--model_name", default="InDiRec", type=str)
    parser.add_argument("--data_dir", default="./datasets/", type=str)
    parser.add_argument("--output_dir", default="output", type=str)
    parser.add_argument("--data_name", default="Beauty", type=str)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--model_idx", default=0, type=int, 
                        help="model idenfier 10, 20, 30...")
    parser.add_argument('--cuda', type=int, default=1,
                            help='cuda device.')

    # robustness experiments
    parser.add_argument("--noise_ratio", type=float, default=0.0,
                        help="percentage of negative interactions in a sequence",)

    ## contrastive learning task args
    parser.add_argument("--temperature", default=1.0, type=float,
                        help="softmax temperature (default:  1.0).")
    parser.add_argument("--intent_num",default=256,type=int,
                        help="the multi intent nums.")
    parser.add_argument("--sim",default='dot',type=str,
                        help="the calculate ways of the similarity.")

    # model args
    parser.add_argument("--hidden_size", type=int, default=64, 
                        help="Number of hidden factors, i.e., embedding size.")
    parser.add_argument("--num_hidden_layers", type=int, default=2, 
                        help="number of layers")
    parser.add_argument("--num_attention_heads", type=int, default=2,
                        help="number of attention heads")
    parser.add_argument("--hidden_act", type=str, default="gelu", 
                        help="active function type.") 

    # dropout
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, 
                        help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, 
                        help="hidden dropout p")
    parser.add_argument('--p', type=float, default=0.1,
                        help='uncondition dropout')

    

    # train args
    parser.add_argument("--lr", type=float, default=0.001, 
                        help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, 
                        help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=50,
                        help="number of epochs")#改为30个#toys:80
    parser.add_argument("--print_log_freq", type=int, default=1, 
                        help="per epoch print res")
    parser.add_argument("--max_seq_length", type=int, default=50, 
                        help="max sequence length")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="weight_decay of adam")
    parser.add_argument("--seed", default=2022, type=int)#原始种子2022

    parser.add_argument("--initializer_range", type=float, default=0.02)


    # loss weight
    parser.add_argument("--rec_weight", type=float, default=1, 
                        help="weight of rating prediction")#1
    parser.add_argument("--diff_weight", type=float, default=1,
                        help="weight of intent-aware diffusion task")#1
    parser.add_argument("--cl_weight", type=float, default=0.2,
                        help="weight of contrastive learning task")#0.2

    # ablation study
    parser.add_argument('--without_segment', action="store_true",
                        help='dropout ')

    # control guidance signal
    parser.add_argument('--w', type=float, default=2.0,
                        help='control the strength of intent-guided signal s')

    # diffusion
    parser.add_argument('--timesteps', type=int, default=200,
                        help='timesteps for diffusion')
    parser.add_argument('--beta_end', type=float, default=0.02,
                        help='beta end of diffusion')
    parser.add_argument('--beta_start', type=float, default=0.0001,
                        help='beta start of diffusion')
    parser.add_argument('--diffuser_type', type=str, default='mlp1',
                        help='type of diffuser.')
    parser.add_argument('--beta_sche', nargs='?', default='exp',
                        help='beta schedule')
    # #粒球参数
    #
    # parser.add_argument("--min_sample", type=int, default=5)

    # 粒球聚类参数
    parser.add_argument("--min_sample", type=int, default=5, help="最小样本数")
    parser.add_argument("--quality_threshold", type=float, default=0.5, help="质量阈值")#video阈值0.5其他都是0.4
    parser.add_argument("--max_depth", type=int, default=15, help="最大深度")
    parser.add_argument("--cluster_save_path", type=str, default="../granular_balls.pkl", help="聚类模型保存路径")

    # 有向窗口共现参数
    parser.add_argument("--cooccur_matrix_save_path", type=str, default=None,
                        help="有向窗口共现矩阵保存路径；为空时自动生成新文件名")
    parser.add_argument("--cooccur_holdout_last", type=int, default=3,
                        help="构造共现时从原始完整序列尾部去掉的项目数")
    parser.add_argument("--cooccur_window_size", type=int, default=3,
                        help="有向窗口共现的窗口大小")
    parser.add_argument("--cooccur_replace_ratio", type=float, default=0.2,
                        help="第二路扩散中，对球内随机序列做 item 替换的比例")

    # 双路扩散 / 双路CL 权重
    parser.add_argument("--center_branch_weight", type=float, default=0.7,
                        help="球中心条件分支的权重")
    parser.add_argument("--aug_branch_weight", type=float, default=0.3,
                        help="球内随机替换序列条件分支的权重")

    parser.add_argument(
    "--condition_cluster_type",type=str,default="granular_ball",choices=["granular_ball", "kmeans"],help="扩散条件使用粒球还是原始KMeans")
    parser.add_argument("--llm_user_pkl", type=str, default=None,
                        help="seq_idx -> {orig_user_id, item_seq, user_embedding} 的 LLM 表示 pkl")
    parser.add_argument("--mse_weight", type=float, default=0.1,
                        help="weight of LLM-sequence MSE alignment")
    parser.add_argument("--ortho_weight", type=float, default=0.1,
                        help="weight of diffusion shared/specific orthogonality loss")
    parser.add_argument("--fourier_alpha", type=float, default=0.7,
                        help="alpha for FourierIntentMix")
    parser.add_argument("--diff_meta_rank", type=int, default=8,
                        help="low-rank bottleneck size for dual-diff shared/specific meta network")



    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    # 设置使用的GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)  # 设置使用 GPU 1
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")  # 打印当前使用的 GPU 名称
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"使用的 cluster_save_path: {args.cluster_save_path}")  # 调试路径
    # print(f"使用的 cooccur_matrix_save_path: {args.cooccur_matrix_save_path}")  # 打印共现矩阵的路径
    # 传递命令行参数到 GranularBallClustering

    set_seed(args.seed)
    check_path(args.output_dir)
    args.data_file = args.data_dir + args.data_name + ".txt"

    if args.without_segment:
        args.segmented_file = args.data_dir + args.data_name + "_random.txt"
        D_random(args.data_file,args.segmented_file,args.max_seq_length)

    else:
        args.segmented_file = args.data_dir + args.data_name + "_s.txt"
    
    if not os.path.exists(args.segmented_file):
        D(args.data_file,args.segmented_file,args.max_seq_length)

    # _, train_seq = get_seqs_and_matrixes("training", args.segmented_file)原
    train_user_ids, train_seq = get_seqs_and_matrixes("training", args.segmented_file)
    _, user_seq, max_item, valid_rating_matrix, test_rating_matrix = get_seqs_and_matrixes("rating", args.data_file)

    args.item_size = max_item + 2
    args.mask_id = max_item + 1

    # ==================== 有向窗口共现：有则读，无则建 ====================
    if args.cooccur_matrix_save_path is None:
        cooccur_dir = os.path.join(args.output_dir, "cooccurrence")
        os.makedirs(cooccur_dir, exist_ok=True)
        args.cooccur_matrix_save_path = os.path.join(
            cooccur_dir,
            f"{args.data_name}_directed_w{args.cooccur_window_size}"
            f"_holdout{args.cooccur_holdout_last}_decay_v1.pkl"
        )

    cooccur_matrix = build_or_load_directed_cooccurrence_matrix(
        data_file=args.data_file,
        item_size=args.item_size,
        save_path=args.cooccur_matrix_save_path,
        holdout_last=args.cooccur_holdout_last,
        window_size=args.cooccur_window_size,
        use_decay=True,
    )

    args_str = f"{args.model_name}-{args.data_name}-{args.model_idx}"
    args.log_file = os.path.join(args.output_dir, args_str + ".txt")
    show_args_info(args)
    with open(args.log_file, "a") as f: 
        f.write(str(args) + "\n")

    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    if args.eval_only:
        cluster_dataloader, training_dataloader, eval_dataloader = [],[],[]
        training_dataset = None
        testing_dataset = DatasetForInDiRec(args, user_seq, data_type="test")
        testing_sampler = RandomSampler(testing_dataset)
        testing_dataloader = DataLoader(testing_dataset, sampler=testing_sampler, batch_size=args.batch_size, drop_last=True)
    else:
        args.train_matrix = valid_rating_matrix

        cluster_dataset = DatasetForInDiRec(args, train_seq, data_type="train")
        cluster_sampler = SequentialSampler(cluster_dataset)
        cluster_dataloader = DataLoader(cluster_dataset, sampler=cluster_sampler, batch_size=args.batch_size, drop_last=True)
        print(f"Cluster dataloader length: {len(cluster_dataloader)}")

        training_dataset = DatasetForInDiRec(args, train_seq, data_type="train")
        training_sampler = RandomSampler(training_dataset)
        training_dataloader = DataLoader(training_dataset, sampler=training_sampler, batch_size=args.batch_size, drop_last=True)
        #使用没有分割的数据进行构建共现矩阵
        # train_dataset = DatasetForInDiRec(args, raw_user_seqs, data_type="train")
        # 评估/测试都基于 user_seq（按行号=样本索引）
        eval_dataset = DatasetForInDiRec(args, user_seq, data_type="valid")
        eval_sampler = RandomSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, drop_last=True)

        testing_dataset = DatasetForInDiRec(args, user_seq, data_type="test")
        testing_sampler = RandomSampler(testing_dataset)
        testing_dataloader = DataLoader(testing_dataset, sampler=testing_sampler, batch_size=args.batch_size, drop_last=True)

    model = InDiRec(device, args=args)

    if args.llm_user_pkl is not None:
        model.load_llm_user_bank(args.llm_user_pkl)

    trainer = InDiRecTrainer(
        model,
        training_dataloader,
        training_dataset,
        cluster_dataloader,
        eval_dataloader,
        testing_dataloader,
        device,
        args
    )

    trainer.train_user_ids = train_user_ids
    trainer.cooccur_matrix = cooccur_matrix

    model.to(device)
    

    if args.eval_only:
        trainer.args.train_matrix = test_rating_matrix
        trainer.load(args.checkpoint_path)

        print(f"==================== Loading trained model from {args.checkpoint_path} for test...")
        scores, log_info = trainer.test(0) # set epoch equal 0 to print results directly.

    else:
        print(f"==================== Training ...")
        start_time = datetime.now()
        with open(args.log_file, "a") as f:
            f.write(f"====================  Training InDiRec at: {start_time} ====================" + "\n")
        
        early_stopping = EarlyStopping(args.checkpoint_path, patience=10, verbose=True)

        best_valid_ndcg = float("-inf")
        best_epoch = -1
        for epoch in range(args.epochs):
            trainer.train(epoch)
            if epoch % 5 == 0:
                scores, _ = trainer.valid(epoch)
                valid_ndcg_20 = round(scores[-1], 4)
                if valid_ndcg_20 > best_valid_ndcg:
                    best_valid_ndcg = valid_ndcg_20
                    trainer.best_epoch = epoch
                    best_epoch = epoch
                    print(f"✨ [Best so far] Epoch {epoch} | NDCG@20={valid_ndcg_20}")

                early_stopping(np.array(scores[-2:]), trainer.model)
                if early_stopping.early_stop:
                    print("==================== Early stopping triggered!")
                    break


        trainer.args.train_matrix = test_rating_matrix
        print("====================  Testing Phase  ====================")
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0)
        
        end_time = datetime.now()
        cost_time = str(end_time-start_time)
        print(args_str)
        print(result_info)
        with open(args.log_file, "a") as f:
            f.write(f"Finished at: {end_time} ==============================================================================================================" + "\n")
            # f.write(f" diff_weight:{args.diff_weight} || cl_weight:{args.cl_weight} || intent_num: {args.intent_num} || dropout: {args.hidden_dropout_prob} || w: {args.w} || p: {args.p} || timesteps: {args.timesteps} || time_cost: {cost_time}" + "\n")
            f.write(f" diff_weight:{args.diff_weight} || cl_weight:{args.cl_weight} || dropout: {args.hidden_dropout_prob} || w: {args.w} || p: {args.p} || timesteps: {args.timesteps} || time_cost: {cost_time}" + "\n")

            f.write(result_info + "\n")
            f.write("======================================================================================================================================================" + "\n")