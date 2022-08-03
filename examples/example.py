import os
import sys
from paddle import rank
from tqdm import tqdm
import numpy as np

# debug code
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rocketqa_path = os.path.join(parent_dir, "rokectqa")
sys.path.append(parent_dir)
sys.path.append(rocketqa_path)
import rocketqa


def train_dual_encoder(base_model, train_set):
    dual_encoder = rocketqa.load_model(
        model=base_model, use_cuda=True, device_id=0, batch_size=2)
    dual_encoder.train(
        train_set, 4, 'de_en_models', save_steps=100,
        learning_rate=1e-5,
        log_folder='de_en_log')


def train_cross_encoder(base_model, train_set):
    cross_encoder = rocketqa.load_model(
        model=base_model, use_cuda=True, device_id=0, batch_size=2)
    cross_encoder.train(
        train_set, 2, 'ce_en_models', save_steps=10,
        learning_rate=1e-5, log_folder='ce_en_log'
    )


def test_dual_encoder(model, q_file, tp_file):
    query_list = []
    para_list = []
    title_list = []
    for line in open(q_file):
        query_list.append(line.strip())

    for line in open(tp_file):
        t, p = line.strip().split('\t')
        para_list.append(p)
        title_list.append(t)

    dual_encoder = rocketqa.load_model(
        model=model, use_cuda=True, device_id=0, batch_size=32)
    # encode query
    dual_encoder.encode_query(query=query_list)
    # encode paragraph
    dual_encoder.encode_para(para=para_list, title=title_list)
    # get match result
    ranking_score = dual_encoder.matching(
        query=query_list,
        para=para_list[:len(query_list)],
        title=title_list[:len(query_list)]
    )


def test_cross_encoder(model, q_file, tp_file):

    query_list = []
    para_list = []
    title_list = []
    for line in open(q_file):
        query_list.append(line.strip())

    for line in open(tp_file):
        t, p = line.strip().split('\t')
        para_list.append(p)
        title_list.append(t)

    cross_conf = rocketqa.load_model(
        model=model, use_cuda=True, device_id=0, batch_size=32)
    cross_encoder = rocketqa.rocketqa.DualEncoder(**cross_conf)
    ranking_score = cross_encoder.matching(
        query=query_list,
        para=para_list[:len(query_list)],
        title=title_list[:len(query_list)]
    )
    ranking_score = list(ranking_score)
    ranks = sorted(ranking_score, reverse=True)
    rank_top = ranks[5]



if __name__ == "__main__":
    import os
    # finetune model
    train_dual_encoder('zh_dureader_de', './examples/data/dual.train.tsv')
    # train_cross_encoder('zh_dureader_ce', './examples/data/cross.train.tsv')

    # test rocketqa model
    #test_dual_encoder('zh_dureader_de_v2', './data/dureader.q', './data/marco.tp.1k')
    #test_cross_encoder('zh_dureader_de_v2', './data/dureader.q', './data/marco.tp.1k')

    # test your own model
    now_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(now_dir)
    data_dir = os.path.join(now_dir, "data")
    query_path = os.path.join(data_dir, "dureader.q")
    paragraph_path = os.path.join(data_dir, "dureader.para")
    config_path = os.path.join(parent_dir, "de_en_models", "config.json")
    # test_dual_encoder(config_path, query_path, paragraph_path)
    # test_dual_encoder('./de_models/config.json', './data/dureader.q', './data/marco.tp.1k')

    # test_cross_encoder(config_path, query_path, paragraph_path)
    # test_cross_encoder('./ce_models/config.json', './data/dureader.q', './data/marco.tp.1k')
