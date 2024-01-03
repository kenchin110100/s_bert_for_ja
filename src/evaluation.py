from pathlib import Path

import polars as pl
import torch
from logzero import logger
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import ParaphraseMiningEvaluator
from torch import Tensor
from train import load_dataset_for_paraphrase_mining_evaluator, load_model

ROOT_DIRPATH = Path(__file__).resolve().parent
VALID_INPUT_FILEPATH = ROOT_DIRPATH / "../data/interium/jsnli_1.1/dev.tsv"
PRECOMPUTED_EMBEDDING_FILEPATH = ROOT_DIRPATH / "../data/interium/openai/jsnli_1.1_dev_with_embedding.parquet"
S_BERT_MODELPATH = ROOT_DIRPATH / "../data/processed/cl-tohoku_bert-base-japanese-v3"
OUTPUT_DIRPATH = ROOT_DIRPATH / "../data/processed/evaluation"

MODEL_NAME = "cl-tohoku/bert-base-japanese-v3"
POOLING_MODE = "mean"


class DummyModel:
    def __init__(self, sentence2embedding: dict[str, list[float]]) -> None:
        self.sentence2embedding = sentence2embedding

    def encode(
        self,
        sentences: list[str],
        show_progress_bar: bool = False,
        batch_size: int = 32,
        convert_to_tensor: bool = True,
    ) -> Tensor:
        """sentence-transformerのParaPhraseMiningEvaluatorを利用するためのダミーのメソッド

        以下を参考に実装
        https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/util.py#L137C11-L137C11

        Args:
            sentences (list[str]): 文字も配列
            show_progress_bar (bool, optional): ダミーの変数、利用しない
            batch_size (int, optional): ダミーの変数、利用しない
            convert_to_tensor (bool, optional): ダミーの変数、利用しない

        Returns:
            Tensor: embeddingされたベクトル
        """
        embeddings = [self.sentence2embedding[sentence] for sentence in sentences]
        return torch.tensor(embeddings)


def load_precomputed_embedding(filepath: Path) -> dict[str, list[float]]:
    """事前計算したembeddingを読み込むための関数（OpenAIのembeddingの精度を評価するために）

    Args:
        filepath (Path): ファイルパス（parquetを想定）

    Returns:
        dict[str, list[float]]: sentenceを入力して、embeddingが返却される辞書
    """
    sentence2embedding: dict[str, list[float]] = {}
    # NOTE: label, sentence1, sentence2, embedding1, embedding2という形式になっている
    for obj in pl.read_parquet(filepath).to_dicts():
        sentence1 = obj["sentence1"]
        sentence2 = obj["sentence2"]
        embedding1 = obj["embedding1"]
        embedding2 = obj["embedding2"]
        sentence2embedding[sentence1] = embedding1
        sentence2embedding[sentence2] = embedding2
    return sentence2embedding


def main() -> None:
    OUTPUT_DIRPATH.mkdir(exist_ok=True)
    sentences_map, duplicates_list = load_dataset_for_paraphrase_mining_evaluator(VALID_INPUT_FILEPATH)

    # openaiモデルの評価
    name = "openai_ada_v2"
    dev_evaluator = ParaphraseMiningEvaluator(sentences_map, duplicates_list, name=name)
    sentence2embedding = load_precomputed_embedding(PRECOMPUTED_EMBEDDING_FILEPATH)
    openai_model = DummyModel(sentence2embedding)
    ap = dev_evaluator(model=openai_model, output_path=OUTPUT_DIRPATH)
    logger.info(f"{name} average precision: {ap}")

    # Pretrainedモデルの評価
    name = MODEL_NAME.replace("/", "_")
    dev_evaluator = ParaphraseMiningEvaluator(sentences_map, duplicates_list, name=name)
    pretrained_model = load_model(model_name=MODEL_NAME, pooling_mode=POOLING_MODE)
    ap = dev_evaluator(model=pretrained_model, output_path=OUTPUT_DIRPATH)
    logger.info(f"{name} average precision: {ap}")

    # 学習したsentence bertの評価
    name = "s_bert"
    dev_evaluator = ParaphraseMiningEvaluator(sentences_map, duplicates_list, name=name)
    s_bert_model = SentenceTransformer(S_BERT_MODELPATH)
    ap = dev_evaluator(model=s_bert_model, output_path=OUTPUT_DIRPATH)
    logger.info(f"{name} average precision: {ap}")


if __name__ == "__main__":
    main()
