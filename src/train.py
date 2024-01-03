import collections
import gc
import math
import random
from enum import Enum
from pathlib import Path
from typing import Self

import numpy as np
import torch
from logzero import logger
from pydantic import BaseModel
from sentence_transformers import InputExample, SentenceTransformer, datasets, losses, models
from sentence_transformers.evaluation import ParaphraseMiningEvaluator
from tqdm import tqdm

MODEL_NAME = "cl-tohoku/bert-base-japanese-v3"
POOLING_MODE = "mean"

SEED = 42

ROOT_DIRPATH = Path(__file__).resolve().parent
TRAIN_INPUT_FILEPATH = ROOT_DIRPATH / "../data/interium/jsnli_1.1/train_w_filtering.tsv"
VALID_INPUT_FILEPATH = ROOT_DIRPATH / "../data/interium/jsnli_1.1/dev.tsv"
# NOTE: スラッシュが含まれていると出力時にエラーになる
VALID_NAME = "jsnli_1.1_dev"

TRAIN_BATCH_SIZE = 128
WARMUP_RATIO = 0.1
NUM_EPOCHS = 1
SCHEDULER = "warmupcosine"

MODEL_OUTPUT_FILEPATH = str(ROOT_DIRPATH / f"../data/processed/{MODEL_NAME.replace('/', '_')}")


def set_seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Label(Enum):
    ENTAILMENT = "entailment"
    NEUTRAL = "neutral"
    CONTRADICTION = "contradiction"


class PairedSentence(BaseModel):
    sentence1: str
    sentence2: str
    label: Label

    @classmethod
    def from_text(cls, text: str) -> Self:
        # NOTE: 判定結果、文1、文2がタブ区切りの1行で保存されている
        # NOTE: 文1, 文2は形態素ごとにスペースで区切られている
        elements = text.strip().split("\t")
        label = elements[0]
        sentence1 = elements[1].replace(" ", "")
        sentence2 = elements[2].replace(" ", "")
        return cls(sentence1=sentence1, sentence2=sentence2, label=Label(label))


class SentenceCollection(BaseModel):
    sentence: str
    entailment: set = set()
    neutral: set = set()
    contradiction: set = set()

    def set_labeled_sentence(self, label: Label, sentence: str) -> None:
        if label == Label.ENTAILMENT:
            self.entailment.add(sentence)
        elif label == Label.NEUTRAL:
            self.neutral.add(sentence)
        elif label == Label.CONTRADICTION:
            self.contradiction.add(sentence)
        else:
            raise KeyError(f"not defined label: {label}")

    @property
    def is_train_target(self) -> bool:
        if len(self.entailment) > 0 and len(self.contradiction) > 0:
            return True
        else:
            return False


def load_model(model_name: str, pooling_mode: str) -> SentenceTransformer:
    word_embedding_model = models.Transformer(model_name_or_path=model_name, max_seq_length=None)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode=pooling_mode)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model


def add_to_sentence_collections(
    sentence_collections: dict[str, SentenceCollection], sentence1: str, sentence2: str, label: Label
) -> dict[str, SentenceCollection]:
    if sentence1 not in sentence_collections:
        sentence_collections[sentence1] = SentenceCollection(sentence=sentence1)
    sentence_collections[sentence1].set_labeled_sentence(label, sentence2)
    return sentence_collections


def load_dataset_for_paraphrase_mining_evaluator(filepath: Path) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """
    https://www.sbert.net/docs/package_reference/evaluation.html#sentence_transformers.evaluation.ParaphraseMiningEvaluator

    Returns:
        sentences_map: A dictionary that maps sentence-ids to sentences, i.e. sentences_map[id] => sentence.
        duplicates_list: Duplicates_list is a list with id pairs [(id1, id2), (id1, id5)] that identifies the duplicates / paraphrases in the sentences_map
    """
    sentence2id: dict[str, int] = collections.defaultdict(lambda: len(sentence2id))
    duplicates_list: list[tuple[str, str]] = []
    with open(filepath, "r") as f:
        # NOTE: メモリセーフにするため、一行ずつ読み込み、整形する
        for line in tqdm(f):
            paired_sentence = PairedSentence.from_text(line)
            ids = (str(sentence2id[paired_sentence.sentence1]), str(sentence2id[paired_sentence.sentence2]))
            if paired_sentence.label == Label.ENTAILMENT:
                duplicates_list.append(ids)
    sentences_map = {str(id_): sentence for sentence, id_ in sentence2id.items()}
    return sentences_map, duplicates_list


def load_dataset_for_dataloader(filepath: Path) -> list[InputExample]:
    sentence_collections: dict[str, SentenceCollection] = {}
    with open(filepath, "r") as f:
        # NOTE: メモリセーフにするため、一行ずつ読み込み、整形する
        for line in tqdm(f):
            paired_sentence = PairedSentence.from_text(line)
            sentence_collections = add_to_sentence_collections(
                sentence_collections, paired_sentence.sentence1, paired_sentence.sentence2, paired_sentence.label
            )
            sentence_collections = add_to_sentence_collections(
                sentence_collections, paired_sentence.sentence2, paired_sentence.sentence1, paired_sentence.label
            )

    samples: list[InputExample] = []
    for sentence1, sentence_collection in tqdm(sentence_collections.items()):
        if sentence_collection.is_train_target:
            samples.append(
                InputExample(
                    texts=[
                        sentence1,
                        random.choice(list(sentence_collection.entailment)),
                        random.choice(list(sentence_collection.contradiction)),
                    ]
                )
            )
            samples.append(
                InputExample(
                    texts=[
                        random.choice(list(sentence_collection.entailment)),
                        sentence1,
                        random.choice(list(sentence_collection.contradiction)),
                    ]
                )
            )

    del sentence_collections
    gc.collect()

    # NOTE: 重複サンプルの削除
    samples = list({"".join(sample.texts): sample for sample in samples}.values())
    return samples


def main() -> None:
    """学習用のコード

    https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/nli/training_nli_v2.py
    https://www.ogis-ri.co.jp/otc/hiroba/technical/similar-document-search/part18.html
    """
    set_seed_everything(seed=SEED)
    logger.info("start loading dataset.")
    model = load_model(model_name=MODEL_NAME, pooling_mode=POOLING_MODE)
    train_samples = load_dataset_for_dataloader(TRAIN_INPUT_FILEPATH)
    train_dataloader = datasets.NoDuplicatesDataLoader(train_samples, batch_size=TRAIN_BATCH_SIZE)
    sentences_map, duplicates_list = load_dataset_for_paraphrase_mining_evaluator(VALID_INPUT_FILEPATH)
    dev_evaluator = ParaphraseMiningEvaluator(sentences_map, duplicates_list, name=VALID_NAME)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    logger.info(
        f"end loading_dataset: [train_size: {len(train_samples)}][train_batch_size: {len(train_dataloader)}][valid_size: {len(duplicates_list)}]"
    )

    warmup_steps = math.ceil(len(train_dataloader) * NUM_EPOCHS * WARMUP_RATIO)

    logger.info(f"train start: [epochs: {NUM_EPOCHS}][warmup_steps: {warmup_steps}]")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=dev_evaluator,
        epochs=NUM_EPOCHS,
        evaluation_steps=int(len(train_dataloader) * 0.1),
        warmup_steps=warmup_steps,
        scheduler=SCHEDULER,
        output_path=MODEL_OUTPUT_FILEPATH,
        # NOTE: Set to True, if your GPU supports FP16 operations
        use_amp=False,
    )
    logger.info("train end.")


if __name__ == "__main__":
    main()
