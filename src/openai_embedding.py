# 比較用にOPENAIのembeddingを取得して精度出力する
# https://platform.openai.com/docs/guides/embeddings/use-cases
import os
from pathlib import Path

import polars as pl
import tiktoken
from dotenv import load_dotenv
from logzero import logger
from openai import OpenAI
from tqdm import tqdm

ROOT_DIRPATH = Path(__file__).resolve().parent
OPENAI_ENVPATH = ROOT_DIRPATH / "../.env"

load_dotenv(OPENAI_ENVPATH, override=True)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

MODEL = "text-embedding-ada-002"

ENCODING_NAME = "cl100k_base"
VALID_INPUT_FILEPATH = ROOT_DIRPATH / "../data/interium/jsnli_1.1/dev.tsv"

OUTPUT_DIRPATH = ROOT_DIRPATH / "../data/interium/openai/"
OUTPUT_FILEPATH = OUTPUT_DIRPATH / "jsnli_1.1_dev_with_embedding.parquet"


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def estimate_tokens() -> None:
    """念のため金額の試算を行う

    186563tokens
    ada v2: 0.0001 / 1k tokens
    -> $ 0.019

    """
    num_total_tokens = 0
    with open(VALID_INPUT_FILEPATH, "r") as f:
        for line in tqdm(f):
            _, sentence1, sentence2 = line.strip().split("\t")
            sentence1 = sentence1.replace(" ", "")
            sentence2 = sentence2.replace(" ", "")
            num_total_tokens += num_tokens_from_string(sentence1, ENCODING_NAME)
            num_total_tokens += num_tokens_from_string(sentence2, ENCODING_NAME)

    logger.info(f"num_total_tokens: {num_total_tokens}")


def get_embedding(text: str) -> list[float]:
    return client.embeddings.create(input=[text], model=MODEL).data[0].embedding


def main() -> None:
    OUTPUT_DIRPATH.mkdir(exist_ok=True)
    df_dev = pl.read_csv(
        VALID_INPUT_FILEPATH, separator="\t", has_header=False, new_columns=["label", "sentence1", "sentence2"]
    )
    df_dev = df_dev.with_columns(
        pl.col("sentence1").str.replace_all(" ", ""), pl.col("sentence2").str.replace_all(" ", "")
    )
    df_dev = df_dev.with_columns(
        pl.col("sentence1").map_elements(get_embedding).alias("embedding1"),
        pl.col("sentence2").map_elements(get_embedding).alias("embedding2"),
    )
    df_dev.write_parquet(OUTPUT_FILEPATH)


if __name__ == "__main__":
    main()
