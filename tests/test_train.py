from pathlib import Path

from sentence_transformers import InputExample

from src.train import load_dataset_for_dataloader, load_dataset_for_paraphrase_mining_evaluator

TEST_DATA_FILEPATH = Path(__file__).resolve().parent / "data/test.tsv"


def test_load_dataset_for_dataloader() -> None:
    actual_samples = load_dataset_for_dataloader(TEST_DATA_FILEPATH)
    expected_samples = [
        InputExample(
            texts=[
                "２匹の犬が店の外で縛られており、自転車が店の壁にもたれかかっています。",
                "２匹の犬が外で縛られています。",
                "２匹の犬が自転車の近くで縛られています。",
            ]
        ),
        InputExample(
            texts=[
                "２匹の犬が外で縛られています。",
                "２匹の犬が店の外で縛られており、自転車が店の壁にもたれかかっています。",
                "２匹の犬が自転車の近くで縛られています。",
            ]
        ),
        InputExample(texts=["人が２頭の馬の間にひざまずいている", "人と二頭の馬がいます。", "二人は緑の森にいます。"]),
        InputExample(texts=["人と二頭の馬がいます。", "人が２頭の馬の間にひざまずいている", "二人は緑の森にいます。"]),
    ]
    assert len(actual_samples) == len(expected_samples)
    for actual_sample, expected_sample in zip(actual_samples, expected_samples):
        assert actual_sample.texts == expected_sample.texts


def test_load_dataset_for_paraphrase_mining_evaluator() -> None:
    actual_sencences_map, actual_duplicates_list = load_dataset_for_paraphrase_mining_evaluator(TEST_DATA_FILEPATH)

    expected_sentences_map = {
        "0": "２匹の犬が店の外で縛られており、自転車が店の壁にもたれかかっています。",
        "1": "２匹の犬が外で縛られています。",
        "2": "２匹の犬が自転車の近くで縛られています。",
        "3": "人が２頭の馬の間にひざまずいている",
        "4": "人と二頭の馬がいます。",
        "5": "二人は緑の森にいます。",
        "6": "人は森の中の崖に登り、他の人は見ています。",
        "7": "人が森の中の崖に登る。",
    }
    expected_duplicates_list = [("0", "1"), ("3", "4")]

    assert actual_sencences_map == expected_sentences_map
    assert actual_duplicates_list == expected_duplicates_list
