S_BERT_FOR_JA
====

日本語ファインチューニングしたsentence bertの作成を行う

## 分析設計

### コード

[train.py](src/train.py)

### モデル

[ci-tohoku/bert-base-japanese-v3](https://huggingface.co/cl-tohoku/bert-base-japanese-v3)

- [apache-2.0 ライセンス](https://choosealicense.com/licenses/apache-2.0/)

### データセット

[日本語SNLI(JSNLI)データセット](https://nlp.ist.i.kyoto-u.ac.jp/index.php?%E6%97%A5%E6%9C%AC%E8%AA%9ESNLI%28JSNLI%29%E3%83%87%E3%83%BC%E3%82%BF%E3%82%BB%E3%83%83%E3%83%88)

- [CC BY-SA 4.0 ライセンス](https://creativecommons.org/licenses/by-sa/4.0/)

### 評価指標

Paraphrase Mining F1 score

https://www.sbert.net/docs/package_reference/evaluation.html#sentence_transformers.evaluation.ParaphraseMiningEvaluator

### 比較モデル

- pretrained ci-tohoku/bert-base-japanese-v3
- openai ada v2 embedding

## 分析結果

### Average Precision

|model|average precision|
|-:|:-|
|openai_ada_v2|0.043452|
|pretrained (cl-tohoku/bert-base-japanese-v3)|0.010030|
|trained (from cl-tohoku/bert-base-japanese-v3)|0.170044|

※ trainedのモデルは評価データでearly stoppingしたのでやや過大評価気味

### 学習中の評価データでのAverage Precisionの推移

![学習中のAverage Precisionの推移](./reports/figures/ap_per_steps.png)

## 参考資料

- https://www.ogis-ri.co.jp/otc/hiroba/technical/similar-document-search/part18.html
- https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/nli/training_nli_v2.py

