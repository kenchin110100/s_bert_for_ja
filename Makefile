download:
	wget -P ./data/raw/ https://nlp.ist.i.kyoto-u.ac.jp/DLcounter/lime.cgi?down=https://nlp.ist.i.kyoto-u.ac.jp/nl-resource/JSNLI/jsnli_1.1.zip&name=JSNLI.zip
	mv ./data/raw/*zip* ./data/raw/jsnli_1.1.zip
	unzip ./data/raw/jsnli_1.1.zip -d ./data/interium/

test:
	poetry run pytest tests/ --cov=src/ --cov-report=term-missing

openai_embedding:
	poetry run python src/openai_embedding.py

train:
	poetry run python src/train.py

evaluation:
	poetry run python src/evaluation.py