# Urdu Sentence Segmentation and Subword Tokenization

A rule-based Urdu sentence segmenter that splits on sentence-ending verbs and auxiliaries while guarding against conjunctions, compared with UrduHack, plus a SentencePiece BPE model trained on an Urdu corpus.

**Interactive demo:** [Run it in your browser](https://www.laibakhawar.com/work/urdu-segmentation#demo)

## How it works

All code is in `i21_1697.ipynb` (run on Google Colab).

**Preprocessing (cell 19).** Adds spaces around `۔`, `؟` and `!`, removes curly quotes, collapses whitespace and puts spaces around digits.

**Rule-based segmenter (cell 22, `improved_custom_sentence_tokenizer`).**

- Walks the text word by word.
- Ends a sentence at an explicit `۔`, `؟` or `!`.
- Also ends a sentence after a word from a set of 28 common Urdu sentence-final words (for example `ہے`, `ہیں`, `تھا`, `تھی`, `گا`, `گی`, `گیا`, `دیا`, `چاہیے`), as long as neither that word nor the word before it is in a set of 26 conjunctions and relative words (for example `اور`, `لیکن`, `کہ`, `جو`, `اگر`, `کیونکہ`, `تاکہ`). Unpunctuated Urdu text often has no `۔`, so these verb endings are the main boundary signal.
- Also runs UrduHack's `sentence_tokenizer` on the same text and returns both outputs for comparison.

**Evaluation (cells 25 and 26).** Three sample passages are embedded in the notebook, each with a raw version and a reference list of sentences. For each sentence the segmenter produces, the notebook builds bag-of-words vectors with `CountVectorizer`, takes the highest cosine similarity against any reference sentence, and reports the mean as "Average Accuracy".

**SentencePiece (cells 29 to 32).** Trains a BPE model on `urdu-corpus.txt` with `vocab_size=500`, `character_coverage=0.9995`, then encodes a passage and splits the token stream back into sentences at punctuation for display.

**Custom encode/decode (cell 35).** Despite the heading, this is a simple word-level ID mapper (regex tokens to incrementing IDs and back), not SentencePiece's own `encode`/`decode`.

## Results

| Passage | Cosine-similarity proxy score | Source |
| --- | --- | --- |
| Passage 3 (`passage_number = 2`) | 75.38% | recorded in cell 26 |
| Passage 1 | 66.37% | reproduced by re-running the notebook code |
| Passage 2 | 67.96% | reproduced by re-running the notebook code |

This score is a **cosine-similarity proxy**, not a segmentation accuracy or boundary F1. It rewards each predicted sentence for overlapping in vocabulary with some reference sentence. It does not penalise missing sentences or check that the boundaries are in the right place, and splitting a sentence into fragments can still score well. For example, the recorded output for passage 3 breaks after `ہوں` in `ہوں گے`, so `گے` becomes its own one-word "sentence".

## Repository contents

- `i21_1697.ipynb`: segmentation, evaluation and SentencePiece notebook.
- `urdu-corpus.txt`: Urdu text corpus (11.5 MB) used to train SentencePiece and as sample input.
- `requirements.txt`: jupyter, matplotlib, numpy, pandas, python-Levenshtein, scikit-learn, sentencepiece, tensorflow.

## Running it

The notebook was run on Google Colab (Python 3.10 according to its outputs).

```bash
git clone https://github.com/LaibaKhawar/Urdu-Segmentation-and-Tokenization.git
cd Urdu-Segmentation-and-Tokenization
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install "urduhack[tf]"
jupyter notebook i21_1697.ipynb
```

Notes:

- `urduhack` is imported but is not in `requirements.txt`; the notebook installs it with `!pip install urduhack[tf]` and pins `tensorflow==2.13` in cell 14. UrduHack downloads its models on first use (`urduhack.download()`).
- Cells 17 and 29 read the corpus from `/content/sample_data/urdu-corpus.txt` (a Colab path). Change it to `urdu-corpus.txt` when running locally.
- Cell 29 calls `spm.SentencePieceTrainer` before `sentencepiece` is imported in cell 31; import it first.
- Start from cell 16. The main pipeline is cells 16 to 35.

## Known limitations

- **Early cells reference missing files.** Cells 4 and 9 come from the assignment's starter template and read `sent-test.txt` and `sent-segmented.txt`, which are not in the repository (cell 4's recorded output is a `FileNotFoundError`). Cell 11 also uses `word_tokenizer` without importing it.
- **The metric is a proxy** (see Results). It cannot be compared with boundary-level precision, recall or F1.
- The evaluation set is three short passages embedded in the notebook.
- The sentence-ending word list is hand-written and will both miss boundaries (verbs not in the list) and over-split (listed words used mid-sentence, as with `ہوں` before `گے` above). `گے` and `گی` appear in both the endings set and the conjunctions set, so they can never trigger a split themselves.
- The SentencePiece section reconstructs "sentences" by splitting on punctuation in the token stream, so it is a tokenization demo rather than a segmentation method.

## Author

[Laiba Khawar](https://github.com/LaibaKhawar) · [LinkedIn](https://www.linkedin.com/in/laibakhawar/)
