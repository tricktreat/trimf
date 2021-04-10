# A Trigger-Sense Memory Flow Framework for Joint Entity andRelation Extraction
Code for TriMF: "A Trigger-Sense Memory Flow Framework for Joint Entity and Relation Extraction". accepted at WWW 2021.

## Description

![](./assets/overview.png)
## Setup
### Requirements

```bash
pip install -r requirements.txt
```

### Dataset Format

```json
{"tokens": ["allan", "chernoff", "live", "from", "the", "new", "york", "stock", "exchange", "with", "more", "."], "entities": [{"type": "PER", "start": 0, "end": 2}, {"type": "FAC", "start": 5, "end": 9}], "relations": [{"type": "PHYS", "head": 0, "tail": 1}], "orig_id": "CNN_ENG_20030530_130025.12-4", "dependency": [{"tail": 0, "head": 1, "type": "nsubj"}, {"tail": 1, "head": 1, "type": "ROOT"}, {"tail": 2, "head": 1, "type": "advmod"}, {"tail": 3, "head": 2, "type": "prep"}, {"tail": 4, "head": 8, "type": "det"}, {"tail": 5, "head": 8, "type": "amod"}, {"tail": 6, "head": 7, "type": "compound"}, {"tail": 7, "head": 8, "type": "compound"}, {"tail": 8, "head": 3, "type": "pobj"}, {"tail": 9, "head": 1, "type": "prep"}, {"tail": 10, "head": 9, "type": "pobj"}, {"tail": 11, "head": 1, "type": "punct"}], "ltokens": ["aol", "time", "warner", "and", "microsoft", "are", "burying", "the", "hatchet", "."], "rtokens": ["bring", "us", "up", "to", "speed", "."]}, ··· ]
```

## Examples

Download checkpoints from this [link](https://drive.google.com/drive/folders/140PqTY417t3wpUYa3Yj-taGsorH5VChV?usp=sharing), and save in the `data` folder.

Train:

```
python trimf.py train --config configs/example.conf
```

Evaluate:

```
python trimf.py eval --config configs/batch_eval.conf
```

## Acknowledgement

SpERT from https://github.com/markus-eberts/spert.git

SciBERT from https://github.com/allenai/scibert