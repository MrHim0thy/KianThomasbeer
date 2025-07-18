# KianThomasbeer

Website: <https://mrhim0thy.github.io/KianThomasbeer/>

## MTG Card Generator

This repository includes a simple character-level LSTM model for generating Magic: The Gathering card names. The model is trained on a small list of existing card names found in `mtg_cards.txt`.

### Requirements
- Python 3.11
- TensorFlow 2.15

Install dependencies with:
```bash
pip install tensorflow-cpu==2.15 markovify
```

### Training
To train the model run:
```bash
python mtg_generator.py train --epochs 30
```
This will create `mtg_name_model.h5`, `char2idx.npy`, and `idx2char.npy`.

### Generating names
After training you can generate new card names:
```bash
python mtg_generator.py generate --start "Black" --length 10 --temperature 0.8
```
`--start` provides an optional seed. `--temperature` controls randomness.
