import os
import numpy as np
import tensorflow as tf

DATA_FILE = 'mtg_cards.txt'
MODEL_FILE = 'mtg_name_model.h5'
CHAR2IDX_FILE = 'char2idx.npy'
IDX2CHAR_FILE = 'idx2char.npy'


def load_data(filepath=DATA_FILE):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    vocab = sorted(set(text))
    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    text_as_int = np.array([char2idx[c] for c in text])
    return text_as_int, char2idx, idx2char, len(vocab)


def build_model(vocab_size, embedding_dim=256, rnn_units=512, batch_size=64):
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True),
        tf.keras.layers.Dense(vocab_size)
    ])


def train(epochs=30):
    text_as_int, char2idx, idx2char, vocab_size = load_data()
    seq_length = 20
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    dataset = sequences.map(split_input_target)
    dataset = dataset.shuffle(1000).batch(1, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    model = build_model(vocab_size, batch_size=1)
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model.fit(dataset, epochs=epochs)

    model.save(MODEL_FILE)
    np.save(CHAR2IDX_FILE, char2idx)
    np.save(IDX2CHAR_FILE, idx2char)


def generate(start_string='', num_generate=20, temperature=1.0):
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError('Model not found. Train the model first.')

    char2idx = np.load(CHAR2IDX_FILE, allow_pickle=True).item()
    idx2char = np.load(IDX2CHAR_FILE)
    vocab_size = len(idx2char)

    model = build_model(vocab_size, batch_size=1)
    model.load_weights(MODEL_FILE)
    model.build(tf.TensorShape([1, None]))

    input_eval = [char2idx.get(s, 0) for s in start_string.lower()]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0) / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    return start_string + ''.join(text_generated)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train or generate MTG card names.')
    subparsers = parser.add_subparsers(dest='command')

    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--epochs', type=int, default=30)

    gen_parser = subparsers.add_parser('generate', help='Generate names')
    gen_parser.add_argument('--start', type=str, default='')
    gen_parser.add_argument('--length', type=int, default=20)
    gen_parser.add_argument('--temperature', type=float, default=1.0)

    args = parser.parse_args()
    if args.command == 'train':
        train(epochs=args.epochs)
    elif args.command == 'generate':
        print(generate(args.start, args.length, args.temperature))
    else:
        parser.print_help()
