from tensorflow import keras
from keras.utils.vis_utils import plot_model

def define_rnn_model(vocab_size, test_len):
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 100, input_length=test_len))
    model.add(keras.layers.SimpleRNN(100))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    plot_model(model, to_file='illu/rnn_model.png', show_shapes=True)
    return model

def define_cnn_model(vocab_size, test_len):
    # Three channel
    sizes = [4, 6, 8]
    flats = []
    inputss = []
    for size in sizes:
        inputs = keras.layers.Input(shape=(test_len,))
        inputss.append(inputs)
        embedding = keras.layers.Embedding(vocab_size, 100)(inputs)
        conv = keras.layers.Conv1D(filters=32, kernel_size=size, activation='relu')(embedding)
        drop = keras.layers.Dropout(0.5)(conv)
        pool = keras.layers.MaxPooling1D(pool_size=2)(drop)
        flat = keras.layers.Flatten()(pool)
        flats.append(flat)
    merged = keras.layers.concatenate(flats)
    dense = keras.layers.Dense(10, activation='relu')(merged)
    outputs = keras.layers.Dense(1, activation='sigmoid')(dense)
    model = keras.Model(inputs=[inputss[0], inputss[1], inputss[2]], outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    plot_model(model, show_shapes=True, to_file='illu/cnn.png')
    return model

def define_lstm_model(vocab_size, test_len):
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 100, input_length=test_len))
    model.add(keras.layers.LSTM(100, dropout=0.3, recurrent_dropout=0))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    model.summary()
    plot_model(model, to_file='illu/lstm_model.png', show_shapes=True)
    return model

def define_gru_model(vocab_size, test_len):
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 100, input_length=test_len))
    model.add(keras.layers.SpatialDropout1D(0.3))
    model.add(keras.layers.GRU(100))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    model.summary()
    plot_model(model, to_file='illu/gru_model.png', show_shapes=True)
    return model

def define_bi_model(vocab_size, test_len):
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 100, input_length=test_len))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(100, dropout=0.3)))
    model.add(keras.layers.Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    model.summary()
    plot_model(model, to_file='illu/bi_directional_model.png', show_shapes=True)
    return model

def define_cnn_rnn_model(vocab_size, test_len):
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 100, input_length=test_len))
    model.add(keras.layers.Conv1D(32, 7, activation='relu'))
    model.add(keras.layers.MaxPooling1D(5))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(100, return_sequences=True, dropout=0.3)))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(100, dropout=0.3)))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    model.summary()
    plot_model(model, to_file='illu/cnn_rnn.png', show_shapes=True)
    return model

