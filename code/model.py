import tensorflow.keras as keras
from tensorflow.keras import layers, regularizers
from positional_encoder import PositionalEncoding

def build_model(model_type: str, input_length: int, n_classes: int,
                n_embeddings:int, n_tokens:int,
                conv_layers:list, conv_activation:str, reg:float,
                rnn_layers:list, rnn_activation:str,
                dense_layers:list, dense_activation:str,
                attention_layers:list):
    """
    Builds and returns a model based on the specified type: 'rnn', 'gru', or 'attention'.
    
    :param model_type: One of 'rnn', 'gru', or 'attention'
    :param input_length: Length of the input sequence
    :param n_tokens: Number of distinct tokens in input
    :param n_classes: Number of output classes
    :return: Compiled keras.Model
    """
    inputs = keras.Input(shape=(input_length,), name='input_tokens')

    # Embedding layer
    x = layers.Embedding(input_dim=n_tokens,
                         output_dim=n_embeddings,
                         input_length=input_length)(inputs)

    # Reshape for Conv1D
    if model_type == 'attention':
        x = PositionalEncoding(max_steps=input_length, max_dims=n_embeddings)(x)

    k_reg = regularizers.l2(reg) if reg else None
        
    for layer in conv_layers:

        x = layers.Conv1D(filters=layer['filters'],
                          kernel_size=layer['kernel_size'],
                          strides=layer['strides'],
                          activation=conv_activation,
                          kernel_regularizer=k_reg)(x)

    # Model building
    if model_type in ['rnn', 'gru']:
        RNNLayer = layers.SimpleRNN if model_type == 'rnn' else layers.GRU
        for layer in rnn_layers:
            x = RNNLayer(units=layer['units'],
                         activation=rnn_activation,
                         return_sequences=True)(x)
            pool_size = layer.get('pool_size', 2)
            x = layers.AveragePooling1D(pool_size=pool_size)(x)

        x = RNNLayer(units=rnn_layers[-1]['units'],
                     activation=rnn_activation,
                     return_sequences=False)(x)
    elif model_type == 'attention':
        for layer in attention_layers:
            x = layers.MultiHeadAttention(num_heads=layer['attention_heads'],
                                          key_dim=layer['key_dim'])(x, x)
            pool_size = layer.get('pool_size', 2)
            x = layers.AveragePooling1D(pool_size=pool_size)(x)
        x = layers.GlobalMaxPooling1D()(x)
    else:
        raise ValueError("Invalid model_type. Choose from 'rnn', 'gru', 'attention'.")

    # Dense layers
    for layer in dense_layers:
        x = layers.Dense(layer['units'], activation=dense_activation)(x)

    # Output
    outputs = layers.Dense(n_classes, activation='softmax')(x)

    # Compile model
    model = keras.Model(inputs=inputs, outputs=outputs, name=f"{model_type}_model")
    model.compile(optimizer=keras.optimizers.Adam(clipnorm=1e-2),
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    
    return model
