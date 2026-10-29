"""Keras encoder compiled from a model plan.

Query is the target player: id, position, current form, and the pooled fixture
history. Current teammates and opposition players are cross-attended with that
query. Pad positions are zeroed and masked.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Embedding, Input, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2 as l2_regularizer

from premierlearning.learning import _output_activation
from premierlearning.learning_v2.plan import (
    SIMPLE_ROOTS, contains_nested_attention, sequence_length, tensor_key,
)


def _hidden_sizes(n_in: int, hparams) -> list:
    net_size = max(8, int(n_in * hparams.width_mult))
    sizes = [net_size]
    if hparams.n_hidden >= 2:
        sizes.append(max(4, net_size // 2))
    if hparams.n_hidden >= 3:
        sizes.append(max(2, net_size // 4))
    if hparams.n_hidden >= 4:
        sizes.append(max(2, net_size // 8))
    return sizes


def _safe_mask(mask):
    """Keep at least one key when a row is entirely padding, so softmax stays finite."""
    mask_f = tf.cast(mask, tf.float32)
    has = tf.reduce_max(mask_f, axis=1, keepdims=True)
    return tf.maximum(mask_f, 1.0 - has)


class _MultiHeadAttention(tf.keras.layers.Layer):
    """Scaled dot-product attention. TensorFlow 2.3 has no MultiHeadAttention layer."""

    def __init__(self, num_heads, key_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = int(num_heads)
        self.key_dim = int(key_dim)
        width = self.num_heads * self.key_dim
        self.query_dense = Dense(width, use_bias=False)
        self.key_dense = Dense(width, use_bias=False)
        self.value_dense = Dense(width, use_bias=False)
        self.output_dense = Dense(width)

    def _split_heads(self, tensor, batch, time):
        tensor = tf.reshape(tensor, [batch, time, self.num_heads, self.key_dim])
        return tf.transpose(tensor, [0, 2, 1, 3])

    def call(self, query, value, key_mask=None, training=None):
        batch = tf.shape(query)[0]
        query_len = tf.shape(query)[1]
        value_len = tf.shape(value)[1]
        queries = self._split_heads(self.query_dense(query), batch, query_len)
        keys = self._split_heads(self.key_dense(value), batch, value_len)
        values = self._split_heads(self.value_dense(value), batch, value_len)
        scores = tf.matmul(queries, keys, transpose_b=True)
        scores = scores * tf.math.rsqrt(tf.cast(self.key_dim, scores.dtype))
        if key_mask is not None:
            mask = tf.cast(key_mask, scores.dtype)
            scores = scores + (1.0 - mask)[:, None, None, :] * -1e9
        weights = tf.nn.softmax(scores, axis=-1)
        mixed = tf.matmul(weights, values)
        mixed = tf.transpose(mixed, [0, 2, 1, 3])
        flat = tf.reshape(mixed, [batch, query_len, self.num_heads * self.key_dim])
        return self.output_dense(flat)


class _HistoryBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, key_dim, dropout, regularizer, **kwargs):
        super().__init__(**kwargs)
        self.attn = _MultiHeadAttention(num_heads=n_heads, key_dim=key_dim)
        self.ffn_hidden = Dense(d_model * 2, activation='relu', kernel_regularizer=regularizer)
        self.ffn_out = Dense(d_model, kernel_regularizer=regularizer)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.drop = Dropout(dropout)

    def call(self, x, key_mask, mask_f, training=None):
        updated = self.attn(x, x, key_mask=key_mask, training=training)
        updated = self.drop(updated, training=training)
        x = self.norm1(x + updated)
        x = x * mask_f[:, :, None]
        hidden = self.ffn_hidden(x)
        hidden = self.drop(hidden, training=training)
        x = self.norm2(x + self.ffn_out(hidden))
        return x * mask_f[:, :, None]


class HistoryEncoder(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, n_layers, dropout, regularizer, **kwargs):
        super().__init__(**kwargs)
        key_dim = max(1, int(d_model) // int(n_heads))
        self.blocks = []
        for index in range(int(n_layers)):
            block = _HistoryBlock(
                d_model, n_heads, key_dim, dropout, regularizer, name='block_%s' % index,
            )
            self.blocks.append(block)
            setattr(self, 'block_%s' % index, block)

    def call(self, x, mask, training=None):
        mask_f = tf.cast(mask, x.dtype)
        x = x * mask_f[:, :, None]
        safe = _safe_mask(mask_f)
        for block in self.blocks:
            x = block(x, safe, mask_f, training=training)
        denom = tf.maximum(tf.reduce_sum(mask_f, axis=1, keepdims=True), 1.0)
        return tf.reduce_sum(x * mask_f[:, :, None], axis=1) / denom


def _layer_name(prefix, path):
    return ('%s_%s' % (prefix, path.replace('.', '_')))[:90]


def build_model(plan, vocab, hparams, loss_name='huber'):
    """Functional model. Inputs are named with ``tensor_key`` paths."""
    if contains_nested_attention(plan):
        raise NotImplementedError('nested squad attention is parsed but not built yet')
    extra = [child.name for child in plan.root.children if child.name not in SIMPLE_ROOTS]
    if extra:
        raise NotImplementedError(
            'simple model does not encode root fields: %s' % ', '.join(extra)
        )
    if plan.config.d_model % plan.config.n_heads != 0:
        raise ValueError('d_model must be divisible by n_heads')

    regularizer = l2_regularizer(float(hparams.l2))
    inputs = {}
    tables = {
        'player': Embedding(
            int(vocab.player_size), plan.config.player_emb_dim, name='player_emb',
        ),
        'team': Embedding(
            int(vocab.team_size), plan.config.team_emb_dim, name='team_emb',
        ),
        'position': Embedding(
            int(vocab.position_size), plan.config.position_emb_dim, name='position_emb',
        ),
    }

    def keras_input(path, shape, dtype):
        key = tensor_key(path)
        if key not in inputs:
            inputs[key] = Input(shape=shape, dtype=dtype, name=key)
        return inputs[key]

    def encode_leaf(node, seq_len):
        if seq_len:
            raw = keras_input(node.path, (seq_len,), 'int32' if node.dtype == 'embedding' else 'float32')
            if node.dtype == 'embedding':
                return tables[node.table](raw)
            return tf.expand_dims(tf.cast(raw, tf.float32), -1)
        raw = keras_input(node.path, (1,), 'int32' if node.dtype == 'embedding' else 'float32')
        if node.dtype == 'embedding':
            return tf.squeeze(tables[node.table](raw), axis=1)
        return tf.cast(raw, tf.float32)

    def encode(node, seq_len, query_d):
        if node.kind == 'leaf':
            return encode_leaf(node, seq_len)
        if node.kind == 'group':
            parts = [encode(child, seq_len, query_d) for child in node.children]
            return tf.concat(parts, axis=-1)
        if node.kind == 'transformer':
            if seq_len:
                raise NotImplementedError('nested transformer at %s' % node.path)
            return encode_history(node)
        if node.kind == 'attention':
            if seq_len:
                raise NotImplementedError(
                    'nested squad attention is parsed but not built yet (%s)' % node.path
                )
            return encode_squad(node, query_d)
        raise ValueError('Unknown node %s' % node.kind)

    def encode_history(node):
        length = sequence_length(plan.config, node)
        token = encode(node.item, length, None)
        mask = keras_input(node.path + '.mask', (length,), 'float32')
        token = token * tf.expand_dims(mask, -1)
        projected = Dense(
            plan.config.d_model, activation='relu', kernel_regularizer=regularizer,
            name=_layer_name('hist_proj', node.path),
        )(token)
        projected = projected * tf.expand_dims(mask, -1)
        return HistoryEncoder(
            plan.config.d_model, plan.config.n_heads, plan.config.n_layers,
            float(hparams.dropout_1), regularizer, name=_layer_name('hist', node.path),
        )(projected, mask)

    def encode_squad(node, query_d):
        length = plan.config.max_squad
        token = encode(node.item, length, None)
        mask = keras_input(node.path + '.mask', (length,), 'float32')
        token = token * tf.expand_dims(mask, -1)
        token = Dense(
            plan.config.d_model, activation='relu', kernel_regularizer=regularizer,
            name=_layer_name('squad_proj', node.path),
        )(token)
        token = token * tf.expand_dims(mask, -1)
        safe = _safe_mask(mask)
        attended = _MultiHeadAttention(
            num_heads=plan.config.n_heads,
            key_dim=plan.config.d_model // plan.config.n_heads,
            name=_layer_name('cross', node.path),
        )(tf.expand_dims(query_d, 1), token, key_mask=safe)
        attended = tf.squeeze(attended, axis=1)
        has = tf.reduce_max(tf.cast(mask, attended.dtype), axis=1, keepdims=True)
        return attended * has

    root = plan.root

    def maybe(name):
        try:
            return root.child(name)
        except KeyError:
            return None

    query_parts = []
    for name in ('player_id_embedding', 'position', 'current_player_features', 'last_season'):
        node = maybe(name)
        if node is not None:
            query_parts.append(encode(node, None, None))
    query_parts.append(encode(root.child('player_fixture_history'), None, None))
    query = tf.concat(query_parts, axis=-1)
    query_d = Dense(
        plan.config.d_model, activation='relu', kernel_regularizer=regularizer, name='query_proj',
    )(query)
    head_in = tf.concat([
        query,
        encode(root.child('current_team'), None, query_d),
        encode(root.child('current_opposition'), None, query_d),
        encode(root.child('match_facts'), None, None),
        encode(root.child('context'), None, None),
    ], axis=-1)

    n_in = head_in.shape[-1]
    n_in = plan.config.d_model if n_in is None else int(n_in)
    hidden = head_in
    dropouts = [float(hparams.dropout_1), float(hparams.dropout_2)]
    for index, units in enumerate(_hidden_sizes(n_in, hparams)):
        hidden = Dense(units, activation='relu', kernel_regularizer=regularizer)(hidden)
        if index < len(dropouts) and dropouts[index] > 0:
            hidden = Dropout(dropouts[index])(hidden)
    points = Dense(
        1, activation=_output_activation(loss_name), name='points',
    )(hidden)
    return Model(inputs=inputs, outputs=points, name='learning_v2_%s' % plan.name)
