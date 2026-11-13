"""Keras encoder compiled from a model plan.

Query is the target player: id, position, current form, and the pooled fixture
history. Current teammates and opposition players are cross-attended with that
query. Pad positions are zeroed and masked.
"""
from __future__ import annotations

import math

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Embedding, Input, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2 as l2_regularizer

from premierlearning.learning import _output_activation, uses_log_targets
from premierlearning.learning_v2.dataset import MINUTES_BAND_EDGES, MINUTES_TARGET_BANDS
from premierlearning.learning_v2.plan import (
    POINTS_DIRECT, POINTS_SCORELINE, POINTS_STAGED, POOL_MEAN, POOL_RECENCY, SIMPLE_ROOTS,
    TEAM_HISTORY_NODES, age_leaf, contains_nested_attention, sequence_length, set_length, tensor_key,
)
from premierlearning.learning_v2.staged import GOAL_BANDS, BandScorelinePoints

# A Premier League side scores about 1.4 goals a match: softplus(1.1) is 1.39.
_GOALS_BIAS = 1.1
# FPL points by position index (0 unknown, 1 GK, 2 DEF, 3 MID, 4 FWD).
CLEAN_SHEET_POINTS = (0.0, 4.0, 4.0, 1.0, 0.0)
# -1 per 2 goals conceded, for goalkeepers and defenders.
CONCEDED_PER_GOAL = (0.0, 0.5, 0.5, 0.0, 0.0)


def _score_net(team_match, d_model, regularizer):
    """Expected goals for and against from the team, opponent and match encodings."""
    hidden = Dense(d_model, activation='relu', kernel_regularizer=regularizer, name='score_hidden')(team_match)
    return Dense(
        2, activation='softplus', name='score',
        bias_initializer=tf.keras.initializers.Constant(_GOALS_BIAS),
    )(hidden)


def masked_poisson(y_true, y_pred):
    """Poisson loss on goals for and against; targets below 0 (unknown score) are skipped."""
    mask = tf.cast(y_true >= 0.0, y_pred.dtype)
    target = tf.maximum(y_true, 0.0)
    loss = (y_pred - target * tf.math.log(y_pred + 1e-7)) * mask
    return tf.reduce_sum(loss, axis=-1) / tf.maximum(tf.reduce_sum(mask, axis=-1), 1.0)


class ScorelinePoints(tf.keras.layers.Layer):
    """Expected points assembled from minutes-band chances and the expected scoreline.

    Inputs: band chances [0, 1-59, 60+], expected goals [for, against], four free
    parts from the shared layers, and the position index. With F and A the
    expected goals for and against:
      60+   : 2 + clean sheet points x e^-A - conceded penalty x A + r x F + other_60
      1-59  : 1 + s x r x F + other_sub
    r (attacking points per team goal) and the others are softplus, s a sigmoid
    share. The fixture reaches points through F and A, so a weak opponent has
    to raise clean-sheet and attacking chances by an amount tied to goals.
    """

    def __init__(self, log_output=True, **kwargs):
        super().__init__(**kwargs)
        self.log_output = bool(log_output)
        self.clean_sheet = tf.constant(CLEAN_SHEET_POINTS, dtype=tf.float32)
        self.conceded = tf.constant(CONCEDED_PER_GOAL, dtype=tf.float32)

    def call(self, inputs):
        bands, goals, parts, position = inputs
        position = tf.reshape(tf.cast(position, tf.int32), [-1])
        position = tf.clip_by_value(position, 0, len(CLEAN_SHEET_POINTS) - 1)
        clean_sheet = tf.gather(self.clean_sheet, position)[:, None]
        conceded = tf.gather(self.conceded, position)[:, None]
        goals_for = goals[:, 0:1]
        goals_against = goals[:, 1:2]
        attack = tf.nn.softplus(parts[:, 0:1])
        sub_share = tf.sigmoid(parts[:, 1:2])
        other_full = tf.nn.softplus(parts[:, 2:3])
        other_sub = tf.nn.softplus(parts[:, 3:4])
        full = (
            2.0 + clean_sheet * tf.exp(-goals_against) - conceded * goals_against
            + attack * goals_for + other_full
        )
        sub = 1.0 + sub_share * attack * goals_for + other_sub
        expected = tf.nn.relu(bands[:, 1:2] * sub + bands[:, 2:3] * full)
        if self.log_output:
            return tf.math.log1p(expected)
        return expected

    def get_config(self):
        config = super().get_config()
        config.update({'log_output': self.log_output})
        return config


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


class RecencyPool(tf.keras.layers.Layer):
    """Sum a history up by attention whose weights fade with each item's age.

    Each head scores an item as (learned query . item key) - rate_h x age, so
    after the softmax its weight is proportional to e^(content) x e^(-rate_h x age):
    an exponential fade at a rate the head learns, which content can still
    override. Heads start at rates spread from slow to fast. The age input may
    be standardised: softmax ignores a shift shared by every item, and the rate
    absorbs the scale. Adds log(1 + matches) / log(1 + slots), so a short
    history reads differently from a long one.
    """

    # Starting fade per head, in age units (standard deviations of age).
    INITIAL_RATES = (0.1, 0.5, 2.0, 8.0)

    def __init__(self, d_model, n_heads, **kwargs):
        super().__init__(**kwargs)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.key_dim = max(1, self.d_model // self.n_heads)
        width = self.n_heads * self.key_dim
        self.key_dense = Dense(width, use_bias=False)
        self.value_dense = Dense(width, use_bias=False)
        self.output_dense = Dense(self.d_model)

    def build(self, input_shape):
        rates = [self.INITIAL_RATES[i % len(self.INITIAL_RATES)] for i in range(self.n_heads)]
        inverse_softplus = [math.log(math.expm1(rate)) for rate in rates]
        self.query = self.add_weight(
            name='query', shape=(self.n_heads, self.key_dim), initializer='glorot_uniform',
        )
        self.rate = self.add_weight(
            name='rate', shape=(self.n_heads,),
            initializer=tf.keras.initializers.Constant(inverse_softplus),
        )
        super().build(input_shape)

    def call(self, x, mask, age=None):
        batch = tf.shape(x)[0]
        length = tf.shape(x)[1]
        mask_f = tf.cast(mask, x.dtype)
        keys = tf.reshape(self.key_dense(x), [batch, length, self.n_heads, self.key_dim])
        values = tf.reshape(self.value_dense(x), [batch, length, self.n_heads, self.key_dim])
        scores = tf.einsum('blhd,hd->bhl', keys, self.query) * (1.0 / math.sqrt(self.key_dim))
        if age is not None:
            rate = tf.nn.softplus(self.rate)
            scores = scores - rate[None, :, None] * tf.cast(age, x.dtype)[:, None, :]
        scores = scores + (1.0 - _safe_mask(mask_f))[:, None, :] * -1e9
        weights = tf.nn.softmax(scores, axis=-1)
        pooled = tf.einsum('bhl,blhd->bhd', weights, values)
        pooled = self.output_dense(tf.reshape(pooled, [batch, self.n_heads * self.key_dim]))
        count = tf.reduce_sum(mask_f, axis=1, keepdims=True)
        pooled = pooled * tf.cast(count > 0, x.dtype)
        slots = tf.cast(length, x.dtype)
        return tf.concat([pooled, tf.math.log1p(count) / tf.math.log1p(slots)], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({'d_model': self.d_model, 'n_heads': self.n_heads})
        return config


class HistoryEncoder(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, n_layers, dropout, regularizer, pool='mean', **kwargs):
        super().__init__(**kwargs)
        key_dim = max(1, int(d_model) // int(n_heads))
        self.blocks = []
        for index in range(int(n_layers)):
            block = _HistoryBlock(
                d_model, n_heads, key_dim, dropout, regularizer, name='block_%s' % index,
            )
            self.blocks.append(block)
            setattr(self, 'block_%s' % index, block)
        self.pool = pool
        self.recency = RecencyPool(d_model, n_heads, name='recency_pool') if pool == POOL_RECENCY else None

    def call(self, x, mask, age=None, training=None):
        mask_f = tf.cast(mask, x.dtype)
        x = x * mask_f[:, :, None]
        safe = _safe_mask(mask_f)
        for block in self.blocks:
            x = block(x, safe, mask_f, training=training)
        if self.recency is not None:
            return self.recency(x, mask_f, age)
        denom = tf.maximum(tf.reduce_sum(mask_f, axis=1, keepdims=True), 1.0)
        return tf.reduce_sum(x * mask_f[:, :, None], axis=1) / denom


def _layer_name(prefix, path):
    return ('%s_%s' % (prefix, path.replace('.', '_')))[:90]


class BlockEmbedding(tf.keras.layers.Layer):
    """An identity vector split by time: a shared base plus a small block part.

    The input carries id * blocks + block (teams.encode_block). The base row is
    learned from every block; the block row is held near zero by an L2 penalty,
    so a block with few matches (a promoted team, the newest block early in a
    season) falls back to the base.
    """

    def __init__(self, ids, blocks, dim, block_l2, **kwargs):
        super().__init__(**kwargs)
        self.ids = int(ids)
        self.blocks = int(blocks)
        self.dim = int(dim)
        self.block_l2 = float(block_l2)
        self.base = Embedding(self.ids, self.dim, name='base')
        self.block = Embedding(
            self.ids * self.blocks, self.dim, name='block',
            embeddings_initializer='zeros',
            embeddings_regularizer=l2_regularizer(self.block_l2) if self.block_l2 > 0 else None,
        )

    def call(self, index):
        index = tf.cast(index, tf.int32)
        return self.base(tf.math.floordiv(index, self.blocks)) + self.block(index)

    def get_config(self):
        config = super().get_config()
        config.update({'ids': self.ids, 'blocks': self.blocks, 'dim': self.dim, 'block_l2': self.block_l2})
        return config


def _identity_table(name, ids, dim, blocks, block_l2):
    if int(blocks) <= 1:
        return Embedding(int(ids), dim, name=name)
    return BlockEmbedding(int(ids), int(blocks), dim, block_l2, name=name)


def points_only(model):
    """The points output of a two-output model, sharing its trained weights."""
    return Model(inputs=model.input, outputs=model.get_layer('points').output, name=model.name)


def minutes_only(model):
    """The minutes head of a two-output model, sharing its trained weights.

    With the bands target this is the softmax over 0, 1-59 and 60+ minutes.
    """
    return Model(
        inputs=model.input, outputs=model.get_layer('minutes').output, name=model.name + '_minutes',
    )


def build_model(plan, vocab, hparams, loss_name='huber', minutes_head=None):
    """Functional model. Inputs are named with ``tensor_key`` paths.

    minutes_head names a minutes target, and adds its output to the last shared
    layer: a softmax over the bands, or a 0-1 scaled log of real minutes. It only
    shapes those layers in training: points_only drops it before predicting.
    """
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
    config = plan.config
    tables = {
        'player': _identity_table(
            'player_emb', vocab.player_size, config.player_emb_dim,
            getattr(config, 'player_blocks', 1), getattr(config, 'block_l2', 0.0),
        ),
        'team': _identity_table(
            'team_emb', vocab.team_size, config.team_emb_dim,
            getattr(config, 'team_blocks', 1), getattr(config, 'block_l2', 0.0),
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

    staged = getattr(config, 'points_structure', POINTS_DIRECT) == POINTS_STAGED
    # Staged plans read both team histories with one set of layers and reuse
    # each history's summary wherever it is needed.
    shared_layers = {}
    history_cache = {}

    def _shared(key, make):
        if key not in shared_layers:
            shared_layers[key] = make()
        return shared_layers[key]

    def encode_history(node):
        if node.path in history_cache:
            return history_cache[node.path]
        share = staged and node.path in TEAM_HISTORY_NODES
        length = sequence_length(plan.config, node)
        token = encode(node.item, length, None)
        mask = keras_input(node.path + '.mask', (length,), 'float32')
        token = token * tf.expand_dims(mask, -1)

        def make_proj(name):
            return Dense(plan.config.d_model, activation='relu', kernel_regularizer=regularizer, name=name)

        if share:
            proj = _shared('hist_proj', lambda: make_proj('hist_proj_team_shared'))
        else:
            proj = make_proj(_layer_name('hist_proj', node.path))
        projected = proj(token)
        projected = projected * tf.expand_dims(mask, -1)
        pool = getattr(plan.config, 'history_pool', POOL_MEAN)
        age = None
        if pool == POOL_RECENCY:
            leaf = age_leaf(node)
            if leaf is not None:
                age = tf.cast(keras_input(leaf.path, (length,), 'int32' if leaf.dtype == 'embedding' else 'float32'),
                              tf.float32)

        def make_encoder(name):
            return HistoryEncoder(
                plan.config.d_model, plan.config.n_heads, plan.config.n_layers,
                float(hparams.dropout_1), regularizer, pool=pool, name=name,
            )

        if share:
            encoder = _shared('hist', lambda: make_encoder('hist_team_shared'))
        else:
            encoder = make_encoder(_layer_name('hist', node.path))
        out = encoder(projected, mask, age=age)
        if staged:
            history_cache[node.path] = out
        return out

    def squad_summary(node):
        """A side's squad, the same for every row of that team: shared layers, no player query.

        Only with scoreline_squads, where the teammates list keeps the player,
        so a team's list matches the one its opponents see.
        """
        length = set_length(plan.config, node)
        token = encode(node.item, length, None)
        mask = keras_input(node.path + '.mask', (length,), 'float32')
        weights = tf.expand_dims(mask, -1)
        proj = _shared('squad', lambda: Dense(
            plan.config.d_model, activation='relu', kernel_regularizer=regularizer, name='squad_proj_team_shared',
        ))
        projected = proj(token * weights) * weights
        count = tf.reduce_sum(weights, axis=1)
        mean = tf.reduce_sum(projected, axis=1) / tf.maximum(count, 1.0)
        # relu outputs are >= 0, so masked slots at 0 never win the max.
        peak = tf.reduce_max(projected, axis=1)
        return tf.concat([mean, peak, count / float(length)], axis=-1)

    def team_side(side, id_name):
        """Team-level inputs of one side: id (if the plan has one), fading
        averages and match history.

        Squads join only with scoreline_squads: otherwise a player's teammates
        list leaves that player out, so it differs between teammates, and the
        scoreline must not.
        """
        parts = [encode(child, None, None) for child in side.children if child.name == id_name]
        for child in side.children:
            if child.name in ('team_features', 'schedule', 'gameweek', 'manager_features'):
                parts.append(encode(child, None, None))
            elif child.name == 'fixture_history':
                if child.kind != 'transformer':
                    raise ValueError('the staged structure reads team histories with a transformer')
                parts.append(encode_history(child))
            elif child.kind == 'attention' and getattr(config, 'scoreline_squads', False):
                parts.append(squad_summary(child))
        return tf.concat(parts, axis=-1)

    def encode_squad(node, query_d):
        length = set_length(plan.config, node)
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
    team_enc = encode(root.child('current_team'), None, query_d)
    opp_enc = encode(root.child('current_opposition'), None, query_d)
    facts_enc = encode(root.child('match_facts'), None, None)
    head_parts = [
        query,
        team_enc,
        opp_enc,
        facts_enc,
        encode(root.child('context'), None, None),
    ]
    if maybe('player_load') is not None:
        head_parts.append(encode(root.child('player_load'), None, None))
    head_in = tf.concat(head_parts, axis=-1)

    n_in = head_in.shape[-1]
    n_in = plan.config.d_model if n_in is None else int(n_in)
    hidden = head_in
    dropouts = [float(hparams.dropout_1), float(hparams.dropout_2)]
    for index, units in enumerate(_hidden_sizes(n_in, hparams)):
        hidden = Dense(units, activation='relu', kernel_regularizer=regularizer)(hidden)
        if index < len(dropouts) and dropouts[index] > 0:
            hidden = Dropout(dropouts[index])(hidden)

    structure = getattr(config, 'points_structure', POINTS_DIRECT)
    score_weight = float(getattr(config, 'score_head_weight', 0.0) or 0.0)
    goals = None
    if structure == POINTS_STAGED:
        side_layer = Dense(config.d_model, activation='relu', kernel_regularizer=regularizer, name='team_side')
        us = side_layer(team_side(root.child('current_team'), 'team_id_embedding'))
        them = side_layer(team_side(root.child('current_opposition'), 'opposition_team_id_embedding'))
        home_key = tensor_key('match_facts.home_away')
        home = tf.cast(inputs[home_key], tf.float32) if home_key in inputs else tf.zeros_like(us[:, :1])
        pair = Dense(config.d_model, activation='relu', kernel_regularizer=regularizer, name='score_pair')
        band_out = Dense(GOAL_BANDS, activation='softmax', name='score_bands')
        goals_for = band_out(pair(tf.concat([us, them, home], axis=-1)))
        goals_against = band_out(pair(tf.concat([them, us, 1.0 - home], axis=-1)))
        goals = tf.keras.layers.Concatenate(name='score')([goals_for, goals_against])
    elif score_weight > 0 or structure == POINTS_SCORELINE:
        goals = _score_net(tf.concat([team_enc, opp_enc, facts_enc], axis=-1), config.d_model, regularizer)

    bands = None
    if minutes_head == MINUTES_TARGET_BANDS or structure in (POINTS_SCORELINE, POINTS_STAGED):
        bands = Dense(len(MINUTES_BAND_EDGES) + 1, activation='softmax', name='minutes')(hidden)

    if structure == POINTS_STAGED:
        if minutes_head and minutes_head != MINUTES_TARGET_BANDS:
            raise ValueError('the staged structure needs the bands minutes target, not %s' % minutes_head)
        position_key = tensor_key('position')
        if position_key not in inputs:
            raise ValueError('the staged structure needs the position input')
        parts = Dense(4, name='scoreline_parts')(hidden)
        points = BandScorelinePoints(log_output=uses_log_targets(loss_name), name='points')(
            [bands, goals, parts, inputs[position_key]],
        )
    elif structure == POINTS_SCORELINE:
        if minutes_head and minutes_head != MINUTES_TARGET_BANDS:
            raise ValueError('the scoreline structure needs the bands minutes target, not %s' % minutes_head)
        position_key = tensor_key('position')
        if position_key not in inputs:
            raise ValueError('the scoreline structure needs the position input')
        parts = Dense(4, name='scoreline_parts')(hidden)
        points = ScorelinePoints(log_output=uses_log_targets(loss_name), name='points')(
            [bands, goals, parts, inputs[position_key]],
        )
    else:
        points = Dense(
            1, activation=_output_activation(loss_name), name='points',
        )(hidden)

    outputs = [points]
    if minutes_head == MINUTES_TARGET_BANDS:
        outputs.append(bands)
    elif minutes_head:
        outputs.append(Dense(1, activation='sigmoid', name='minutes')(hidden))
    if score_weight > 0:
        outputs.append(goals)
    return Model(
        inputs=inputs, outputs=outputs[0] if len(outputs) == 1 else outputs,
        name='learning_v2_%s' % plan.name,
    )
