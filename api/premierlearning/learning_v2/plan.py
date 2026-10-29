"""Load a model-plan JSON and walk it into nodes the dataset and Keras builder share."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json

PLANS_DIR = Path(__file__).resolve().parent / 'model_plans'
LEAF_TYPES = ('embedding', 'int', 'float')

# Root fields the simple builder knows how to place. Anything else is refused
# so a new column cannot silently skip the network.
SIMPLE_ROOTS = (
    'player_id_embedding',
    'position',
    'current_player_features',
    'last_season',
    'player_fixture_history',
    'current_team',
    'current_opposition',
    'match_facts',
    'context',
)


def sequence_length(config: PlanConfig, node: Node) -> int:
    """Fixture history and season history are different padded lengths."""
    if node.path == 'last_season' or node.name == 'last_season':
        return int(config.season_len)
    return int(config.history_len)


def tensor_key(path: str) -> str:
    """Keras input names cannot contain dots."""
    return path.replace('.', '__')


def embedding_table(field_name: str) -> str:
    """Shared table for every id of the same kind, including opposition ids."""
    if field_name == 'position':
        return 'position'
    if field_name in ('player_id_embedding', 'opposition_player_id_embedding'):
        return 'player'
    if field_name in ('team_id_embedding', 'opposition_team_id_embedding'):
        return 'team'
    if field_name == 'manager_id_embedding':
        return 'manager'
    raise ValueError('No embedding table for %s' % field_name)


@dataclass
class PlanConfig:
    history_len: int = 50
    season_len: int = 8
    team_form_n: int = 5
    max_squad: int = 20
    player_emb_dim: int = 16
    team_emb_dim: int = 8
    position_emb_dim: int = 4
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 1

    @classmethod
    def from_dict(cls, raw) -> 'PlanConfig':
        config = cls()
        raw = raw or {}
        for key in (
            'history_len', 'season_len', 'team_form_n', 'max_squad',
            'player_emb_dim', 'team_emb_dim', 'position_emb_dim',
            'd_model', 'n_heads', 'n_layers',
        ):
            if raw.get(key) is None:
                continue
            setattr(config, key, int(raw[key]))
        if (
            config.history_len < 1 or config.max_squad < 1
            or config.team_form_n < 1 or config.season_len < 1
        ):
            raise ValueError('history_len, season_len, max_squad and team_form_n must be positive')
        if config.n_heads < 1 or config.d_model < 1 or config.n_layers < 1:
            raise ValueError('d_model, n_heads and n_layers must be positive')
        return config


@dataclass
class Node:
    kind: str
    name: str
    path: str
    dtype: str = ''
    table: str = ''
    children: list = field(default_factory=list)
    item: 'Node | None' = None

    def child(self, name: str) -> 'Node':
        for node in self.children:
            if node.name == name:
                return node
        raise KeyError(name)


@dataclass
class InputSpec:
    key: str
    path: str
    shape: tuple
    dtype: str
    kind: str


@dataclass
class Plan:
    name: str
    config: PlanConfig
    root: Node

    def leaves(self):
        return list(_leaves(self.root))


def _leaves(node: Node):
    if node.kind == 'leaf':
        yield node
        return
    if node.kind == 'group':
        for child in node.children:
            yield from _leaves(child)
        return
    if node.item is not None:
        yield from _leaves(node.item)


def contains_nested_attention(plan: Plan) -> bool:
    """True when a set-attention block sits inside a history transformer."""
    def visit(node: Node, inside_sequence: bool) -> bool:
        if node.kind == 'attention' and inside_sequence:
            return True
        if node.kind == 'transformer':
            return visit(node.item, True)
        if node.kind == 'attention':
            return visit(node.item, inside_sequence)
        return any(visit(child, inside_sequence) for child in node.children)

    return visit(plan.root, False)


def _parse_group(path: str, name: str, obj: dict) -> Node:
    node = Node(kind='group', name=name, path=path)
    for key, value in obj.items():
        child_path = '%s.%s' % (path, key) if path else key
        node.children.append(_parse_node(key, value, child_path))
    return node


def _parse_node(name: str, value, path: str) -> Node:
    if isinstance(value, str):
        if value not in LEAF_TYPES:
            raise ValueError('Unknown leaf type %r at %s' % (value, path))
        table = embedding_table(name) if value == 'embedding' else ''
        return Node(kind='leaf', name=name, path=path, dtype=value, table=table)
    if not isinstance(value, dict):
        raise ValueError('Plan node %s must be a type string or object' % path)
    keys = list(value.keys())
    if keys == ['attention']:
        body = value['attention']
        if not isinstance(body, list) or len(body) != 1 or not isinstance(body[0], dict):
            raise ValueError('attention at %s must be a one-item list of fields' % path)
        node = Node(kind='attention', name=name, path=path)
        node.item = _parse_group(path, name, body[0])
        return node
    if keys == ['transformer']:
        body = value['transformer']
        if not isinstance(body, list) or len(body) != 1 or not isinstance(body[0], dict):
            raise ValueError('transformer at %s must be a one-item list of fields' % path)
        node = Node(kind='transformer', name=name, path=path)
        node.item = _parse_group(path, name, body[0])
        return node
    return _parse_group(path, name, value)


def plan_from_dict(data: dict, name: str = 'plan') -> Plan:
    raw = dict(data)
    config = PlanConfig.from_dict(raw.pop('_config', {}) or {})
    if 'team_fixture_history' in raw:
        raise ValueError('team_fixture_history is not part of the v2 plans')
    root = _parse_group('', 'root', raw)
    return Plan(name=name, config=config, root=root)


def load_plan(name: str) -> Plan:
    path = PLANS_DIR / ('%s.json' % name)
    if not path.is_file():
        raise SystemExit('Unknown model plan %s (%s)' % (name, path))
    with path.open(encoding='utf-8') as handle:
        return plan_from_dict(json.load(handle), name)


def iter_input_specs(plan: Plan):
    """One model input per leaf, plus a mask for each sequence or set."""
    config = plan.config

    def visit(node: Node, seq_len):
        if node.kind == 'leaf':
            shape = (seq_len,) if seq_len else (1,)
            dtype = 'int32' if node.dtype == 'embedding' else 'float32'
            yield InputSpec(tensor_key(node.path), node.path, shape, dtype, node.dtype)
            return
        if node.kind == 'group':
            for child in node.children:
                yield from visit(child, seq_len)
            return
        if node.kind == 'transformer':
            length = sequence_length(config, node)
            mask_path = node.path + '.mask'
            yield InputSpec(tensor_key(mask_path), mask_path, (length,), 'float32', 'mask')
            yield from visit(node.item, length)
            return
        if node.kind == 'attention':
            length = config.max_squad
            mask_path = node.path + '.mask'
            yield InputSpec(tensor_key(mask_path), mask_path, (length,), 'float32', 'mask')
            yield from visit(node.item, length)
            return
        raise ValueError('Unknown plan node %s' % node.kind)

    yield from visit(plan.root, None)


def input_specs(plan: Plan) -> list:
    return list(iter_input_specs(plan))
