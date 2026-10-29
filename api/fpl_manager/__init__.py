"""PPO fantasy manager, kept separate from the points model.

The points model, the PuLP squad optimiser and the historical replay stay
where they are. This package decides the one high-level action each gameweek
and asks those pieces to carry it out.

`python -m fpl_manager` trains the masked policy. The debugger entry is "fantasy manager".
"""
