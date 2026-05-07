"""Thin wrapper around :class:`torch.utils.data.DataLoader` for evaluation tasks."""

from torch.utils.data import DataLoader as TorchDataLoader


class EvalDataLoader(TorchDataLoader):
    """:class:`torch.utils.data.DataLoader` subclass reserved for eval-task customization."""
