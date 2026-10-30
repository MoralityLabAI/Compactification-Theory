"""SelfModelGym: reproducible, theory-neutral mechanistic research protocols."""
from .env import SelfModelEnv
from .protocol import PROTOCOL_ID, protocol_digest

__all__ = ["SelfModelEnv", "PROTOCOL_ID", "protocol_digest"]
