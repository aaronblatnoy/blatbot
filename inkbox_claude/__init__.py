"""Inkbox bridge for Claude Code — email, SMS, iMessage, and voice."""

__version__ = "0.2.10"

# One codebase runs several assistants. INSTANCE_PROFILE names the running instance's profile
# (deploy/instances/<name>.yaml); it is applied here, before any module reads the environment.
# Unset, this does nothing.
from . import profile as _profile

_profile.apply_from_environment()
