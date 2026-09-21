"""Blatbot Gate: no-tools router, human approval gate, deterministic executor.

Replaces the Claude-per-session model when INKBOX_MODE=gate. The only path
to Claude Code is the executor, and the only way into the executor is a
Request that is either from the approver or was approved by the approver.
"""
