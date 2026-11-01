"""Owner/assistant identity, read from the environment. This is a leaf module
(imports only `os`, no package imports) so config.py can import it without a
cycle through realtime.py. Every public symbol here has a generic default so
the package runs with no personal data baked in; the real deployment sets the
env vars and gets byte-identical prompts/labels/strings to what used to be
hardcoded.
"""

from __future__ import annotations

import os

OWNER_NAME = os.getenv("OWNER_NAME") or "the owner"
OWNER_FIRST_NAME = os.getenv("OWNER_FIRST_NAME") or OWNER_NAME
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME") or "the assistant"
ASSISTANT_EMAIL = os.getenv("ASSISTANT_EMAIL") or "assistant@example.com"
SERVER_NAME = os.getenv("SERVER_NAME") or "the host"
# How callers say the assistant's name (for the voice vocabulary) and any short
# names that count as addressing it at the start of a group message.
ASSISTANT_PRONUNCIATION = os.getenv("ASSISTANT_PRONUNCIATION") or ASSISTANT_NAME
ASSISTANT_NAME_ALIASES = [a.strip().lower() for a in (os.getenv("ASSISTANT_NAME_ALIASES") or "").split(",") if a.strip()]

# Derived forms used where the old code had a literal upper/lower-cased or
# possessive spelling baked into a string, so substituting these keeps the
# assembled text byte-identical to the original when the env is set to the
# real values.
OWNER_NAME_UPPER = OWNER_NAME.upper()
OWNER_FIRST_UPPER = OWNER_FIRST_NAME.upper()
ASSISTANT_NAME_UPPER = ASSISTANT_NAME.upper()
ASSISTANT_NAME_LOWER = ASSISTANT_NAME.lower()
SERVER_NAME_UPPER = SERVER_NAME.upper()

# "<FIRST>'S", e.g. "JANE'S"
OWNER_FIRST_UPPER_POSSESSIVE = f"{OWNER_FIRST_UPPER}'S"
# "<FULL NAME> HIMSELF", e.g. "JANE DOE HIMSELF"
OWNER_NAME_UPPER_HIMSELF = f"{OWNER_NAME_UPPER} HIMSELF"
# "<ASSISTANT>'s", e.g. "NOVA's"
ASSISTANT_NAME_UPPER_POSSESSIVE = f"{ASSISTANT_NAME_UPPER}'s"

# "<assistant>\nExecutive Assistant to <owner>" signature block used in
# outbound mail/sms/imessage (manager.py, sessions.py).
EA_SIGNATURE = f"{ASSISTANT_NAME}\nExecutive Assistant to {OWNER_NAME}"
