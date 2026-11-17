"""Who this running instance is, from the environment.

One codebase runs more than one assistant (Blatbot, Gondroid, ...). Everything that names the
assistant to people or to a model, its own mailbox, or the server it runs on reads it from here.
Unset, every value is the original Blatbot one, so an instance that sets nothing behaves exactly
as before.

  ASSISTANT_NAME   the name it signs and introduces itself with   (default "Blatbot")
  ASSISTANT_EMAIL  its own Inkbox mailbox address                 (default "blatbot@inkboxmail.com")
  SERVER_NAME      the machine it runs on, as people call it      (default "black-sky")

Other per-instance settings already come from the environment: INKBOX_IDENTITY, INKBOX_API_KEY,
INKBOX_BRIDGE_PORT, INKBOX_CLAUDE_HOME (state dir), GATE_DB_PATH / GATE_PG_URL (database),
GATE_SCOPES_FILE (scope registry), GATE_STANDING_PATH, GATE_EXEC_DIR, GRAPHITI_URL and
GRAPHITI_GROUP (long-term memory), TELEGRAM_BOT_TOKEN / TELEGRAM_BOT_NAME, OWNER_NAME.
"""

from __future__ import annotations

import os

ASSISTANT = (os.getenv("ASSISTANT_NAME") or "").strip() or "Blatbot"
ASSISTANT_UPPER = ASSISTANT.upper()
ASSISTANT_EMAIL = (os.getenv("ASSISTANT_EMAIL") or "").strip() or "blatbot@inkboxmail.com"
SERVER_NAME = (os.getenv("SERVER_NAME") or "").strip() or "black-sky"
OWNER_NAME = (os.getenv("OWNER_NAME") or "").strip() or "Aaron Blatnoy"
GRAPHITI_GROUP = (os.getenv("GRAPHITI_GROUP") or "").strip() or "blatbot"
