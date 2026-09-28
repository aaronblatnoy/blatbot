# Blatbot

A personal AI assistant that many people can reach, and only one person can command.

Blatbot has its own email address, phone number, and iMessage line. Anyone can write to it or call it. It answers on my behalf, keeps track of what each person asked for, and can act on my calendar, inbox, and documents. The catch that makes this safe: when someone other than me asks it to *do* something, nothing happens until I approve the exact action from my phone.

> **Built on Inkbox.** This project is a fork of [inkbox-ai/claude-code-plugin](https://github.com/inkbox-ai/claude-code-plugin), the Inkbox bridge that gives a Claude Code agent a mailbox, a phone number, SMS, iMessage, and a tunnel. All of the transport (receiving webhooks, sending messages, carrying call audio) is Inkbox's work. This fork adds the layer on top: a router, an approval gate, a scoped executor, a task ledger, and a phone surface. The `main` branch is Inkbox's code, untouched. The `blatbot` branch is this project. [See exactly what was added.](https://github.com/aaronblatnoy/blatbot/compare/main...blatbot) Inkbox's original README is kept at [docs/inkbox-plugin-README.md](docs/inkbox-plugin-README.md).

## The problem

Most assistant demos have one user. The interesting case is an assistant that lives at a public address. Candidates email it to book time with me. Colleagues text it. I call it while walking down the street. They are all talking to the same agent, and that agent can send email and edit my calendar.

So the real question is not "can the model do the task." It is "who is allowed to make it act, and how do I know what it is about to do." A system prompt that says "only obey Aaron" is not an answer, because prompts can be argued with. Blatbot answers it in code.

## How it works

```
  PEOPLE                                   SURFACES
 ┌─────────────────┐
 │ Owner           │──── iMessage ─────┐
 │ trusted number  │──── phone call ───┼──────────────┐
 └─────────────────┘                   │              │
 ┌─────────────────┐                   │              │
 │ Anyone else     │──── email ────────┤              │
 │                 │──── SMS ──────────┤              │
 │                 │──── phone call ───┼──────────────┤
 └─────────────────┘                   │              │
                                       v              v
                         ┌──────────────────┐  ┌───────────────────────────────┐
                         │ INKBOX           │  │ INKBOX phone line             │
                         │ mailbox, SMS,    │  │ raw audio, 8 kHz              │
                         │ iMessage         │  │                               │
                         └────────┬─────────┘  └───────────────┬───────────────┘
                                  │ webhook                    │ websocket
                                  │                            v
                                  │            ┌───────────────────────────────┐
                                  │            │ VOICE AGENT   (OpenAI         │
                                  │            │               GPT-Live)       │
                                  │            │ runs the whole conversation   │
                                  │            │ knows nothing about what is   │
                                  │            │   behind it                   │
                                  │            │ no tools: it DELEGATES        │
                                  │            └───────────────┬───────────────┘
                                  │                            │ only when there
                                  │                            │ is real work
══════════════════════════════════╪════════════════════════════╪═════════════════════
  YOUR SERVER                     │                            │
                                  v                            v
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │ GATEWAY  (code, no model)                          every turn, cannot be skipped
 │  1. identify sender      2. write inbound      3. load context                │
 │     trusted = owner's       to TASK LEDGER         ledger + date/time         │
 │     iMessage thread or                             + recent messages          │
 │     owner's phone number                           + standing instructions    │
 └───────────────────────────────────┬───────────────────────────────────────────┘
                                     v
                    ┌─────────────────────────────────┐
                    │ ROUTER        (any cheap LLM)   │
                    │ NO TOOLS. outputs:              │
                    │   reply  and/or                 │
                    │   task = prompt + scopes        │
                    └───────┬─────────────────┬───────┘
                   reply    │                 │  task
                            │                 v
                            │   ┌──────────────────────────────────┐
                            │   │ GATE  (code)                     │
                            │   │ store prompt + hash              │
                            │   │ from owner ──> approved, run now │
                            │   │ from others ─> HOLD              │
                            │   │    text owner: their message,    │
                            │   │    exact prompt, scopes          │
                            │   │    "#N yes" / "no" / "edit: ..." │
                            │   │    expires in 24h                │
                            │   └────────────────┬─────────────────┘
                            │                    v  approved
                            │   ┌──────────────────────────────────┐
                            │   │ EXECUTOR   (Claude Code)         │
                            │   │ verifies prompt hash             │
                            │   │ only the tools the scopes allow  │
                            │   │ ends with STATUS: OK / FAILED    │
                            │   └────────────────┬─────────────────┘
                            │                    v
                            │      result -> TASK LEDGER (done / failed)
                            v                    v
                   reply sent on the same channel the person used
```

## Design rules

These are the decisions the whole thing rests on.

1. **The model that reads strangers' messages has no tools.** The router sees every inbound message, including hostile ones. All it can produce is text: a reply, and optionally a proposed task. It cannot touch anything.
2. **The gate is code, not a prompt.** Whether a request runs is decided by comparing the sender against the owner's iMessage thread or phone number. No wording in a message can change that comparison. An email that claims to be from the owner, even from the owner's real address, is not the owner.
3. **The owner approves the exact prompt, and only that prompt runs.** The approval text shows the person's message, what the assistant wants to do, the tool scopes, and the literal prompt. The prompt is hashed when stored and checked again before execution.
4. **Every task gets the smallest set of tools that covers it.** Scopes map to tool lists. A tool outside the granted scopes is denied at call time, in code.
5. **One ledger per person, across every channel.** Tasks are keyed by the person's email or phone, so a call, a text, and an email from the same person share one record. The ledger is loaded before every turn, which is what lets the assistant pick a task back up hours later.
6. **The phone is a surface, not a second brain.** The voice model runs the conversation and is deliberately told nothing about the system behind it. When something real is needed it hands off, and that hand-off enters the same gateway as a text message, under the same rules. A caller who talks the voice model into something gains nothing, because the gate still checks the real caller number.
7. **The assistant never claims work it has not done.** Replies can only report an action as complete when the ledger shows it completed.

## A worked example

1. A candidate emails: "Could we do Tuesday at 6 for a coffee chat?"
2. The gateway logs it to that candidate's ledger and sends me a one-line heads-up on iMessage.
3. The router replies to them, "Let me confirm that with Aaron," and proposes a task: book a 30 minute event Tuesday at 6 PM and email a confirmation. Scopes: calendar, send email.
4. I get a text with their message, the summary, the scopes, and the exact prompt. I reply `#12 yes`.
5. The executor runs that prompt with only calendar and email tools, books the event, sends the confirmation, and reports `STATUS: OK`.
6. The ledger marks it done. The candidate gets their confirmation. I get a one-line "done."
7. Two days later they write "can we push it 30 minutes?" The router opens the ledger, sees the booking, and proposes the change. It does not ask them what meeting they mean.

If I had sent the same request from my own iMessage thread, steps 3 to 5 would run immediately with no approval.

## What is in this fork

Everything custom lives in one folder plus one file.

| Path | What it is |
|---|---|
| `inkbox_claude/gate/manager.py` | The core: a session per person, the owner trust check, approval commands, the phone surface |
| `inkbox_claude/gate/router.py` | The zero-tool router, its rules, and the date and time line |
| `inkbox_claude/gate/store.py` | SQLite: the task ledger, requests and their states, threads |
| `inkbox_claude/gate/executor.py` | Runs one approved, hash-checked prompt through Claude Code |
| `inkbox_claude/gate/jevagent.py` | Optional executor without a chat model: TypeSafe judgments pick tools, code fills arguments, prose only on demand |
| `inkbox_claude/gate/taskpick.py` | TypeSafe judgments for the ledger: which task, what kind of event, which scopes, whether action is needed |
| `inkbox_claude/gate/scopes.py` | Scope names mapped to tool lists |
| `inkbox_claude/live.py` | The phone bridge for OpenAI GPT-Live, using client delegation |
| `tests/test_gate.py`, `tests/test_live.py` | Tests, including simulated phone calls with fake sockets |
| `docs/blatbot-architecture.md` | Longer design notes and the Live versus Realtime comparison |

The remaining edits are small hooks inside Inkbox's `gateway.py`, `realtime.py`, `config.py`, `tools.py`, and `sessions.py` that plug the gate in. The compare link above shows all of it.

## Running it

Start with Inkbox's own setup, which creates the agent identity and installs the bridge. Follow [docs/inkbox-plugin-README.md](docs/inkbox-plugin-README.md). Then turn on the gate by adding these to the bridge's `.env`:

```
# Turn on the gate
INKBOX_MODE=gate

# The router. Any OpenAI-compatible chat model works; DeepSeek is the default.
DEEPSEEK_API_KEY=

# Who the owner is. These two values are the entire trust boundary.
INKBOX_APPROVER_PHONE=+15550100001
INKBOX_APPROVER_IMESSAGE_CONVERSATION_ID=

# Trust the owner's number on phone calls too (caller ID can be spoofed; off by default)
GATE_VOICE_TRUST_APPROVER=1

# Phone voice. "live" is OpenAI GPT-Live, "realtime" is the older Realtime API.
OPENAI_API_KEY=
INKBOX_REALTIME_ENABLED=1
INKBOX_VOICE_API=live
INKBOX_LIVE_VOICE=cedar

# Optional
GATE_VOICE_VOCABULARY="Names and terms callers use, with pronunciation hints"
GATE_ORG_GOOGLE_ACCOUNT=
GATE_OWNER_GOOGLE_ACCOUNT=
```

Put your own standing instructions (tone, signature, house rules) in `standing.md` next to the `.env`. The router reads it fresh on every turn, so you can change the assistant's behavior by editing a text file.

Run the tests with `pytest tests/test_gate.py tests/test_live.py`.

### Honest caveats

- **The scope map is wired to my setup.** `scopes.py` references the specific Google Workspace tool servers I run. To use this yourself, edit that file to point at your own tools. Making this configurable is the next piece of work.
- **Some prompts still say "Blatbot" and "Aaron."** The persona and a few rules are written for me. They are being moved into config.
- **Caller ID is not authentication.** Trusting a phone number on voice calls is a convenience with a known weakness, which is why it is a separate switch.
- **It is a single-owner design.** One assistant, one person who can approve. It is not a multi-tenant service.

## Credits and license

The bridge this is built on is [inkbox-ai/claude-code-plugin](https://github.com/inkbox-ai/claude-code-plugin) by [Inkbox](https://inkbox.ai). Their code is included here as a fork so the project runs as a whole, and all credit for the email, SMS, iMessage, voice transport, and tunnel belongs to them. The upstream repository does not currently state a license, so their code remains theirs and this fork makes no claim over it. If you want to reuse the transport, go to the upstream project.

The gate, router, executor, ledger, scopes, and GPT-Live bridge were written by [Aaron Blatnoy](https://github.com/aaronblatnoy).
