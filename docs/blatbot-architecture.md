# Blatbot architecture

As deployed, 2026-09-28. Gate mode (`inkbox_claude/gate/`), decisions and
execution by TypeSafe Jev judgments (`taskpick.py`, `jevagent.py`), replies by
DeepSeek (`router.py`), Claude Code as fallback (`executor.py`), voice via
OpenAI GPT-Live (`inkbox_claude/live.py`), with the older Realtime bridge
(`inkbox_claude/realtime.py`) selectable by env.

Everything below the double line runs on the owner's server. Everything above it is a third
party: Inkbox carries messages and audio, OpenAI runs the voice agent. Text and email
hit the gateway directly. A phone call reaches the gateway only when the voice agent
delegates. From the gateway down, the path is identical for all channels.

```
  PEOPLE                                   SURFACES
 ┌─────────────────┐
 │ Aaron           │──── iMessage ─────┐
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
                                  │            │ VOICE AGENT        (OpenAI)   │
                                  │            │ gpt-live-1, voice: Cedar      │
                                  │            │                               │
                                  │            │ knows: date/time, who's       │
                                  │            │   calling, vocabulary,        │
                                  │            │   recent task summary         │
                                  │            │ does:  all conversation       │
                                  │            │ full duplex; no tools: it     │
                                  │            │   DELEGATES to the gateway    │
                                  │            └───────────────┬───────────────┘
                                  │                            │ delegation event (id only);
                                  │                            │ bridge builds the request
                                  │                            │ from the call transcript,
                                  │                            │ only when there is real work
══════════════════════════════════╪════════════════════════════╪═════════════════════
  YOUR SERVER   systemd user unit "blatbot"   tunnel: <agent>.inkboxwire.com
                                  │                            │
                                  v                            v
 ┌───────────────────────────────────────────────────────────────────────────────────┐
 │ GATEWAY   code, no model. Runs on every turn; nothing below can be skipped.       │
 │                                                                                   │
 │  sender ──> trusted?  owner's iMessage thread or owner's phone number ──> OWNER   │
 │                       anything else, including email from the owner's   ──> OTHER │
 │                       own address                                                 │
 │  message ──> stored on its thread (email reply chains linked by Message-ID)       │
 │  context ──> task ledger for this person/thread · last 20 messages                │
 │              · New York date and time · standing instructions · contact notes     │
 └─────────────────────────────────────────┬─────────────────────────────────────────┘
                                           v
 ┌───────────────────────────────────────────────────────────────────────────────────┐
 │ DECIDE   TypeSafe Jev. Each answer is a pick from a list or a probability,        │
 │          never free text. About 0.2 s each.                                       │
 │                                                                                   │
 │  1  which task?      state: the message, last 6 turns, each candidate task's      │
 │                      id, title, where it stands, people, age. Choice over them    │
 │                      + new_task + no_task. Under 0.55 confidence: abstain.        │
 │  2  needs a tool?    state: message, turns, the task from 1. Noul. p ≥ 0.5 ──> a  │
 │                      request is built; else the reply writer answers in words.    │
 │  3  what happened?   event kind: asked / provided info / confirmed / changed /    │
 │                      declined / progress / conversation                           │
 │                      who it now waits on: owner / other person / nobody           │
 │                      does this resolve the task?  ──> written to the ledger       │
 └────────────────┬──────────────────────────────────────────┬───────────────────────┘
                  │ decision                                 │ needs a tool
                  v                                          v
 ┌──────────────────────────────────┐   ┌──────────────────────────────────────────────┐
 │ REPLY WRITER   DeepSeek          │   │ REQUEST   code, no model                     │
 │                                  │   │                                              │
 │ in:  the decision ("a request    │   │ prompt  = sender's exact words               │
 │      was created and will run /  │   │           + the conversation so far          │
 │      will be shown to Aaron" or  │   │           + this task's record               │
 │      "no action"), the ledger,   │   │ summary = the task's title                   │
 │      the conversation            │   │ scopes  = Jev, one yes/no per scope          │
 │ out: the words sent back, and a  │   │           (calendar, email send, Drive read, │
 │      title for a brand-new task  │   │           site admin, ...) granted at 0.6,   │
 │ cannot: create a request, call   │   │           plus every scope this task's       │
 │      a tool, claim work is done  │   │           earlier requests had               │
 └────────────────┬─────────────────┘   │ stored with a SHA-256 of the prompt          │
                  │                     └──────────────────────┬───────────────────────┘
                  │                                            v
                  │                     ┌──────────────────────────────────────────────┐
                  │                     │ GATE   code                                  │
                  │                     │                                              │
                  │                     │ OWNER ──────────────────> approved, run now  │
                  │                     │ OTHER ──> HOLD, text the owner:              │
                  │                     │           their message · the task ·         │
                  │                     │           the tools it will get              │
                  │                     │           "#N yes"  ─> run                   │
                  │                     │           "#N no"   ─> drop, tell them       │
                  │                     │           "#N edit: <instructions>" ─>       │
                  │                     │              appended to the prompt,         │
                  │                     │              shown again                     │
                  │                     │           24 h ─> expires                    │
                  │                     └──────────────────────┬───────────────────────┘
                  │                                            v  approved
                  │  ┌────────────────────────────────────────────────────────────────┐
                  │  │ AGENT   Jev judgments + code. No chat model drives the loop.  │
                  │  │                                                                │
                  │  │  hash of the prompt re-checked ──> mismatch: refuse            │
                  │  │  tools = only what the scopes map to; schemas read live        │
                  │  │                                                                │
                  │  │  ┌─ loop, at most 8 steps ─────────────────────────────────┐  │
                  │  │  │ done?        yes/no over goal + steps so far   ≥0.6 stop │  │
                  │  │  │ next tool    Choice over the allowed tools + give up     │  │
                  │  │  │ arguments    ONE batched request, per argument:         │  │
                  │  │  │    optional?     yes/no: supply it or leave it           │  │
                  │  │  │    id/email/tel  Choice over values already in play      │  │
                  │  │  │                  (the request, earlier results)         │  │
                  │  │  │    date bound    Choice over ranges computed from the    │  │
                  │  │  │                  clock: today, this week, last week...  │  │
                  │  │  │    enum/bool     Choice                                  │  │
                  │  │  │    must be       DeepSeek writes the bare value (an     │  │
                  │  │  │    written       email body, a title); a sentence in an │  │
                  │  │  │                  id slot or UNKNOWN is rejected          │  │
                  │  │  │ guard        send aimed at the requester ──> refused     │  │
                  │  │  │              identical repeat of a done step ──> stop    │  │
                  │  │  │              same tool twice in a row ──> next must differ│  │
                  │  │  │              a value already tried ──> not offered again  │  │
                  │  │  │              error text in a result ──> a failed step     │  │
                  │  │  │ budget       state over ~100k chars ──> largest result   │  │
                  │  │  │              becomes a digest; a result over budget is   │  │
                  │  │  │              narrowed: split, Jev scores parts, descend  │  │
                  │  │  │ call         Inkbox tools in-process; Google and site   │  │
                  │  │  │              servers over MCP stdio; result kept whole  │  │
                  │  │  └──────────────────────────────────────────────────────────┘  │
                  │  │                                                                │
                  │  │  ends:  OK with every result in full                           │
                  │  │         FAILED: give up, a tool failed twice, an argument      │
                  │  │         could not be filled, step limit                        │
                  │  │  FAILED and nothing written ──> Claude Code runs the same     │
                  │  │         prompt with the same tools, same send guard            │
                  │  │  FAILED after a write ──> stop, report; never retried         │
                  │  └──────────────────────────────┬─────────────────────────────────┘
                  │                                 v
                  │                   result ──> TASK LEDGER: done / failed
                  │                                 │
                  v                                 v
 ┌───────────────────────────────────────────────────────────────────────────────────┐
 │ DELIVERY   exactly once                                                           │
 │  the reply writer phrases the result for whoever asked; sent on their channel     │
 │  executors cannot message the requester (denied at the tool call, in code)        │
 │  someone else's task: the owner gets a one-line done/failed note                  │
 │  phone: the result goes back to the voice agent as tool output; it says it        │
 └───────────────────────────────────────────────────────────────────────────────────┘

 ┌───────────────────────────────────────────────────────────────────────────────┐
 │ TASK LEDGER   SQLite: ~/.inkbox-claude/gate.db                                │
 │ a task = a unit of work; participants = people (one person, many contacts)    │
 │ every inbound is assigned to a task by Jev; every request is written to one   │
 │ events: inbound · outbound · request · approved · rejected · done · failed    │
 │         · expired · call_ended, typed by Jev (asked / provided / confirmed..) │
 │ full-text search over title, summary, people, events; dates; any_of filters  │
 │ read by: gateway every turn, judgments, reply writer, agent.                  │
 │ NOT exposed to voice agent                                                    │
 │ except the short recent-task summary at call pickup                           │
 └───────────────────────────────────────────────────────────────────────────────┘

  HOUSEKEEPING   watchdog restarts on stale log (~35 min) · expiry sweep
```

## Text / email vs phone

| | Text / email | Phone call |
|---|---|---|
| First thing to see the words | Gateway code | OpenAI voice model |
| Who handles small talk | Router, every message | Voice agent, never reaches the gate |
| Who decides it is a task | Jev (task pick + action judgment) | Voice agent escalates, then Jev decides the same way |
| Who writes the words delivered | Reply writer (DeepSeek), sent verbatim | Voice agent, paraphrasing the result |
| Models involved | Jev, DeepSeek; Sonnet only as fallback | gpt-live-1, Jev, DeepSeek; Sonnet only as fallback |
| Trust | Aaron's iMessage thread only | Aaron's phone number (`GATE_VOICE_TRUST_APPROVER=1`) |
| Other people's requests | Held for Aaron's yes | Held for Aaron's yes, same approval text |
| FYI to Aaron per message | Yes | No, and no transcripts |
| Wait for results | No limit | About 75 s on the line, then by text |
| Record kept | Full thread, verbatim | Only what was escalated, plus results |

## Design rules

- The voice agent is a surface. Its only reach into Aaron's systems is a delegation,
  which the bridge turns into a transcript excerpt for the gateway. It does not judge feasibility; it escalates and the
  gateway reports what happened, including "can't" and "needs Aaron's approval".
- The tool description is deliberately general ("your escalation path"). Capabilities
  live in the Claude Code session config, not in a second list on the voice side.
- Nothing behind the gateway is exposed to the voice agent except read-only context at
  pickup: date/time, who is calling (a courtesy hint; the real trust check is in the
  gate, from the actual caller number), vocabulary, and a short recent-task summary.
- A phone call never changes a thread's stored text/email route.
- An email from Aaron's address is not Aaron. Caller ID can be spoofed; removing
  `GATE_VOICE_TRUST_APPROVER` puts Aaron's calls back behind the iMessage yes.

## Key files and settings

| What | Where |
|---|---|
| Gate: sessions, approvals, phone surface | `inkbox_claude/gate/manager.py` |
| Router prompt and date/time line | `inkbox_claude/gate/router.py` |
| Executor (Claude Code) | `inkbox_claude/gate/executor.py` |
| Executor (Jev agent, `GATE_EXECUTOR=jev`) | `inkbox_claude/gate/jevagent.py` |
| Ledger and requests schema | `inkbox_claude/gate/store.py` |
| Voice bridge, prompt, tool schema | `inkbox_claude/realtime.py` |
| Voice persona | `_BLATBOT_VOICE` in `inkbox_claude/config.py` |
| Per-call notes for the voice agent | `GateSession.voice_briefing` |
| Tests | `tests/test_gate.py` |
| Env (`~/.inkbox-claude/.env`) | `INKBOX_VOICE_STACK`, `INKBOX_REALTIME_ENABLED`, `OPENAI_API_KEY`, `INKBOX_REALTIME_VOICE`, `GATE_VOICE_TRUST_APPROVER`, `GATE_VOICE_VOCABULARY`, `INKBOX_APPROVER_PHONE` |

## Deciding a turn: Jev first, the router only writes

With the task picker on (`TYPESAFE_API_KEY` set), a message is decided in this order
(`GateSession.decide`):

1. Jev picks the task (Choice over candidates, new, none). No router hint.
2. Jev judges whether a tool action is needed (Noul; 0.5 is the line).
3. DeepSeek runs once, told the decision ("a request has been created and will run /
   be shown to Aaron" or "no action"), and writes only the reply plus, for a new task,
   the title and where-it-stands line. Any request it outputs is discarded.
4. If action: code builds the request. Summary is the task title, prompt is the
   verbatim message plus conversation plus ledger, scopes are Jev's judgment over
   that prompt (no router list; undecided falls back to the two likeliest, else
   `web`). Counterpart is an email in the message when Aaron is asking.
5. Jev classifies the event on the task.

`GATE_ROUTER_DEFINES_REQUEST=1` restores the older order (router proposes the
request, Jev checks task, action and scopes afterwards). With the picker off that
older order is the only one.

## Executor: Jev agent vs Claude Code

`GATE_EXECUTOR=jev` (default `claude`) runs approved requests through
`inkbox_claude/gate/jevagent.py` instead of Claude Code. No chat model drives the
loop. Each step is one TypeSafe request:

1. Is the goal achieved by the steps so far? (yes/no judgment; done at 0.6)
2. Which tool next, out of the scope's list, or give up? (a choice; unsure below
   `JEV_AGENT_MIN_CONF`, 0.45)
3. For that tool, one batched request fills the arguments: a yes/no per optional
   argument (supply it?), a choice among values already in play (ids, addresses,
   phones found in the request and earlier results), a choice for enums.
4. Only arguments that must be written (a body, a title, a date string) go to
   DeepSeek, one short call each, and a value that comes back as prose or
   UNKNOWN is rejected.

Code owns the loop, the schemas (read live from each MCP server), and every call.
Inkbox tools are called in-process; Google and site-admin servers are started
once per request from the same `~/.claude.json` config Claude Code uses. Rules:
read before write; a step that would repeat identically means the work is done;
a tool failing twice, an unfillable required argument, or a give-up ends the run
as FAILED. With `GATE_EXECUTOR_FALLBACK=claude` (default) a failed run that made
no write is re-run through Claude Code; after a successful write there is no
fallback (no double sends). `JEV_AGENT_MAX_STEPS` caps the loop (8).

Arguments that are range bounds (`time_min`, `time_max`) are a Choice over
named ranges (today, this week, last week, next 7 days, ...) whose ISO bounds
code computes from the clock, so no text model ever writes a date. Prose values
for one call are requested concurrently. Both executors refuse any send aimed
at the requester (address, phone, conversation, or the owner's), so a result is
delivered exactly once, by the gate.

### Limits and the breakdown

TypeSafe accepts about 32k input tokens per request (measured: 32,787 passed, the
next step up returned `max_tokens_exceeded`); a Choice takes at most 255 options.
The agent keeps every judgment under that without truncating anything a person or
the prose model sees:

- A judgment state under ~100k characters is sent raw.
- Over that, `_fit` replaces the largest value with a digest (kind, size, keys,
  item counts, header lines, counts of ids and addresses), largest first, until it
  fits. The full results stay in the agent's facts and in the report.
- When a value must be selected from a single result that is itself over budget,
  `narrow` runs a search in the shape of a binary search: split the text on line
  boundaries into parts that fit, ask Jev in parallel whether each part contains
  what the argument needs, descend into the best part, split again, down to a
  6k-character leaf. Candidates are extracted from that leaf. A 4,000-row result
  resolves in about four rounds.
- Candidates are newest-result-first and capped at 250.
- If TypeSafe still refuses a request, one retry replaces oversized strings with
  markers; any other judgment failure counts as unsure.

Other guards in the loop: a tool result that reads as an error (HTTP errors, "Error
calling tool") is a failed step; a value a tool already received for an argument is
not offered again, and the prose model is told which queries found nothing; after two
calls of the same tool in a row the next step must be a different tool; an unsure
step after reads finishes only when the done score is at least 0.3, otherwise the
run fails over. Step budget 12.

### One request, one response

`Store.running_for_thread` blocks a second request on a thread while one is
executing: the new message is written to the task and acknowledged, and if the
running request fails, the latest such message is re-decided as a fresh turn.
`GateSessionManager._run_executor` returns exactly one status; Claude Code, when it
runs, runs inside that call as an escalation (status carries `escalated: true` and
the Jev attempt as metadata) and nothing is delivered, noted, or phrased for the
Jev attempt. Both executors refuse sends to the requester at the tool call.

### Browser

`browser_read` (navigate, snapshot, find, screenshot, wait, tabs) and `browser_act`
(adds click, type, forms, select, keys, hover, drag, upload, dialogs) map to a
headless, isolated Playwright MCP server (`npx @playwright/mcp --headless
--isolated --browser chromium`) registered in the machine Claude config next to the
other servers. The code-execution tools are in neither scope and on the executor
deny list. Browser actions count as writes. No saved logins: pages behind a sign-in
need a persistent profile, not set up.

### Measured

Same box, same tool servers, 2026-09-28. Executor time only; the reply
phrasing after it is the same for both.

| request | Jev agent | Claude Code |
|---|---|---|
| TAMID board members and titles | 1.4 s | 14.1 s |
| Is the TAMID application open | 1.6 s | 8.0 s |
| SJBA upcoming events | 2.0 s | 9.5 s |
| TAMID calendar this week | 4.9 s | 8.4 s |
| Stern calendar today | 4.7 s | 12.5 s |
| Latest Stern email from a sender | 5.3 s | 10.7 s |
| Coffee chats last week | 9.4 s | 47 s (with the old fallback) |

Live request table, approval to result: Claude Code over 29 requests, median
18.3 s, mean 22.1 s, range 6 to 56 s. Jev agent over the requests since the
switch, median 3.5 s, mean 4.0 s, range 1.8 to 8.7 s. Site reads make zero
text-model calls; calendar lookups make zero since the date-range change.

## Voice engine: Live vs Realtime

Toggle in `~/.inkbox-claude/.env`, then `systemctl --user restart blatbot`:

| Setting | Meaning |
|---|---|
| `INKBOX_VOICE_API=live` | GPT-Live (`gpt-live-1`). Default. Falls back to Realtime if the Live connect fails. |
| `INKBOX_VOICE_API=realtime` | The older single-model Realtime bridge. |
| `INKBOX_LIVE_VOICE` | Voice for Live only (Live has extra voices, e.g. `meridian`). Unset = use `INKBOX_REALTIME_VOICE`. |
| `INKBOX_LIVE_MODEL` | Override the Live model id. |

Both engines call the same consult callback, so everything from the gateway down is shared.

| | Realtime (`realtime.py`) | Live (`live.py`) |
|---|---|---|
| Shape | One model: speech, reasoning, tools | Full-duplex voice model that delegates to a backend |
| How work reaches the gateway | Function tool `consult_agent(query)` | `session.delegation.created` (id only); bridge rebuilds the request from transcript deltas |
| How results return | Function output, then `response.create` | `session.commentary.append` with the delegation id; model paraphrases |
| Turn-taking, interruptions, noise | Bridge code: VAD thresholds, sustained-sound barge-in, playback tracking, wait-for-opening | The model's job; steered only by the prompt's interruption policy |
| Greeting | `response.create` with instructions | `session.instructions.append` after `session.started` |
| Hang up | `hang_up_call` tool | Caller hangs up (no tool) |
| Billing | Per audio token | $0.05 per minute |

None of the Realtime workarounds (noise reduction and VAD tuning, the half-second barge-in rule,
playback-aware result timing, injected hold lines, forced tool choice) exist in `live.py`.
They remain in `realtime.py` and only run when `INKBOX_VOICE_API=realtime`.
