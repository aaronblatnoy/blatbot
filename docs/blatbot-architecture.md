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
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │ GATEWAY  (code, no model)                          every turn, cannot be skipped
 │                                                                               │
 │  1. identify sender      2. write inbound      3. load context                │
 │     trusted = Aaron's       to TASK LEDGER         ledger + NY date/time      │
 │     iMessage thread or                             + last 20 messages         │
 │     Aaron's phone number                           + standing instructions    │
 │                                                    + contact notes            │
 │  4. FYI to Aaron if sender is someone else   (text/email only, not calls)     │
 └───────────────────────────────────┬───────────────────────────────────────────┘
                                     v
        ┌────────────────────────────────────────────────────────┐
        │ DECIDE   (TypeSafe Jev: typed judgments, no text)      │
        │  which task is this?        Choice over the ledger     │
        │  does it need a tool?       yes / no                   │
        └──────────────┬─────────────────────────┬───────────────┘
                       │                         │ yes
                       v                         v
        ┌──────────────────────────┐   ┌──────────────────────────────────┐
        │ REPLY WRITER (DeepSeek)  │   │ REQUEST   (code, no model)       │
        │ told the decision;       │   │ prompt = their exact words       │
        │ writes only the words    │   │   + conversation + task record   │
        │ sent back                │   │ scopes = Jev, one yes/no each    │
        └──────────────┬───────────┘   │ hashed and stored                │
                       │               └────────────────┬─────────────────┘
                       │                                v
                       │               ┌──────────────────────────────────┐
                       │               │ GATE  (code)                     │
                       │               │ from owner ──> approved, run now │
                       │               │ from others ─> HOLD              │
                       │               │    text owner: their message,    │
                       │               │    what it will do, the tools    │
                       │               │    "#N yes" / "no" / "edit: ..." │
                       │               └────────────────┬─────────────────┘
                       │                                v  approved
                       │               ┌──────────────────────────────────┐
                       │               │ AGENT  (Jev judgments + code)    │
                       │               │ loop: done? -> next tool ->      │
                       │               │   fill arguments -> call         │
                       │               │ arguments are SELECTED from      │
                       │               │   values in play, never written  │
                       │               │ prose model only for text that   │
                       │               │   must be composed (an email)    │
                       │               │ only the tools the scopes allow  │
                       │               │ fallback: Claude Code, only if   │
                       │               │   nothing was written yet        │
                       │               └────────────────┬─────────────────┘
                       │                                v
                       │      result -> TASK LEDGER (done / failed)
                       │                          │
                       v                          v
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │ DELIVERY   exactly once: executors are denied any send to the requester       │
 │  text/email : reply writer's words sent verbatim, same channel, signature     │
 │  phone      : result returned to voice agent as tool output, it says it       │
 │               its own way. Hung up or over ~75s: iMessage (Aaron) / SMS       │
 │  to Aaron   : done/failed note on iMessage when the task was someone else's   │
 └───────────────────────────────────────────────────────────────────────────────┘

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
