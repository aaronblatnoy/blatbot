# Blatbot

A personal AI assistant that many people can reach, and only one person can command.

Blatbot has its own email address, phone number, and iMessage line. Anyone can write to it or call it. It answers on my behalf, keeps track of what each person asked for, and can act on my calendar, inbox, and documents. The catch that makes this safe: when someone other than me asks it to *do* something, nothing happens until I approve the exact action from my phone.

> **Built on Inkbox.** This project is a fork of [inkbox-ai/claude-code-plugin](https://github.com/inkbox-ai/claude-code-plugin), the Inkbox bridge that gives a Claude Code agent a mailbox, a phone number, SMS, iMessage, and a tunnel. All of the transport (receiving webhooks, sending messages, carrying call audio) is Inkbox's work. This fork adds the layer on top: typed judgments that decide each turn, a reply writer, an approval gate, a task ledger, a tool-calling agent that runs without a chat model, and a phone surface. The `main` branch is Inkbox's code, untouched. The `blatbot` branch is this project. [See exactly what was added.](https://github.com/aaronblatnoy/blatbot/compare/main...blatbot) Inkbox's original README is kept at [docs/inkbox-plugin-README.md](docs/inkbox-plugin-README.md).

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
                  │  │  │ done?        yes/no over goal + steps so far   ≥0.6 stop│  │
                  │  │  │ next tool    Choice over the allowed tools + give up    │  │
                  │  │  │ arguments    ONE batched request, per argument:         │  │
                  │  │  │    optional?     yes/no: supply it or leave it          │  │
                  │  │  │    id/email/tel  Choice over values already in play     │  │
                  │  │  │                  (the request, earlier results)         │  │
                  │  │  │    date bound    Choice over ranges computed from the   │  │
                  │  │  │                  clock: today, this week, last week...  │  │
                  │  │  │    enum/bool     Choice                                 │  │
                  │  │  │    must be       DeepSeek writes the bare value (an     │  │
                  │  │  │    written       email body, a title); a sentence in an │  │
                  │  │  │                  id slot or UNKNOWN is rejected         │  │
                  │  │  │ guard        send aimed at the requester ──> refused    │  │
                  │  │  │              identical repeat of a done step ──> stop   │  │
                  │  │  │              same tool twice in a row: the next must dif│  │
                  │  │  │              a value already tried: not offered again   │  │
                  │  │  │              error text in a result: a failed step      │  │
                  │  │  │ budget       state over ~100k chars: largest result beco│  │
                  │  │  │              a digest; one result over budget is narrowe│  │
                  │  │  │              split, Jev scores the parts, descend, repea│  │
                  │  │  │ call         Inkbox tools in-process; Google and site   │  │
                  │  │  │              servers over MCP stdio; result kept whole  │  │
                  │  │  └──────────────────────────────────────────────────────────┘  │
                  │  │                                                                │
                  │  │  ends:  OK with every result in full                           │
                  │  │         FAILED: give up, a tool failed twice, an argument      │
                  │  │         could not be filled, step limit                        │
                  │  │  FAILED and nothing written ──> Claude Code runs the same     │
                  │  │         prompt inside this same call: an escalation, not a    │
                  │  │         second execution; ONE status leaves, nothing is       │
                  │  │         delivered for the Jev attempt                          │
                  │  │  FAILED after a write ──> stop, report; never retried         │
                  │  └──────────────────────────────┬─────────────────────────────────┘
                  │                                 v
                  │                   result ──> TASK LEDGER: done / failed
                  │                                 │
                  v                                 v
 ┌───────────────────────────────────────────────────────────────────────────────────┐
 │ DELIVERY   exactly once, grounded                                                 │
 │  the reply writer phrases the result; Jev checks every claim against the results  │
 │  (unsupported: rewritten from the results only, else the raw result is sent)      │
 │  executors cannot message the requester (denied at the tool call, in code)        │
 │  someone else's task: the owner gets a one-line done/failed note                  │
 │  phone: the result goes back to the voice agent as tool output; it says it        │
 └───────────────────────────────────────────────────────────────────────────────────┘
```


### One request, every call

What actually happened for one owner text on 2026-09-28, from the log. Times are wall
clock on the server.

```
03:24:35  iMessage from the owner's thread: "give me a list of all coffee chats from last week."
03:24:35  gateway: sender matches INKBOX_APPROVER_IMESSAGE_CONVERSATION_ID -> OWNER
          message stored on the thread; ledger loaded: 12 open tasks, 17,710 chars
03:24:36  ledger search (DeepSeek planned the filter, code ran the SQL):
            {text: "coffee chat", states: [live], touched_within_days: 3} -> 0 new matches
03:24:37  Jev #1 which task?   state = {message, last turns, candidates: [T62 "List
            Aaron's coffee chats from last week", T65 "List TAMID board members", ...]}
            answer: T62 (0.96)
03:24:37  Jev #2 needs a tool?  state = {message, turns, task: T62}   answer: yes (0.96)
03:24:38  DeepSeek reply writer, told "a request was created on T62 and will run now":
            reply = "On it, pulling last week's coffee chats now."      -> sent
03:24:38  code builds the request:
            prompt  = the message verbatim + last turns + T62's record
            summary = "List Aaron's coffee chats from last week"
03:24:38  Jev #6 scopes, one Noul each over that prompt:
            stern_calendar 0.90  imessage_send 0.69  sms_send 0.62  inbox_read 0.20 ...
            granted: stern_calendar (+ imessage_send, sms_send that night; since fixed:
            sending is only for messaging other people)
03:24:39  Jev #3-5 event on T62: kind=asked, waits on=owner, resolved=0.1 -> ledger
03:24:39  gate: OWNER -> approved, run now.  prompt hashed and stored as request #155
03:24:39  AGENT starts. tools = the scope's list; stern-drive MCP server started (2 s cold)
03:24:41  Jev #8 next tool?  state = {goal, T62 record, now, steps: []}
            answer: get_events (0.66)
03:24:42  Jev #9-12 arguments for get_events, one batched call:
            user_google_email      fixed by code: the owner's account
            calendar_id  optional  supply? 0.31 -> omitted (primary calendar)
            time_min     range?    last_week (0.88) -> 2026-09-21T00:00:00-04:00
            time_max     range?    last_week (0.91) -> 2026-09-28T00:00:00-04:00
            query        optional  supply? 0.72 -> must be written:
                                   DeepSeek -> "coffee"
03:24:45  tool call get_events(...)  -> 6 events, kept in full
03:24:46  Jev #7 done?  state = {goal, steps: [get_events -> 6 events ...]}  p = 0.54
03:24:46  Jev #8 next tool?  answer: give_up (0.79)  [it wanted to "send" the list]
          -> all steps so far were successful reads: finish with the findings
03:24:47  request #155 done. result = the six events, whole.
03:24:47  DeepSeek reply writer, told "no action" with the result as a system note:
            "Coffee chats last week (Mon 9/21 to Fri 9/25): 1. ... 6. ..."   -> sent
```

Eleven Jev calls, two DeepSeek calls for words and one for a search term, one tool
call, no Anthropic call. About 12 s including a cold MCP server start; 5 s warm.

### Every judgment, and what code does with it

Jev (TypeSafe's System One model) never writes text. Each call is a question over a JSON
state and returns a pick with probabilities, or a probability of yes. Code owns every
threshold and every consequence. This is the complete list.

| # | when | state it sees | question | answers | code does |
|---|---|---|---|---|---|
| 1 | every message | message, last turns, candidate tasks (title, where it stands, participants, age) | which task is this? | `T12` ... / `new_task` / `no_task` | picks the task; below 0.55 confidence abstains and a new task is started; permission still checked in code |
| 2 | every message | message, last turns, the task from #1 | does this need a tool action now? | probability of yes | ≥0.5 a request is built; else reply only |
| 3 | every message | message, the task | what did this message do to the task? | kind: asked / provided_information / confirmed / changed / declined / progress / conversation | writes the typed event to the ledger |
| 4 | every message | same | who does the task wait on now? | owner / other / nobody | updates the "where it stands" line |
| 5 | every message | same | is the task resolved? | probability | ≥0.8 marks it done |
| 6 | request built | the full prompt, the task title, each scope's description | does carrying this out require this capability? one per scope, asked together | probability per scope | grants every scope ≥0.6, plus the scopes this task's earlier requests had; if none clears, the two likeliest, else read-only web |
| 7 | agent, each step | goal, task record, date, steps so far with results | is the goal fully achieved? | probability | ≥0.6 stop, report |
| 8 | agent, each step | same, plus every allowed tool's description | which tool next? | one of the allowed tools / give up | top probability <0.45 after reads: stop with findings; give up after reads: same; give up otherwise: FAILED |
| 9 | agent, per tool, one batched call | goal, the tool's schema, all facts and results | for each optional argument: supply it? | probability | <0.5 omitted |
| 10 | same call | same | for each id / email / phone argument: which of these values? | the candidate values found in the request and earlier results, or "write new" | copies the pick verbatim; top <0.3 on an optional argument omits it |
| 11 | same call | same | for each date-bound argument: which range? | today / tomorrow / yesterday / this week / last week / next week / next 7 days / past 7 days / this month / next 30 days / "specific" | fills the start or end from ranges computed off the clock |
| 12 | same call | same | for each enum or boolean: which value? | the enum | copies it |
| 13 | only when one result is over the API budget | goal, what the argument needs, one part of the result | does this part contain it? one per part, in parallel | probability per part | descends into the best part, splits again, down to a 6k leaf; candidates come from the leaf |
| 14 | agent, when #8 has no clear pick | same as #8 | of the three likeliest, which is the most useful next attempt? | one of three / give up | tries it; give up here means escalation |
| 15 | before a result reply is sent | the question, the retrieved results, the draft reply | is every factual claim in the draft supported by the results? | probability | under 0.5 the reply is rewritten from the results only; still under, the raw result is sent with no interpretation |

Jev's ceiling is about 32k input tokens per request. A judgment state under that goes
in raw. Over it, code replaces the largest value with a structural digest (kind, size,
keys, counts, header lines, how many ids and addresses it holds), largest first, until
it fits. Only the judgment is shrunk; the full results still reach the prose model, the
reply model, and the owner.

Anything not on this list is either code (hashing, permissions, the send guard, the
loop limit, calling the tool) or DeepSeek writing words: the reply to the sender, a
title for a new task, and a bare value the agent could not select (an email body, a
subject, a search query).

## Design rules

These are the decisions the whole thing rests on.

1. **The models that read strangers' messages have no tools.** A typed-judgment model decides which task a message belongs to and whether anything needs doing; a text model writes the reply. Neither can call a tool. The judgment model cannot even produce free text: its answers are a choice from a list or a probability.
2. **The gate is code, not a prompt.** Whether a request runs is decided by comparing the sender against the owner's iMessage thread or phone number. No wording in a message can change that comparison. An email that claims to be from the owner, even from the owner's real address, is not the owner.
3. **No model paraphrases a request.** What runs is built by code: the sender's exact words, the recent conversation, and the task's record. The owner approves that, sees the tools it will get, and it is hashed when stored and checked again before execution.
4. **Every task gets the smallest set of tools that covers it.** Scopes map to tool lists, chosen by one yes/no judgment per scope. A tool outside the granted scopes is denied at call time, in code, and so is any send aimed at the person who asked: the gate delivers results itself, once.
5. **A task is a unit of work, not a person.** One person can have several tasks; a task can involve several people, each reachable by email, phone, and iMessage. Every inbound message is assigned to a task before anything else happens, and every request is written to one. The ledger is full-text searchable and loaded before every turn, which is what lets the assistant pick a task back up hours later.
6. **The phone is a surface, not a second brain.** The voice model runs the conversation and is deliberately told nothing about the system behind it. When something real is needed it hands off, and that hand-off enters the same gateway as a text message, under the same rules. A caller who talks the voice model into something gains nothing, because the gate still checks the real caller number.
7. **The assistant never claims work it has not done.** Replies can only report an action as complete when the ledger shows it completed.
8. **One request at a time per thread, one response per request.** A message that arrives while a request is running joins the task and is acknowledged; it never starts a second run. Execution returns exactly one status. When the agent gives up without having written anything, Claude Code runs as an escalation inside that same execution, and only its status leaves. The result is phrased and sent once.
9. **No confident wrong answers.** A result reply is checked by a judgment against the retrieved results before it is sent: names, titles, numbers, dates and "current" claims must appear in the results, or the reply is rewritten from the results only, or replaced by the raw result. In the agent, a web search result is a lead, not an answer; the page is opened and read before the run can end.
10. **Tools are called by code, not by a chat model.** The agent that runs an approved request is a loop of typed judgments: is the goal met, which tool next, which of the values already in play fills each argument. Dates come from the clock, ids from earlier results. A text model is called only for text that must be composed, such as an email body. Claude Code remains as a fallback for what the loop cannot do, and only when nothing has been written yet.

## A worked example

1. A candidate emails: "Could we do Tuesday at 6 for a coffee chat?"
2. The gateway logs it to that candidate's ledger and sends me a one-line heads-up on iMessage.
3. Jev assigns the message to that candidate's coffee-chat task and judges that it needs an action. The reply writer, told this, answers them "Let me confirm that with Aaron." Code builds the request from their words and the task record; Jev grants the calendar and send-email scopes.
4. I get a text with their message, the task, and the tools. I reply `#12 yes`.
5. The agent lists the calendar, picks Tuesday's slot from a range computed off the clock, creates the event, asks the text model for a two-line confirmation, sends it, and reports `STATUS: OK`.
6. The ledger marks it done. The candidate gets their confirmation. I get a one-line "done."
7. Two days later they write "can we push it 30 minutes?" Jev assigns it to the same task, the request inherits that task's scopes, and the agent selects the existing event id from the record. It does not ask them what meeting they mean.

If I had sent the same request from my own iMessage thread, steps 3 to 5 would run immediately with no approval.

## What is in this fork

Everything custom lives in one folder plus one file.

| Path | What it is |
|---|---|
| `inkbox_claude/gate/manager.py` | The core: a session per person, the owner trust check, approval commands, the phone surface |
| `inkbox_claude/gate/router.py` | The reply writer (DeepSeek): told the decision, writes only the words sent back; also the ledger query planner |
| `inkbox_claude/gate/store.py` | SQLite: the task ledger, requests and their states, threads |
| `inkbox_claude/gate/taskpick.py` | The typed judgments (TypeSafe Jev): which task, whether action is needed, which scopes, what the message did to the task |
| `inkbox_claude/gate/jevagent.py` | The executor: a loop of typed judgments that picks tools and selects arguments, calls MCP tools directly, prose only on demand |
| `inkbox_claude/gate/executor.py` | The Claude Code fallback for a hash-checked prompt, tools limited to the scopes |
| `inkbox_claude/gate/scopes.py` | Scope names mapped to tool lists, including a headless browser (Playwright over MCP) as read-only and interactive scopes, and the send-to-requester guard |
| `inkbox_claude/live.py` | The phone bridge for OpenAI GPT-Live, using client delegation |
| `tests/test_gate.py`, `tests/test_jevagent.py`, `tests/test_live.py` | Tests, including scripted judgments and simulated phone calls with fake sockets |
| `docs/blatbot-architecture.md` | Longer design notes and the Live versus Realtime comparison |

The remaining edits are small hooks inside Inkbox's `gateway.py`, `realtime.py`, `config.py`, `tools.py`, and `sessions.py` that plug the gate in. The compare link above shows all of it.

## Running it

Start with Inkbox's own setup, which creates the agent identity and installs the bridge. Follow [docs/inkbox-plugin-README.md](docs/inkbox-plugin-README.md). Then turn on the gate by adding these to the bridge's `.env`:

```
# Turn on the gate
INKBOX_MODE=gate

# The reply writer and prose model. Any OpenAI-compatible chat model works; DeepSeek is the default.
DEEPSEEK_API_KEY=

# The judgment model (TypeSafe Jev). Without it, DeepSeek decides tasks and
# proposes requests, and Claude Code runs them.
TYPESAFE_API_KEY=
GATE_EXECUTOR=jev              # jev (default when the key is set) or claude
GATE_EXECUTOR_FALLBACK=claude  # or none
GATE_SEARCH_URL=http://127.0.0.1:8888/search   # a local SearXNG; web search runs through the browser against it

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

Run the tests with `pytest tests`.

### Speed

Measured on the deployed box, executor time from approval to result. Claude Code, 29 requests: median 18 s, range 6 to 56 s. Jev agent: site and roster reads 1.5 to 6 s; calendar and inbox questions 5 s; a multi-hop sheet question (find the form, its responses sheet, the right tab, read it) about 10 s; a web lookup that opens and reads a page 4 to 12 s. Escalation to Claude Code is the last resort and now starts from the agent's findings. Details and the per-request table are in [docs/blatbot-architecture.md](docs/blatbot-architecture.md).

### Honest caveats

- **The scope map is wired to my setup.** `scopes.py` references the specific Google Workspace tool servers I run. To use this yourself, edit that file to point at your own tools. Making this configurable is the next piece of work.
- **Some prompts still say "Blatbot" and "Aaron."** The persona and a few rules are written for me. They are being moved into config.
- **Caller ID is not authentication.** Trusting a phone number on voice calls is a convenience with a known weakness, which is why it is a separate switch.
- **It is a single-owner design.** One assistant, one person who can approve. It is not a multi-tenant service.

## Credits and license

The bridge this is built on is [inkbox-ai/claude-code-plugin](https://github.com/inkbox-ai/claude-code-plugin) by [Inkbox](https://inkbox.ai). Their code is included here as a fork so the project runs as a whole, and all credit for the email, SMS, iMessage, voice transport, and tunnel belongs to them. The upstream repository does not currently state a license, so their code remains theirs and this fork makes no claim over it. If you want to reuse the transport, go to the upstream project.

The gate, the judgments, the agent, the reply writer, the ledger, the scopes, and the GPT-Live bridge were written by [Aaron Blatnoy](https://github.com/aaronblatnoy).
