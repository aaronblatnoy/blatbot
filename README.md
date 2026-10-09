# Blatbot

A personal AI assistant that many people can reach, and only one person can command.

Blatbot has its own email address, phone number, iMessage line, and Telegram account. Anyone can write to it or call it. It answers on my behalf, keeps track of what each person asked for, and can act on my calendar, inbox, and documents. The catch that makes this safe: when someone asks it to *do* something, nothing happens until I approve the exact action, by replying to a text or clicking it in the console. The exception is someone I have deliberately given a role to, and only for the scopes that role carries.

A surface is how a message arrived, not who it is with. Everything I send or am sent, on any surface, is one conversation written down once. Other people share a record wherever Inkbox resolves them to the same contact; Telegram is still kept separately.

> **Built on Inkbox.** This project is a fork of [inkbox-ai/claude-code-plugin](https://github.com/inkbox-ai/claude-code-plugin), the Inkbox bridge that gives a Claude Code agent a mailbox, a phone number, SMS, iMessage, and a tunnel. All of the transport (receiving webhooks, sending messages, carrying call audio) is Inkbox's work. This fork adds the layer on top: typed judgments that decide each turn, a reply writer, an approval gate, a task ledger, a tool-calling agent that runs without a chat model, and a phone surface. The `main` branch is Inkbox's code, untouched. The `blatbot` branch is this project. [See exactly what was added.](https://github.com/aaronblatnoy/blatbot/compare/main...blatbot) Inkbox's original README is kept at [docs/inkbox-plugin-README.md](docs/inkbox-plugin-README.md).

## The problem

Most assistant demos have one user. The interesting case is an assistant that lives at a public address. Candidates email it to book time with me. Colleagues text it. I call it while walking down the street. They are all talking to the same agent, and that agent can send email and edit my calendar.

So the real question is not "can the model do the task." It is "who is allowed to make it act, and how do I know what it is about to do." A system prompt that says "only obey Aaron" is not an answer, because prompts can be argued with. Blatbot answers it in code.

## How it works

```
  PEOPLE                                   SURFACES
 ┌─────────────────┐
 │ Owner           │──── iMessage ─────┐
 │ trusted number  │──── Telegram ─────┤
 │ trusted tg id   │──── phone call ───┼──────────────┐
 └─────────────────┘                   │              │
 ┌─────────────────┐                   │              │
 │ Trusted people  │──── email ────────┤              │
 │ roles + scopes  │──── SMS ──────────┤              │
 ├─────────────────┤──── Telegram ─────┤              │
 │ Anyone else     │──── phone call ───┼──────────────┤
 │ nothing by      │                   │              │
 │ default         │                   │              │
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
 │  sender ──> trusted?  owner's iMessage thread, telegram user id, or phone  ──> OWNER │
 │                       a person he has given a role, for those scopes   ──> TRUSTED│
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
 │  1  which task?      state: the message, last 8 stored messages, each candidate's │
 │                      id, title, where it stands, people, age. Choice over them    │
 │                      + new_task + no_task. Under 0.55 confidence: abstain.        │
 │  2  needs a tool?    state: message, turns, the task from 1. Noul. ≥0.65 yes, ≤0.35 no;  │
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
 │      the conversation            │   │ scopes  = Jev over a tree of systems          │
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
                  │                     │ TRUSTED, scopes covered ─> approved, run now │
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
                  │  │ EXECUTOR   Claude Code by default (see below), running the     │
                  │  │            hash-checked prompt with only the granted tools.     │
                  │  │            Jev picks Opus or Sonnet per request.                │
                  │  │                                                                │
                  │  │   denied at the tool call, in code, whichever engine runs:      │
                  │  │     a tool outside the scopes · anything destructive ·          │
                  │  │     a message to the requester · a send to anyone else when     │
                  │  │     the owner asked for it (held, shown to him, his yes runs it)│
                  │  │                                                                │
                  │  │ ALTERNATIVE   Jev judgments + code. No chat model drives it.   │
                  │  │                                                                │
                  │  │  hash of the prompt re-checked ──> mismatch: refuse            │
                  │  │  tools = only what the scopes map to; schemas read live        │
                  │  │                                                                │
                  │  │  ┌─ loop, at most 12 steps (JEV_AGENT_MAX_STEPS) ──────────┐  │
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
                  │  │  FAILED, gave up, or findings that do not answer the goal, │
                  │  │         prompt inside this same call: an escalation, not a    │
                  │  │         second execution; ONE status leaves, nothing is       │
                  │  │         delivered for the Jev attempt                          │
                  │  │  (this loop runs only when the engine is set to jev)           │
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
 │  someone else's task: the owner gets a done/failed note with the result                   │
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
threshold and every consequence. These were the judgments at the time of writing; several have been added since, including whether a request can be acted on as it stands, whether to speak in a group, whether a drafted reply promises work, which model a request needs, and how to search the ledger.

| # | when | state it sees | question | answers | code does |
|---|---|---|---|---|---|
| 1 | every message | message, last turns, candidate tasks (title, where it stands, participants, age) | which task is this? | `T12` ... / `new_task` / `no_task` | picks the task; below 0.55 confidence abstains and a new task is started; permission still checked in code |
| 2 | every message | message, last turns, the task from #1 | does this need a tool action now? | probability of yes | ≥0.5 a request is built; else reply only |
| 2b | result reply | the question, the results, the draft | (DeepSeek, not Jev) reasoning first: facts established, what follows, what stays unknown; then the reply as a conclusion | text | the reasoning is never sent |
| 3 | every message | message, the task | what did this message do to the task? | kind: asked / provided_information / confirmed / changed / declined / progress / conversation | writes the typed event to the ledger |
| 4 | every message | same | who does the task wait on now? | owner / other / nobody | updates the "where it stands" line |
| 5 | every message | same | is the task resolved? | probability | ≥0.8 marks it done |
| 6 | request built | the full prompt, the task title, each scope's description | does carrying this out require this capability? one per scope, asked together | probability per scope | grants every scope ≥0.6, plus the scopes this task's earlier requests had; if none clears, the two likeliest, else read-only web |
| 7 | agent, each step | goal, task record, background, steps so far with results | can the goal be answered with confidence from the evidence, every needed fact in hand? | probability | ≥0.7 stop, report; below, keep collecting; out of moves: "insufficient evidence" with what was found |
| 7b | agent, each step | same, plus each read tool's purpose | would calling this tool now add needed, not-yet-gathered information? one per read tool | probability per tool | ≥0.7 run it in parallel with the pick (up to 3) |
| 8 | agent, each step | same, plus every allowed tool's description | which tool next? | one of the allowed tools / give up | top probability <0.45 after reads: stop with findings; give up after reads: same; give up otherwise: FAILED |
| 9 | agent, per tool, one batched call | goal, the tool's schema, all facts and results | for each optional argument: supply it? | probability | <0.5 omitted |
| 10 | same call | same | for each id / email / phone argument: which of these values? | the candidate values found in the request and earlier results, or "write new" | copies the pick verbatim; top <0.3 on an optional argument omits it |
| 11 | same call | same | for each date-bound argument: which range? | today / tomorrow / yesterday / this week / last week / next week / next 7 days / past 7 days / this month / next 30 days / "specific" | fills the start or end from ranges computed off the clock |
| 12 | same call | same | for each enum or boolean: which value? | the enum | copies it |
| 13 | only when one result is over the API budget | goal, what the argument needs, one part of the result | does this part contain it? one per part, in parallel | probability per part | descends into the best part, splits again, down to a 6k leaf; candidates come from the leaf |
| 14 | agent, when #8 has no clear pick | same as #8 | of the three likeliest, which is the most useful next attempt? | one of three / give up | tries it; give up here means escalation |
| 15 | before a result reply is sent | the question, the retrieved results, the draft reply | is every claim supported by the results, counting stated inferences whose premises are in the results? | probability | under 0.5 the reply is rewritten (reason first, results only); a rewrite at 0.35+ passes; else the reply goes out with a one-line caveat |
| 16 | before a destructive call | the exact call and what earlier results say it refers to | (the owner, not Jev): "#N yes" runs that one call, "#N no" leaves it | text | nothing is deleted, cancelled or replaced without it |

Jev's ceiling is about 32k input tokens per request. A judgment state under that goes
in raw. Over it, code replaces the largest value with a structural digest (kind, size,
keys, counts, header lines, how many ids and addresses it holds), largest first, until
it fits. Only the judgment is shrunk. The whole result reaches the prose and
reply models up to 60k characters, and goal-relevant evidence above that; the full text is
always kept in the request record.

Anything not on this list is either code (hashing, permissions, the send guard, the
loop limit, calling the tool) or DeepSeek writing words: the reply to the sender, a
title for a new task, and a bare value the agent could not select (an email body, a
subject, a search query).

## Design rules

These are the decisions the whole thing rests on.

1. **The models that decide and reply before approval have no tools.** A typed-judgment model decides which task a message belongs to and whether anything needs doing; a text model writes the reply. Neither can call a tool. The judgment model cannot even produce free text: its answers are a choice from a list or a probability.
2. **The gate is code, not a prompt.** Whether a request runs is decided by comparing the sender against the owner's iMessage thread, his Telegram user id, or his phone number, and against the roles he has granted. No wording in a message can change that comparison. An email that claims to be from the owner, even from the owner's real address, is not the owner.
3. **No model paraphrases a request.** What runs is built by code: the sender's exact words, the recent conversation, and the task's record. The owner approves that, sees the tools it will get, and it is hashed when stored and checked again before execution.
4. **Every task gets a deliberately narrow set of tools, enforced as a hard maximum.** Scopes are chosen by a judgment over a tree of systems and leaves, widened on purpose in two ways: a task inherits the scopes its earlier requests had, and a secondary system can be included. The result is a ceiling, not a guess: A tool outside the granted scopes is denied at call time, in code, and so is any send aimed at the person who asked: the gate delivers results itself, once.
5. **A task is a unit of work, not a person.** One person can have several tasks; a task can involve several people, each reachable by email, phone, and iMessage. Every inbound message is assigned to a task before anything else happens, and every request is written to one. The ledger is full-text searchable and loaded before every turn, which is what lets the assistant pick a task back up hours later.
6. **The phone is a surface, not a second brain.** The voice model runs the conversation and is deliberately told nothing about the system behind it. When something real is needed it hands off, and that hand-off enters the same gateway as a text message, under the same rules. A caller who talks the voice model into something gains nothing, because the gate still checks the real caller number.
7. **The assistant never claims work it has not done.** Replies can only report an action as complete when the ledger shows it completed.
8. **One request at a time per thread, one response per request.** A message that arrives while a request is running joins the task and is acknowledged; it never starts a second run. Execution returns exactly one status. When the agent gives up without having written anything, Claude Code runs as an escalation inside that same execution, and only its status leaves. The result is phrased and sent once.
9. **Nothing is deleted, cancelled or replaced without the owner's yes, even when the owner asked.** The agent stops before a destructive call, texts the exact action and what it refers to (the event's title and time, the row), and performs that one call only on "#N yes". Claude Code cannot make such calls at all. Every result report opens with the writes performed, or "none".
10. **No confident wrong answers.** A result reply is checked by a judgment against the retrieved results before it is sent: names, titles, numbers, dates and "current" claims must appear in the results, or the reply is rewritten from the results only, or replaced by the raw result. In the agent, a web search result is a lead, not an answer; the page is opened and read before the run can end.
11. **Nothing goes out to another person without me seeing it, including when I asked for it.** A message the owner asks to have sent is him speaking to someone through the assistant, so the send is held at the tool boundary, the whole draft comes back with every recipient and the text verbatim, and only his yes releases that exact call. A request from anyone else was already read and approved by him in full, so answering it is untouched. The hold is enforced by withholding send tools from the pre-approved list, because a tool the runtime has been told to allow never reaches a permission hook.
12. **Everyone starts with no permissions.** A role names a level of trust and the scopes it carries; a person is trusted by handle, so the same person is recognised by email, phone, Telegram id or name. A request whose scopes fall entirely inside what they were given runs immediately; anything outside it still waits. Trusting someone with the calendar does not let them send mail.
13. **A surface is how a message arrived, not who it is with.** Email, iMessage, SMS, Telegram and the phone are one conversation with the owner, written to one record, so a notice sent on one is visible from another. A reply still goes back out the way it came. A group chat is its own conversation even when he is the one talking, because what is said in front of other people does not belong in his private record.
14. **In a group it decides whether to speak, and that decision is only for groups, and today only on Telegram.** A direct message is always answered. In a Telegram group, a message naming it is answered; one that does not goes to a single judgment asking whether its silence would be the worse answer, and an undecided or failed judgment stays quiet. It hears everything either way, and records what it chose not to answer, so a later follow-up has something to refer back to.
15. **An interrupted run is reported, not forgotten.** A run the process died inside leaves someone watching a typing indicator that never resolves. On startup an ordinary run is named on its own thread and offered again. A scheduled run is completed as a failed schedule run and reported to the owner. Neither is retried on its own, because a half-finished run may already have changed something.
16. **Tools are called by code, not by a chat model.** The agent that runs an approved request is a loop of typed judgments: is the goal met, which tool next, which of the values already in play fills each argument. Dates come from the clock, ids from earlier results. A text model is called only for text that must be composed, such as an email body. Claude Code remains as a fallback for what the loop cannot do, and only when nothing has been written yet.

## A worked example

1. A candidate emails: "Could we do Tuesday at 6 for a coffee chat?"
2. The gateway logs it to that candidate's ledger and sends me a one-line heads-up on iMessage.
3. Jev assigns the message to that candidate's coffee-chat task and judges that it needs an action. The reply writer, told this, answers them "Let me confirm that with Aaron." Code builds the request from their words and the task record; Jev grants the calendar and send-email scopes.
4. I get a text with their message, the task, and the tools. I reply `#12 yes`.
5. The agent lists the calendar, picks Tuesday's slot from a range computed off the clock, creates the event, asks the text model for a two-line confirmation, sends it, and reports `STATUS: OK`.
6. The ledger marks it done. The candidate gets their confirmation. I get a one-line "done."
7. Two days later they write "can we push it 30 minutes?" Jev assigns it to the same task, the request inherits that task's scopes, and the agent selects the existing event id from the record. It does not ask them what meeting they mean.

If I had sent the same request from my own iMessage thread, steps 3 to 5 would run immediately with no approval.

## Which engine runs the work

Two executors exist, and the running one is a setting.

**Claude Code is the default today.** It runs the hash-checked prompt with only the granted
tools, and a judgment picks Opus or Sonnet per request, so most work lands on Sonnet and only
a request judged hard gets the expensive model.

**The Jev agent is the alternative.** A loop of typed judgments that picks the tool and fills
each argument from values already in play, calling MCP tools directly with no chat model
driving it. A text model is asked only for text that must be composed.

They were measured against each other on fifteen read-only requests: Jev correct on thirteen
at about 7.6 s each, Claude Code correct on eleven at about 17 s and roughly $0.17 a request.
Jev is faster and far cheaper; Claude Code is more reliable on anything it has not seen before.
Reliability won, so the engine is `claude`. Switch it in the console under Settings, Engine; it is
read when the process starts, so it applies after a restart.

The safety rules do not depend on which one runs. A tool outside the granted scopes, anything
destructive, a message to the requester, and an outbound send the owner asked for are all
refused at the tool call in both engines.

## The console

A private web console, reachable only over the tailnet. It exists so permissions are
something I can see rather than something I remember.

- **People and permissions.** Pick a person, give them a role, tick extra scopes. For each
  one it spells out what runs on its own and what still interrupts me. Adding a person means
  adding a handle: an email, a phone, a Telegram id, or a name. "Refresh from Inkbox" pages
  every channel the agent identity has ever spoken on (`console/sync.py` via
  `console/directory.py`'s real `InkboxSDKDirectoryClient`) and folds the results into the
  people list by the same handle-normalization trust keys use, without ever touching trust
  or role assignment on its own.
- **Requests.** What is waiting on me at the top, with the sender, the surface, the scopes and
  their full message, approved or rejected inline. That calls the same command a `#N yes` text
  does, so a yes cannot come to mean two different things.
- **Tasks** with their ledgers, and **health**: the tunnel, uptime, interrupted runs, recent
  errors.
- **Settings.** The knobs that change how it behaves, grouped by what they affect, each with a
  plain label and a sentence of what it does. Most take effect on the next turn; the four that are read when the process starts say so
  on the control itself. The registry is the whole surface, so a value out of range is refused with the
  reason and a name not in it is refused outright: no key or path can be edited by a mis-click.

### Split architecture: this repo is the API only

The console's HTML/CSS/JS now live in a **separate frontend repo**,
[`blatbot-console`](https://github.com/aaronblatnoy/blatbot-console) (plain HTML/CSS/JS, no
build step, no framework), served by its own tiny static server on its own port. This repo
(`inkbox_claude/console/`) mounts only `/console/api/*` and `/console/events` -- no page
routes, no static asset routes. The gateway's public tunnel hostname does **not** serve any
`/console` path at all anymore; the console (both the page and the API) is reachable only at
the gateway's Tailscale address, never at the public hostname.

- `inkbox_claude/console/routes.py` -- `register(app, gateway)` mounts the API and the event
  stream on an existing `aiohttp.web.Application`.
- `inkbox_claude/console/access.py` -- `tailnet_gate` (peer must be in `100.64.0.0/10`, else
  404, never 403) and `cors_gate` (the Origin on a mutation must be the console's own origin
  or one named in `CONSOLE_ALLOWED_ORIGINS`). Both run on every `/console/api/*` and
  `/console/events` request, read or write.
- `inkbox_claude/console/sync.py` + `directory.py` -- the people sync and the real Inkbox SDK
  paging client behind it.
- `inkbox_claude/console/events.py` -- the server-sent-event hub for live refresh hints; sets
  its own CORS header before `response.prepare()`, since a streaming response's headers are
  already flushed by the time the `cors_gate` middleware would otherwise add one.

It holds no state of its own, and it is not a database editor: everything it changes goes
through the same commands the messaging surfaces use.

**`CONSOLE_ALLOWED_ORIGINS`** (env var on this process) must list the frontend's own origin,
e.g. `CONSOLE_ALLOWED_ORIGINS=http://100.64.0.10:8793`, or every mutation from the frontend
is refused with "cross-origin mutation rejected" even though reads still work. This is on top
of, not instead of, `tailnet_gate`: an allowed Origin header from a non-tailnet peer still
gets a 404 before the Origin check ever runs.

**`scripts/count_inkbox_people.py`** runs only the collection step the console's people-sync
uses (paging every Inkbox channel, then the same dedupe key `sync_people` uses) and prints
counts per channel plus a distinct total -- no names, numbers, or addresses, and no write to
the gate database. Run it yourself against the real account
(`INKBOX_API_KEY=... INKBOX_IDENTITY=... .venv/bin/python scripts/count_inkbox_people.py`); it
is intentionally not run by the test suite or by any Claude Code session in this repo.

### Deploying both sides on the same box

1. This process (the backend) keeps running as it always has, bound to the box's Tailscale
   address, with `CONSOLE_ALLOWED_ORIGINS` set to the frontend's origin.
2. The frontend (`blatbot-console`) runs as its own static file server (`serve.py`) under its
   own systemd user unit (`systemd/blatbot-console-frontend.service` in that repo), with
   `BIND_HOST` set to the same Tailscale address and its own port (8793 by convention, chosen
   to avoid every other port already in use on black-sky). `static/config.js` in that repo
   points at this backend's `/console/api` and `/console/events` on its Tailscale address.
3. There is a rerunnable end-to-end check in `blatbot-console/e2e/` that boots both sides
   against a throwaway store and the real backend code, and checks every page, the role/member
   CRUD, the Inkbox refresh, the live event stream, and both the tailnet gate and the
   cross-origin gate -- see that repo's README.

A private web console on the tailnet, never the public tunnel: live updates are server-sent
events and plain DOM, no build step, no framework, nothing fetched from the network at
runtime on either side.

## Schedules and continuing work

Aaron can ask Blatbot in his private thread to do something once at an absolute
time, on a recurring five-field cron cadence, or as continuing work. The same
forms are available on the console's Schedules page. Creating a schedule does
not run it. It enters the ordinary approval queue with the exact prompt, frozen
scopes, report mode, cadence rendered from the stored specification, and the
next three fire times computed in its IANA timezone. Only Aaron may create,
edit, pause, resume, run, or delete a schedule. An edit to its prompt, scopes,
cadence, or limits returns it to the approval queue.

A due schedule creates a normal request on its task and sends that request
through the existing executor, grounding, delivery, and ledger paths. Schedule
approval covers later sends and writes inside the approved scopes, but a delete,
cancel, replacement, or other destructive call still stops for Aaron's yes.
Scheduled requests do not receive the extra read scopes normally granted to an
owner request, and a failed run cannot grant itself a missing scope.

Recurring schedules use a dependency-free cron parser with numbers, wildcards,
lists, ranges, and steps. Day-of-month and day-of-week use the standard cron OR
rule. Times that do not exist during a daylight-saving change are skipped, and
a repeated local time fires once. Downtime collapses missed slots into one run.
The manager checks every 30 seconds, isolates errors by schedule, skips an
overlapping firing with a ledger note, honors a global pause, and pauses a
schedule after three consecutive failed runs.

Report mode `always` reports every run. Report mode `changed` reports failed
runs and runs with a successful write, but not read-only runs or the
`schedule_continue` wake-up itself. A failed final once run and every final
continue run are always reported. If a held destructive call is declined or
expires, the run closes without counting as a failure; recurring schedules
keep their already-computed next firing.

A continue schedule may use the `schedule_continue` host tool to set its next
wake-up and leave a progress note. The next run receives all notes in full. The
tool exists only for requests created by a continue schedule. A run that does
not call it closes the schedule. Continue schedules have an approval-time run
limit and deadline, defaulting to 20 runs and seven days, and wake-ups must be at
least five minutes apart.

## What is in this fork

The custom code spans the gate package, the console package, the Telegram and voice surfaces, the scope registry, and the tests.

| Path | What it is |
|---|---|
| `inkbox_claude/gate/manager.py` | The core: a session per person, the owner trust check, approval commands, the phone surface |
| `inkbox_claude/gate/router.py` | The reply writer (DeepSeek): told the decision, writes only the words sent back; also the ledger query planner |
| `inkbox_claude/gate/store.py` | SQLite: the task ledger, requests and their states, threads |
| `inkbox_claude/gate/schedules.py`, `cron.py` | Schedule approval, firing, continuation bounds, and timezone-aware cron calculation |
| `inkbox_claude/gate/taskpick.py` | The typed judgments (TypeSafe Jev): which task, whether action is needed, which scopes, what the message did to the task |
| `inkbox_claude/gate/jevagent.py` | The alternative executor: a loop of typed judgments that picks tools and selects arguments, calls MCP tools directly, prose only on demand |
| `inkbox_claude/gate/executor.py` | The Claude Code executor, the default: a hash-checked prompt, tools limited to the scopes, and the hook that holds an outbound send |
| `inkbox_claude/gate/scopes.yaml`, `scopes.resolved.json` | The scope registry and the tool lists resolved from the live MCP servers; `scopes_cli.py` lists, checks and re-syncs them |
| `inkbox_claude/gate/hosttools.py` | In-process tools the executor always has: host status, and exact filtering and counting over a result it already gathered |
| `inkbox_claude/gate/scopes.py` | Scope names mapped to tool lists, including a headless browser (Playwright over MCP) as read-only and interactive scopes, and the send-to-requester guard |
| `inkbox_claude/gate/decidegraph.py` | The gateway turn as a LangGraph: the judgments in parallel, the reply writer, the request, and the join that keeps the words and the work agreeing |
| `inkbox_claude/gate/jevgraph.py` | The executor as a graph: collect, judge, act, repeat |
| `inkbox_claude/gate/rooms.py` | What it means to be in a group chat, for any channel |
| `inkbox_claude/gate/settings.py` | The knobs the owner may turn, read from the database first and the environment second |
| `inkbox_claude/console/` | The console API only (`/console/api/*`, `/console/events`): read models, routes, the tailnet/CORS access gate, the server-sent event hub, and the Inkbox people sync. The page itself lives in the separate `blatbot-console` frontend repo. |
| `inkbox_claude/telegram.py` | Telegram as a surface: webhook, send, typing, and whether a group message was addressed to it |
| `inkbox_claude/live.py` | The phone bridge for OpenAI GPT-Live, using client delegation |
| `tests/test_gate.py`, `tests/test_schedules.py`, `tests/test_jevagent.py`, `tests/test_live.py` | Tests, including schedules, scripted judgments, and simulated phone calls with fake sockets |
| `docs/blatbot-architecture.md` | Longer design notes and the Live versus Realtime comparison |

The remaining edits are small hooks inside Inkbox's `gateway.py`, `realtime.py`, `config.py`, `tools.py`, and `sessions.py` that plug the gate in. The compare link above shows all of it.

## Running it

Start with Inkbox's own setup, which creates the agent identity and installs the bridge. Follow [docs/inkbox-plugin-README.md](docs/inkbox-plugin-README.md). Then turn on the gate by adding these to the bridge's `.env`:

```
# Turn on the gate
INKBOX_MODE=gate

# The reply writer and prose model. The primary is DeepSeek's OpenAI-compatible API; the
# timeout fallback is separately configurable (ROUTER_FALLBACK_*).
DEEPSEEK_API_KEY=

# The judgment model (TypeSafe Jev). Without it, DeepSeek decides tasks and
# proposes requests, and Claude Code runs them.
TYPESAFE_API_KEY=
GATE_EXECUTOR=claude           # claude (what runs today) or jev; also in the console
GATE_EXECUTOR_FALLBACK=claude  # or none; only used when the engine is jev
GATE_SEARCH_URL=http://127.0.0.1:8888/search   # a local SearXNG; web search runs through the browser against it

# Who the owner is. These values are the entire trust boundary.
INKBOX_APPROVER_PHONE=+15550100001
INKBOX_APPROVER_IMESSAGE_CONVERSATION_ID=
GATE_APPROVER_TELEGRAM_ID=      # the numeric user id Telegram stamps on every message

# Telegram as a surface. Turn privacy mode off in BotFather for group chats, then remove
# and re-add the bot, since the setting only applies on a fresh join.
TELEGRAM_BOT_TOKEN=
TELEGRAM_WEBHOOK_SECRET=        # generated if absent

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

Open the console frontend (the separate `blatbot-console` repo's own static server, e.g.
`http://<your-tailnet-host>:8793/console/`), pointed via `static/config.js` at this gateway's
`/console/api` on the same Tailscale address, with that frontend's origin added to
`CONSOLE_ALLOWED_ORIGINS` here. The public tunnel does not serve `/console` at all. Anything
listed in Settings can be changed there instead of in the `.env`, and the database wins over
the environment for those. Most apply on the next turn; which engine runs, which task picker,
the decision graph and the engine fallback are read at startup and the console says so.
Permissions live only in the console: everyone starts with none.

Run the tests with `pytest tests`.

### Speed

Measured on the deployed box, executor time from approval to result. Claude Code, the engine in use, 29 requests: median 18 s, range 6 to 56 s. Jev agent: site and roster reads 1.5 to 6 s; calendar and inbox questions 5 s; a multi-hop sheet question (find the form, its responses sheet, the right tab, read it) about 10 s; a web lookup that opens and reads a page 4 to 12 s. Escalation to Claude Code is the last resort and now starts from the agent's findings. Details and the per-request table are in [docs/blatbot-architecture.md](docs/blatbot-architecture.md).

### Honest caveats

- **Gate mode needs the source tree.** The scope registry is read from `gate/scopes.yaml` at import, and that file is not in the package manifest, so a built wheel will not start in gate mode.
- **The scope map is wired to my setup.** `scopes.py` references the specific Google Workspace tool servers I run. To use this yourself, edit that file to point at your own tools. Making this configurable is the next piece of work.
- **Some prompts still say "Blatbot" and "Aaron."** The persona and a few rules are written for me. They are being moved into config.
- **The safety classes are name-based.** Whether a tool is destructive, or sends to a person, is decided by its name and arguments against a maintained pattern, not by understanding what it does. A newly added tool with an unfamiliar name can slip past the hold until the pattern is updated.
- **A name is not a principal.** A person can be trusted by email, phone, Telegram id or name. Only the first three are stamped by a provider; a display name is whatever its owner types, so granting scopes to a name is not safe.
- **Caller ID is not authentication.** Trusting a phone number on voice calls is a convenience with a known weakness, which is why it is a separate switch.
- **It is a single-owner design.** One assistant, one person who can approve. Other people can be given roles, but only the owner grants them. It is not a multi-tenant service.
- **The console has no auth of its own.** It is safe because it is only reachable on a private network. Do not put it on the public internet as it stands.
- **A dropped tunnel still costs a run.** When the tunnel goes, the gateway exits so systemd can restart it with a fresh one, and a request in flight dies with it. It is now reported rather than silent, but not resumed.

## Credits and license

The bridge this is built on is [inkbox-ai/claude-code-plugin](https://github.com/inkbox-ai/claude-code-plugin) by [Inkbox](https://inkbox.ai). Their code is included here as a fork so the project runs as a whole, and all credit for the email, SMS, iMessage, voice transport, and tunnel belongs to them. The upstream repository does not currently state a license, so their code remains theirs and this fork makes no claim over it. If you want to reuse the transport, go to the upstream project.

The gate, the judgments, the agent, the reply writer, the ledger, the scopes, and the GPT-Live bridge were written by [Aaron Blatnoy](https://github.com/aaronblatnoy).
