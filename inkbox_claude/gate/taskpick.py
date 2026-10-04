"""Task selection as a typed judgment (TypeSafe System One, model Jev).

The router (a chat LLM) writes the reply and the request. Which task a message
belongs to is a narrower question: given the candidate tasks, pick one or say
"new". That is a Choice over a defined set, which is what Jev is built for. It
returns the pick, a probability for every option, and a confidence; code owns
the threshold and the permission checks.

Configuration (environment):
  TYPESAFE_API_KEY          required to enable; unset means the router's own pick is used
  TYPESAFE_MODEL            default jev-latest
  TYPESAFE_TASK_MIN_CONF    default 0.55; below this the pick is treated as undecided
  GATE_TASK_PICKER          "jev" (default when the key is set) or "router"
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
NEW = "new_task"
NO_TASK = "no_task"


def enabled() -> bool:
    if (os.getenv("GATE_TASK_PICKER") or "").strip().lower() == "router":
        return False
    return bool((os.getenv("TYPESAFE_API_KEY") or "").strip())


def _age(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 3600:
        return f"{seconds // 60} minutes ago"
    if seconds < 48 * 3600:
        return f"{seconds // 3600} hours ago"
    return f"{seconds // 86400} days ago"


def candidate_view(t: Dict[str, Any], now: Optional[float] = None) -> Dict[str, Any]:
    """The facts about one task that decide whether a message continues it."""
    now = now or time.time()
    who = [p.get("display") or p.get("key") for p in (t.get("participants") or [])]
    events = t.get("events") or []
    return {
        "title": t.get("title") or "",
        "where_it_stands": t.get("summary") or "",
        "people_involved": who or ["nobody in particular"],
        "state": t.get("state"),
        "last_touched": _age(now - float(t.get("updated_at") or now)),
        "recent_events": [f"{e.get('kind')}: {str(e.get('text') or '')[:160]}" for e in events[-4:]],
    }


EVENT_KINDS = {
    "asked_for_something": "The message asks for work to be done, information, or a decision.",
    "provided_information": "The message supplies information the task was waiting for (details, an answer, a file, availability).",
    "confirmed": "The message agrees to or confirms something already proposed on the task.",
    "changed_or_corrected": "The message changes or corrects something already on the task (a time, a detail, a decision).",
    "declined_or_cancelled": "The message declines, cancels or withdraws.",
    "reported_progress": "The message reports that something was done or is in progress.",
    "conversation_only": "Small talk, thanks, acknowledgement, or nothing that changes the task.",
}


def _last_request_outcome(task: Optional[Dict[str, Any]]) -> str:
    """What the task's most recent request did, for the action judgment: a
    correction after a run means run again; before any run it is just detail."""
    if not task:
        return "none"
    for e in reversed(task.get("events") or []):
        k = e.get("kind")
        if k in ("done", "failed"):
            return f"{k}: {str(e.get('text') or '')[:200]}"
        if k in ("request", "approved"):
            return f"requested, not finished: {str(e.get('text') or '')[:160]}"
    return "nothing has run on this task yet"


from .scopes import SCOPES, SCOPE_TREE_ALWAYS, SCOPE_TREE_LEAVES, SCOPE_TREE_SYSTEMS, WHERE_THINGS_LIVE  # the registry


SCOPE_MIN_YES = float(os.getenv("TYPESAFE_SCOPE_MIN_YES") or 0.6)
SCOPE_SECONDARY_MIN = float(os.getenv("TYPESAFE_SCOPE_SECONDARY_MIN") or 0.3)


# Who the assistant is in a room full of people. Not a rule about when to talk: it is
# what any participant knows about their own standing and reach before deciding to speak.
_DEFAULT_ROLE = (
    "Executive assistant to one member of this chat. Works on their behalf across their "
    "email, calendar, contacts, files and the organisations they run, can look things up "
    "and act in those systems, and holds the record of the work it has been given. The "
    "others in the room have none of that reach."
)


def _history_text(history: List[Dict[str, Any]], limit: int = 12) -> str:
    """The conversation as plain lines, for a judgment that has to read it."""
    out = []
    for row in (history or [])[-limit:]:
        who = "them" if (row.get("kind") or row.get("role")) in ("inbound", "user") else "you"
        text = (row.get("text") or row.get("content") or "").strip()
        if text:
            out.append(f"{who}: {text}")
    return "\n".join(out)


class TaskPicker:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 min_confidence: Optional[float] = None, timeout: float = 20.0):
        self.api_key = (api_key or os.getenv("TYPESAFE_API_KEY") or "").strip()
        self.model = (model or os.getenv("TYPESAFE_MODEL") or "jev-latest").strip()
        self.min_confidence = float(min_confidence if min_confidence is not None
                                    else (os.getenv("TYPESAFE_TASK_MIN_CONF") or 0.55))
        self.timeout = timeout

    async def judge_grounded(self, *, reply: str, results: str, question: str) -> Dict[str, Any]:
        """Before a result is delivered: is every factual claim in `reply` supported
        by `results` (the tool output it was written from)? Returns {"grounded":
        bool|None, "p": float}. Names, titles, numbers, dates and 'current'/'former'
        claims that the results do not contain count as unsupported."""
        if not self.api_key:
            return {"grounded": None, "p": 0.0, "reason": "disabled"}
        if len(results) > 90000:
            # Over TypeSafe's request ceiling even after the agent's narrowing: undecided, not a 400.
            return {"grounded": None, "p": 0.0, "reason": "results too large to judge"}
        state = {"question_asked": question, "results_retrieved": results, "draft_reply": reply}
        body = {"state": state, "model": self.model, "questions": {"grounded": {
            "type": "noul",
            "instructions": {
                "question": "Is every factual claim in `draft_reply` directly supported by `results_retrieved`?",
                "count_as_no": [
                    "A name, title, number, date, email, or 'current'/'former' status in the reply that does not "
                    "appear in the results.",
                    "The reply states as fact something the results only suggest (a search snippet, a stale page, "
                    "a partial list).",
                    "The reply claims a lookup was done or a source was read that the results do not show.",
                ],
                "count_as_yes": [
                    "Every specific claim can be pointed to in the results.",
                    "The reply says plainly what was NOT found or not confirmed.",
                    "Summaries, counts and comparisons computed from the results themselves.",
                    "An inference stated as one (\"so\", \"which means\", \"likely\") whose premises are in the "
                    "results: an email sent to a list reached the people on it; a person with no slot has not "
                    "booked; an event on the calendar means the meeting is scheduled.",
                ],
            },
            "criteria": {"true": "All claims are supported by the results.",
                         "false": "At least one claim is not supported by the results."},
        }}}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(TYPESAFE_URL, headers={"Authorization": f"Bearer {self.api_key}"}, json=body)
                r.raise_for_status()
                p = float(r.json()["answers"]["grounded"].get("noul") or 0.0)
        except Exception as exc:
            logger.warning("grounding judgment via TypeSafe failed: %s", exc)
            return {"grounded": None, "p": 0.0, "reason": f"error: {exc}"}
        logger.info("grounding judgment: p(supported)=%.2f", p)
        return {"grounded": p >= 0.5, "p": p, "reason": "ok"}

    async def judge_action(self, *, message: str, sender_label: str, history: List[Dict[str, Any]],
                           task: Optional[Dict[str, Any]], router_said_action: Optional[bool] = None) -> Dict[str, Any]:
        """Does this message call for the assistant to DO something with its tools
        (look up, book, send, change), as opposed to being answerable in conversation?
        Returns {"needs_action": bool|None, "p": float}. None = undecided."""
        if not self.api_key:
            return {"needs_action": None, "p": 0.0, "reason": "disabled"}
        state = {
            "sender": sender_label,
            "conversation_so_far": [f"[{m.get('kind')}] {str(m.get('text') or '')[:300]}" for m in history[-6:]],
            "new_message": message,
            "task_it_belongs_to": candidate_view(task) if task else "none",
            "last_action_on_that_task": _last_request_outcome(task),
            "last_thing_the_assistant_said": next((str(m.get("text") or "")[:400] for m in reversed(history)
                                                   if m.get("kind") == "outbound"), ""),
            "another_model_said_action_needed": router_said_action,
        }
        body = {"state": state, "model": self.model, "questions": {"needs_action": {
            "type": "noul",
            "instructions": {
                "question": "Should the assistant now go and DO something for this message, using its tools?",
                "principle": "The assistant is a general-purpose executive assistant with a computer, a browser, "
                             "Aaron's accounts, and a coding agent behind it. It is NOT limited to scheduling. Any "
                             "message that can only be satisfied by doing something or by fetching a fact from "
                             "outside this conversation is a yes. The lists below are illustrations, not the "
                             "boundary: if a capable human assistant at a laptop could act on it, count it as yes.",
                "the_assistant_can": [
                    "Anything a person can do at a computer: read and write files and documents, run scripts and "
                    "commands, research, compute, compare, draft, format, convert, summarize a source it fetches.",
                    "TAMID Google Calendar and Aaron's Stern calendar: list events for a day or range, free/busy, "
                    "create, move, shorten, cancel events.",
                    "TAMID Google Drive: find sheets, docs, forms and folders; read a sheet tab's rows (rosters, "
                    "schedules, trackers, form responses); read docs; write cells, append rows, create sheets, "
                    "docs and folders; read a form's questions and every response.",
                    "TAMID Gmail and Aaron's Stern Gmail: search messages and threads, read one in full.",
                    "Blatbot's own Inkbox mailbox, texts and iMessages: list and read threads, send email, "
                    "SMS or iMessage to a third party; look up, create and update contacts and their notes.",
                    "TAMID and SJBA website admin: list and read board members and bios, events, members, "
                    "semesters, site settings, contact-form submissions, newsletter signups; create, update "
                    "and delete any of those; replace headshots and flyers.",
                    "Public web: search, open a page, read and find text on it, click, type and fill forms.",
                    "The black-sky server: what is running (containers, services, uptime, disk, memory, GPUs).",
                    "TAMID's Instagram and LinkedIn: account, recent posts, insights, audience; stage and publish posts.",
                    "TAMID's Google Analytics property: dimensions, metrics, key events, streams, access; change them.",
                    "Aaron's NYU Brightspace: courses, assignments, due dates, grades, announcements, discussions, "
                    "content; submit, post, mark complete.",
                    "Coolify (black-sky and Vox): apps, services, databases, deployments, logs, env vars; deploy, "
                    "restart, stop.",
                ],
                "count_as_yes": [
                    "The sender asks for anything to be done, made, fetched, booked, moved, cancelled, sent, "
                    "looked up, checked, changed, added, closed, counted, computed, drafted or found out.",
                    "Any task with a concrete deliverable that does not yet exist in the conversation: a document, "
                    "a list, a summary of a source, a comparison, a file, a number, a status.",
                    "Writing on request is a deliverable and a yes: draft an email, a bio, an announcement, a "
                    "message to send, a post, a script, a report. The assistant produces it as a task result.",
                    "A request outside the examples below but within reach of a computer and Aaron's accounts.",
                    "The sender asks a question whose answer must be looked up (a calendar, an inbox, a sheet, "
                    "a website), even if phrased casually or as a fragment.",
                    "Calendar: 'when is X's interview', 'what's on Friday', 'is 2:30 free', 'move X to 9', "
                    "'make it 30 minutes', 'cancel X's slot but keep the other one', 'add a room to the event'.",
                    "Sheets and forms: 'how many people applied', 'who hasn't responded', 'what school is X in', "
                    "'find X's email', 'which slots are open', 'mark X as interviewed', 'add a row for X', "
                    "'what's the link to that sheet', 'what's the exact title of the form'.",
                    "Mail: 'did we email X', 'what did X say', 'has X replied', 'forward me the thread', "
                    "'find the invitation we sent'.",
                    "Websites: 'who is on the board', 'update X's bio', 'add the event', 'who contacted us "
                    "through the site', 'take X off the board page', 'swap the headshot'.",
                    "Contacts and messaging: 'text X that ...', 'email X the list', 'what's X's number', "
                    "'add a note to X's contact'.",
                    "Public web: 'look up X', 'who is the president of Y', 'find Z's LinkedIn', 'confirm that "
                    "roster on their site'.",
                    "Server: 'what's running on black sky', 'is the gateway up', 'how much disk is left'.",
                    "Social: 'what did we post last', 'how did the recruitment post do', 'how many followers', "
                    "'post this to instagram', 'draft a linkedin post about X'.",
                    "School: 'what's due this week', 'did I get a grade on X', 'any new announcements in Y', "
                    "'submit the memo', 'when is the quiz'.",
                    "Deployments: 'is the billing frontend up', 'redeploy the validator', 'show me the last "
                    "deploy log', 'what env vars does X have', 'restart the gateway app'.",
                    "The sender supplies information that `task_it_belongs_to` was waiting for, so its pending "
                    "step can now be carried out (an application, availability, a confirmation, details).",
                    "The sender wants to schedule or set up something (a chat, a meeting, a call).",
                    "'yes', 'ok', 'sure', 'go ahead', 'do it', 'please' after `last_thing_the_assistant_said` offered "
                    "to do or check something: the sender is accepting the offer, so do it.",
                    "A correction, addition or 'try again' on a task whose `last_action_on_that_task` already "
                    "ran (done or failed): the sender wants it done again with the change ('should be 30 minutes', "
                    "'the other one', 'give it those scopes', 'no, Friday', 'yes, resubmit').",
                    "A follow-up question whose answer is not in the conversation and must come from the same "
                    "source again ('what was the document called', 'give me the link', 'and their emails?').",
                ],
                "count_as_no": [
                    "Thanks, greetings, acknowledgements, small talk.",
                    "A question about the assistant itself or about what it will do, answerable in words.",
                    "A question already answered by facts in `conversation_so_far`: rephrasing, filtering or "
                    "reformatting what was just delivered ('without Dylan and Sean', 'same thing but shorter').",
                    "A clarification on a task whose next step has NOT run yet and still cannot run.",
                ],
            },
            "criteria": {
                "true": "Yes: at least one tool action should happen now because of this message.",
                "false": "No: a reply in words is the right response; no tool action is needed yet.",
            },
        }}}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(TYPESAFE_URL, headers={"Authorization": f"Bearer {self.api_key}"}, json=body)
                r.raise_for_status()
                p = float(r.json()["answers"]["needs_action"].get("noul") or 0.0)
        except Exception as exc:
            logger.warning("action judgment via TypeSafe failed: %s", exc)
            return {"needs_action": None, "p": 0.0, "reason": f"error: {exc}"}
        logger.info("action judgment: p(needs tool)=%.2f", p)
        if p >= 0.65:
            return {"needs_action": True, "p": p, "reason": "ok"}
        if p <= 0.35:
            return {"needs_action": False, "p": p, "reason": "ok"}
        return {"needs_action": None, "p": p, "reason": "undecided"}

    async def judge_scopes(self, *, prompt: str, summary: str, scopes: Dict[str, str],
                           router_scopes: Optional[List[str]] = None) -> Dict[str, Any]:
        """Which tool scopes does this task prompt need? One yes/no (Noul) per scope,
        all asked together over the same prompt. Returns {"scopes": [...], "probabilities": {...}}.
        A scope is granted when its yes-probability is at or above SCOPE_MIN_YES.
        The router's own list is a hint in the state and the caller's fallback."""
        if not self.api_key or not scopes:
            return {"scopes": None, "probabilities": {}, "reason": "disabled"}
        state = {
            "task_summary": summary,
            "task_prompt": prompt,
            "another_model_suggested": router_scopes or [],
            "delivery": "The gateway itself delivers the task's answer to whoever asked. Sending capabilities "
                        "(email, text, iMessage) are needed only when the task must message SOMEONE ELSE, never "
                        "to report the result back to the requester.",
        }
        questions: Dict[str, Any] = {}
        for name, desc in scopes.items():
            questions[name] = {
                "type": "noul",
                "instructions": {
                    "capability": desc,
                    "question": "Does carrying out `task_prompt` require this capability? Judge by what the task "
                                "must actually do, not by what is merely mentioned.",
                },
                "criteria": {
                    "true": "At least one step of the task cannot be completed without this capability.",
                    "false": "The task can be completed fully without it, or it is only referenced in passing.",
                },
            }
        body = {"state": state, "model": self.model, "questions": questions}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(TYPESAFE_URL, headers={"Authorization": f"Bearer {self.api_key}"}, json=body)
                r.raise_for_status()
                answers = r.json()["answers"]
        except Exception as exc:
            logger.warning("scope judgment via TypeSafe failed: %s", exc)
            return {"scopes": None, "probabilities": {}, "reason": f"error: {exc}"}
        probs = {k: float((answers.get(k) or {}).get("noul") or 0.0) for k in scopes}
        chosen = [k for k, v in probs.items() if v >= SCOPE_MIN_YES]
        logger.info("scope judgment: %s | top=%s", chosen, sorted(probs.items(), key=lambda kv: -kv[1])[:5])
        return {"scopes": chosen, "probabilities": probs, "reason": "ok"}

    async def judge_scopes_tree(self, *, prompt: str, summary: str) -> Dict[str, Any]:
        """Scopes as a two-level tree. Level 1: which SYSTEMS the task touches (a yes/no
        per system, seven at most). Level 2, only for the systems chosen: read or write,
        and which channel. Each judgment is a small question over a few options; the
        scope names are assembled by code from the answers. Returns the same shape as
        judge_scopes: {"scopes": [...], "probabilities": {...}, "reason": str}."""
        if not self.api_key:
            return {"scopes": None, "probabilities": {}, "reason": "disabled"}
        state = {
            "task_summary": summary,
            "task_prompt": prompt,
            "delivery": "The gateway itself delivers the task's answer to whoever asked. Messaging is needed only "
                        "when the task must contact SOMEONE ELSE, never to report back to the requester.",
            "where_things_live": WHERE_THINGS_LIVE,
        }

        def noul(capability: str) -> Dict[str, Any]:
            return {"type": "noul",
                    "instructions": {"capability": capability,
                                     "question": "Does carrying out `task_prompt` require this? Judge by what the task "
                                                 "must actually do, not by what is merely mentioned."},
                    "criteria": {"true": "At least one step of the task cannot be done without it.",
                                 "false": "The task can be done fully without it."}}

        async def ask(questions: Dict[str, Any]) -> Dict[str, float]:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(TYPESAFE_URL, headers={"Authorization": f"Bearer {self.api_key}"},
                                      json={"state": state, "model": self.model, "questions": questions})
                r.raise_for_status()
                answers = r.json()["answers"]
            return {k: float((answers.get(k) or {}).get("noul") or 0.0) for k in questions}

        # One request: every system noul and every leaf noul together. The tree is applied
        # in code afterwards: a leaf counts only when its system opened.
        questions: Dict[str, Any] = {f"sys::{k}": noul(v) for k, v in SCOPE_TREE_SYSTEMS.items()}
        questions["wants_change"] = {"type": "noul",
                                     "instructions": {"question": "Does `task_prompt` (with the conversation it quotes) ask for "
                                                                  "something to be CHANGED, created, moved, cancelled, sent or "
                                                                  "written, as opposed to only looked up or counted?"},
                                     "criteria": {"true": "Something must be written or sent somewhere.",
                                                  "false": "Only reading, finding, counting or answering."}}
        for sysname, leaves in SCOPE_TREE_LEAVES.items():
            for leaf, (capability, _) in leaves.items():
                questions[f"leaf::{sysname}::{leaf}"] = noul(capability)
        try:
            answers = await ask(questions)
        except Exception as exc:
            logger.warning("scope tree via TypeSafe failed: %s", exc)
            return {"scopes": None, "probabilities": {}, "reason": f"error: {exc}"}
        systems = {k: answers.get(f"sys::{k}", 0.0) for k in SCOPE_TREE_SYSTEMS}
        wants_change = answers.get("wants_change", 0.0) >= 0.5
        chosen_systems = [k for k, v in systems.items() if v >= SCOPE_MIN_YES]
        # Secondary systems: plausible but not certain. Granting them costs nothing now
        # that the agent picks tools by system; missing them loses the right source.
        secondary = [k for k, v in systems.items() if SCOPE_SECONDARY_MIN <= v < SCOPE_MIN_YES]
        if chosen_systems and secondary:
            chosen_systems += secondary
        if not chosen_systems:
            # A request exists, so something is needed: take the likeliest system when it is
            # not implausible, rather than granting nothing and failing the run for want of a tool.
            top, p = max(systems.items(), key=lambda kv: kv[1])
            if p >= 0.4:
                chosen_systems = [top]
        logger.info("scope tree: systems %s wants_change=%.2f | top=%s", chosen_systems, answers.get("wants_change", 0.0),
                    sorted(systems.items(), key=lambda kv: -kv[1])[:4])
        probs: Dict[str, float] = {}
        scopes: List[str] = []
        leaves_log = []
        for sysname in chosen_systems:
            for sc in SCOPE_TREE_ALWAYS.get(sysname, []):
                if sc not in scopes:
                    scopes.append(sc)
                probs[sc] = systems[sysname]
            strong = systems[sysname] >= SCOPE_MIN_YES
            for leaf, (_, granted) in SCOPE_TREE_LEAVES.get(sysname, {}).items():
                p = answers.get(f"leaf::{sysname}::{leaf}", 0.0)
                leaves_log.append((leaf, round(p, 2)))
                for sc in granted:
                    probs[sc] = max(probs.get(sc, 0.0), p)
                # A leaf under a system the task clearly touches opens on a lower bar: a write
                # the agent never needs costs nothing (it only writes when the goal is an action,
                # and destructive calls still wait for the owner), while a missing one fails the task.
                is_write = any(not SCOPES.get(sc, {}).get("read", False) for sc in granted) if granted else False
                # The task is a change: every write leaf of a clearly chosen system opens, since
                # which write it is was already decided by the system and the goal.
                if p >= SCOPE_MIN_YES or (strong and p >= SCOPE_SECONDARY_MIN) or (strong and wants_change and is_write and p >= 0.1):
                    for sc in granted:
                        if sc not in scopes:
                            scopes.append(sc)
        if leaves_log:
            logger.info("scope tree: leaves %s", sorted(leaves_log, key=lambda kv: -kv[1]))
        logger.info("scope judgment (tree): %s", scopes)
        return {"scopes": scopes, "probabilities": probs, "reason": "ok"}

    async def judge_reply(self, *, reply: str, message: str) -> Dict[str, Any]:
        """Two nouls about a drafted reply, one call: does it promise work the assistant is
        about to do, and is it only an acknowledgement (no facts, no question)? None for
        either when undecided; the caller falls back to its word rules."""
        if not self.api_key or not reply:
            return {"promises_action": None, "is_acknowledgement": None, "reason": "disabled"}
        state = {"message_being_answered": message, "draft_reply": reply}
        body = {"state": state, "model": self.model, "questions": {
            "promises": {"type": "noul", "instructions": {
                "question": "Does `draft_reply` tell the sender that the assistant is now going to do, check, look up, "
                            "send, submit or change something (work it is about to carry out)?"},
                "criteria": {"true": "It announces work the assistant will now do ('on it', 'pulling that up', "
                                     "'let me check', 'submitting now', 'I'll look').",
                             "false": "It answers, asks, acknowledges, or offers ('want me to...?') without claiming "
                                      "work is underway."}},
            "ack": {"type": "noul", "instructions": {
                "question": "Is `draft_reply` ONLY a brief acknowledgement that work is underway, with no facts, no "
                            "answer and no question in it?"},
                "criteria": {"true": "A short holding line: 'on it', 'checking now', 'pulling your calendar up'.",
                             "false": "It states a fact, gives an answer, lists anything, or asks a question."}},
        }}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(TYPESAFE_URL, headers={"Authorization": f"Bearer {self.api_key}"}, json=body)
                r.raise_for_status()
                a = r.json()["answers"]
        except Exception as exc:
            logger.warning("reply judgment via TypeSafe failed: %s", exc)
            return {"promises_action": None, "is_acknowledgement": None, "reason": f"error: {exc}"}
        p1 = float((a.get("promises") or {}).get("noul") or 0.0)
        p2 = float((a.get("ack") or {}).get("noul") or 0.0)
        logger.info("reply judgment: p(promises)=%.2f p(ack)=%.2f", p1, p2)
        return {"promises_action": p1 >= 0.5, "is_acknowledgement": p2 >= 0.5, "reason": "ok"}

    async def judge_actionable(self, *, message: str, history: List[Dict[str, Any]],
                               task: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Before anything is handed to the execution layer: can it be carried out as it
        stands? One noul. A request missing the thing it acts on is better answered with a
        question than with a run that fails and asks the same question afterwards.
        Returns {"actionable": bool|None, "p": float}."""
        if not self.api_key or not message:
            return {"actionable": None, "p": 0.0, "reason": "disabled"}
        state = {"message": message, "conversation": _history_text(history),
                 "task_in_hand": (task or {}).get("title") or ""}
        body = {"state": state, "model": self.model, "questions": {"clear": {
            "type": "noul",
            "instructions": {
                "question": "Someone with full access to the accounts, files and calendars is about to go and "
                            "do what `message` asks. Reading it with `conversation`, do they know what to do, "
                            "or would they have to come back and ask what was meant?",
                "guidance": [
                    "Judge it as the one who has to go and do it. What is missing has to be missing: a "
                    "detail you can settle by looking, or that the conversation already supplies, is not.",
                ],
            },
            "criteria": {"true": "They could go and do it.",
                         "false": "They would have to ask first."},
        }}}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(TYPESAFE_URL, headers={"Authorization": f"Bearer {self.api_key}"}, json=body)
                r.raise_for_status()
                a = r.json()["answers"]
        except Exception as exc:
            logger.warning("actionable judgment via TypeSafe failed: %s", exc)
            return {"actionable": None, "p": 0.0, "reason": f"error: {exc}"}
        p = float((a.get("clear") or {}).get("noul") or 0.0)
        floor = float(os.getenv("GATE_ACTIONABLE_MIN") or 0.4)
        logger.info("actionable judgment: p(clear enough to run)=%.2f floor=%.2f", p, floor)
        return {"actionable": p >= floor, "p": p, "reason": "ok"}

    async def judge_group_reply(self, *, message: str, sender: str, group: str,
                                recent: str = "", bot_name: str = "", role: str = "") -> Dict[str, Any]:
        """A group message that did not name the assistant: is it nonetheless something the
        assistant should answer? One noul. Returns {"should_reply": bool|None, "p": float}.
        Undecided (None) means stay quiet: in a group, silence is the safe default."""
        if not self.api_key or not message:
            return {"should_reply": None, "p": 0.0, "reason": "disabled"}
        state = {"group_name": group, "sender": sender, "message": message,
                 "recent_messages": recent, "assistant_name": bot_name or "Blatbot",
                 # Who the assistant is in this room. Without it, "is this mine to answer?"
                 # has no referent and the judgment is guessing at its own job.
                 "assistant_role": role or (os.getenv("GATE_ASSISTANT_ROLE") or _DEFAULT_ROLE)}
        body = {"state": state, "model": self.model, "questions": {"reply": {
            "type": "noul",
            "instructions": {
                "question": "You are in this group chat in the capacity described by `assistant_role`, "
                            "reading it as it happens. `message` has just been sent. Do you speak?",
                "guidance": [
                    "You speak when your silence would be the worse answer. Where you sit cuts both ways: "
                    "being in the room is not being one of the party, so what is theirs you let them have, "
                    "and what you are there for is yours to carry whether or not anyone turns to you for "
                    "it. Judge it in the moment, not by working anything out.",
                ],
            },
            "criteria": {"true": "You would say something.",
                         "false": "You would let it pass."},
        }}}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(TYPESAFE_URL, headers={"Authorization": f"Bearer {self.api_key}"}, json=body)
                r.raise_for_status()
                a = r.json()["answers"]
        except Exception as exc:
            logger.warning("group reply judgment via TypeSafe failed: %s", exc)
            return {"should_reply": None, "p": 0.0, "reason": f"error: {exc}"}
        p = float((a.get("reply") or {}).get("noul") or 0.0)
        floor = float(os.getenv("GATE_GROUP_REPLY_MIN") or 0.45)
        logger.info("group reply judgment: p(should reply)=%.2f floor=%.2f", p, floor)
        return {"should_reply": p >= floor, "p": p, "reason": "ok"}

    async def judge_complexity(self, *, prompt: str, summary: str, scopes: List[str]) -> Dict[str, Any]:
        """Does this request need the stronger (slower, dearer) model? One noul over the
        request. Returns {"complex": bool|None, "p": float}."""
        if not self.api_key:
            return {"complex": None, "p": 0.0, "reason": "disabled"}
        state = {"task_summary": summary, "task_prompt": prompt, "capabilities_granted": scopes}
        body = {"state": state, "model": self.model, "questions": {"complex": {
            "type": "noul",
            "instructions": {
                "question": "Does carrying out `task_prompt` call for the strongest reasoning model, rather than the "
                            "fast one?",
                "count_as_yes": [
                    "Several sources must be combined or reconciled (a roster against responses, a sheet against a calendar).",
                    "Many separate changes or sends (a batch of events, a mail merge), or a write whose wrong "
                    "execution is costly (deleting, moving, publishing, emailing many people).",
                    "Writing something substantial: a bio, an announcement, a long reply, a document.",
                    "The request is ambiguous or underspecified and needs careful interpretation.",
                    "Long documents or many rows must be read and judged, not just filtered.",
                ],
                "count_as_no": [
                    "One lookup in one place: a time, a name, a phone, an email, a count, a status.",
                    "A single straightforward change with the target already named.",
                ],
            },
            "criteria": {"true": "The strongest model is worth its time and cost here.",
                         "false": "The fast model handles this fine."},
        }}}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(TYPESAFE_URL, headers={"Authorization": f"Bearer {self.api_key}"}, json=body)
                r.raise_for_status()
                p = float(r.json()["answers"]["complex"].get("noul") or 0.0)
        except Exception as exc:
            logger.warning("complexity judgment via TypeSafe failed: %s", exc)
            return {"complex": None, "p": 0.0, "reason": f"error: {exc}"}
        logger.info("complexity judgment: p(needs strongest model)=%.2f", p)
        return {"complex": p >= 0.5, "p": p, "reason": "ok"}

    async def judge_event(self, *, message: str, sender_label: str, task: Dict[str, Any],
                          history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Given the task the message was assigned to, judge what the message did to it
        and whether the task is now waiting on the owner, on the other person, or done.

        Returns {"kind": <EVENT_KINDS key>, "kind_conf": float, "waiting_on": "owner"|"other"|"nobody"|None,
                 "resolves_task": float}. Text for the event line is composed by code."""
        if not self.api_key:
            return {"kind": None, "kind_conf": 0.0, "waiting_on": None, "resolves_task": 0.0, "reason": "disabled"}
        state = {
            "task": candidate_view(task),
            "sender": sender_label,
            "conversation_so_far": [f"[{m.get('kind')}] {str(m.get('text') or '')[:300]}" for m in history[-6:]],
            "new_message": message,
        }
        body = {
            "state": state, "model": self.model,
            "questions": {
                "kind": {
                    "type": "choice",
                    "instructions": "What did `new_message` do to `task`?",
                    "criteria": EVENT_KINDS,
                },
                "waiting_on": {
                    "type": "choice",
                    "instructions": "After `new_message`, who does `task` wait on next?",
                    "criteria": {
                        "owner": "The owner (the assistant's boss) must decide, approve, or act next.",
                        "other": "The other person on the task must reply or provide something next.",
                        "nobody": "Nothing is outstanding right now; the next step is the assistant's own or the task is done.",
                    },
                },
                "resolves_task": {
                    "type": "noul",
                    "instructions": "Does `new_message` complete the work described in `task`, so nothing more is needed on it?",
                    "criteria": {"true": "The task's work is finished by this message.",
                                 "false": "Work on the task remains."},
                },
            },
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(TYPESAFE_URL, headers={"Authorization": f"Bearer {self.api_key}"}, json=body)
                r.raise_for_status()
                a = r.json()["answers"]
        except Exception as exc:
            logger.warning("event judgment via TypeSafe failed: %s", exc)
            return {"kind": None, "kind_conf": 0.0, "waiting_on": None, "resolves_task": 0.0, "reason": f"error: {exc}"}
        kind, kc = a["kind"].get("choice"), float(a["kind"].get("confidence") or 0)
        wo, wc = a["waiting_on"].get("choice"), float(a["waiting_on"].get("confidence") or 0)
        res = float(a["resolves_task"].get("noul") or 0)
        logger.info("event judgment: kind=%s (%.2f) waiting_on=%s (%.2f) resolves=%.2f", kind, kc, wo, wc, res)
        return {"kind": kind if kc >= self.min_confidence else None, "kind_conf": kc,
                "waiting_on": wo if wc >= self.min_confidence else None, "resolves_task": res, "reason": "ok"}

    async def plan_lookup(self, *, message: str, history: List[Dict[str, Any]], sender_label: str,
                          in_view_titles: List[str], candidates: List[str]) -> Dict[str, Any]:
        """Should the ledger be searched for a task not already in view, and for what?

        Two typed judgments in one call: `lookup` (no lookup / live tasks / any state)
        and `terms` (which phrase from the message to search for, chosen from
        `candidates`, all taken from words in play). Returns {"lookup": "none" |
        "live" | "any" | None, "text": str, "confidence": float, "reason": str}.
        None means undecided: the caller falls back to its own planner."""
        if not self.api_key:
            return {"lookup": None, "text": "", "confidence": 0.0, "reason": "disabled"}
        NONE_TERM = "(none)"
        terms = [c for c in candidates if c][:40]
        state = {
            "sender": sender_label,
            "tasks_already_in_view": in_view_titles[:60] or ["(no tasks on record)"],
            "conversation_so_far": [f"[{m.get('kind')}] {str(m.get('text') or '')[:300]}" for m in history[-8:]],
            "new_message": message,
        }
        questions: Dict[str, Any] = {
            "lookup": {
                "type": "choice",
                "instructions": {
                    "question": "Does acting on `new_message` need a task that is NOT in `tasks_already_in_view`?",
                    "how_to_decide": [
                        "The view already holds every live task for this sender. A follow-up, correction, "
                        "or status question about work listed there needs no lookup.",
                        "A person, subject, or piece of work named in the message that no title in the view "
                        "covers needs a lookup.",
                        "Choose any_state when the message refers to work that may already be finished or failed "
                        "(\"what happened with\", \"last week\", \"did you ever\").",
                    ],
                },
                "criteria": {
                    "no_lookup": "Every task the message could concern is already in view, or the message is "
                                 "only conversation.",
                    "live_tasks": "Search open, waiting and running tasks for what the message names.",
                    "any_state": "Search tasks in every state, including done and failed.",
                },
            },
        }
        if terms:
            questions["terms"] = {
                "type": "choice",
                "instructions": {
                    "question": "If a lookup is made, which phrase from `new_message` identifies the task or "
                                "person to search for?",
                    "how_to_decide": ["Prefer a person's name or email, then the subject of the work.",
                                      f"Choose {NONE_TERM} when no phrase would find anything useful."],
                },
                "criteria": {t: f"search the ledger for \"{t}\"" for t in terms} | {NONE_TERM: "no useful search phrase"},
            }
        body = {"state": state, "model": self.model, "questions": questions}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(TYPESAFE_URL, headers={"Authorization": f"Bearer {self.api_key}"}, json=body)
                r.raise_for_status()
                answers = r.json()["answers"]
        except Exception as exc:
            logger.warning("lookup planning via TypeSafe failed: %s", exc)
            return {"lookup": None, "text": "", "confidence": 0.0, "reason": f"error: {exc}"}
        look = answers.get("lookup") or {}
        choice = str(look.get("choice") or "")
        conf = float(look.get("confidence") or 0.0)
        term = str(((answers.get("terms") or {}).get("choice")) or "")
        text = "" if term == NONE_TERM else term
        logger.info("lookup plan: %s (conf %.2f) terms=%r", choice, conf, text)
        if conf < self.min_confidence:
            return {"lookup": None, "text": text, "confidence": conf, "reason": "low confidence"}
        kind = {"no_lookup": "none", "live_tasks": "live", "any_state": "any"}.get(choice)
        return {"lookup": kind, "text": text, "confidence": conf, "reason": "ok"}

    async def pick(self, *, message: str, history: List[Dict[str, Any]], sender_label: str,
                   candidates: List[Dict[str, Any]], router_hint: Optional[str] = None,
                   proposed_title: Optional[str] = None) -> Dict[str, Any]:
        """Return {"choice": "T12" | "new" | None, "confidence": float, "probabilities": {...}}.

        ``None`` means undecided (below threshold or no candidates and no request),
        and the caller falls back to its own rule."""
        if not self.api_key:
            return {"choice": None, "confidence": 0.0, "probabilities": {}, "reason": "disabled"}
        now = time.time()
        options: Dict[str, Any] = {}
        for t in candidates[:40]:
            options[f"T{t['id']}"] = candidate_view(t, now)
        options[NEW] = {
            "meaning": "This message is about a different piece of work from every task listed: "
                       "a new job, even if it is with the same person.",
            "proposed_title": proposed_title or "",
        }
        options[NO_TASK] = "This message is only conversation (a greeting, thanks, small talk, a question " \
                           "answered without doing anything) and does not belong to any piece of work."
        state = {
            "sender": sender_label,
            "conversation_so_far": [
                f"[{m.get('kind')}] {str(m.get('text') or '')[:300]}" for m in history[-8:]
            ],
            "new_message": message,
            "router_suggestion": router_hint or "(none)",
        }
        body = {
            "state": state,
            "model": self.model,
            "questions": {
                "task": {
                    "type": "choice",
                    "instructions": {
                        "question": "Which task does `new_message` belong to?",
                        "how_to_decide": [
                            "A task is a unit of work, not a person. The same person can have several tasks.",
                            "Continue an existing task when the message is about that same work: a follow-up, "
                            "a correction, a status question, or the next step of it.",
                            f"Choose {NEW} when the message asks for a different job, even from someone who "
                            "already has tasks.",
                            f"Choose {NO_TASK} only when nothing needs doing or tracking.",
                            "`router_suggestion` is a hint from another model and may be wrong.",
                        ],
                    },
                    "criteria": options,
                },
            },
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(TYPESAFE_URL, headers={"Authorization": f"Bearer {self.api_key}"}, json=body)
                r.raise_for_status()
                ans = r.json()["answers"]["task"]
        except Exception as exc:
            logger.warning("task pick via TypeSafe failed: %s", exc)
            return {"choice": None, "confidence": 0.0, "probabilities": {}, "reason": f"error: {exc}"}
        choice = str(ans.get("choice") or "")
        conf = float(ans.get("confidence") or 0.0)
        probs = ans.get("probabilities") or {}
        logger.info("task pick: %s (conf %.2f) top=%s", choice, conf,
                    sorted(probs.items(), key=lambda kv: -kv[1])[:3])
        if conf < self.min_confidence:
            return {"choice": None, "confidence": conf, "probabilities": probs, "reason": "low confidence"}
        if choice == NEW:
            return {"choice": "new", "confidence": conf, "probabilities": probs, "reason": "ok"}
        if choice == NO_TASK:
            return {"choice": "none", "confidence": conf, "probabilities": probs, "reason": "ok"}
        return {"choice": choice, "confidence": conf, "probabilities": probs, "reason": "ok"}
