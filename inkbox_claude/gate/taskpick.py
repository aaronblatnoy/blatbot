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


SCOPE_MIN_YES = float(os.getenv("TYPESAFE_SCOPE_MIN_YES") or 0.6)


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
            "another_model_said_action_needed": router_said_action,
        }
        body = {"state": state, "model": self.model, "questions": {"needs_action": {
            "type": "noul",
            "instructions": {
                "question": "Should the assistant now go and DO something for this message, using its tools?",
                "the_assistant_can": "read and change calendars, inboxes, contacts, documents, spreadsheets, forms "
                                     "and website admin pages; search the web; send email and texts.",
                "count_as_yes": [
                    "The sender asks for anything to be booked, moved, cancelled, sent, looked up, checked, "
                    "changed, added, closed, or found out.",
                    "The sender asks a question whose answer must be looked up (a calendar, an inbox, a sheet, "
                    "a website), even if phrased casually.",
                    "The sender supplies information that `task_it_belongs_to` was waiting for, so its pending "
                    "step can now be carried out (e.g. an application, availability, a confirmation, details).",
                    "The sender wants to schedule or set up something (a chat, a meeting, a call).",
                ],
                "count_as_no": [
                    "Thanks, greetings, acknowledgements, small talk.",
                    "A question about the assistant itself or about what it will do, answerable in words.",
                    "A clarification that adds nothing actionable yet.",
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
