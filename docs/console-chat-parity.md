# Console and chat parity

Everything the web console shows or changes, and the text Aaron sends Blatbot to do the
same. Owner only, direct chat only: in a group chat these get one short line
("I only change or show my setup in our direct chat. Ask me there.") and nothing changes,
except a room's own mode ("set this chat to silent"), which belongs to that room.

Every reply goes through `gate/chatreply.py` (`finish()`): answer first, at most 8 list
lines then "and N more" with where to see the rest, no symbols, no raw ids, no scope
slugs, times as "today 9:00 AM". Changes that the console asks to confirm (delete a role,
remove or merge a person, stop a schedule, widen memory or sending) ask once with
"#id yes / no".

Status: "before" = could already be done by text before 2026-11-18; "added" = added in
this change; "no" = not covered yet.

| Console page / action | Console route | Text phrasing | Covered |
|---|---|---|---|
| Home overview | GET /console/api/home | "overview", "what's going on", "dashboard" | added |
| Overview counts | GET /console/api/overview | "overview" (same reply as Home) | added |
| Health | GET /console/api/health | "are you ok", "health" | before |
| People list | GET /console/api/people | "show people" | before (reshaped) |
| One person | GET /console/api/people (row) | "show Sean", "what can Sean Parnell do" | added / before |
| Add person | POST /console/api/people/create | "add Jane Doe to people" | before |
| Edit person name or note | POST /console/api/people/update | "rename X to Y", "set X's note to ..." | before |
| Remove person | POST /console/api/people/remove, /people/delete | "delete X from people" (asks #id yes/no) | before |
| Add, remove, move contact | POST /console/api/people/contact/* | "add the email a@b.com to X", "move ... from X to Y" | before |
| Edit a contact in place | POST /console/api/people/contact/update | remove then add the new value | before (two steps) |
| Merge two people | POST /console/api/people/link | "merge X into Y" (asks #id yes/no) | before |
| Sync people from Inkbox | POST /console/api/people/sync | "refresh people" | before |
| Directory (Inkbox contacts) | GET /console/api/people/directory | "refresh people", then "show people" | before |
| Avatar image | GET /console/api/avatar | not a text thing (a picture) | n/a |
| Roles list | GET /console/api/roles, /roles/summary | "show roles" | before |
| One role | GET /console/api/roles | "what can the interns role do" | before |
| Create role | POST /console/api/roles | "create a role called interns" | before |
| Rename role | POST /console/api/roles/rename | "rename the interns role to volunteers" | before |
| Delete role | POST /console/api/roles/delete (+ delete_preview) | "delete the interns role" (asks #id yes/no, names who holds it) | before |
| Add or remove a person's role | POST /console/api/roles/members/* | "make Sean a corgi teammate", "Sean is no longer a general" | before |
| Grant or remove role abilities | POST /console/api/roles/scopes/* | "let the interns role read the TAMID calendar", "take X away from the interns role" | before |
| Role access level per system | POST /console/api/roles/set_level | "set the interns role's calendar to read" | before |
| Role may send from a system | POST /console/api/roles/set_send | "let the interns role send from email" | before |
| Role notes access | POST /console/api/roles/set_notes | "set the interns role's notes to all / none" | added |
| Role memory reach | POST /console/api/roles/set_memory | "set the interns role's memory to own / groups / everything" (widening asks #id yes/no) | added |
| Who can do X | GET /console/api/roles/who_can | "who has access to my calendar", "who can read the TAMID calendar" | before |
| Who can see notes and memory | GET /console/api/knowledge | "who can see my notes" | added |
| Systems and abilities | GET /console/api/scopes, /scopes/breakdown | "show tools", "what systems can you use" | added |
| Requests waiting on Aaron | GET /console/api/requests | "what's pending", "show pending requests" | before |
| Approve or decline | POST /console/api/requests/decide | "#521 yes", "#521 no", "#521 edit: ..." | before |
| Failed requests | GET /console/api/requests?state=failed | "show failed requests", "what's stuck" | before |
| Retry a failed request | POST /console/api/requests/retry | "retry #521" | before |
| Dismiss a failed request | POST /console/api/requests/dismiss | "dismiss #521" | added |
| Tasks | GET /console/api/tasks | "tasks", "show tasks" | added |
| One task's detail | GET /console/api/tasks/{id} | ask about it by name ("what happened with the Abhinash reminder") through the normal chat path | before (via chat, not a fixed command) |
| Schedules list | GET /console/api/schedules | "show schedules" | added |
| Schedule run history | GET /console/api/schedules/{id}/runs | "when did the corgi digest last run" | added |
| Create a schedule | POST /console/api/schedules | "every weekday at 9 send me ..." (proposal, then yes) | before |
| Edit a schedule | POST /console/api/schedules/edit | "make the digest run at 8 instead" | before |
| Pause, resume, run now | POST /console/api/schedules/action | "pause the 9am digest", "run the digest now" | before |
| Stop a schedule | POST /console/api/schedules/action (stop) | "stop the digest" (asks #id yes/no) | before |
| Approve a proposed schedule | POST /console/api/schedules/approve | "#id yes" on the proposal | before |
| Repair a schedule | GET+POST /console/api/schedules/repair | "repair it", "repair the digest" | before |
| Pause or resume every schedule | POST /console/api/schedules/global | "pause every schedule" | before |
| Inbox | GET /console/api/inbox | "inbox", "what's in the inbox" | added |
| Outbox | GET /console/api/outbox | "outbox", "retry outbox #12" | before (reshaped) |
| Follow-up timers | GET /console/api/followups | "follow-ups", "what are you waiting on" | added |
| Memory | GET /console/api/memory | "memory", "what do you remember" | added |
| Published pages | GET /console/api/pages | "pages" | added |
| Settings list | GET /console/api/settings | "settings", "show all my settings" | before |
| Change a setting | POST /console/api/settings | "turn off group chatter", "set the speaking bar to 0.6" | before |
| Group chats and modes | GET /console/api/groups | "show group chats" | before (reshaped) |
| Set a group's mode | POST /console/api/groups/mode | "set the TAMID board chat to silent" | before |
| Live event stream | GET /console/events | not a text thing (pushes page updates) | n/a |
| What can I do from here | (the console menu) | "what can I do from here", "help" | added |

Counts: 38 covered before, 15 added in this change, 0 missing; 2 rows are not text
things (avatar image, live event stream).
