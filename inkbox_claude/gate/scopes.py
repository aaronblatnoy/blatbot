"""Fixed tool scopes. A scope is the only unit of capability the executor
accepts. The global deny list (executor workdir .claude/settings.json) is
applied by Claude Code on top of every scope and cannot be widened here."""

from __future__ import annotations

import os
from typing import Dict, List

# Which Google accounts the workspace tools act as. Set in the environment; the
# defaults are placeholders so no real address lives in the source.
ORG_ACCOUNT = os.getenv("GATE_ORG_GOOGLE_ACCOUNT") or "org@example.org"
OWNER_ACCOUNT = os.getenv("GATE_OWNER_GOOGLE_ACCOUNT") or "owner@example.edu"

SCOPES: Dict[str, Dict[str, object]] = {
    "calendar": {
        "description": f"read and write the {ORG_ACCOUNT} Google Calendar (list, free/busy, create/update events)",
        "tools": [
            "mcp__tamid-drive__list_calendars",
            "mcp__tamid-drive__get_events",
            "mcp__tamid-drive__query_freebusy",
            "mcp__tamid-drive__manage_event",
        ],
    },
    "inbox_read": {
        "description": "read Blatbot's own Inkbox mailbox blatbot@inkboxmail.com (list recent inbound/outbound emails, read one by id) and its SMS/iMessage threads",
        "tools": [
            "mcp__inkbox__inkbox_list_emails",
            "mcp__inkbox__inkbox_get_email",
            "mcp__inkbox__inkbox_list_text_conversations",
            "mcp__inkbox__inkbox_get_text_conversation",
            "mcp__inkbox__inkbox_list_imessage_conversations",
            "mcp__inkbox__inkbox_get_imessage_conversation",
        ],
    },
    "email_send": {
        "description": "send email from the Inkbox mailbox blatbot@inkboxmail.com and read that mailbox",
        "tools": [
            "mcp__inkbox__inkbox_send_email",
            "mcp__inkbox__inkbox_list_emails",
            "mcp__inkbox__inkbox_get_email",
        ],
    },
    "imessage_send": {
        "description": "send iMessages from Blatbot's line and read its iMessage threads",
        "tools": [
            "mcp__inkbox__inkbox_send_imessage",
            "mcp__inkbox__inkbox_list_imessage_conversations",
            "mcp__inkbox__inkbox_get_imessage_conversation",
        ],
    },
    "sms_send": {
        "description": "send SMS from Blatbot's number and read its text threads",
        "tools": [
            "mcp__inkbox__inkbox_send_sms",
            "mcp__inkbox__inkbox_list_text_conversations",
            "mcp__inkbox__inkbox_get_text_conversation",
        ],
    },
    "contacts": {
        "description": "look up, create, or update Inkbox address-book contacts and their notes",
        "tools": [
            "mcp__inkbox__inkbox_lookup_contact",
            "mcp__inkbox__inkbox_list_contacts",
            "mcp__inkbox__inkbox_get_contact",
            "mcp__inkbox__inkbox_create_contact",
            "mcp__inkbox__inkbox_update_contact",
        ],
    },
    "tamid_drive_read": {
        "description": f"read TAMID Google Drive, Sheets, Docs, Forms, and Gmail ({ORG_ACCOUNT})",
        "tools": [
            "mcp__tamid-drive__search_drive_files",
            "mcp__tamid-drive__list_drive_items",
            "mcp__tamid-drive__get_drive_file_content",
            "mcp__tamid-drive__list_spreadsheets",
            "mcp__tamid-drive__get_spreadsheet_info",
            "mcp__tamid-drive__read_sheet_values",
            "mcp__tamid-drive__list_sheet_tables",
            "mcp__tamid-drive__search_docs",
            "mcp__tamid-drive__list_docs_in_folder",
            "mcp__tamid-drive__get_doc_content",
            "mcp__tamid-drive__get_doc_as_markdown",
            "mcp__tamid-drive__get_form",
            "mcp__tamid-drive__list_form_responses",
            "mcp__tamid-drive__get_form_response",
            "mcp__tamid-drive__search_gmail_messages",
            "mcp__tamid-drive__get_gmail_message_content",
            "mcp__tamid-drive__get_gmail_thread_content",
        ],
    },
    "tamid_drive_write": {
        "description": "write to TAMID Google Sheets, Docs, Forms, and Drive folders (no sharing, no Gmail send)",
        "tools": [
            "mcp__tamid-drive__modify_sheet_values",
            "mcp__tamid-drive__append_table_rows",
            "mcp__tamid-drive__create_spreadsheet",
            "mcp__tamid-drive__create_sheet",
            "mcp__tamid-drive__create_doc",
            "mcp__tamid-drive__modify_doc_text",
            "mcp__tamid-drive__insert_doc_elements",
            "mcp__tamid-drive__create_form",
            "mcp__tamid-drive__batch_update_form",
            "mcp__tamid-drive__set_publish_settings",
            "mcp__tamid-drive__create_drive_folder",
        ],
    },
    "stern_calendar": {
        "description": f"read and write Aaron's own NYU Stern Google Calendar (tool address {OWNER_ACCOUNT}): list, free/busy, create/update events and invites",
        "tools": [
            "mcp__stern-drive__list_calendars",
            "mcp__stern-drive__get_events",
            "mcp__stern-drive__query_freebusy",
            "mcp__stern-drive__manage_event",
        ],
    },
    "stern_email_read": {
        "description": f"read Aaron's Stern Gmail (tool address {OWNER_ACCOUNT}); no sending",
        "tools": [
            "mcp__stern-drive__search_gmail_messages",
            "mcp__stern-drive__get_gmail_message_content",
            "mcp__stern-drive__get_gmail_thread_content",
        ],
    },
    "sjba_site_read": {
        "description": "read the SJBA (Stern Jewish Business Association) website admin backend: board members and bios, events, club members, semesters, site config, contact requests, newsletter signups",
        "tools": ['mcp__sjba-admin__sjba_list_board_members', 'mcp__sjba-admin__sjba_get_board_member', 'mcp__sjba-admin__sjba_list_events', 'mcp__sjba-admin__sjba_list_upcoming_events', 'mcp__sjba-admin__sjba_get_event', 'mcp__sjba-admin__sjba_list_members', 'mcp__sjba-admin__sjba_list_semesters', 'mcp__sjba-admin__sjba_list_site_config', 'mcp__sjba-admin__sjba_get_site_config', 'mcp__sjba-admin__sjba_list_contact_requests', 'mcp__sjba-admin__sjba_get_contact_request', 'mcp__sjba-admin__sjba_list_newsletter_signups', 'mcp__sjba-admin__sjba_get_newsletter_signup'],
    },
    "sjba_site_write": {
        "description": "change the SJBA (Stern Jewish Business Association) website: create/update/delete board members (incl. bios and headshots), events and flyers, members, semesters, site config; manage contact requests and newsletter signups. Includes read.",
        "tools": ['mcp__sjba-admin__sjba_list_board_members', 'mcp__sjba-admin__sjba_get_board_member', 'mcp__sjba-admin__sjba_list_events', 'mcp__sjba-admin__sjba_list_upcoming_events', 'mcp__sjba-admin__sjba_get_event', 'mcp__sjba-admin__sjba_list_members', 'mcp__sjba-admin__sjba_list_semesters', 'mcp__sjba-admin__sjba_list_site_config', 'mcp__sjba-admin__sjba_get_site_config', 'mcp__sjba-admin__sjba_list_contact_requests', 'mcp__sjba-admin__sjba_get_contact_request', 'mcp__sjba-admin__sjba_list_newsletter_signups', 'mcp__sjba-admin__sjba_get_newsletter_signup', 'mcp__sjba-admin__sjba_create_board_member', 'mcp__sjba-admin__sjba_update_board_member', 'mcp__sjba-admin__sjba_delete_board_member', 'mcp__sjba-admin__sjba_replace_board_member_headshot', 'mcp__sjba-admin__sjba_create_event', 'mcp__sjba-admin__sjba_update_event', 'mcp__sjba-admin__sjba_delete_event', 'mcp__sjba-admin__sjba_replace_event_flyer', 'mcp__sjba-admin__sjba_create_member', 'mcp__sjba-admin__sjba_update_member', 'mcp__sjba-admin__sjba_delete_member', 'mcp__sjba-admin__sjba_create_semester', 'mcp__sjba-admin__sjba_update_semester', 'mcp__sjba-admin__sjba_delete_semester', 'mcp__sjba-admin__sjba_create_site_config', 'mcp__sjba-admin__sjba_update_site_config', 'mcp__sjba-admin__sjba_delete_site_config', 'mcp__sjba-admin__sjba_update_contact_request', 'mcp__sjba-admin__sjba_delete_contact_request', 'mcp__sjba-admin__sjba_create_newsletter_signup', 'mcp__sjba-admin__sjba_update_newsletter_signup', 'mcp__sjba-admin__sjba_delete_newsletter_signup'],
    },
    "tamid_site_read": {
        "description": "read the TAMID at NYU website admin backend: board members and bios, events, club members, semesters, site config, contact requests, newsletter signups",
        "tools": ['mcp__tamid-admin__tamid_list_board_members', 'mcp__tamid-admin__tamid_get_board_member', 'mcp__tamid-admin__tamid_list_events', 'mcp__tamid-admin__tamid_list_upcoming_events', 'mcp__tamid-admin__tamid_get_event', 'mcp__tamid-admin__tamid_list_members', 'mcp__tamid-admin__tamid_list_semesters', 'mcp__tamid-admin__tamid_list_site_config', 'mcp__tamid-admin__tamid_get_site_config', 'mcp__tamid-admin__tamid_list_contact_requests', 'mcp__tamid-admin__tamid_get_contact_request', 'mcp__tamid-admin__tamid_list_newsletter_signups', 'mcp__tamid-admin__tamid_get_newsletter_signup'],
    },
    "tamid_site_write": {
        "description": "change the TAMID at NYU website: create/update/delete board members (incl. bios and headshots), events and flyers, members, semesters, site config; manage contact requests and newsletter signups. Includes read.",
        "tools": ['mcp__tamid-admin__tamid_list_board_members', 'mcp__tamid-admin__tamid_get_board_member', 'mcp__tamid-admin__tamid_list_events', 'mcp__tamid-admin__tamid_list_upcoming_events', 'mcp__tamid-admin__tamid_get_event', 'mcp__tamid-admin__tamid_list_members', 'mcp__tamid-admin__tamid_list_semesters', 'mcp__tamid-admin__tamid_list_site_config', 'mcp__tamid-admin__tamid_get_site_config', 'mcp__tamid-admin__tamid_list_contact_requests', 'mcp__tamid-admin__tamid_get_contact_request', 'mcp__tamid-admin__tamid_list_newsletter_signups', 'mcp__tamid-admin__tamid_get_newsletter_signup', 'mcp__tamid-admin__tamid_create_board_member', 'mcp__tamid-admin__tamid_update_board_member', 'mcp__tamid-admin__tamid_delete_board_member', 'mcp__tamid-admin__tamid_replace_board_member_headshot', 'mcp__tamid-admin__tamid_create_event', 'mcp__tamid-admin__tamid_update_event', 'mcp__tamid-admin__tamid_delete_event', 'mcp__tamid-admin__tamid_replace_event_flyer', 'mcp__tamid-admin__tamid_create_member', 'mcp__tamid-admin__tamid_update_member', 'mcp__tamid-admin__tamid_delete_member', 'mcp__tamid-admin__tamid_create_semester', 'mcp__tamid-admin__tamid_update_semester', 'mcp__tamid-admin__tamid_delete_semester', 'mcp__tamid-admin__tamid_create_site_config', 'mcp__tamid-admin__tamid_update_site_config', 'mcp__tamid-admin__tamid_delete_site_config', 'mcp__tamid-admin__tamid_update_contact_request', 'mcp__tamid-admin__tamid_delete_contact_request', 'mcp__tamid-admin__tamid_create_newsletter_signup', 'mcp__tamid-admin__tamid_update_newsletter_signup', 'mcp__tamid-admin__tamid_delete_newsletter_signup'],
    },
    "web": {
        "description": "search the web and open web pages: find a person, company, article, LinkedIn profile, or any public page, then read it in the headless browser",
        "tools": ["WebSearch", "mcp__playwright__browser_search", "mcp__playwright__browser_navigate", "mcp__playwright__browser_snapshot",
                  "mcp__playwright__browser_find", "mcp__playwright__browser_wait_for", "mcp__playwright__browser_navigate_back"],
    },
    "browser_read": {
        "description": "open web pages in a headless browser and read them: navigate, take a snapshot of the page, find elements, scroll, wait, screenshot. Read-only: no clicking, typing or form filling. For pages that need a real browser (JavaScript sites, portals) rather than a plain web search.",
        "tools": ["mcp__playwright__browser_search", "mcp__playwright__browser_navigate", "mcp__playwright__browser_navigate_back", "mcp__playwright__browser_snapshot",
                  "mcp__playwright__browser_find", "mcp__playwright__browser_take_screenshot", "mcp__playwright__browser_wait_for",
                  "mcp__playwright__browser_tabs", "mcp__playwright__browser_resize", "mcp__playwright__browser_console_messages",
                  "mcp__playwright__browser_network_requests", "mcp__playwright__browser_close"],
    },
    "browser_act": {
        "description": "operate a headless browser like a person: click, type, fill and submit forms, select options, press keys, hover, drag, upload files, handle dialogs, on any website. Use for tasks that must be done through a website's UI (sign-ups, portals, checkouts, admin pages with no API). Includes browser_read.",
        "tools": ["mcp__playwright__browser_search", "mcp__playwright__browser_navigate", "mcp__playwright__browser_navigate_back", "mcp__playwright__browser_snapshot",
                  "mcp__playwright__browser_find", "mcp__playwright__browser_take_screenshot", "mcp__playwright__browser_wait_for",
                  "mcp__playwright__browser_tabs", "mcp__playwright__browser_resize", "mcp__playwright__browser_console_messages",
                  "mcp__playwright__browser_network_requests", "mcp__playwright__browser_close",
                  "mcp__playwright__browser_click", "mcp__playwright__browser_type", "mcp__playwright__browser_fill_form",
                  "mcp__playwright__browser_select_option", "mcp__playwright__browser_press_key", "mcp__playwright__browser_hover",
                  "mcp__playwright__browser_drag", "mcp__playwright__browser_drop", "mcp__playwright__browser_file_upload",
                  "mcp__playwright__browser_handle_dialog", "mcp__playwright__browser_emulate_media"],
    },
}


def tools_for(scopes: List[str]) -> List[str]:
    out: List[str] = []
    for s in scopes:
        for t in SCOPES[s]["tools"]:  # type: ignore[union-attr]
            if t not in out:
                out.append(t)
    return out


_SEND_TOOLS = ("inkbox_send_imessage", "inkbox_send_sms", "inkbox_send_email")


def _norm(v: str) -> str:
    v = (v or "").strip().lower()
    digits = "".join(ch for ch in v if ch.isdigit())
    return digits[-10:] if digits and "@" not in v and len(digits) >= 7 else v


def sends_to_requester(tool_name: str, args: dict, protected: list) -> bool:
    """True when a send tool is aimed at the person who asked (their address, phone
    or conversation). The gateway delivers results itself; an executor sending the
    answer as well means the requester gets it twice."""
    short = tool_name.split("__")[-1]
    if short not in _SEND_TOOLS:
        return False
    keys = {_norm(p) for p in protected if p}
    targets = []
    to = args.get("to")
    targets += to if isinstance(to, list) else ([to] if to else [])
    if args.get("conversation_id"):
        targets.append(args["conversation_id"])
    return any(_norm(str(t)) in keys for t in targets)


# What each tool is FOR, in the words a task uses. Shown to the judgment that picks
# the next tool instead of the servers' own (often long, implementation-flavoured)
# descriptions. Tools not listed fall back to the server's description.
TOOL_PURPOSE: Dict[str, str] = {
    # web
    "mcp__playwright__browser_search": "SEARCH THE WEB: type a query, get result titles, links and snippets. First step for any 'look up', 'find online', LinkedIn, company, person, or news question.",
    "mcp__playwright__browser_navigate": "OPEN A WEB PAGE by URL in the browser (a search result, a profile, an article). Follow with browser_snapshot or browser_find to read it.",
    "mcp__playwright__browser_snapshot": "READ THE CURRENT PAGE: the full text and links of the page that is open.",
    "mcp__playwright__browser_find": "FIND TEXT ON THE CURRENT PAGE: locate a name, number or phrase on the open page and read around it.",
    "mcp__playwright__browser_wait_for": "WAIT for the page to finish loading or for text to appear (only after a page opened blank).",
    "mcp__playwright__browser_navigate_back": "GO BACK to the previous page.",
    "mcp__playwright__browser_click": "CLICK a button or link on the open page.",
    "mcp__playwright__browser_type": "TYPE into a field on the open page.",
    "mcp__playwright__browser_fill_form": "FILL a form's fields on the open page.",
    # Google Drive / Sheets / Docs / Forms (TAMID account)
    "mcp__tamid-drive__search_drive_files": "FIND A FILE in TAMID Drive by name or words: sheets, docs, forms, folders. Returns names and ids. Use before opening anything whose id is unknown.",
    "mcp__tamid-drive__list_drive_items": "LIST the files inside one Drive folder by folder id.",
    "mcp__tamid-drive__list_spreadsheets": "LIST recent spreadsheets in TAMID Drive with their ids.",
    "mcp__tamid-drive__get_spreadsheet_info": "SHEET STRUCTURE: the tabs of a spreadsheet, their names and row/column counts, by spreadsheet id. Answers 'how many rows'.",
    "mcp__tamid-drive__read_sheet_values": "READ A SHEET'S CELLS: the rows of a spreadsheet tab (a roster, responses, a tracker) by spreadsheet id and range.",
    "mcp__tamid-drive__get_drive_file_content": "READ A WHOLE FILE'S text by file id (a doc, a sheet export). Large; prefer read_sheet_values for sheets.",
    "mcp__tamid-drive__get_doc_content": "READ A GOOGLE DOC's text by document id.",
    "mcp__tamid-drive__search_docs": "FIND A GOOGLE DOC by words in its name.",
    "mcp__tamid-drive__get_form": "FORM DEFINITION: a Google Form's title, questions and settings, by form id.",
    "mcp__tamid-drive__list_form_responses": "WHO ANSWERED A FORM: every response to a Google Form (respondent email, answers, time), by form id. Answers 'who filled it out', 'who has not responded'.",
    "mcp__tamid-drive__get_form_response": "ONE FORM RESPONSE in full, by form id and response id.",
    "mcp__tamid-drive__get_events": "TAMID CALENDAR EVENTS: list or search events on the TAMID calendar (a query word, a date range, or an event id). Returns titles, times, attendees, ids.",
    "mcp__tamid-drive__list_calendars": "WHICH CALENDARS the TAMID account has and their ids (needed before get_events on a non-primary calendar).",
    "mcp__tamid-drive__manage_event": "CREATE, UPDATE or DELETE an event on the TAMID calendar.",
    "mcp__tamid-drive__search_gmail_messages": "SEARCH THE TAMID INBOX: find emails by sender, words, or date (Gmail search syntax). Returns message ids and headers. Use to find someone's email address or what they sent.",
    "mcp__tamid-drive__get_gmail_message_content": "READ ONE EMAIL in the TAMID inbox in full, by message id.",
    "mcp__tamid-drive__get_gmail_thread_content": "READ A WHOLE EMAIL THREAD in the TAMID inbox, by thread id.",
    "mcp__tamid-drive__send_gmail_message": "SEND AN EMAIL from the TAMID account to someone else.",
    # Stern account
    "mcp__stern-drive__get_events": "AARON'S STERN CALENDAR: list or search his events (a query word, a date range). Returns titles, times, ids.",
    "mcp__stern-drive__list_calendars": "WHICH CALENDARS Aaron's Stern account has and their ids.",
    "mcp__stern-drive__manage_event": "CREATE, UPDATE or DELETE an event on Aaron's Stern calendar.",
    "mcp__stern-drive__search_gmail_messages": "SEARCH AARON'S STERN INBOX: find emails by sender, words, or date. Returns message ids and headers.",
    "mcp__stern-drive__get_gmail_message_content": "READ ONE EMAIL in Aaron's Stern inbox in full, by message id.",
    "mcp__stern-drive__send_gmail_message": "SEND AN EMAIL from Aaron's Stern account to someone else.",
    # Inkbox (Blatbot's own mailbox, phone, contacts)
    "mcp__inkbox__inkbox_get_contact": "LOOK UP A CONTACT in Blatbot's address book by name, email or phone: returns their email, phone and notes. First stop for 'what is X's email / number'.",
    "mcp__inkbox__inkbox_list_contacts": "LIST Blatbot's contacts (names, emails, phones).",
    "mcp__inkbox__inkbox_list_emails": "LIST recent emails in Blatbot's own mailbox.",
    "mcp__inkbox__inkbox_send_email": "SEND AN EMAIL from Blatbot's mailbox to someone else.",
    "mcp__inkbox__inkbox_send_sms": "SEND A TEXT (SMS) from Blatbot's number to someone else.",
    "mcp__inkbox__inkbox_send_imessage": "SEND AN IMESSAGE from Blatbot's line to someone else.",
    # site admin
    "mcp__tamid-admin__tamid_list_board_members": "TAMID BOARD ROSTER from the website: every board member with name, title, email, bio.",
    "mcp__tamid-admin__tamid_list_members": "TAMID MEMBER ROSTER from the website: club members with names and emails.",
    "mcp__tamid-admin__tamid_list_events": "TAMID EVENTS listed on the website.",
    "mcp__tamid-admin__tamid_list_site_config": "TAMID WEBSITE SETTINGS: application open/closed, deadlines, labels.",
    "mcp__sjba-admin__sjba_list_board_members": "SJBA BOARD ROSTER from the website: every board member with name, title, email, bio.",
    "mcp__sjba-admin__sjba_list_events": "SJBA EVENTS listed on the website.",
    "mcp__sjba-admin__sjba_list_upcoming_events": "SJBA UPCOMING EVENTS on the website.",
}
