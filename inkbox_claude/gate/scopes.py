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
        "description": "search the web",
        "tools": ["WebSearch"],
    },
}


def tools_for(scopes: List[str]) -> List[str]:
    out: List[str] = []
    for s in scopes:
        for t in SCOPES[s]["tools"]:  # type: ignore[union-attr]
            if t not in out:
                out.append(t)
    return out
