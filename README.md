# Lead Qualifier Agent

AI agent that qualifies incoming leads via Telegram and logs them to Google Sheets.

## What it does

Listens for a `/qualify` command in Telegram. When triggered with a message describing a lead, the agent:
- Extracts service type, budget, and lead temperature via GPT-4o-mini
- Logs structured data to Google Sheets (Timestamp, Name, Service, Budget, Temperature, Summary)
- Sends a formatted qualification report back to Telegram

Any message **without** `/qualify` is handled by a fallback branch that returns a short AI-generated summary of the text.

## Stack

- n8n (self-hosted, Docker)
- OpenAI GPT-4o-mini
- Telegram Bot API
- Google Sheets API (OAuth2)

## Flow

```
Telegram Trigger
    └── IF /qualify
           ├── TRUE  → GPT-4o-mini → Google Sheets + Telegram reply
           └── FALSE → GPT Summarize → Telegram reply
```

## What GPT extracts

| Field | Description |
|---|---|
| `service` | Type of service requested |
| `budget` | Mentioned budget or "not specified" |
| `temperature` | hot / warm / cold |
| `summary` | 1-sentence lead summary |

## Setup

1. Import `lead-qualifier.json` into your n8n instance
2. Configure credentials: OpenAI API key, Telegram Bot token, Google Sheets OAuth2
3. Set your Telegram webhook URL (ngrok for local, domain for VPS)
4. Activate the workflow

## Notes

- Prompt instructs GPT to return raw JSON only (no markdown fences)
- `.replace('/qualify ', '')` applied before sending to GPT and writing to Sheets
- Tested on n8n self-hosted via Docker + WSL2
