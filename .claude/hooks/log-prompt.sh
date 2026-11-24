#!/usr/bin/env bash
# UserPromptSubmit hook: appende il messaggio dell'utente, così come ricevuto,
# al log giornaliero indicato da `conversationLog` in .bemyagent/settings.json.
# Non blocca mai il prompt: ogni errore esce con 0.
set -u
root="${CLAUDE_PROJECT_DIR:-$(pwd)}"
input="$(cat)"
dir="$(jq -r '.conversationLog // ""' "$root/.bemyagent/settings.json" 2>/dev/null)" || exit 0
[ -n "$dir" ] || exit 0
case "$dir" in /*) ;; *) dir="$root/$dir" ;; esac
prompt="$(printf '%s' "$input" | jq -r '.prompt // empty' 2>/dev/null)" || exit 0
[ -n "$prompt" ] || exit 0
session="$(printf '%s' "$input" | jq -r '.session_id // "?"' 2>/dev/null | cut -c1-8)"
mkdir -p "$dir" 2>/dev/null || exit 0
{
  printf '## %s · sessione %s\n\n' "$(date +%H:%M)" "$session"
  printf '%s\n\n' "$prompt"
} >> "$dir/$(date +%F).md" 2>/dev/null
exit 0
