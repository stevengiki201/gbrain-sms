# USB SMS Chatbot

A Python chatbot that runs on your Linux PC and communicates with an Android phone over USB via ADB. It reads incoming SMS (via the [adbsms](https://github.com/gonodono/adbsms) companion app), generates human-like replies using Ollama or a rule-based fallback, remembers past conversations locally, and supports scheduled responses.

## Features

- **ADB SMS read/send** — polls inbox via `content://adbsms`, sends via intent or `service call isms`
- **Human-like replies** — Ollama local LLM with casual tone; rule-based fallback when offline
- **Local memory** — SQLite storage with keyword retrieval and preference learning
- **Scheduled replies** — delay responses to a specific time; DND window support
- **Stability** — auto-reconnect on disconnect, deduplication, rotating logs

## Project layout

```
gbrain-sms/
├── bot/
│   ├── main.py              # CLI + event loop
│   ├── adb_connector.py     # ADB device, SMS, WhatsApp notifs
│   ├── chat_engine.py       # Ollama + rule fallback
│   ├── memory_store.py      # SQLite persistence
│   ├── scheduler.py         # Async delayed send queue
│   └── config.py            # Config loader
├── config.example.yaml
├── requirements.txt
└── README.md
```

---

## 1. Linux / Ubuntu setup

### Install system packages

```bash
sudo apt update
sudo apt install android-tools-adb python3 python3-venv python3-pip
```

### USB udev rules (optional but recommended)

If `adb devices` shows `no permissions`, add a udev rule:

```bash
# Find your phone vendor ID
lsusb

# Create rule (replace VENDOR_ID)
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="VENDOR_ID", MODE="0666", GROUP="plugdev"' | \
  sudo tee /etc/udev/rules.d/51-android.rules

sudo usermod -aG plugdev $USER
sudo udevadm control --reload-rules
# Replug USB cable after logging out/in
```

### Python environment

```bash
cd gbrain-sms
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config.example.yaml config.yaml
```

### Optional: Ollama (local LLM)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b
ollama serve   # usually runs as a systemd service after install
```

If Ollama is not running, the bot automatically falls back to rule-based replies.

---

## 2. Android phone setup (ADB + adbsms)

### Enable USB debugging

1. Open **Settings → About phone** → tap **Build number** 7 times
2. Go to **Developer options** → enable **USB debugging**
3. Connect phone via USB → accept the RSA fingerprint prompt

### Verify ADB connection

```bash
adb devices
```

Expected output:

```
List of devices attached
XXXXXXXX    device
```

If you see `unauthorized`, check the phone screen for the debugging prompt.

### Install adbsms companion app

Modern Android blocks direct `content://sms` access from the shell. Install [gonodono/adbsms](https://github.com/gonodono/adbsms):

1. Download the APK from the [releases page](https://github.com/gonodono/adbsms/releases)
2. Install: `adb install adbsms.apk`
3. Open the app on your phone and grant **READ_SMS** permission

### Verify SMS read access

```bash
adb shell content query --uri content://adbsms/inbox \
  --projection address:body --sort "date DESC"
```

You should see rows like `Row: address=+123..., body=Hello`.

### Test SMS send

**Recommended — use the bot CLI** (verifies delivery in the sent folder):

```bash
python -m bot.main send --to +255772691677 --msg "Test from ADB"
```

The bot tries `intent` → `compose` → `isms` and only reports success when the message appears in **adbsms sent**. The `isms` method often logs success without actually delivering on modern Android.

**Manual ADB tests** — note that `adb shell` splits on spaces, so multi-word bodies break unless quoted:

```bash
# Single-word body (works without extra quoting)
adb shell am start -a android.intent.action.SENDTO -d sms:+255772691677 \
  --es sms_body Test --ez exit_on_sent true
adb shell input keyevent 66

# Multi-word body — wrap entire command in single quotes on the host
adb shell 'am start -a android.intent.action.SENDTO -d sms:+255772691677 --es sms_body "Test from ADB" --ez exit_on_sent true'
adb shell input keyevent 66

# Headless send via isms (no UI; may be blocked on some devices)
adb shell service call isms 5 i32 0 s16 "com.android.mms.service" s16 "null" \
  s16 "+255772691677" s16 "null" s16 "Test from ADB" s16 "null" s16 "null" i32 0 i64 0
```

The SMS app may flash briefly with intent/compose methods — this is normal.

### WhatsApp auto-reply (optional)

WhatsApp replies read incoming messages from the **notification shade** and reply without you typing. Enable in `config.yaml`:

```yaml
whatsapp_enabled: true
```

**Phone requirements:**

1. WhatsApp installed and logged in
2. **Notifications enabled** for WhatsApp (Settings → Apps → WhatsApp → Notifications)
3. Incoming messages must show a notification (not silenced / DND)
4. Phone screen can be off, but direct reply works best while the notification is still visible

**How it works:**

1. Bot polls `adb shell dumpsys notification --noredact`
2. Parses new WhatsApp message text and sender name
3. Generates a reply (Ollama or rules)
4. Sends via **DirectReply** (silent, no app open) or opens the chat and types if that fails

**Limitations:**

- Only works while the WhatsApp notification is active (usually a few minutes)
- Group chats reply to the group thread (contact name = group title)
- If DirectReply is blocked by your Android version, the UI fallback briefly opens WhatsApp
- Restart the bot after changing `whatsapp_enabled`

---

## 3. Configuration

Edit `config.yaml` (copied from `config.example.yaml`):


| Key                     | Description                                                                                       |
| ----------------------- | ------------------------------------------------------------------------------------------------- |
| `poll_interval_sec`     | How often to check for new SMS (default: 3)                                                       |
| `auto_reply_enabled`    | Enable/disable automatic replies                                                                  |
| `ollama_model`          | Model name for Ollama                                                                             |
| `dnd_start` / `dnd_end` | Do-not-disturb window (e.g. `22:00` / `07:00`)                                                    |
| `whatsapp_enabled`      | Poll notification shade and auto-reply to WhatsApp messages                                       |
| `reply_cooldown_sec`    | Minimum seconds between replies to same contact                                                   |
| `sms_send_strategies`   | Send methods in order. Default `[isms]` = silent/headless. Add `intent`/`compose` as UI fallbacks |


Environment overrides: `BOT_AUTO_REPLY_ENABLED=false`, `BOT_POLL_INTERVAL_SEC=5`, `BOT_SMS_SEND_STRATEGIES=isms,intent,compose`, etc.

---

## 4. Example usage

Activate the virtual environment first:

```bash
source .venv/bin/activate
```

### Start the bot (main loop)

```bash
python -m bot.main run
```

The bot will:

1. Wait for an ADB-connected device
2. Poll SMS inbox every few seconds
3. Generate and send replies (or queue them if a time hint is detected)

### Check status

```bash
python -m bot.main status
```

### Send an SMS manually

```bash
python -m bot.main send --to +1234567890 --msg "Hey, running late!"
```

### Schedule a message

```bash
python -m bot.main schedule --to +1234567890 --at 19:00 --msg "Don't forget dinner!"
python -m bot.main schedule --to +1234567890 --at "2026-06-09 08:30" --msg "Good morning!"
```

### View conversation history

```bash
python -m bot.main history --contact +1234567890 --limit 20
```

### Toggle auto-reply

```bash
python -m bot.main toggle-auto-reply
```

### Cancel a scheduled reply

```bash
python -m bot.main cancel --id abc12345
```

Task ids are shown by `status` and `schedule` output.

### Export messages to JSON

```bash
python -m bot.main export --out data/backup.json
```

### Natural scheduling in messages

When someone texts "reply at 7pm" or "in 30 minutes", the bot parses the hint and queues the response accordingly.

Example conversation:

```
Them: I'm tired
You:  Long day?

Them: Yes school was hard
You:  What part was the hardest?
```

---

## 5. Troubleshooting


| Problem                                | Solution                                                                                                               |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `adb: command not found`               | Install `android-tools-adb`                                                                                            |
| Device shows `unauthorized`            | Unplug/replug USB; accept prompt on phone                                                                              |
| Device keeps disconnecting             | Use a data-capable USB cable; set USB mode to **File transfer / MTP**; run `adb kill-server && adb start-server`       |
| Empty SMS inbox query                  | Install adbsms; grant READ_SMS permission                                                                              |
| `Permission Denial` on `content://sms` | Use adbsms URI (`content://adbsms/inbox`) — already configured                                                         |
| `pkg=from` or `pkg=BODY` in send error | Shell split your message on spaces; use `python -m bot.main send` or wrap the full `am start` command in single quotes |
| SMS send fails                         | Bot tries isms → intent → compose automatically; check default SMS app permissions                                     |
| Ollama not used                        | Run `ollama serve`; check `ollama list`; bot falls back to rules                                                       |
| Duplicate replies                      | Bot deduplicates by SMS `_id`; check `processed_sms` table                                                             |
| Encoding garbled text                  | Messages are normalized to NFC; check phone locale                                                                     |


Logs are written to `logs/bot.log` (rotating, 2 MB × 3 files).

---

## 6. Known limitations

- **adbsms required** on modern Android for reliable SMS read
- **SMS send via ADB** varies by OEM/Android version; intent method may briefly show the SMS UI
- **WhatsApp** via notifications is experimental and fragile (needs active notification)
- **Learning** is local retrieval + preferences, not model fine-tuning
- **Fully offline** works with rule-based mode; Ollama requires local install but no cloud API

---

## 7. Dependencies

- Python 3.10+
- `pyyaml` — config file parsing
- `httpx` — Ollama HTTP client
- Everything else uses the Python standard library

---

## License

MIT