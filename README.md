# GBrain SMS

A Python chatbot that runs on your Linux PC and talks to an Android phone over
the **local Wi-Fi network** — no USB cable and no ADB required. The phone runs
the GBrain AudioLink companion app, discovers the PC over mDNS, pairs once with a
6-digit code, and keeps an authenticated WebSocket open for SMS.

The bot reads incoming SMS on the phone, generates human-like replies with Ollama
or a rule-based fallback, remembers past conversations locally, and supports
scheduled responses and do-not-disturb windows.

## Features

- **Wi-Fi SMS read/send** — the phone sends `sms.inbound` and executes `sms.send`
  through `SmsManager`; the PC never touches ADB for normal SMS
- **No USB required** — automatic mDNS discovery on the same LAN, TLS-pinned
  pairing, and an authenticated WSS signaling channel
- **Human-like replies** — Ollama local LLM with casual tone; rule-based fallback
  when offline
- **Local memory** — SQLite storage with keyword retrieval and preference learning
- **Scheduled replies** — delay responses to a specific time; DND window support
- **Stability** — auto-reconnect on disconnect, deduplication, rotating logs
- **Optional legacy ADB** — calls, WhatsApp notification replies, and the old
  `adb` SMS transport still work if you plug in a USB cable

## How the Wi-Fi bridge works

```
┌─────────────┐    mDNS _gbrain-alink._tcp.local.    ┌──────────────┐
│  Linux PC   │ ◄──────────────────────────────────► │ Android phone│
│  bot.main   │   HTTPS :8787  pairing + identity    │ AudioLink app│
│  run        │   WSS   :8787  /v1/signaling         │              │
└─────────────┘                                      └──────────────┘
        ▲                                                    │
        │  sms.inbound  (phone → PC)                         │
        └──────────────  sms.send  (PC → phone) ─────────────┘
```

1. `python -m bot.main run` starts one HTTPS server on `0.0.0.0:8787` with a
   self-signed certificate and advertises it over mDNS.
2. The Android app discovers the PC on the same Wi-Fi network and pins the
   advertised certificate fingerprint.
3. The PC shows a 6-digit pairing code in the log; enter it in the app.
4. The app opens an authenticated WebSocket. Incoming SMS arrives as
   `sms.inbound`; generated replies go back as `sms.send` and are sent with the
   phone's own `SmsManager`.
5. Scheduled messages use the same socket.

## Project layout

```
gbrain-sms/
├── bot/
│   ├── main.py              # CLI + event loop + Wi-Fi bridge bootstrap
│   ├── api.py               # HTTPS API mounted with the bridge
│   ├── bridge/              # mDNS, pairing, TLS, signaling, SMS tasks
│   ├── adb_connector.py     # legacy USB/ADB path
│   ├── chat_engine.py       # Ollama + rule fallback
│   ├── memory_store.py      # SQLite persistence
│   ├── scheduler.py         # Async delayed send queue
│   └── config.py            # Config loader
├── gbrain_cli/              # Website relay agent for the SMS dashboard
├── config.example.yaml
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 1. Linux / Ubuntu setup

### Install system packages

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip
```

`android-tools-adb` is only needed if you plan to use the optional legacy USB
path (calls, WhatsApp notifications, or `sms_transport: adb`).

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

## 2. Android phone setup (Wi-Fi only)

### Install the GBrain AudioLink companion app

Install the app from the sibling `audiolink/android/` project (build the debug
APK or install the release APK). The app is the only phone-side component you
need.

### Permissions

Open the app and grant:

1. **SMS** permissions (`READ_SMS`, `RECEIVE_SMS`, `SEND_SMS`)
2. **Notifications** (so you can see connection state)
3. **Local network / mDNS** access if your Android version asks for it

Disable battery optimization for the app so the WebSocket stays alive in the
background.

### Pair over Wi-Fi

1. Make sure the PC and phone are on the **same Wi-Fi/LAN**.
2. Start the bot on the PC:

   ```bash
   source .venv/bin/activate
   python -m bot.main run
   ```

3. In the Android app, tap **Search** (or **Discover**). The PC appears by name.
4. A 6-digit code is printed in the PC log:

   ```
   pairing request from <phone> (<device_id>): show code 123456 to the user
   ```

5. Type that code into the app and confirm. The app stores a bearer token and
   pins the TLS certificate.
6. Once connected, the app reports online in the PC log and `status` output.

No USB cable, udev rules, or `adb devices` step is involved.

### Verify from the PC

```bash
python -m bot.main status
```

You should see `SMS transport: bridge`, `Wi-Fi bridge: enabled`, and
`Paired phones: 1`.

### Test sending over Wi-Fi

With `python -m bot.main run` still active in another terminal:

```bash
python -m bot.main send --to +255772691677 --msg "Test over Wi-Fi"
```

The CLI posts to the local bridge API, the PC pushes `sms.send` to the paired
phone, and the phone sends the SMS with its own SIM. If no phone is paired the
command tells you to pair first.

### Remote website dashboard

Install the bundled CLI so this PC can connect to your GBrain website account:

```bash
pip install -e .
gbrain-cli sign-in --server https://gbrain.tech
gbrain-cli daemon
```

`sign-in` prompts for the password without echoing it. For non-interactive setup,
set `GBRAIN_CLI_PASSWORD` in the environment instead; do not pass passwords as
command arguments because process lists expose them. `sign-in` registers this PC
against your website account. Run it on every PC you want to manage; the
dashboard supports multiple PCs per user. The daemon makes outbound-only
requests, so no router port forwarding is needed. It reads the local AudioLink
admin token and certificate and talks only to `https://127.0.0.1:8787` by default.

On the website, sign in and open `/dashboard/sms`. Each connected PC appears as
a separate card. When a pairing request is pending on the PC, the daemon forwards
the **existing 6-digit AudioLink phone pairing code** to the dashboard; enter
that code in the AudioLink phone app to pair the phone. The dashboard never sees
the local admin token.

The daemon also caches the phone's SMS inbox and sent folder on the website so
the browser can show recent messages. To use a different relay during testing:

```bash
gbrain-cli sign-in --server http://localhost:3000
gbrain-cli daemon
```

### Hosted service (no local PC)

A confirmed website account can register a phone from the dashboard QR code.
The QR contains a short-lived, single-use enrollment token; it never contains a
password or session cookie. The phone posts that token to the hosted AudioLink
bridge, which verifies it against the website with `GBRAIN_SERVICE_TOKEN` and
pairs the phone directly to the server-side bridge.

Run the hosted bridge (existing `python -m bot.main run` server) with:

```bash
export GBRAIN_WEBSITE_URL="https://gbrain.tech"
export GBRAIN_SERVICE_TOKEN="same-token-as-the-website"
```

Run the server-side CLI relay with:

```bash
export GBRAIN_WEBSITE_URL="https://gbrain.tech"
export GBRAIN_AUDIOLINK_BRIDGE_URL="https://sms.gbrain.tech"
export GBRAIN_SERVICE_TOKEN="same-token-as-the-website"
gbrain-cli hosted-daemon
```

The daemon sends a `hosted-cli` heartbeat to `/api/internal/services/heartbeat`
every 30 seconds. The company admin console shows its status, device count, and
bridge health. The website must set `GBRAIN_SERVICE_TOKEN` to the same value.

The relay fetches only hosted devices owned by confirmed users, polls their
dashboard commands, and dispatches SMS through the hosted WSS bridge. The
website also requires `GBRAIN_CONFIRMED_EMAILS` (or `GBRAIN_ADMIN_EMAILS`) to
mark accounts as confirmed.

---

## 3. Configuration

Edit `config.yaml` (copied from `config.example.yaml`). The Wi-Fi-relevant keys:

| Key | Description |
| --- | --- |
| `sms_transport` | `bridge` for Wi-Fi (default); `adb` for legacy USB |
| `gbrain_api_port` | HTTPS/WSS port for pairing, signaling, and SMS (default `8787`) |
| `bridge_enabled` | Enable the AudioLink bridge (default `true`) |
| `bridge_mdns_enabled` | Advertise and browse with mDNS (default `true`) |
| `bridge_device_name` | Optional display name for the PC on the network |
| `bridge_auto_reply_enabled` | Let the bridge generate replies for inbound SMS |
| `bridge_pairing_code_ttl_sec` | How long the 6-digit pairing code stays valid |
| `poll_interval_sec` | Poll cadence for the legacy ADB path |
| `auto_reply_enabled` | Enable/disable automatic replies |
| `ollama_model` | Model name for Ollama |
| `dnd_start` / `dnd_end` | Do-not-disturb window (e.g. `22:00` / `07:00`) |
| `reply_cooldown_sec` | Minimum seconds between replies to the same contact |

Environment overrides work the same way: `BOT_AUTO_REPLY_ENABLED=false`,
`BOT_POLL_INTERVAL_SEC=5`, etc.

### Optional legacy ADB keys

Only used when `sms_transport: adb`, or when `whatsapp_enabled` /
`call_answer_enabled` are on:

| Key | Description |
| --- | --- |
| `adb_path` | Path to the `adb` binary (default `adb`) |
| `sms_uri_authority` | `adbsms` or `sms` content provider authority |
| `whatsapp_enabled` | Poll the notification shade over ADB |
| `call_answer_enabled` | Answer calls over ADB keyevents |
| `sms_send_strategies` | ADB send methods: `respond`, `compose`, `intent`, `isms` |

---

## 4. Example usage

Activate the virtual environment first:

```bash
source .venv/bin/activate
```

### Start the bot and Wi-Fi bridge

```bash
python -m bot.main run
```

The bot will:

1. Start the mDNS advertisement and HTTPS/WSS server on port `8787`
2. Wait for a paired phone to connect over Wi-Fi
3. Generate and send replies as `sms.inbound` messages arrive
4. Queue replies when a time hint is detected

### Check status

```bash
python -m bot.main status
```

### Show the bridge identity and fingerprint

```bash
python -m bot.main bridge-identity
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

When someone texts "reply at 7pm" or "in 30 minutes", the bot parses the hint
and queues the response accordingly.

Example conversation:

```
Them: I'm tired
You:  Long day?

Them: Yes school was hard
You:  What part was the hardest?
```

---

## 5. Troubleshooting (Wi-Fi)

| Problem | Solution |
| --- | --- |
| Phone cannot find the PC | Confirm both are on the same Wi-Fi/LAN and not on a guest or client-isolated network; some routers block mDNS |
| `mDNS advertising unavailable` in the log | Install/allow mDNS on the PC; check `avahi-daemon` and that UDP `5353` is not blocked |
| Phone finds the PC but pairing fails | Re-read the 6-digit code from the PC log; codes expire after `bridge_pairing_code_ttl_sec` |
| `certificate fingerprint mismatch` | The PC regenerated its identity or you paired with a different PC; unpair in the app and pair again |
| Connection drops when the phone sleeps | Disable battery optimization for the AudioLink app |
| `No Wi-Fi paired phone reachable` from `send` | Start `python -m bot.main run` and pair the app first; the CLI sends through the running bridge |
| `connection refused` on port `8787` | Allow TCP `8787` through the PC firewall and make sure the bot is running |
| SMS auto-reply not firing | Check the app has SMS permissions and the phone is connected in `python -m bot.main status` |
| Replies look wrong | Try `python -m bot.main try-reply --msg "..."` locally; switch `reply_backend` or install Ollama |

Legacy ADB problems:

| Problem | Solution |
| --- | --- |
| `adb: command not found` | Install `android-tools-adb` (legacy path only) |
| Device shows `unauthorized` | Replug USB and accept the debugging prompt |
| Empty SMS inbox query | Install the `adbsms` companion app and grant `READ_SMS` |
| SMS send fails over ADB | Prefer Wi-Fi `bridge`; ADB send varies by OEM and Android version |

Logs are written to `logs/bot.log` (rotating, 2 MB × 3 files).

---

## 6. Real-time voice calls (optional, legacy ADB)

This feature still uses ADB for call control. A non-rooted Android phone does not
expose cellular call audio over ADB, and Linux cannot present itself as a
Bluetooth HFP headset. The supported bridge is acoustic:

```
caller ── cellular ──> phone on speakerphone ── room ──> PC microphone
caller <── cellular ── phone microphone <── room ── PC speakers (TTS)
```

Put the phone on speakerphone next to the computer, unmute the PC microphone
and speakers, and keep the phone roughly 20–40 cm from the microphone.

### Install the speech runtime

The system Python is PEP 668 managed, so Whisper is installed inside the
project (the voice engine adds this directory to `sys.path`):

```bash
python3 -m pip install --target tools/voice/pylibs --upgrade faster-whisper
sudo apt install pipewire-utils alsa-utils   # pw-record/pw-play or arecord/aplay
```

The first transcription downloads the Whisper model into `data/models/`
(`medium` ≈ 1.5 GB, `small` ≈ 460 MB; `tiny`/`base` are faster and less accurate).

Natural TTS (Piper) instead of espeak-ng:

```bash
python3 -m pip install --target tools/voice/pylibs --upgrade piper-tts
mkdir -p tools/voice/voices/sw_CD
# voices: https://huggingface.co/rhasspy/piper-voices  (Swahili: sw/sw_CD)
# fetch sw_CD-lanfrica-medium.onnx and sw_CD-lanfrica-medium.onnx.json into that folder
```

Set `voice_tts_provider: piper` and `voice_piper_model: tools/voice/voices/sw_CD/sw_CD-lanfrica-medium.onnx`.

### Use

```bash
python -m bot.main voice-check                 # devices, STT, TTS, call state
python -m bot.main voice-selftest --seconds 6  # mic -> Whisper -> LLM -> speakers
python -m bot.main voice-run                   # live call, Ctrl+C to stop
python -m bot.main voice-run --hangup-after 90
```

With `call_answer_enabled: true`, an answered call starts the voice engine
automatically. Without auto-answer, run `voice-run` while a call is active.

### Limitations of the bridge

- **Half-duplex while replying** — capture is dropped during playback so the
  bot does not transcribe itself.
- Echo and room noise degrade recognition; a headset-style microphone aimed at
  the phone works better than a laptop's built-in array.
- Swahili accuracy (measured on five natural-voice sentences): word error rate
  0.35 with `medium` vs 0.53 with `small`.
- Audio quality on the far end is limited by speakerphone + microphone pickup.
- The PC must keep running for the bridge to work.

---

## 7. Known limitations

- **mDNS required** — the phone and PC must be on the same LAN and the network
  must allow multicast DNS for automatic discovery
- **Battery optimization** can suspend the phone app; disable it for reliable
  background SMS delivery
- **WhatsApp and call answering** still use the optional legacy ADB path
- **Learning** is local retrieval + preferences, not model fine-tuning
- **Fully offline** works with rule-based mode; Ollama requires a local install
  but no cloud API

---

## 8. Dependencies

- Python 3.10+
- `pyyaml` — config file parsing
- `httpx` — Ollama HTTP client
- `fastapi`, `uvicorn` — HTTPS/WSS bridge and API
- `zeroconf` — mDNS discovery
- `cryptography` — TLS identity and certificate pinning
- `aiortc`, `av` — optional AudioLink media features
- Voice engine (optional): `faster-whisper` and `piper-tts` in
  `tools/voice/pylibs`, `espeak-ng` (bundled in `tools/tts`) as the offline
  fallback, PipeWire or ALSA utilities

---

## License

MIT
