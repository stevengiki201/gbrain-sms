from __future__ import annotations

import pytest

from bot.bridge.errors import ValidationFailed
from bot.bridge.operations import AudioLinkOperations
from bot.bridge.systemd import render_unit


async def test_all_documented_operations_exist(service):
    ops = AudioLinkOperations(service)
    expected = {
        "audiolink.get_status",
        "audiolink.start_service",
        "audiolink.stop_service",
        "audiolink.discover_devices",
        "audiolink.get_paired_devices",
        "audiolink.pair_device",
        "audiolink.connect",
        "audiolink.disconnect",
        "audiolink.start_audio_session",
        "audiolink.end_audio_session",
        "audiolink.set_microphone_enabled",
        "audiolink.set_speaker_enabled",
        "audiolink.set_volume",
        "audiolink.set_audio_routing",
        "audiolink.get_audio_devices",
        "audiolink.get_diagnostics",
        "audiolink.send_signal",
        # SMS integration
        "audiolink.sms_list_tasks",
        "audiolink.sms_cancel_task",
        "audiolink.sms_send_now",
        "audiolink.sms_set_auto_reply",
        "audiolink.sms_schedule",
        "audiolink.sms_instruct",
    }
    assert set(ops.names) == expected
    assert {op["operation"] for op in ops.describe()} == expected


async def test_unknown_operation_and_missing_params(service):
    ops = AudioLinkOperations(service)
    with pytest.raises(ValidationFailed):
        await ops.execute("audiolink.nope")
    with pytest.raises(ValidationFailed) as exc:
        await ops.execute("audiolink.connect", {})
    assert "device_id" in exc.value.details["missing"]


async def test_pair_device_operation_returns_verification_code(service):
    ops = AudioLinkOperations(service)
    result = await ops.execute("audiolink.pair_device", {"device_id": "phone-9"})
    assert len(result["code"]) == 6
    assert result["device_id"] == "phone-9"


def test_systemd_unit_quotes_paths_with_spaces():
    unit = render_unit(
        python_executable="/home/hp/My Env/bin/python",
        root="/home/hp/project z/audiolink",
    )
    assert 'ExecStart="/home/hp/My Env/bin/python" -m bot.main run' in unit
    assert 'WorkingDirectory="/home/hp/project z/audiolink"' in unit


def test_systemd_unit_is_a_user_service():
    unit = render_unit(python_executable="/opt/venv/bin/python", root="/srv/audiolink")
    assert "ExecStart=/opt/venv/bin/python -m bot.main run" in unit
    assert "WorkingDirectory=/srv/audiolink" in unit
    assert "WantedBy=default.target" in unit
    assert "Restart=on-failure" in unit
    assert "network-online.target" in unit
