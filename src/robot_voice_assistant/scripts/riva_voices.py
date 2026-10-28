"""List the NVIDIA Riva voices in NVIDIA's cloud, or save a sample of one to listen to.

Needs the NVIDIA_API_KEY environment variable (create a key on build.nvidia.com). Run with the workspace
environment:

    python riva_voices.py --list
    python riva_voices.py --voice Magpie-Multilingual.EN-US.Aria --say "Hello, I am R1." --out aria.wav
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import wave

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from robot_voice_assistant.speech_pipeline import (  # noqa: E402
    RIVA_CLOUD_SERVER,
    RIVA_MAGPIE_FUNCTION_ID,
    RivaSpeechEngine,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--list", action="store_true", help="list the available voices")
    parser.add_argument("--voice", default="Magpie-Multilingual.EN-US.Aria")
    parser.add_argument("--language", default="en-US")
    parser.add_argument("--say", default="Hello! I am R1, a robot made by Unitree. How can I help you today?")
    parser.add_argument("--out", default="riva_sample.wav")
    args = parser.parse_args()
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        sys.exit("Set NVIDIA_API_KEY first (see README, 'NVIDIA Riva voices').")

    if args.list:
        import riva.client
        from riva.client.proto import riva_tts_pb2

        auth = riva.client.Auth(
            uri=RIVA_CLOUD_SERVER,
            use_ssl=True,
            metadata_args=[["function-id", RIVA_MAGPIE_FUNCTION_ID], ["authorization", f"Bearer {api_key}"]],
        )
        service = riva.client.SpeechSynthesisService(auth)
        config = service.stub.GetRivaSynthesisConfig(
            riva_tts_pb2.RivaSynthesisConfigRequest(), metadata=auth.get_auth_metadata()
        )
        for model in config.model_config:
            parameters = dict(model.parameters)
            voices = parameters.get("subvoices") or parameters.get("voice_name", "")
            print(f"{parameters.get('language_code', '?')}: {json.dumps(voices) if not isinstance(voices, str) else voices}")
        return

    engine = RivaSpeechEngine(api_key, voice=args.voice, language=args.language)
    audio = engine.synthesize(args.say)
    with wave.open(args.out, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(engine.sample_rate)
        file.writeframes(audio.tobytes())
    print(f"saved {len(audio) / engine.sample_rate:.1f} s of {args.voice} to {args.out}")


if __name__ == "__main__":
    main()
