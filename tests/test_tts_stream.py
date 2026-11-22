"""Exercise the installed entity methods without starting Home Assistant."""

import ast
import asyncio
from contextlib import aclosing
import logging
from pathlib import Path
from types import SimpleNamespace
import unittest


def load_entity():
    """Load the actual entity class with small platform and network doubles."""
    path = Path(__file__).parents[1] / "custom_components/edge_tts/tts.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    entity = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        entity,
    ], type_ignores=[])
    namespace = {
        "TextToSpeechEntity": object,
        "HomeAssistantError": RuntimeError,
        "_LOGGER": logging.getLogger(__name__),
        "aclosing": aclosing,
        "time": __import__("time"),
        "CONF_LANG": "language",
        "SUPPORTED_VOICES": {"ko-KR-SunHiNeural": "ko-KR"},
        "SUPPORTED_LANGUAGES": {"ko-KR": "ko-KR-SunHiNeural"},
        "DEFAULT_LANG": "ko-KR",
        "DEFAULT_VOICE": "ko-KR-SunHiNeural",
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["EdgeTTSEntity"], namespace


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    """Catch dropped text, buffering regressions and leaked streams."""

    def setUp(self):
        entity_class, self.namespace = load_entity()
        self.entity = object.__new__(entity_class)
        self.entity._config_entry = SimpleNamespace(options={"rate": "+5%"})
        self.entity._style_options = ["style", "styledegree", "role"]
        self.calls = []
        self.closed = []
        self.allow_finish = None
        self.error = None
        test = self

        class Communicate:
            def __init__(self, message, **options):
                test.calls.append((message, options))

            async def stream(self):
                try:
                    if test.error:
                        raise test.error
                    yield {"type": "audio", "data": b"first"}
                    if test.allow_finish:
                        await test.allow_finish.wait()
                    yield {"type": "SentenceBoundary"}
                    yield {"type": "audio", "data": b"last"}
                finally:
                    test.closed.append(True)

        self.no_audio = type("NoAudioReceived", (Exception,), {})
        self.namespace["edge_tts"] = SimpleNamespace(
            Communicate=Communicate,
            exceptions=SimpleNamespace(NoAudioReceived=self.no_audio),
        )

    def request(self, parts, language="ko-KR", options=None):
        async def messages():
            for part in parts:
                yield part
        return SimpleNamespace(message_gen=messages(), language=language, options=options or {})

    async def test_fragmented_long_korean_text_is_complete_and_bounded(self):
        text = ("첫 번째 소식입니다. 관련 사실을 자세히 전합니다.\n\n" * 60)
        audio = [chunk async for chunk in self.entity._process_tts_stream(
            self.request(list(text))
        )]
        self.assertGreater(len(self.calls), 10)
        self.assertTrue(all(len(message) <= 200 for message, _ in self.calls))
        combined = "".join(message for message, _ in self.calls)
        self.assertEqual("".join(combined.split()), "".join(text.split()))
        self.assertEqual(len(audio), len(self.calls) * 2)

    async def test_unbroken_text_and_whitespace(self):
        text = "가" * 601
        await self.collect(self.request([" " * 450, text, "\n\n"]))
        self.assertTrue(all(0 < len(message) <= 200 for message, _ in self.calls))
        self.assertEqual("".join(message for message, _ in self.calls), text)

    async def collect(self, request):
        return [chunk async for chunk in self.entity._process_tts_stream(request)]

    async def test_first_audio_arrives_before_batch_finishes_and_close_propagates(self):
        self.allow_finish = asyncio.Event()
        stream = self.entity._process_tts_stream(self.request(["짧은 뉴스입니다."]))
        self.assertEqual(await asyncio.wait_for(anext(stream), 1), b"first")
        self.assertFalse(self.allow_finish.is_set())
        await stream.aclose()
        self.assertEqual(self.closed, [True])

    async def test_cancellation_closes_edge_connection(self):
        self.allow_finish = asyncio.Event()
        stream = self.entity._process_tts_stream(self.request(["취소될 뉴스입니다."]))
        await anext(stream)
        task = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.closed, [True])

    async def test_empty_input_does_not_call_edge(self):
        self.assertEqual(await self.collect(self.request([" ", "\n"])), [])
        self.assertEqual(self.calls, [])

    async def test_voice_and_prosody_override_defaults(self):
        await self.collect(self.request(["안녕하세요."], language="ko-KR-SunHiNeural",
                                        options={"rate": "+10%", "pitch": "+2Hz"}))
        self.assertEqual(self.calls[0][1], {
            "voice": "ko-KR-SunHiNeural", "rate": "+10%",
            "pitch": "+2Hz", "volume": "+0%",
        })

    async def test_no_audio_is_reported(self):
        self.error = self.no_audio()
        with self.assertRaisesRegex(RuntimeError, "no audio"):
            await self.collect(self.request(["오디오 오류입니다."]))


if __name__ == "__main__":
    unittest.main()
