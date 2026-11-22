"""The speech service."""
import logging
import time
from typing import Any
from collections.abc import AsyncGenerator
from contextlib import aclosing
from homeassistant.exceptions import HomeAssistantError
from homeassistant.components.tts import (
    CONF_LANG,
    TextToSpeechEntity,
    TtsAudioType,
    TTSAudioRequest,
    TTSAudioResponse,
)
from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry
from homeassistant.util import ulid
from homeassistant.helpers.device_registry import DeviceEntryType
from homeassistant.helpers.entity_platform import AddEntitiesCallback

import edge_tts
from .const import DOMAIN, SUPPORTED_VOICES, DEFAULT_LANG, DEFAULT_VOICE


_LOGGER = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = {
    **dict(zip(SUPPORTED_VOICES.values(), SUPPORTED_VOICES.keys())),
    DEFAULT_LANG: DEFAULT_VOICE,
}

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Edge TTS entity from a config entry."""
    entity = EdgeTTSEntity(hass, config_entry)
    async_add_entities([entity])


class EdgeTTSEntity(TextToSpeechEntity):
    """The Edge TTS entity."""

    _attr_name = "Edge TTS"
    
    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry) -> None:
        """Initialize Edge TTS entity."""
        self.hass = hass
        self._config_entry = config_entry
        self._attr_unique_id = f"{config_entry.entry_id}-tts"

        self._attr_device_info = {
            "identifiers": {(DOMAIN, self._config_entry.entry_id)},
            "name": "Edge TTS Service",
            "manufacturer": "Edge TTS Community",
            "model": "Cloud TTS",
            "sw_version": edge_tts.__version__,
            "entry_type": DeviceEntryType.SERVICE,
        }
        self._attr_extra_state_attributes = {}

        # Prosody and style options
        self._prosody_options = ['pitch', 'rate', 'volume']
        self._style_options = ['style', 'styledegree', 'role']

    async def async_added_to_hass(self) -> None:
        domain_data = self.hass.data.setdefault(DOMAIN, {})
        domain_data["tts_entity_id"] = self.entity_id
        access_tokens = domain_data.setdefault("access_tokens", {
            "temp": ulid.ulid_hex(),
            "long": self.hass.data["core.uuid"],
        })
        self._attr_extra_state_attributes["access_tokens"] = access_tokens.copy()

    @property
    def default_language(self) -> str:
        """Return the default language from options."""
        return self._config_entry.options.get(CONF_LANG, DEFAULT_LANG)

    @property
    def supported_languages(self) -> list[str]:
        """Return list of supported languages."""
        return list([*SUPPORTED_LANGUAGES.keys(), *SUPPORTED_VOICES.keys()])

    @property
    def supported_options(self) -> list[str]:
        """Return a list of supported options."""
        return ['voice'] + self._prosody_options

    async def async_get_tts_audio(
        self, message: str, language: str, options: dict[str, Any]
    ) -> TtsAudioType:
        return "mp3", await self.async_process_tts_audio(message, language, options)

    async def async_process_tts_audio(
        self, message: str, language: str, options: dict[str, Any]
    ) -> bytes | None:
        return await self.hass.async_add_executor_job(
            self._process_tts_audio,
            message,
            language,
            options,
        )

    # Share voice and prosody resolution across both audio paths. Related: config_flow.py, const.py.
    def _create_communicate(
        self, message: str, language: str, options: dict[str, Any]
    ) -> edge_tts.Communicate:
        # Reject non-speech input before opening a network request. Related: intent.py.
        if not any(char.isalnum() for char in message):
            raise HomeAssistantError("Edge TTS requires text containing letters or numbers")
        opt = {CONF_LANG: language}
        if language in SUPPORTED_VOICES:
            opt[CONF_LANG] = SUPPORTED_VOICES[language]
            opt['voice'] = language
        opt = {**self._config_entry.options, **opt, **(options or {})}

        lang = opt.get(CONF_LANG) or language or DEFAULT_LANG
        voice = opt.get('voice') or SUPPORTED_LANGUAGES.get(lang) or DEFAULT_VOICE

        for f in self._style_options:
            if f in opt:
                _LOGGER.warning(
                    'Edge TTS options style/styledegree/role are no longer supported, '
                    'please remove them from your automation or script. '
                    'See: https://github.com/hasscc/hass-edge-tts/issues/8'
                )
                break

        return edge_tts.Communicate(
            message,
            voice=voice,
            pitch=opt.get('pitch', '+0Hz'),
            rate=opt.get('rate', '+0%'),
            volume=opt.get('volume', '+0%'),
        )

    def _process_tts_audio(
        self, message: str, language: str, options: dict[str, Any]
    ) -> bytes | None:
        tts = self._create_communicate(message, language, options)
        mp3 = b''
        start_time = time.perf_counter()
        try:
            for chunk in tts.stream_sync():
                if chunk["type"] == "audio":
                    mp3 += chunk["data"]
                else:
                    _LOGGER.debug("Edge TTS metadata: %s", chunk)
        except edge_tts.exceptions.NoAudioReceived as exc:
            _LOGGER.warning("No audio received for text: %s", message)
            raise HomeAssistantError(f"{self.name}: No audio received: {message}") from exc
        end_time = time.perf_counter()
        elapsed_time = (end_time - start_time) * 1000
        _LOGGER.debug("load tts elapsed_time: %sms", elapsed_time)
        if not mp3:
            raise HomeAssistantError("Edge TTS received no audio")
        return mp3

    async def async_stream_tts_audio(self, request: TTSAudioRequest) -> TTSAudioResponse:
        # Validate the first audio before HA starts MP3 conversion. Related: tests/test_tts_stream.py.
        stream = self._process_tts_stream(request)
        try:
            first_audio = await anext(stream)
        except StopAsyncIteration as exc:
            raise HomeAssistantError(
                "Edge TTS requires text containing letters or numbers"
            ) from exc
        except BaseException:
            await stream.aclose()
            raise

        async def audio_stream():
            async with aclosing(stream):
                yield first_audio
                async for audio in stream:
                    yield audio

        return TTSAudioResponse("mp3", audio_stream())

    # Bound text batches and forward audio immediately. Related: tests/test_tts_stream.py.
    async def _stream_message(
        self, message: str, request: TTSAudioRequest
    ) -> AsyncGenerator[bytes]:
        """Forward audio while Edge is still synthesizing the current batch."""
        # Skip punctuation-only tails; retry only before any audio is delivered.
        if not any(char.isalnum() for char in message):
            _LOGGER.debug("Skipping non-speech TTS batch: %s characters", len(message))
            return
        for attempt in range(2):
            tts = self._create_communicate(message, request.language, request.options)
            start_time = time.perf_counter()
            audio_bytes = 0
            _LOGGER.debug(
                "Starting TTS batch: %s characters, attempt %s", len(message), attempt + 1
            )
            try:
                async with aclosing(tts.stream()) as stream:
                    async for chunk in stream:
                        if chunk["type"] == "audio" and chunk["data"]:
                            if not audio_bytes:
                                _LOGGER.debug(
                                    "First TTS audio after %.0f ms",
                                    (time.perf_counter() - start_time) * 1000,
                                )
                            audio_bytes += len(chunk["data"])
                            yield chunk["data"]
                if not audio_bytes:
                    raise edge_tts.exceptions.NoAudioReceived("Empty audio stream")
                _LOGGER.debug("Completed TTS batch: %s audio bytes", audio_bytes)
                return
            except edge_tts.exceptions.NoAudioReceived as exc:
                _LOGGER.warning(
                    "Edge TTS batch failed: characters=%s, audio_bytes=%s, attempt=%s",
                    len(message), audio_bytes, attempt + 1,
                )
                if audio_bytes or attempt:
                    raise HomeAssistantError(
                        "Edge TTS received no audio or an incomplete batch"
                    ) from exc

    async def _process_tts_stream(self, request: TTSAudioRequest) -> AsyncGenerator[bytes]:
        """Generate speech from an incoming message."""
        _LOGGER.debug("Starting TTS stream")
        separators = ["\n", "。", ". ", "，", ", ", "；", "; ", "！", "! ", "？", "? ", "、"]
        buffer = ""
        async for message in request.message_gen:
            for char in message:
                buffer += char
                msg = buffer.strip()
                if len(buffer) >= 200 or (
                    len(msg) >= 20
                    and (char in separators or buffer[-2:] in separators)
                ):
                    buffer = ""
                    if msg:
                        async with aclosing(self._stream_message(msg, request)) as stream:
                            async for audio in stream:
                                yield audio
        if msg := buffer.strip():
            async with aclosing(self._stream_message(msg, request)) as stream:
                async for audio in stream:
                    yield audio
