import grpc
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "output" / "yandex"))
yandex_path = Path(__file__).parent / "yandex"
if yandex_path.exists():
    sys.path.insert(0, str(yandex_path.parent))

import yandex.cloud.ai.stt.v3.stt_pb2 as stt_pb2
import yandex.cloud.ai.stt.v3.stt_service_pb2_grpc as stt_service_pb2_grpc


from typing import Generator, List, Tuple
from utils.logger import logger
from config.settings import settings


# gRPC channel options для длинных стримов
GRPC_CHANNEL_OPTIONS = [
    ('grpc.max_send_message_length', 100 * 1024 * 1024),    # 100 MB
    ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100 MB
    ('grpc.keepalive_time_ms', 30_000),                       # пинг каждые 30с
    ('grpc.keepalive_timeout_ms', 10_000),                    # ждём ответ 10с
    ('grpc.keepalive_permit_without_calls', 1),               # пинг даже без активных вызовов
    ('grpc.http2.max_pings_without_data', 0),                 # без ограничений на пинги
]


class YandexSTTClient:
    """Клиент для работы с Yandex Speech-to-Text API"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.logger = logger.getChild("yandex_stt")

    def _create_streaming_options(self) -> stt_pb2.StreamingOptions:
        """Создание настроек распознавания"""
        return stt_pb2.StreamingOptions(
            recognition_model=stt_pb2.RecognitionModelOptions(
                audio_format=stt_pb2.AudioFormatOptions(
                    raw_audio=stt_pb2.RawAudio(
                        audio_encoding=stt_pb2.RawAudio.LINEAR16_PCM,
                        sample_rate_hertz=settings.SAMPLE_RATE,
                        audio_channel_count=settings.AUDIO_CHANNELS
                    )
                ),
                text_normalization=stt_pb2.TextNormalizationOptions(
                    text_normalization=stt_pb2.TextNormalizationOptions.TEXT_NORMALIZATION_ENABLED,
                    profanity_filter=True,
                    literature_text=False
                ),
                language_restriction=stt_pb2.LanguageRestrictionOptions(
                    restriction_type=stt_pb2.LanguageRestrictionOptions.WHITELIST,
                    language_code=['ru-RU']
                ),
                audio_processing_type=stt_pb2.RecognitionModelOptions.FULL_DATA
            )
        )

    def _audio_generator(self, audio_file_path: str) -> Generator[stt_pb2.StreamingRequest, None, None]:
        """Генератор для потоковой передачи аудио"""
        yield stt_pb2.StreamingRequest(session_options=self._create_streaming_options())

        with open(audio_file_path, 'rb') as f:
            while True:
                data = f.read(settings.CHUNK_SIZE)
                if not data:
                    break
                yield stt_pb2.StreamingRequest(chunk=stt_pb2.AudioChunk(data=data))

    def _transcribe_single(self, audio_file_path: str, save_intermediate: bool = False) -> Tuple[str, List[str]]:
        """Транскрибирование одного аудиофайла (сегмента)."""
        self.logger.info(f"Начало транскрибирования: {audio_file_path}")

        cred = grpc.ssl_channel_credentials()
        channel = grpc.secure_channel(
            settings.YANDEX_ENDPOINT, cred, options=GRPC_CHANNEL_OPTIONS
        )
        stub = stt_service_pb2_grpc.RecognizerStub(channel)

        stream = stub.RecognizeStreaming(
            self._audio_generator(audio_file_path),
            metadata=(('authorization', f'Api-Key {self.api_key}'),)
        )

        segments = []
        intermediate_texts = []

        try:
            for response in stream:
                event_type = response.WhichOneof('Event')

                if event_type == 'partial' and len(response.partial.alternatives) > 0:
                    text = response.partial.alternatives[0].text
                    if save_intermediate:
                        intermediate_texts.append(text)
                        self.logger.debug(f"Промежуточный результат: {text}")

                elif event_type == 'final' and len(response.final.alternatives) > 0:
                    text = response.final.alternatives[0].text
                    segments.append(text)
                    self.logger.info(f"Финальный сегмент ({len(segments)}): {text[:80]}...")

                elif event_type == 'final_refinement':
                    if len(response.final_refinement.normalized_text.alternatives) > 0:
                        refined = response.final_refinement.normalized_text.alternatives[0].text
                        self.logger.debug(f"Уточненный результат: {refined}")

        except grpc._channel._Rendezvous as err:
            self.logger.error(f"Ошибка gRPC: код={err._state.code}, сообщение={err._state.details}")
            raise
        finally:
            channel.close()

        full_text = ' '.join(segments)
        self.logger.info(f"Транскрибирование завершено. Сегментов: {len(segments)}, символов: {len(full_text)}")

        return full_text, segments

    def transcribe(self, audio_file_path: str, save_intermediate: bool = False) -> Tuple[str, List[str]]:
        """
        Транскрибирование аудиофайла.
        Если передан список сегментов — обрабатывает каждый последовательно.
        Returns: (full_text, segments)
        """
        return self._transcribe_single(audio_file_path, save_intermediate)

    def transcribe_segments(self, segment_paths: List[str], save_intermediate: bool = False) -> Tuple[str, List[str]]:
        """
        Транскрибирование нескольких сегментов последовательно.
        Returns: (full_text, all_segments)
        """
        all_segments = []
        for i, path in enumerate(segment_paths):
            self.logger.info(f"Обработка сегмента {i+1}/{len(segment_paths)}: {path}")
            _, segments = self._transcribe_single(path, save_intermediate)
            all_segments.extend(segments)

        full_text = ' '.join(all_segments)
        self.logger.info(
            f"Все сегменты обработаны. Итого: {len(all_segments)} фрагментов, {len(full_text)} символов"
        )
        return full_text, all_segments
