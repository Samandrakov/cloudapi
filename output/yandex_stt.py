import grpc
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "output" / "yandex"))
yandex_path = Path(__file__).parent / "yandex"
if yandex_path.exists():
    sys.path.insert(0, str(yandex_path.parent))

import yandex.cloud.ai.stt.v3.stt_pb2 as stt_pb2
import yandex.cloud.ai.stt.v3.stt_service_pb2_grpc as stt_service_pb2_grpc


from typing import Generator, List, Tuple, Optional
from utils.logger import logger
from config.settings import settings


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
                audio_processing_type=stt_pb2.RecognitionModelOptions.REAL_TIME
            )
        )

    def _audio_generator(self, audio_file_path: str) -> Generator[stt_pb2.StreamingRequest, None, None]:
        """Генератор для потоковой передачи аудио"""
        # Отправляем настройки
        yield stt_pb2.StreamingRequest(session_options=self._create_streaming_options())

        # Читаем и отправляем аудио чанками
        with open(audio_file_path, 'rb') as f:
            while True:
                data = f.read(settings.CHUNK_SIZE)
                if not data:
                    break
                yield stt_pb2.StreamingRequest(chunk=stt_pb2.AudioChunk(data=data))

    def transcribe(self, audio_file_path: str, save_intermediate: bool = False) -> Tuple[str, List[str]]:
        """
        Транскрибирование аудиофайла
        Returns: (full_text, segments)
        """
        self.logger.info(f"Начало транскрибирования файла: {audio_file_path}")

        # Настраиваем соединение
        cred = grpc.ssl_channel_credentials()
        channel = grpc.secure_channel(settings.YANDEX_ENDPOINT, cred)
        stub = stt_service_pb2_grpc.RecognizerStub(channel)

        # Запускаем распознавание
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
                    self.logger.info(f"Добавлен финальный сегмент ({len(segments)}): {text}")

                elif event_type == 'final_refinement':
                    if len(response.final_refinement.normalized_text.alternatives) > 0:
                        refined = response.final_refinement.normalized_text.alternatives[0].text
                        self.logger.debug(f"Уточненный результат: {refined}")

        except grpc._channel._Rendezvous as err:
            self.logger.error(f"Ошибка gRPC: код={err._state.code}, сообщение={err._state.details}")
            raise

        full_text = ' '.join(segments)

        self.logger.info(f"Транскрибирование завершено. Сегментов: {len(segments)}, символов: {len(full_text)}")

        return full_text, segments