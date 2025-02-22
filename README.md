# Демка для детекции животных

## Описание

Это запасная демка для стенда на AIJ. Её ценность состоит в том, что демку на gradio достаточно легко поднять, и она поддерживает кастомизацию при помощи JS.

## Что я изменил в исходниках

1. PytorchWoldlife предлагает несколько предобученных классификаторов, я заменил их на наш со второго этапа НТО. Сам класс живёт в [здесь](PytorchWildlife/models/classification/ds_hub.py). Для этой демки нам требуется просто обернуть модель в интерфейс с методом "single_image_classification".
2. По какой-то причине внутри докера не находились чекпоинты от torchhub. Я не смог быстро найти объяснение этопу эффекту. Вместо этого я залез в кишки модели и заставил её подгружать чекпоинты в директорию "./checkpoints", которую создаю при запуске gui. **Важно** по умолчанию мы поддерживаем MegaDetectorV6, который является наслежником YOLO8. Веса подгружаются в конструкторе именно YOLO, поэтому я таким образом пропатчил только [этот класс](PytorchWildlife/models/detection/ultralytics_based/yolov8_base.py). Для других моделей, которые не наследники YOLO8, могут наблюдаться проблемы с загрузкой чекпонтов.
3. Вместо [оригинального докерфайла](https://github.com/microsoft/CameraTraps/blob/main/Dockerfile) я написал свой, который 100% работает

## Как собирать и запускать

Чтобы собрать образ выволните команду:

```bash
docker build -t gradio_detection_demo .
```

Для запуска демки выполните команду:

```bash
docker run --gpus "0" --rm -it -v $(pwd):/app -p 80:80 gradio_detection_demo:latest
```

**Важно**: 
1. id GRU следует указать таким, какой девайс вы хотите использовать.
2. Порты также можно менять. Внутри докерфайла прописан порт 80, но благодаря маппингу портов нам важен только порт слева от ":".