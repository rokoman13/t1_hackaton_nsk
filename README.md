Для запуска:

python3 run_with_gpu.py --port=port --address=adress

Т.к. у браузеров есть защита от передачи видео без шифрования (по http), нужно произвести следующие настройки:

**Firefox:**

1. Открыть новую вкладку
2. Ввести about:config
3. Установить в true значения media.devices.insecure.enabled и media.getusermedia.insecure.enabled

**Chrome:**

1. В новой вкладке открыть chrome://flags/#unsafely-treat-insecure-origin-as-secure
2. Установить значение во "включено" и перезапустить браузер
