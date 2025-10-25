#!/usr/bin/env python3
"""
Скрипт для запуска Streamlit
"""

import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='Запуск Streamlit')
    parser.add_argument('--port', type=int, default=8333, help='Порт для Streamlit')
    parser.add_argument('--address', default='0.0.0.0', help='Адрес для Streamlit')
    args = parser.parse_args()

    print("=" * 50)
    print("ЗАПУСК")
    print("=" * 50)
    
    # Запускаем Streamlit
    try:
        streamlit_args = [
            "streamlit", "run", "app.py",
            f"--server.port={args.port}",
            f"--server.address={args.address}",
            "--browser.serverAddress=localhost",
            "--theme.base=light"
        ]
        
        print(f"Запуск на: http://{args.address}:{args.port}")
        
        subprocess.run(streamlit_args)
        
    except KeyboardInterrupt:
        print("\nПриложение остановлено")
    except Exception as e:
        print(f"Ошибка запуска: {e}")

if __name__ == "__main__":
    main()