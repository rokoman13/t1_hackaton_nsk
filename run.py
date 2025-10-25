#!/usr/bin/env python3
import os
import sys

def main():
    print("Запуск системы замены фона...")
    print("Запуск на порту 8333...")
    print("Откройте: http://localhost:8333")
    
    os.system("streamlit run app.py --server.port=8333 --server.address=0.0.0.0")

if __name__ == "__main__":
    main()