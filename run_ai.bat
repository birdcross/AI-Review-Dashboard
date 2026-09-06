@echo off
chcp 65001 > nul
cd /d %~dp0
if not exist .env (
  echo [안내] 프로젝트 폴더에 .env 파일이 없습니다.
  echo .env.example 파일을 복사하여 .env를 만들고 API 키를 입력하세요.
  echo 예: GEMINI_API_KEY=본인_API_KEY
  echo.
)
py -3.13 main.py
pause
