# 테스트 결과

## 초기 데이터 상태

- clean 리뷰: 7,044건
- 초기 AI 분석 완료: 0건
- 초기 extracts: 0건
- 평균 고객 별점: 약 4.37

기존 데모/offline 감정값은 모두 제거하여 제품을 처음 열었을 때 실제 고객 별점과 리뷰만 표시하도록 초기화했습니다.

## 쇼핑몰 메뉴 테스트

검증 흐름:

1. `python main.py`
2. 제품 둘러보기
3. 제품 선택
4. 제품 상세 화면에서 `AI 분석 상태 : 0/N건` 확인
5. `AI 리뷰 분석 실행` 선택
6. API 키가 없을 때 환경변수 설정 안내 확인
7. 작업 후 clean_reviews.jsonl의 분석 건수가 여전히 0건인지 확인

결과: PASS

## argparse 테스트

```text
python main.py --help
python main.py analyze --help
```

필수 서브커맨드 확인:

- import
- clean
- analyze
- extract
- list
- show
- stats
- dashboard
- export

analyze 옵션 확인:

- --all
- --id
- --unanalyzed
- --limit
- --force
- --product
- --brand
- --date-from
- --date-to

결과: PASS

## AI 처리 흐름 모의 API 테스트

외부 API 호출 대신 동일한 응답 형식의 Fake AIClient를 사용하여 저장 로직을 검증했습니다.

1. 3개 미분석 리뷰 입력
2. AI 결과 positive / negative / neutral 반환
3. confidence 저장
4. `analysis_provider=openai_api` 저장
5. 키워드/요약/개선제안 추출
6. `extracts.jsonl`에 `provider=openai_api` 저장

결과: PASS

실제 API 호출은 사용자의 `GEMINI_API_KEY`가 있어야 실행됩니다.
