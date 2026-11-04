# 태풍 이동 경로 분석 및 예측 웹앱

## 구성 파일
GitHub 저장소의 최상위 폴더에 다음 파일이 있어야 합니다.

- `app.py`
- `requirements.txt`
- `README.md`
- `typhoon_data.csv`
- `typhoon_rf_model.pkl`

## 데이터와 모델 파일 만들기
Google Colab에서 전처리된 데이터프레임 이름이 `typhoon_df`, 학습 완료 모델 이름이 `model`이라면 다음 코드를 실행합니다.

```python
import joblib

typhoon_df.to_csv("typhoon_data.csv", index=False)
joblib.dump(model, "typhoon_rf_model.pkl")
```

두 파일을 다운로드한 다음 GitHub 저장소의 최상위 폴더에 업로드합니다. 데이터프레임과 모델 변수 이름이 다르다면 실제 사용한 변수 이름으로 바꿔야 합니다.

## Streamlit 배포
1. https://share.streamlit.io/ 에 접속해 GitHub 계정으로 로그인합니다.
2. 저장소 `yuniQ-152/typhoon-prediction`을 선택합니다.
3. Branch는 `main`, Main file path는 `app.py`로 설정합니다.
4. Deploy를 누르고 로그에서 오류가 있는지 확인합니다.

## 주의
이 앱은 교육·탐구용 예측 시연이며 공식 기상 예보가 아닙니다. 실제 안전 판단이나 대피 결정에 사용하지 마세요.
