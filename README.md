
# IMBK_Bank_Customer_Churn_ML

## 1. 프로젝트명
고객 이탈 분류 ML 및 인사이트 분석 (Bank Customer Churn Prediction)

## 2. 기간
2026년 4월 10일

## 3. 기술 스택 (Tech Stack)
* **Data Manipulation:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`, `shap`
* **Machine Learning & AutoML:** `scikit-learn`, `pycaret`, `catboost`, `lightgbm`, `xgboost`
* **Hyperparameter Tuning:** `optuna`

## 4. 데이터 출처
* **데이터셋:** 캐글 Bank Customer Churn Dataset (Row: 10000, Col: 12)
* **URL:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/data)

## 5. 데이터 전처리 (Data Preprocessing)
* **불필요한 컬럼 제거:** 데이터 간의 독립성을 보장하는 고유 식별자(`customer_id`)는 모델 학습에 불필요하여 제거.
* **범주형 변수 수치화:** `country`(거주 국가) 및 `gender`(성별)와 같은 범주형 데이터는 패턴 학습을 위해 `LabelEncoder`를 적용하여 수치형으로 변환.
* **데이터 분할:** 타겟 데이터의 불균형을 고려하여 `stratify=y` 옵션을 적용, Train과 Validation 데이터를 8:2 비율로 분할.
* **데이터 스케일링:** 변수 간의 단위 차이로 인한 모델 편향을 방지하기 위해 `StandardScaler`를 사용하여 데이터 표준화 적용.

## 6. EDA 및 해석 (Exploratory Data Analysis)
<img width="1088" height="576" alt="바그래프" src="https://github.com/user-attachments/assets/7afb70f9-2041-4d02-8dbd-03a0e816254f" />
<img width="1050" height="575" alt="꺾은선그래프" src="https://github.com/user-attachments/assets/6919cccb-8350-4cd0-9d8d-6b573fd6eec1" />

* **국가별 이탈자 수 분석 (Countplot):**
    * 특정 국가에서 타 국가 대비 이탈자의 절대적 수와 비율이 뚜렷하게 높게 나타남.
    * **해석:** 해당 지역의 금융 환경이나 경쟁사의 마케팅 등 외부 요인이 이탈에 주요한 영향을 줄 수 있음을 시사함.
* **나이대별 평균 이탈률 변화 분석 (Lineplot):**
    * 나이가 증가함에 따라 이탈률이 점진적으로 상승하며, 특히 50~60대 구간에서 최고치를 기록함.
    * **해석:** 고령층 고객을 위한 맞춤형 자산 관리 서비스나 유지 보상 프로그램의 부재가 이탈의 주요 원인일 수 있음을 파악.

## 7. 파이프라인 (AutoML – Tuning – Stacking – Shap value)
* **AutoML (PyCaret):** 단순 정확도(Accuracy)보다 이탈 고객을 정확히 찾아내어 비즈니스 손실을 줄이는 데 중요한 **F1-Score**를 핵심 평가 지표로 설정. 전수 모델 비교 결과 성능이 우수한 상위 4개 부스팅 모델(CatBoost, LightGBM, GradientBoosting, XGBoost)을 최종 후보로 선정.
* **Hyperparameter Tuning (Optuna):** 4개의 후보 모델 각각에 대해 Optuna를 활용하여 하이퍼파라미터 최적화 진행 (반복 횟수, 트리 깊이, 학습률 조절 등).
* **SHAP Value 분석:** CatBoost 모델에 `shap.TreeExplainer`를 적용하여 기여도 시각화. 분석 결과 `Age`(나이), `Balance`(잔고), `Active Member`(활동 회원 여부)가 고객 이탈 결정에 가장 큰 영향을 미치는 변수로 확인됨.
* **Stacking Pipeline:** 최적화된 4개의 단일 모델을 전방 모델(Base Estimator)로 배치하고, `LogisticRegression`을 후방 모델(Meta Model)로 결합하여 스태킹 앙상블 수행. 이를 통해 단일 모델 대비 성능 편차를 줄이고 예측의 일반화 성능 및 신뢰도를 향상함. (최종 Stacking F1-Score: 0.6027)

## 8. 인사이트 제안
데이터 및 SHAP 분석 결과를 바탕으로 기존 고객 이탈 방지(Retention)를 위한 구체적 솔루션을 제안합니다.

* **고연령층 타겟 맞춤형 서비스 강화:** 이탈 기여도가 가장 높게 나타난 50~60대 고객을 대상으로 전담 직원을 배치하여 VIP 자산 관리 및 연금 관리 서비스를 강화.
* **특화 금융 상품 신설:** 고령층 어르신을 주요 타겟으로 하여 특정 연령 이상만 가입 가능한 특화 상품이나 유지 우대 금리 혜택을 제공하여 고객 이탈을 효과적으로 방지(Lock-in).
* **Active Member 전환 프로모션:** 가입은 되어있으나 활동성이 낮은 고객군에게 모바일 앱 사용 혜택 및 리워드를 제공하여 활성 회원(Active Member)으로 전환을 유도하는 전략 수립 필요.

## 9. Reference
* **Dataset:** [Kaggle - Bank Customer Churn Dataset](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/data)
* **Libraries:** PyCaret, Optuna, Scikit-Learn, SHAP, CatBoost, LightGBM, XGBoost Official Documentation.
