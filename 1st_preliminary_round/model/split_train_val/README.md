# Train & Validation 셋 분리 테스트

해당 로그 : `team_41/ir_ph1_v2/55`

## 변경 사항

### 1. 하이퍼 파라미터 위로 옮김

135~137 lines

``` python
EPOCHS = 5
BATCH_SIZE = 16
VAL_RATIO = 0.05
```

* EPOCH, BATCH_SIZE를 일일이 찾기 싫어 메인의 제일 상단으로 옮겼습니다.
* Validation Ratio를 지정할 수 있도록 `VAL_RATIO`를 추가하였습니다.

### 2. 

200~213 lines

``` python
# split train & validation
# shuffle
s = np.arange(x_all.shape[0])
np.random.shuffle(s)
x_all = x_all[s]
y_all = y_all[s]
# split
split_i = int(x_all.shape[0] * (1.0 - VAL_RATIO))
x_train = x_all[:split_i]
y_train = y_all[:split_i]
x_val = x_all[split_i:]
y_val = y_all[split_i:]
print("Train shape:", x_train.shape)
print("Validation shape:", x_val.shape)
```

* 아직 nsml에 sklearn을 어떻게 설치할 줄 몰라 `numpy`를 사용해 나누었습니다.
* `numpy`로 먼저 한번 `shuffle`을 진행한 후 split 하였습니다. 