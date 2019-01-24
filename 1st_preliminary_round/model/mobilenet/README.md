# 모바일 넷 코드 변경 사항

해당 모델

1. `team_41/ir_ph1_v2/48/19`
1. `team_41/ir_ph1_v2/57/19`
1. `team_41/ir_ph1_v2/60` : epochs:100, val_ratio:0.15

## Update: 19.01.16.

### 1. Split Train & Validation Set

### 2. Add Hyper Parameters

lines 135~140

``` python
EPOCHS = 100
BATCH_SIZE = 16
VAL_RATIO = 0.15
LR = 0.0001
REDUCE_STEP = 15
REDUCE_FACT = 0.5
```

* 메인 시작하는 부분에서 위를 수정하셔서 사용하시면 됩니다.
* `REDUCE_STEP`, `REDUCE_FACT`는 epoch이 step만큼 진행하면 `LR`에 fact만큼 곱해서 학습률을 조정하는 하이퍼 파라미터 입니다.

### 3. Add Result logs

* 마지막에 모든 결과가 나올 수 있도록 수정하였습니다.


## Update: 19.01.05.

### 1. `infer` code

`main.py` 57 line
``` python
get_feature_layer = K.function([model.layers[0].input] + [K.learning_phase()], [model.layers[-1].output])
```

피쳐 익스트랙션 만 쓰던 부분에서 마지막 레이어까지 쓸 수 있도록 수정

###  2. `Mobilenet` 모델 변경

간단하게 프로토 타이핑 해 볼 용도로 `imagenet` weights init하고 20 epoch으로 학습