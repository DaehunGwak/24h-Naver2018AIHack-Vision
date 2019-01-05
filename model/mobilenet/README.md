# 모바일 넷 코드 변경 사항

해당 모델 : `team_41/ir_ph1_v2/48/19`

## 1. `infer` code

`main.py` 57 line
``` python
get_feature_layer = K.function([model.layers[0].input] + [K.learning_phase()], [model.layers[-1].output])
```

피쳐 익스트랙션 만 쓰던 부분에서 마지막 레이어까지 쓸 수 있도록 수정

## 2. `Mobilenet` 모델 변경

간단하게 프로토 타이핑 해 볼 용도로 `imagenet` weights init하고 20 epoch으로 학습