# 24h-Naver2018AIHack-Vision
"24시간이모자라" 팀의 Naver AI Hackathon 2018 코드 및 기록입니다.

## Results

https://github.com/DaehunGwak/24h-Naver2018AIHack-Vision/issues/2

## Logs

### 1차 
- 1000 classification : 0.65

### 2차
- 1383 classification : 0.28
- siamese network, triplet loss metric learning : 0.53
- euclidean distance & l2_norm : 0.72
- semi hard sampling : 0.823

### 결선
- train classification & metric fine tuning : 0.858
- base change/densenet169 : 0.869
- euclidean -> cossim : 0.889
- query expansion(step 2) : 0.890
- query expansion(step 5, n 5) + cossim : 0.918
- query expansion(step 10, n 5) + cossim : 0.9201
- ensemblem : 0.9380
