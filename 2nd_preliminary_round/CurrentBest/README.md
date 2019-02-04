# Current Best Model

## Model
- globalAveragePooling()
- Dropout(0.2)
- BatchNormalization()
- Dense(embedding_dim)

## Hyper-parameters
- batch_size: 32
- learning_rate: 0.00005
- embedding_dim: 2048
- Mobilenet, pretrain: True

## Score
- team_41/ir_ph2/143 3, 0.48872538085417566
- team_41/ir_ph2/143/12, 0.5144610022392557
- team_41/ir_ph2/143/36, 0.5230737106786733
