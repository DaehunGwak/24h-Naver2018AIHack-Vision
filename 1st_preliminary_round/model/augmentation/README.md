# 어그멘테이션

`keras.preprocessing.image.ImageDataGenerator`를 이용한 이미지 어그멘테이션을 적용해보았습니다.


``` python
# create aug.
datagen = ImageDataGenerator(zoom_range=0.3,
                                shear_range=0.3,
                                rotation_range=180,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                horizontal_flip=True,
                                vertical_flip=True)

# Train
res = model.fit_generator(generator=datagen.flow(x_train, y_train,                                               batch_size=BATCH_SIZE),
                                steps_per_epoch=x_train.shape[0],
                                initial_epoch=epoch,
                                epochs=epoch + 1,
                                callbacks=[reduce_lr],
                                validation_data=(x_val, y_val),
                                verbose=1,
                                shuffle=True)
```

* 위와 같이 제너레이터를 생성하고 `fit_generator`에 끼워넣어서 사용할 수 있습니다.
* 한 이미지당 배치사이즈만큼 부풀릴 수 있도록 만들었습니다.
