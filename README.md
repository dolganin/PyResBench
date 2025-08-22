# PyResBench

В играх все постоянно меряются количеством FPS, и это стало стандартом оценки производительности систем.  
А вот в машинном обучении такого удобного и наглядного эталона нет: все бенчмарки либо про синтетические FLOPS, либо про голые GPU/CPU тесты.  
**PyResBench** закрывает эту дыру — это простой инструмент, который показывает, как быстро твоя система учит реальные нейросети.  

             
## Установка
```bash
git clone https://github.com/dolganin/PyResBench.git
pip3 install -e .
```

## Использование
Запуск с дефолтным датасетом (кошки/собаки):  
```bash
PyResBench
```

Выбор датасета:  
```bash
PyResBench --dataset cifar10
```

## Пример вывода
```
✓ Epoch 1 finished: time=3.29s, val_acc=100.00%
✓ Epoch 2 finished: time=2.73s, val_acc=100.00%
✓ Epoch 3 finished: time=2.74s, val_acc=100.00%
✓ Epoch 4 finished: time=2.77s, val_acc=100.00%
✓ Epoch 5 finished: time=2.75s, val_acc=100.00%
✓ Epoch 6 finished: time=2.75s, val_acc=100.00%
✓ Epoch 7 finished: time=2.73s, val_acc=100.00%
✓ Epoch 8 finished: time=2.71s, val_acc=100.00%
✓ Epoch 9 finished: time=2.75s, val_acc=100.00%
✓ Epoch 10 finished: time=2.75s, val_acc=100.00%
```

## Зачем это нужно?
Чтобы мериться не FPS, а реальной скоростью обучения нейросетей.  
Теперь ты можешь честно сравнить свою тачку с чужой не только в играх, но и в ML.
