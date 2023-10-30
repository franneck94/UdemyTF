# Performances

## 1: Early Stopping

- Active: 0.85605

## 2: Weights Init

- GlorotUniform*: 0.8688
- GlorotNormal: 0.8302
- HeUniform: 0.8337
- HeNormal: 0.8392
- LecunUniform: 0.8593
- LecunNormal: 0.8780

## 3: Activation Functions

- RELU*: 0.8541
- LEAKY_RELU: 0.8534
- ELU: 0.7848

## 4: Dropout

- 0.0: 0.8557
- 0.1: 0.8540
- 0.2: 0.8437
- 0.3: 0.8413

## 5: Batch Norm

- Active: 0.9020

## 6: LR Schedule

- 1: 0.8697
- 2: 0.8885
- 3: 0.8937
- 4: 0.8908

## 7: LR Plateau

- 1: 0.8692
- 2: 0.8836
- 3: 0.9034
- 4: 0.9020

## 8: Global Pooling

- Dense, GP: 0.9005
  - Total params: 1,340,962
- No Dense, GP: 0.9249
  - Total params: 1,272,098

## 9: Final

- Test Performance: 0.928456
