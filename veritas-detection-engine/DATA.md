# Data sources

Raw datasets are **not** stored in this repository.

## CIFAKE (training and in-distribution test)
- Kaggle: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
- 100,000 train / 20,000 test images, 32x32. Real images come from CIFAR-10; fake images were
  generated with Stable Diffusion v1.4.
- Citation: Bird, J.J. and Lotfi, A., "CIFAKE: Image Classification and Explainable
  Identification of AI-Generated Synthetic Images".
- Limitation: all fakes come from one generator, so strong results here do not show the model
  detects AI images in general.

## GenImage subset (cross-generator evaluation)
- Kaggle: https://www.kaggle.com/datasets/renhuang8/genimage-subset-detection
- Official repo: https://github.com/GenImage-Dataset/GenImage
- Used only for zero-shot evaluation on generators unseen during training.

Used for non-commercial research and education. Please check each dataset's page for its
current license terms.
