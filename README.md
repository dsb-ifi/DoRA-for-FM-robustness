# DoRA for FM robustness
GitHub repo for the paper "Low-Rank Adaptations for increased Generalization in Foundation Model features"

## How to use this code
#### Running experiments
Use tools/slide_level_tasks/crossvalidation_external.py to run experiments with distinct train/test datasets.   
Then change parameters in conf/slide_level_task/cross_validation/config_extreme.yaml and conf/slide_level_task/cross_validation/task/os_prediction_extreme.yaml for different models/datasets/hyperparameters.

Use tools/slide_level_tasks/cross_validation.py for internal validation experiments.

## Acknowledgements
A large part of this codebase is derived from [Owkin/HistoSSLscaling (Phikon)](https://github.com/owkin/HistoSSLscaling).
Dino tuning is derived from [facebookresearch/dino](https://github.com/facebookresearch/dino)


## Attribution & License

This page includes material from the project:

**Filiot et al. (2023). _Scaling Self-Supervised Learning for Histopathology with Masked Image Modeling_. medRxiv.**  
[DOI: 10.1101/2023.07.21.23292757](https://www.medrxiv.org/content/early/2023/07/26/2023.07.21.23292757)

Source: [Phikon GitHub Repository](https://github.com/owkin/HistoSSLscaling)  
License: [Owkin Non-Commercial License](https://github.com/owkin/HistoSSLscaling/tree/main?tab=License-1-ov-file)

This material is licensed for **Non-Commercial use only**. You may reproduce and share it, or share any derivative work, under the same Non-Commercial terms and with attribution to the original authors.  

If you reuse or adapt this material, please also cite the Phikon paper:  
Alexandre Filiot, Ridouane Ghermi, Antoine Olivier, Paul Jacob, Lucas Fidon, Axel Camara, Alice Mac Kain, Charlie Saillard, and Jean-Baptiste Schiratti. Scaling selfsupervised learning for histopathology with masked image modeling. medRxiv, pages
2023–07, 2023.
