---
layout: project_page
permalink: /

title: Low-Rank Adaptations for increased Generalization in Foundation Model features
authors:
  - name: Vilde Schulerud Bøe
    affiliation: 1, 3
  - name: Andreas Kleppe
    affiliation: 1, 2, 3
  - name: Sebastian Foersch
    affiliation: 4
  - name: Daniel-Christoph Wagner
    affiliation: 4
  - name: Lill-Tove Rasmussen Busund
    affiliation: 5, 6
  - name: Adín Ramírez Rivera
    affiliation: 1, 3
affiliations:
    - name: DSB group, Department of Informatics, University of Oslo, Norway
      link: https://www.mn.uio.no/ifi/forskning/grupper/dsb/
    - name: Institute for Cancer Genetics and Informatics, Oslo University Hospital, Norway
    - name: SFI Visual Intelligence
      link: https://www.visual-intelligence.no/
    - name: Institute of Pathology, University Medical Center, Mainz, Germany
    - name: Department of Medical Biology, UiT The Arctic University of Norway, Tromsø, Norway
    - name: Department of Clinical Pathology, University Hospital of North Norway, Tromsø, Norway
paper: https://openreview.net/pdf?id=0BJTRUVDf4
code: https://github.com/dsb-ifi/DoRA-for-FM-robustness

abstract: |
  For foundation models (FMs) to truly advance computational pathology, they must deliver consistent and reliable predictions under diverse, unseen test conditions. Without
  such robustness, clinical trust and widespread adoption remain out of reach. Although
  many FMs for histopathology now exist, they have to our knowledge not been systematically tested for robustness by external researchers on independent datasets. In this study,
  we evaluate the robustness of foundation model features on three separate histopathology
  datasets and find that their performance drops on external data. Our analysis also reveals
  that these models often encode dataset-specific information, limiting their generalizability.
  To address this issue, we train a Weight-Decomposed Low-Rank Adaptation (DoRA) with
  strong data augmentations to improve feature robustness. Our experiments show that
  models trained with this adapter exhibit fewer signs of dataset-specific information and
  may generate more robust features across domains. These results highlight the need for
  robustness testing and encourage incorporating robustness considerations into the development, training, and tuning of FMs for histopathology.

---

![Figure 1](fig1.png)
*Figure 1: Tokenized image and attributions from standard Vision Transformer (ViT), a transformer trained with random Voronoi tesselations as tokens (RViT), and our proposed superpixel tokens (SPiT). We use attribution methods inherent in the transformer architecture (Att.Flow), PCA with prototypes (Proto. PCA), and local interpretable model-agnostic explanations LIME using independently computed superpixels using simple linear iterative clustering (SLIC).*


The Vision Transformer (ViT) has largely superseded convolutional neural networks (CNN) for state-of-the-art performance in vision tasks. The ViT applies a novel way of processing images; instead of applying a deep sequence of learnable local convolutional filters, the strategy is instead to partition the image into *square patches* to be treated as *discrete tokens*. These are processed in parallel through *global attention operators*, for which the model learns to share information between tokens optimally for a specific downstream task, e.g. classification. Hence, the ViT represents a move from a focus on local processing to a focus on global processing of spatial data.

Since its introduction, the ViT model has been leveraged successfully for a variety of downstream tasks, including classification, object detection, segmentation, and has been shown to be particularly well suited to self-supervised learning. Importantly, since the transformer architecture is used in language tasks, this allows for multimodality in language and vision.

# Robustness of Foundation Models



# DoRA for improved generalization

## DoRA

## Conclusion

Add a ashort conclusion here

## Citation
{% raw %}
```
@inproceedings{Boee2025,
  title={Low-Rank Adaptations for increased Generalization in Foundation Model features},
  author={B\o{}e, Vilde Schulerud and Kleppe, Andreas and Foersch, Sebastian and Wagner, Daniel-Christoph and Busund, Lill-Tove Rasmussen and Ram\'irez Rivera, Ad\'in},
  year={2025}
}
```
{% endraw %}
