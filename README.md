# RemarqueGPT

Repo contains code and notebooks used for my personal experiments on how to create GPT model from scratch and train on a small dataset of books. The purpose of the repo is solely educational and experimental, aimed at better understanding of the transformer architecture.

## Dataset 
Here is the list of the books used to train the model:
| Title (EN)                  | Title (RU)                        | Alt Title (RU)            | Year |                                                                |
|-----------------------------|-----------------------------------|---------------------------|------|------------------------------------------------------------------------------|
| The Dream Room              | Приют грёз                        | Мансарда снов             | 1920 
| Station at the Horizon      | Станция на горизонте              |                           | 1928 | 
| All Quiet on the Western Front | На Западном фронте без перемен | На Западе без перемен     | 1929 |  
| The Road Back               | Возвращение                       |                           | 1931 |                                                                              |
| Three Comrades              | Три товарища                      |                           | 1936 |                                                                              |

### Dataset Preprocessing  
For simplicity sake, the content of the books was preprocessed by removing all non-cyrillic letters. Phrases in foreign languages were translated. Each book was split into train/dev/test parts with the following proportions: 80/10/10.

# Implementation details

The implementation follows the original GPT decoder architecture.

## Single Head Attention Graph

Visualization of attention calculation for a single head:
![Single Head Attention Graph](images/single_head_attention_graph.png "Single Head Attention Graph")

# Suplementary materials
The Illustrated Transformer by Jay Alammar: http://jalammar.github.io/illustrated-transformer/    
Let's build GPT: from scratch, in code, spelled out. by Andrej Karpathy: https://youtu.be/kCc8FmEb1nY