# [AAAI24] Dual-channel Learning Framework for Drug-Drug Interaction Prediction via Relation-aware Heterogeneous Graph Transformer 

TIGER leverages the Transformer architecture to effectively exploit the structure of heterogeneous graph, which allows it direct learning of long dependencies and high-order structures. Furthermore, TIGER incorporates a relation-aware self-attention mechanism, capturing a diverse range of semantic relations that exist between pairs of nodes in heterogeneous graph. In addition to these advancements, TIGER enhances predictive accuracy by modeling DDI prediction task using a dual-channel network, where drug molecular graph and biomedical knowledge graph are fed into two respective channels. By incorporating embeddings obtained at graph and node levels, TIGER can benefit from structural properties of drugs as well as rich contextual information provided by biomedical knowledge graph. Extensive experiments conducted on three real-world datasets demonstrate the effectiveness of TIGER in DDI prediction. Furthermore, case studies highlight its ability to provide a deeper understanding of underlying mechanisms of DDIs.

![TIGER framework](https://github.com/Blair1213/TIGER/blob/main/AAAI.jpeg)

# Installation & Dependencies

TIGER is mainly tested on a Linux OS with NVIDIA A100 40G and its dependecies are below.

|Package|Version|
|-----:|-------|
|python| 3.6.13|
|rdkit|2021.9.4|
|pytorch| 1.10.2|
|cuda|11.6.2|
|torch-cluster|1.5.9|
|torch-geometric| 2.0.2|
|torch-scatter |2.0.9|
|torch-sparse| 0.6.12|
|torchvision| 0.11.3|

In addition, TIGER can also be trained on a Mac OS machine equipped with 16GB RAM.

# Datasets

TIGER is trained and tested on three datasets, including DrugBank, KEGG, and OGB-biokg. The networks are availiable at [datasets](https://drive.google.com/file/d/13ZFDZ28Eam5C5gs-yw-UZ6Yi_X2jkN69/view?usp=share_link).

# Reproducibility

To reproduce the results of TIGER or train TIGER, you are supposed to download above datasets first, and put it into a file "datasets/". The directory structure of TIGER is shown below:

```
.
├── README.md
├── best_save
├── data
├── dataset
│   └── drugbank
│   └── kegg
│   └── ogbl-biokg
├── model
├── randomWalk
├── main.py
├── train_eval.py
├── utils.py
└── data_process.py

```
Then, you can train TIGER with the following command:

```
python main.py
```
or
```
python main.py -dataset drugbank/kegg/ogbl-biokg -extractor khop-subtree/randomWalk/probability
```

The hyper-parameters used to train TIGER on above three datasets are shown in our paper.

