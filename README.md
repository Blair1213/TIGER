# [AAAI24] Dual-channel Learning Framework for Drug-Drug Interaction Prediction via Relation-aware Heterogeneous Graph Transformer 

TIGER leverages the Transformer architecture to effectively exploit the structure of heterogeneous graph, which allows it direct learning of long dependencies and high-order structures. Furthermore, TIGER incorporates a relation-aware self-attention mechanism, capturing a diverse range of semantic relations that exist between pairs of nodes in heterogeneous graph. In addition to these advancements, TIGER enhances predictive accuracy by modeling DDI prediction task using a dual-channel network, where drug molecular graph and biomedical knowledge graph are fed into two respective channels. By incorporating embeddings obtained at graph and node levels, TIGER can benefit from structural properties of drugs as well as rich contextual information provided by biomedical knowledge graph. Extensive experiments conducted on three real-world datasets demonstrate the effectiveness of TIGER in DDI prediction. Furthermore, case studies highlight its ability to provide a deeper understanding of underlying mechanisms of DDIs.

# Installation & Dependencies

TIGER is mainly tested on both Linux and Mac OS.

TIGER has the following dependencies on Mac OS:

|Package|Version|
|-----:|-------|
|python| 3.7.16|
|rdkit||
|torch| 1.10.1|
|torch-cluster |1.5.9|
|torch-geometric| 2.0.3|
|torch-scatter |2.0.9|
|torch-sparse| 0.6.12|
|torch-spline-conv |1.2.1|
|torchvision| 0.11.2|
|pandas |1.3.5|
|numpy| 1.19.5|


