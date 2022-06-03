# A Hybrid Method Based on Semi-Supervised Learning for Relation Extraction in Chinese EMRs

# Background: 
Building a large-scale medical knowledge graphs needs to automatically extract the relations between entities from electronic medical records(EMRs) . The main challenges are the scarcity of available labeled corpus and the identification of complexity semantic relations in text of Chinese EMRs. A hybrid method based on semi-supervised learning is proposed to extract the medical entity relations from small-scale complex Chinese EMRs.
# Methods: 
The semantic features of sentences are extracted by a residual network(ResNet) and the long dependent information is captured by bidirectional gated recurrent unit(BiGRU). Then the attention mechanism is used to assign weights for the extracted features respectively, and the output of two attention mechanisms is integrated for relation prediction. We adjusted the training process with manually annotated small-scale relational corpus and bootstrapping semi-supervised learning algorithm, and continuously expanded the datasets during the training process.
# The architecture of the ResGRU-Att model
