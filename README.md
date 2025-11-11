# SFGA: Similarity-Constrained Fusion Learning for Unsupervised Anomaly Detection in Multiplex Graphs
Multiplex graphs are widely used to model multi-relational complex systems and play an important role in various real-world scenarios, such as financial systems and social networks. Hence, detecting anomalous samples in multiplex graph becomes crucial to ensure cybersecurity and stability. Although existing homogeneous graph anomaly detection (GAD) methods can be applied to deal with multiplex graphs, they still face two major challenges: 1) Due to the multiplicity and complexity of relations in multiplex graphs, homogeneous GAD models fail to effectively capture anomalous behaviors that correlate with diverse relational patterns. 2) In real-world applications, malicious entities usually disguise themselves through various camouflage strategies, making it difficult to capture subtle anomalous features via single-relation analysis. To address these challenges, we propose a novel unsupervised anomaly detection method for multiplex graphs based on Similarity-constrained Fusion Graph Autoencoder (SFGA). In SFGA, we design a multiplex graph autoencoder and introduced a cross-plex attention module at the model bottleneck to achieve comprehensive modeling of cross-relation anomaly patterns. Then, a similarity balancing strategy is proposed to constrain node representations at the bottleneck from both local and global perspectives, enhancing the discriminative power against camouflaged anomalies of autoencoder and enabling more effective identification of anomalous nodes with overlapping or deceptive patterns. Extensive experiments are conducted on both synthetic and real-world datasets at varying scales, and the results demonstrate that our proposed method outperforms state-of-the-art approaches by a large margin.

## Environment Installation
To set up and launch the experimental environment, you can first execute the following commands:
```
conda create -n SFGA python=3.9
conda activate SFGA
```
Then you can set up the running environment by executing:
```
pip install -r requirements.txt
```
Requirements.txt only provides recommended installation versions, or you can install versions compatible with your machine.

## Datasets Preparation
> For readers who want to reproduce the experimental results, we provide pre-processed IMDB and Freebase datasets with injected anomalies in the project, readers can extract the datasets from the zip files in the `./data/.../raw/` folder respectively. <br>
> Additionally, Amazon-fraud and Yelpchi-fraud datasets can be downloaded from https://github.com/YingtongDou/CARE-GNN .<br>

## Experimental Infrastructures
> For small- and medium-scale datasets (i.e., IMDB, Freebase, and Amazon-fraud), all the experiments are run with one NVIDIA GeForce RTX 3090 24GB GPU. <br>
> For large-scale graph dataset (i.e., YelpChi-fraud), the experiments are run with one H20-NVLink GPU (96GB memory). <br>

## Run for Reproducibility
For IMDB with injected anomalies, you can run experiments using the following script:
```
python main.py model_name sfga dataset_name imdb_injected
```
For Freebase with injected anomalies, you can run experiments using the following script:
```
python main.py model_name sfga dataset_name freebase_injected
```
For Amazon-fraud, you can run experiments using the following script:
```
python main.py model_name sfga dataset_name Amazon_fraud
```
For YelpChi-fraud, you can run experiments using the following script:
```
python main.py model_name sfga dataset_name YelpChi_fraud
```


