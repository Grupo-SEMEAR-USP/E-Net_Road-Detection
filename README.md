# E-Net - Road Detection
 
Implementação da **Rede Neural Convolucional** E-Net (*Efficient Neural Network*) com o intuito de ser utilizada na tarefa de detecção de região navegável por robô autônomo. <br/>
Propondo uma grande redução de parâmetros com o intuito de ser utilizado em dispostivos embarcados, esse repositório possui com intuito reproduzir o modelo e propor modificações a fim de ser implementado em um robô autônomo utilizando de hardware de relativo baixo custo.

## Como Usar

O modelo está divido entre os arquivos .py na pasta implementation, para treinar o modelo é só acessar o arquivo jupyter notebook *E_net_train.ipynb* e seguir as instruções que indicam os parâmetros utilizados e depois executar o bloco que contém o algoritmo de treino. Para o teste, equivale o mesmo para o arquivo E_net_test.ipynb

## Referências
1. A. Paszke, A. Chaurasia, S. Kim, and E. Culurciello.
Enet: A deep neural network architecture
for real-time semantic segmentation. arXiv preprint
arXiv:1606.02147, 2016.

