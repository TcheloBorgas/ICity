# Sistema de Detecção de Acidentes de Trânsito


## Descrição do Projeto
Este projeto é uma implementação de um sistema de detecção de acidentes de trânsito que usa aprendizado de máquina. Ele utiliza um modelo de rede neural convolucional (CNN) treinado para identificar acidentes em vídeos de trânsito.


# Características


## Rede neural convolucional (CNN): O sistema utiliza uma CNN treinada para detectar acidentes em vídeos. A CNN analisa cada frame do vídeo para identificar possíveis acidentes. O modelo de CNN usado é baseado no VGG16, com camadas adicionais específicas para a tarefa de detecção de acidentes.


## Flask API: A implementação usa uma API Flask que recebe requisições contendo um vídeo, analisa esse vídeo usando a CNN, e retorna uma resposta indicando se um acidente foi detectado.

## Amazon SNS: Quando um acidente é detectado, o sistema usa Amazon SNS para enviar uma mensagem de texto alertando sobre o acidente.


## Processamento de vídeo: O vídeo recebido é processado frame a frame. Cada frame é preparado e alimentado na CNN ##para predição. Se a predição ultrapassa um limiar pré-definido, um acidente é considerado detectado.


# Uso
## Pré-requisitos
### Você precisa ter Python e Flask instalados em seu computador. Além disso, você precisará configurar uma conta na AWS para usar o Amazon SNS.


# Instalação
Clone este repositório em sua máquina local.
bash
Copy code
git 
clone
 https://github.com/TcheloBorgas/Icity.git
Entre na pasta do projeto.
bash
Copy code
cd
 Icity
Instale as dependências.
Copy code
pip install -r requirements.txt

## Execute a aplicação.

Copy code
python app.py

## Envie uma requisição POST para a API com o vídeo que você deseja analisar.
Contribuição
Contribuições são bem-vindas! Para contribuir:


# Fork este repositório.
Crie uma nova branch com suas alterações: git checkout -b my-feature
Commit suas alterações: git commit -m 'Add some feature'
Push para a branch: git push origin my-feature
Abra um Pull Request.


# Licença
Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.