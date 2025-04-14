# Integrantes

Matheus Perestrelo - 93260
Lucas Santana de Paula - 95338
José Victor - 95633
Gabriel Toledo - 93152
Guilherme Tomé - 94186

# Vídeo Funcionamento

https://share.cleanshot.com/tJm6ydYH


# Ideia por trás do projeto

Traffic Sense

Utilizando a visão computacional podemos auxiliar na engenheria de transito, podendo mapear aréas com maior tráfego afim de melhorar o fluxo de automoveis nas grandes capitais.

# Detector de Objetos com YOLOv8

Este projeto implementa um sistema de detecção de objetos em tempo real usando YOLOv8, capaz de identificar e classificar objetos tanto em vídeos quanto através da webcam.

## Funcionalidades

- Detecção de objetos em tempo real
- Suporte para webcam e arquivos de vídeo
- Interface em português
- Opções de otimização de performance
- Suporte a GPU (quando disponível)

## Pré-requisitos

- Python 3.8 ou superior
- OpenCV
- PyTorch
- Ultralytics YOLO

## Instalação

1. Clone o repositório:
```bash
git clone <url-do-repositorio>
cd checkpoint2-ia
```

2. Crie e ative um ambiente virtual:
```bash
# No Windows
python -m venv venv
venv\Scripts\activate

# No Linux/MacOS
python3 -m venv venv
source venv/bin/activate
```

3. Instale as dependências do projeto:
```bash
pip install -r requirements.txt
```

## Como Usar

1. Execute o script principal:
```bash
python3 object_detection.py
```

2. Escolha o modo de entrada:
   - Opção 1: Usar webcam
   - Opção 2: Processar arquivo de vídeo
     - Ao escolher esta opção, você deverá informar o caminho completo do arquivo de vídeo
     - Exemplo: /Users/videos/exemplo.mp4 ou C:\Videos\exemplo.mp4

3. Para sair do programa:
   - Pressione 'q' durante a execução

## Configurações

Ao usar um arquivo de vídeo, você pode ajustar:
- Fator de redimensionamento (0.1-1.0)
- Número de frames para pular (1-10)
- Uso de GPU

## Observações

- O modelo YOLOv8 será baixado automaticamente no primeiro uso
- As detecções são mostradas em tempo real com caixas delimitadoras e rótulos em português
