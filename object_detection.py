import cv2
import numpy as np
import os
import torch
from ultralytics import YOLO
import warnings

# Suprimir FutureWarnings relacionados ao torch.cuda.amp.autocast
warnings.filterwarnings('ignore', category=FutureWarning)

class ObjectDetector:
    def __init__(self, model_path):
        """
        Inicializa o detector de objetos com o modelo YOLOv8
        
        :param model_path: Caminho para o modelo YOLOv8
        """
        # Carregar modelo YOLOv8 usando a biblioteca Ultralytics
        self.model = YOLO(model_path)
        
        # Configurações de detecção
        self.confidence_threshold = 0.5
        
        # Dicionário para tradução de classes comuns (inglês -> português)
        self.translations = {
            'person': 'pessoa',
            'bicycle': 'bicicleta',
            'car': 'carro',
            'motorcycle': 'moto',
            'airplane': 'avião',
            'bus': 'ônibus',
            'train': 'trem',
            'truck': 'caminhão',
            'boat': 'barco',
            'traffic light': 'semáforo',
            'fire hydrant': 'hidrante',
            'stop sign': 'placa de pare',
            'parking meter': 'parquímetro',
            'bench': 'banco',
            'bird': 'pássaro',
            'cat': 'gato',
            'dog': 'cachorro',
            'horse': 'cavalo',
            'sheep': 'ovelha',
            'cow': 'vaca',
            'elephant': 'elefante',
            'bear': 'urso',
            'zebra': 'zebra',
            'giraffe': 'girafa',
            'backpack': 'mochila',
            'umbrella': 'guarda-chuva',
            'handbag': 'bolsa',
            'tie': 'gravata',
            'suitcase': 'mala',
            'bottle': 'garrafa',
            'wine glass': 'taça de vinho',
            'cup': 'xícara',
            'fork': 'garfo',
            'knife': 'faca',
            'spoon': 'colher',
            'bowl': 'tigela',
            'banana': 'banana',
            'apple': 'maçã',
            'sandwich': 'sanduíche',
            'orange': 'laranja',
            'broccoli': 'brócolis',
            'carrot': 'cenoura',
            'hot dog': 'cachorro-quente',
            'pizza': 'pizza',
            'donut': 'rosquinha',
            'cake': 'bolo',
            'chair': 'cadeira',
            'couch': 'sofá',
            'potted plant': 'planta',
            'bed': 'cama',
            'dining table': 'mesa de jantar',
            'toilet': 'vaso sanitário',
            'tv': 'televisão',
            'laptop': 'laptop',
            'mouse': 'mouse',
            'remote': 'controle remoto',
            'keyboard': 'teclado',
            'cell phone': 'celular',
            'microwave': 'micro-ondas',
            'oven': 'forno',
            'toaster': 'torradeira',
            'sink': 'pia',
            'refrigerator': 'geladeira',
            'book': 'livro',
            'clock': 'relógio',
            'vase': 'vaso',
            'scissors': 'tesoura',
            'teddy bear': 'ursinho de pelúcia',
            'hair drier': 'secador de cabelo',
            'toothbrush': 'escova de dentes'
        }

    def detect_objects(self, image):
        """
        Detecta objetos na imagem usando YOLOv8
        
        :param image: Imagem de entrada para detecção
        :return: Lista de objetos detectados
        """
        # Executar a detecção com YOLOv8
        results = self.model(image, conf=self.confidence_threshold)
        
        # Processamento final das detecções
        detected_objects = []
        
        # Extrair informações das detecções
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extrair coordenadas do retângulo
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                
                # Extrair classe e confiança
                class_id = int(box.cls[0].item())
                label = result.names[class_id]
                confidence = float(box.conf[0].item())
                
                # Traduzir nome da classe se disponível
                translated_label = self.translations.get(label.lower(), label)
                
                detected_objects.append({
                    'label': translated_label,
                    'original_label': label,
                    'confidence': confidence,
                    'bbox': (x, y, w, h)
                })
        
        return detected_objects
    
    def draw_detections(self, image, detections):
        """
        Desenha caixas de detecção na imagem
        
        :param image: Imagem original
        :param detections: Lista de objetos detectados
        :return: Imagem com detecções
        """
        for obj in detections:
            x, y, w, h = obj['bbox']
            label = obj['label']
            confidence = obj['confidence']
            
            # Cores mais vibrantes para melhor visibilidade (azul)
            box_color = (255, 0, 0)  # Azul em BGR
            text_color = (255, 255, 255)  # Branco em BGR
            
            # Desenhar retângulo com linha mais espessa
            cv2.rectangle(image, (x, y), (x + w, y + h), box_color, 3)
            
            # Texto da classe e confiança com fonte maior
            text = f"{label}: {confidence:.2f}"
            
            # Configurações de texto
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_thickness = 2
            
            # Obter tamanho do texto para criar o fundo
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            
            # Desenhar retângulo de fundo para o texto
            cv2.rectangle(image, 
                         (x, y - text_height - 10), 
                         (x + text_width, y), 
                         box_color, 
                         -1)  # -1 para preencher o retângulo
            
            # Posicionar o texto sobre o fundo
            cv2.putText(image, text, (x, y - 5), 
                        font, font_scale, text_color, font_thickness)
        
        return image

def process_video_file(detector, video_path, resize_factor=0.5, skip_frames=2, use_gpu=True):
    """
    Processa um arquivo de vídeo e exibe as detecções
    
    :param detector: Instância do detector de objetos
    :param video_path: Caminho para o arquivo de vídeo
    :param resize_factor: Fator para redimensionar o frame (0.5 = metade do tamanho)
    :param skip_frames: Número de frames para pular (1 = processar todos, 2 = processar um a cada 2)
    :param use_gpu: Tentar usar GPU para aceleração se disponível
    """
    # Configurar para usar GPU se disponível e solicitado
    if use_gpu and torch.cuda.is_available():
        print("Usando GPU para aceleração")
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        if use_gpu:
            print("GPU não disponível, usando CPU")
        else:
            print("Usando CPU para processamento")
    
    # Abrir o arquivo de vídeo
    cap = cv2.VideoCapture(video_path)
    
    # Verificar se o vídeo foi aberto corretamente
    if not cap.isOpened():
        print(f"Erro ao abrir o arquivo de vídeo: {video_path}")
        return
    
    # Obter informações do vídeo
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calcular novas dimensões
    new_width = int(frame_width * resize_factor)
    new_height = int(frame_height * resize_factor)
    
    print(f"Processando vídeo: {video_path}")
    print(f"Resolução original: {frame_width}x{frame_height}, FPS: {fps}")
    print(f"Resolução reduzida: {new_width}x{new_height}")
    print(f"Processando 1 a cada {skip_frames} frames")
    
    # Contador de frames
    frame_count = 0
    
    # Para calcular FPS do processamento
    start_time = cv2.getTickCount()
    processed_frames = 0
    
    while True:
        # Ler o próximo frame
        ret, frame = cap.read()
        
        # Verificar se chegamos ao final do vídeo
        if not ret:
            print("Fim do vídeo.")
            break
        
        # Incrementar contador de frames
        frame_count += 1
        
        # Pular frames para acelerar processamento
        if frame_count % skip_frames != 0:
            continue
        
        # Redimensionar o frame para acelerar o processamento
        if resize_factor != 1.0:
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Detectar objetos no frame
        detections = detector.detect_objects(frame)
        
        # Desenhar as detecções no frame
        frame_with_detections = detector.draw_detections(frame, detections)
        
        # Calcular e mostrar FPS de processamento
        processed_frames += 1
        if processed_frames % 10 == 0:
            current_time = cv2.getTickCount()
            elapsed_time = (current_time - start_time) / cv2.getTickFrequency()
            processing_fps = processed_frames / elapsed_time
            cv2.putText(frame_with_detections, 
                       f"FPS: {processing_fps:.1f}", 
                       (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       1, 
                       (0, 255, 0), 
                       2)
        
        # Mostrar o frame com as detecções
        cv2.imshow('Detecção de Objetos - Vídeo', frame_with_detections)
        
        # Controlar a velocidade de exibição para aproximar do FPS original
        # Usar um valor pequeno para waitKey para manter a responsividade
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

def process_webcam(detector):
    """
    Processa o vídeo da webcam e exibe as detecções
    
    :param detector: Instância do detector de objetos
    """
    # Capturar vídeo da webcam
    cap = cv2.VideoCapture(0)
    
    # Verificar se a webcam foi aberta corretamente
    if not cap.isOpened():
        print("Erro ao abrir a webcam.")
        return
    
    while True:
        # Ler o próximo frame
        ret, frame = cap.read()
        
        # Verificar se o frame foi lido corretamente
        if not ret:
            print("Erro ao capturar frame da webcam.")
            break
        
        # Detectar objetos no frame
        detections = detector.detect_objects(frame)
        
        # Desenhar as detecções no frame
        frame_with_detections = detector.draw_detections(frame, detections)
        
        # Mostrar o frame com as detecções
        cv2.imshow('Detecção de Objetos - Webcam', frame_with_detections)
        
        # Sair se a tecla 'q' for pressionada
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Caminho para o modelo YOLOv8
    model_path = 'yolo12x.pt'  # Modelo nano (mais leve)
    
    # Verificar se o modelo existe, caso contrário, baixá-lo
    if not os.path.exists(model_path):
        print(f"Modelo {model_path} não encontrado. Baixando...")
        try:
            # Baixar o modelo YOLOv8 nano
            model = YOLO('yolo12x.pt')
            print(f"Modelo baixado com sucesso para {model_path}")
        except Exception as e:
            print(f"Erro ao baixar o modelo: {e}")
            return
    
    # Inicializar detector
    detector = ObjectDetector(model_path)
    
    # Perguntar ao usuário se deseja usar webcam ou arquivo de vídeo
    print("Escolha o modo de entrada:")
    print("1. Webcam")
    print("2. Arquivo de vídeo")
    
    choice = input("Digite sua escolha (1 ou 2): ")
    
    if choice == '1':
        # Usar webcam
        process_webcam(detector)
    elif choice == '2':
        # Usar arquivo de vídeo
        video_path = input("Digite o caminho completo para o arquivo de vídeo: ")
        if os.path.exists(video_path):
            # Configurações de otimização
            print("\nConfigurações de otimização para processamento em tempo real:")
            print("Valores recomendados: resize_factor=0.5, skip_frames=2")
            
            try:
                resize_factor = float(input("Fator de redimensionamento (0.1-1.0, menor = mais rápido): ") or 0.5)
                skip_frames = int(input("Pular frames (1-10, maior = mais rápido): ") or 2)
                use_gpu = input("Usar GPU se disponível? (s/n): ").lower() in ['s', 'sim', 'y', 'yes', '']
                
                # Validar entradas
                resize_factor = max(0.1, min(1.0, resize_factor))
                skip_frames = max(1, min(10, skip_frames))
                
                process_video_file(detector, video_path, resize_factor, skip_frames, use_gpu)
            except ValueError:
                print("Valores inválidos. Usando configurações padrão.")
                process_video_file(detector, video_path)
        else:
            print(f"Arquivo não encontrado: {video_path}")
    else:
        print("Escolha inválida. Saindo.")

if __name__ == "__main__":
    main()