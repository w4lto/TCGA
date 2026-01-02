import numpy as np
import tensorflow as tf
from typing import Tuple, Optional
import cv2


class RandStainNA:
    """
    Random Stain Normalization and Augmentation.
    Aplica transformações aleatórias no espaço LAB de cores para simular
    variações de staining entre diferentes scanners e laboratórios.
    """
    
    def __init__(
        self,
        yaml_file: Optional[str] = None,
        std_hyper: float = 0.0,
        probability: float = 1.0,
        distribution: str = "normal",
        is_train: bool = True
    ):
        self.std_hyper = std_hyper
        self.probability = probability
        self.distribution = distribution
        self.is_train = is_train
        
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica stain augmentation à imagem.
        
        Args:
            image: Imagem RGB uint8 [H, W, 3] com valores [0, 255]
        
        Returns:
            Imagem augmentada RGB uint8 [H, W, 3]
        """
        if not self.is_train or np.random.rand() > self.probability:
            return image
        
        # Converter para float [0, 1]
        img = image.astype(np.float32) / 255.0
        
        # Converter RGB para LAB
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        
        # Separar canais
        L, A, B = cv2.split(img_lab)
        
        # Aplicar perturbações nos canais A e B (cor)
        # Canal L (luminosidade) é menos perturbado para preservar estrutura
        if self.distribution == "normal":
            # Perturbação gaussiana
            delta_A = np.random.normal(0, self.std_hyper * 10)
            delta_B = np.random.normal(0, self.std_hyper * 10)
            scale_A = np.random.normal(1, self.std_hyper * 0.3)
            scale_B = np.random.normal(1, self.std_hyper * 0.3)
        else:
            # Perturbação uniforme
            delta_A = np.random.uniform(-self.std_hyper * 10, self.std_hyper * 10)
            delta_B = np.random.uniform(-self.std_hyper * 10, self.std_hyper * 10)
            scale_A = np.random.uniform(1 - self.std_hyper * 0.3, 1 + self.std_hyper * 0.3)
            scale_B = np.random.uniform(1 - self.std_hyper * 0.3, 1 + self.std_hyper * 0.3)
        
        # Aplicar transformações
        A = A * scale_A + delta_A
        B = B * scale_B + delta_B
        
        # Clip para range válido LAB
        A = np.clip(A, 0, 255)
        B = np.clip(B, 0, 255)
        
        # Recombinar canais
        img_lab = cv2.merge([L, A, B])
        
        # Converter de volta para RGB
        img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        
        # Clip e converter para uint8
        img_rgb = np.clip(img_rgb * 255, 0, 255).astype(np.uint8)
        
        return img_rgb


def tf_randstaina(
    image: tf.Tensor,
    std_hyper: float = 0.4,
    probability: float = 0.8
) -> tf.Tensor:
    """
    Wrapper TensorFlow-friendly para RandStainNA.
    
    Args:
        image: Tensor TF [H, W, 3] com valores [0, 1] (float32)
        std_hyper: Intensidade da augmentation (0.3-0.5 recomendado)
        probability: Probabilidade de aplicar (0.8 recomendado)
    
    Returns:
        Tensor augmentado [H, W, 3] com valores [0, 1]
    """
    
    def _apply_randstaina(img_tensor):
        # Converter para numpy uint8
        img_np = (img_tensor.numpy() * 255).astype(np.uint8)
        
        # Aplicar RandStainNA
        augmenter = RandStainNA(
            std_hyper=std_hyper,
            probability=probability,
            is_train=True
        )
        img_augmented = augmenter(img_np)
        
        # Converter de volta para float [0, 1]
        return img_augmented.astype(np.float32) / 255.0
    
    # Aplicar via py_function
    img_augmented = tf.py_function(
        _apply_randstaina,
        [image],
        tf.float32
    )
    
    # Restaurar shape
    img_augmented.set_shape(image.shape)
    
    return img_augmented


def mixup_batch(
    images: tf.Tensor,
    labels: tf.Tensor,
    alpha: float = 0.2
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Aplica Mixup INTRA-BATCH (recomendado para histopatologia).
    
    Mixup cria exemplos sintéticos interpolando pares de imagens:
        mixed_image = lambda * image_i + (1 - lambda) * image_j
        mixed_label = lambda * label_i + (1 - lambda) * label_j
    
    Referência:
    - mixup: Beyond Empirical Risk Minimization (ICLR 2018)
    - Aplicado em histopatologia: MICCAI 2024 workshop
    
    Args:
        images: Batch de imagens [B, H, W, 3]
        labels: Labels one-hot [B, num_classes] ou sparse [B]
        alpha: Parâmetro da distribuição Beta. Valores típicos:
               - 0.2: Mixup conservador (recomendado para histopatologia)
               - 0.4: Mixup moderado
               - 1.0: Mixup agressivo
    
    Returns:
        (mixed_images, mixed_labels)
    """
    batch_size = tf.shape(images)[0]
    
    # Sample lambda from Beta distribution
    lambda_dist = tf.compat.v1.distributions.Beta(alpha, alpha)
    lambda_value = lambda_dist.sample()
    
    # Garante que lambda está em [0, 1]
    lambda_value = tf.clip_by_value(lambda_value, 0.0, 1.0)
    
    # Reshape lambda para broadcasting
    lambda_img = tf.reshape(lambda_value, [1, 1, 1, 1])
    lambda_lbl = tf.reshape(lambda_value, [1, 1])
    
    # Shuffle indices para criar pares aleatórios
    indices = tf.random.shuffle(tf.range(batch_size))
    
    # Pegar imagens e labels shuffled
    images_shuffled = tf.gather(images, indices)
    labels_shuffled = tf.gather(labels, indices)
    
    # Converter labels para one-hot se necessário
    if len(labels.shape) == 1:
        num_classes = tf.reduce_max(labels) + 1
        labels = tf.one_hot(labels, num_classes)
        labels_shuffled = tf.one_hot(labels_shuffled, num_classes)
    
    # Aplicar mixup
    mixed_images = lambda_img * images + (1.0 - lambda_img) * images_shuffled
    mixed_labels = lambda_lbl * tf.cast(labels, tf.float32) + \
                   (1.0 - lambda_lbl) * tf.cast(labels_shuffled, tf.float32)
    
    return mixed_images, mixed_labels


def cutmix_batch(
    images: tf.Tensor,
    labels: tf.Tensor,
    alpha: float = 1.0
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Aplica CutMix: recorta região de uma imagem e cola em outra.
    
    CutMix é particularmente efetivo em histopatologia pois preserva
    features locais enquanto aumenta diversidade.
    
    Referência:
    - CutMix: Regularization Strategy to Train Strong Classifiers (ICCV 2019)
    
    Args:
        images: Batch de imagens [B, H, W, 3]
        labels: Labels one-hot ou sparse
        alpha: Parâmetro Beta (1.0 recomendado)
    
    Returns:
        (cutmix_images, cutmix_labels)
    """
    batch_size = tf.shape(images)[0]
    image_height = tf.shape(images)[1]
    image_width = tf.shape(images)[2]
    
    # Sample lambda from Beta distribution
    lambda_dist = tf.compat.v1.distributions.Beta(alpha, alpha)
    lambda_value = lambda_dist.sample()
    
    # Calcular tamanho do box
    cut_rat = tf.sqrt(1.0 - lambda_value)
    cut_h = tf.cast(tf.cast(image_height, tf.float32) * cut_rat, tf.int32)
    cut_w = tf.cast(tf.cast(image_width, tf.float32) * cut_rat, tf.int32)
    
    # Centro aleatório do box
    cx = tf.random.uniform([], 0, image_width, dtype=tf.int32)
    cy = tf.random.uniform([], 0, image_height, dtype=tf.int32)
    
    # Coordenadas do box
    x1 = tf.clip_by_value(cx - cut_w // 2, 0, image_width)
    y1 = tf.clip_by_value(cy - cut_h // 2, 0, image_height)
    x2 = tf.clip_by_value(cx + cut_w // 2, 0, image_width)
    y2 = tf.clip_by_value(cy + cut_h // 2, 0, image_height)
    
    # Shuffle para pegar imagens de mistura
    indices = tf.random.shuffle(tf.range(batch_size))
    images_shuffled = tf.gather(images, indices)
    labels_shuffled = tf.gather(labels, indices)
    
    # Converter labels para one-hot se necessário
    if len(labels.shape) == 1:
        num_classes = tf.reduce_max(labels) + 1
        labels = tf.one_hot(labels, num_classes)
        labels_shuffled = tf.one_hot(labels_shuffled, num_classes)
    
    # Criar máscara
    mask = tf.ones_like(images)
    mask = tf.tensor_scatter_nd_update(
        mask,
        [[i, j, k, l] for i in range(batch_size) 
                      for j in range(y1, y2) 
                      for k in range(x1, x2) 
                      for l in range(3)],
        tf.zeros((batch_size, y2-y1, x2-x1, 3))
    )
    
    # Aplicar CutMix
    cutmix_images = images * mask + images_shuffled * (1.0 - mask)
    
    # Calcular lambda real baseado na área
    lambda_real = 1.0 - tf.cast((x2 - x1) * (y2 - y1), tf.float32) / \
                        tf.cast(image_height * image_width, tf.float32)
    
    cutmix_labels = lambda_real * tf.cast(labels, tf.float32) + \
                    (1.0 - lambda_real) * tf.cast(labels_shuffled, tf.float32)
    
    return cutmix_images, cutmix_labels


# Wrapper para uso fácil no pipeline
def apply_stain_augmentation(
    image: tf.Tensor,
    mode: str = "randstaina",
    probability: float = 0.8,
    intensity: float = 0.4
) -> tf.Tensor:
    """
    Aplica stain augmentation com configurações otimizadas para histopatologia.
    
    Args:
        image: Tensor [H, W, 3] com valores [0, 1]
        mode: "randstaina", "none"
        probability: Probabilidade de aplicar
        intensity: Intensidade (0.0-1.0)
    
    Returns:
        Imagem augmentada
    """
    if mode == "none":
        return image
    
    if mode == "randstaina":
        # Aplicar com probabilidade
        should_apply = tf.random.uniform([]) < probability
        return tf.cond(
            should_apply,
            lambda: tf_randstaina(image, std_hyper=intensity, probability=1.0),
            lambda: image
        )
    
    return image