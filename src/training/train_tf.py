from __future__ import annotations

import cv2


import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Silencia avisos do TensorFlow (CUDA/XLA)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'  # Desabilita XLA auto-jit

import argparse
import random
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from src.utils.config_utils import load_config, TrainConfig
from src.utils.logging_utils import logger
from src.utils.mlflow_utils import init_mlflow, start_run, log_metrics
from src.utils.tf_device_utils import setup_tf_device

from src.data.tf_dataset import (
    create_tf_datasets_from_clickhouse
)
from src.models.kimianet_backbone_tf import build_kimianet_backbone_tf
from src.models.resnet_backbone_tf import build_resnet_backbone_tf
from src.models.ensemble_tf import build_ensemble_model_tf
from src.data.ch_utils import (
    ClickHouseClient    
)
from src.training.metrics_utils import (
    compute_slide_metrics_multiclass,
    aggregate_by_slide,
)

"""
Sistema de fine-tuning progressivo integrado.
Segue princípios de MLOps: single pipeline, config-driven, reproduzível.
"""
from enum import Enum
from typing import List, Tuple, Dict, Any
import os


class TrainingStrategy(Enum):
    """Estratégias de treinamento disponíveis."""
    LINEAR_PROBING = "linear_probing"      # Apenas classificador
    PROGRESSIVE = "progressive"             # Fine-tuning progressivo (3 fases)
    FULL_FINETUNING = "full_finetuning"    # 50% das camadas de uma vez
    STANDARD = "standard"                   # Compatibilidade com código antigo

def get_all_trainable_layers_recursive(model: keras.Model) -> List[keras.layers.Layer]:
    all_layers = []
    
    def _recursive_get_layers(layer_or_model):
        """Função recursiva para percorrer hierarquia."""
        # Se for um Model, descer para suas camadas
        if isinstance(layer_or_model, keras.Model):
            for sublayer in layer_or_model.layers:
                _recursive_get_layers(sublayer)
        # Se for uma camada treinável (Conv, Dense, etc), adicionar
        elif hasattr(layer_or_model, 'trainable') and hasattr(layer_or_model, 'weights'):
            # Filtrar apenas camadas com pesos (Conv2D, Dense, etc)
            trainable_types = (
                keras.layers.Conv2D,
                keras.layers.Dense,
                keras.layers.DepthwiseConv2D,
                keras.layers.Conv1D,
                keras.layers.DepthwiseConv1D
            )
            if isinstance(layer_or_model, trainable_types):
                all_layers.append(layer_or_model)
    
    _recursive_get_layers(model)
    return all_layers


def apply_unfreezing_strategy(
    model: keras.Model,
    strategy: TrainingStrategy,
    phase: int = 1
) -> Tuple[keras.Model, int]:
    logger.info(f"Aplicando estratégia: {strategy.value}, fase: {phase}")
    
    # Identificar backbones do ensemble
    backbones = []
    for layer in model.layers:
        if isinstance(layer, keras.Model) and layer.name in ['kimianet_tf', 'resnet_tf']:
            backbones.append(layer)
    
    if not backbones:
        logger.warning("Nenhum backbone identificado. Aplicando ao modelo completo.")
        backbones = [model]
    
    # Aplicar estratégia a cada backbone
    for backbone in backbones:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processando backbone: {backbone.name}")
        logger.info(f"{'='*60}")
        
        # PASSO 1: Congelar TUDO primeiro
        for layer in backbone.layers:
            layer.trainable = False
        
        # PASSO 2: Buscar recursivamente TODAS as camadas treináveis
        all_trainable_layers = get_all_trainable_layers_recursive(backbone)
        
        logger.info(
            f"Camadas treináveis encontradas recursivamente: "
            f"{len(all_trainable_layers)}"
        )
        
        # Debug: Mostrar primeiras e últimas 5 camadas
        if len(all_trainable_layers) > 10:
            logger.info("Primeiras 5 camadas:")
            for i, layer in enumerate(all_trainable_layers[:5]):
                logger.info(f"  [{i}] {layer.name} ({layer.__class__.__name__})")
            logger.info("...")
            logger.info("Últimas 5 camadas:")
            for i, layer in enumerate(all_trainable_layers[-5:], start=len(all_trainable_layers)-5):
                logger.info(f"  [{i}] {layer.name} ({layer.__class__.__name__})")
        else:
            for i, layer in enumerate(all_trainable_layers):
                logger.info(f"  [{i}] {layer.name} ({layer.__class__.__name__})")
        
        # PASSO 3: Aplicar estratégia de descongelamento
        if len(all_trainable_layers) == 0:
            logger.error(f"ERRO: Nenhuma camada treinável encontrada em {backbone.name}!")
            continue
        
        if strategy == TrainingStrategy.LINEAR_PROBING:
            # Apenas última camada Dense
            logger.info(f"Estratégia: Linear Probing - Apenas classificador")
            for layer in reversed(all_trainable_layers):
                if isinstance(layer, keras.layers.Dense):
                    layer.trainable = True
                    logger.info(f"  ✓ Descongelada: {layer.name}")
                    break
                    
        elif strategy == TrainingStrategy.FULL_FINETUNING:
            layers_to_unfreeze = min(15, max(3, len(all_trainable_layers) // 4))
            logger.info(
                f"Estratégia: Full Fine-tuning - "
                f"Descongelando {layers_to_unfreeze}/{len(all_trainable_layers)} camadas"
            )
            
            unfrozen = []
            for layer in reversed(all_trainable_layers):
                if len(unfrozen) >= layers_to_unfreeze:
                    break
                if not isinstance(layer, keras.layers.BatchNormalization):
                    layer.trainable = True
                    unfrozen.append(layer.name)
            
            logger.info(f"  ✓ Descongeladas {len(unfrozen)} camadas:")
            for name in unfrozen[:3]:
                logger.info(f"    - {name}")
            if len(unfrozen) > 3:
                logger.info(f"    ... (+{len(unfrozen)-3} camadas)")
                    
        elif strategy == TrainingStrategy.PROGRESSIVE:
            if phase == 1:
                # Fase 1: Apenas classificador
                logger.info(f"Fase 1: Linear Probing - Apenas classificador")
                for layer in reversed(all_trainable_layers):
                    if isinstance(layer, keras.layers.Dense):
                        layer.trainable = True
                        logger.info(f"  ✓ Descongelada: {layer.name}")
                        break
                        
            elif phase == 2:
                # Fase 2: Últimas 5 camadas
                layers_to_unfreeze = min(5, max(2, len(all_trainable_layers) // 10))
                logger.info(
                    f"Fase 2: Descongelando {layers_to_unfreeze}/{len(all_trainable_layers)} camadas"
                )
                
                unfrozen = []
                for layer in reversed(all_trainable_layers):
                    if len(unfrozen) >= layers_to_unfreeze:
                        break
                    if not isinstance(layer, keras.layers.BatchNormalization):
                        layer.trainable = True
                        unfrozen.append(layer.name)
                
                for name in unfrozen:
                    logger.info(f"  ✓ Descongelada: {name}")
                    
            elif phase == 3:
                layers_to_unfreeze = min(8, max(3, len(all_trainable_layers) // 7))
                logger.info(
                    f"Fase 3: Descongelando {layers_to_unfreeze}/{len(all_trainable_layers)} camadas"
                )
                
                unfrozen = []
                for layer in reversed(all_trainable_layers):
                    if len(unfrozen) >= layers_to_unfreeze:
                        break
                    if not isinstance(layer, keras.layers.BatchNormalization):
                        layer.trainable = True
                        unfrozen.append(layer.name)
                
                for name in unfrozen:
                    logger.info(f"  ✓ Descongelada: {name}")
        
        else:  # STANDARD
            logger.info(f"Estratégia: Standard - Mantendo configuração atual")
    
    # Contar parâmetros treináveis
    trainable_count = sum([tf.size(var).numpy() for var in model.trainable_variables])
    non_trainable_count = sum([tf.size(var).numpy() for var in model.non_trainable_variables])
    total_count = trainable_count + non_trainable_count
    
    logger.info(f"\n{'='*60}")
    logger.info(f"RESUMO DE PARÂMETROS:")
    logger.info(f"  Treináveis: {trainable_count:,} ({trainable_count/total_count*100:.2f}%)")
    logger.info(f"  Congelados: {non_trainable_count:,} ({non_trainable_count/total_count*100:.2f}%)")
    logger.info(f"{'='*60}\n")
    
    return model, trainable_count

def get_phase_config(strategy: TrainingStrategy, phase: int = 1) -> Dict[str, Any]:
    """
    Retorna configuração de hiperparâmetros OTIMIZADA por estratégia/fase.
    
    AJUSTE: Reduzir épocas e LR da Fase 3 para prevenir overfitting.
    """
    configs = {
        TrainingStrategy.LINEAR_PROBING: {
            "learning_rate": 1e-3,
            "epochs": 5,
            "patience": 3
        },
        TrainingStrategy.FULL_FINETUNING: {
            "learning_rate": 1e-4,
            "epochs": 15,  # REDUZIDO de 20
            "patience": 5
        },
        TrainingStrategy.PROGRESSIVE: {
            1: {
                "learning_rate": 1e-3,
                "epochs": 5,
                "patience": 3
            },
            2: {
                "learning_rate": 1e-4,
                "epochs": 25,
                "patience": 8
            },
            3: {
                "learning_rate": 2e-5,  
                "epochs": 8,            
                "patience": 4,          
                "dropout": 0.5          
            }
        },
        TrainingStrategy.STANDARD: {
            "learning_rate": 5e-5,
            "epochs": 20,
            "patience": 5
        }
    }
    
    if strategy == TrainingStrategy.PROGRESSIVE:
        return configs[strategy][phase]
    return configs[strategy]


def apply_randstaina_tf(image: tf.Tensor, std_hyper: float = 0.4) -> tf.Tensor:
    """
    RandStainNA: Random Stain Normalization and Augmentation.
    
    Simula variações de coloração histológica entre diferentes
    scanners e laboratórios para melhorar generalização.
    
    Referência: ICCV 2023, validado em TCGA-BRCA (+12.83% improvement)
    """
    def _randstaina_numpy(img):
        # Converter para numpy uint8
        img_np = (img.numpy() * 255).astype(np.uint8)
        
        # Converter RGB para LAB
        img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Separar canais
        L, A, B = cv2.split(img_lab)
        
        # Aplicar perturbações nos canais de cor (A e B)
        delta_A = np.random.normal(0, std_hyper * 10)
        delta_B = np.random.normal(0, std_hyper * 10)
        scale_A = np.random.normal(1, std_hyper * 0.3)
        scale_B = np.random.normal(1, std_hyper * 0.3)
        
        A = np.clip(A * scale_A + delta_A, 0, 255)
        B = np.clip(B * scale_B + delta_B, 0, 255)
        
        # Recombinar e converter de volta para RGB
        img_lab = cv2.merge([L, A, B]).astype(np.uint8)
        img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        
        return (img_rgb.astype(np.float32) / 255.0)
    
    # Aplicar via py_function
    img_augmented = tf.py_function(_randstaina_numpy, [image], tf.float32)
    img_augmented.set_shape(image.shape)
    
    return img_augmented

def add_data_augmentation(
    train_ds: tf.data.Dataset,
    enable_stain_aug: bool = True,
    stain_intensity: float = 0.4,
    stain_probability: float = 0.8
) -> tf.data.Dataset:
    """
    Data augmentation AVANÇADO para histopatologia.
    
    Pipeline otimizado baseado em MICCAI 2024.
    """
    def augment(image, label):
        # CRÍTICO: Garantir que imagem tem shape correto [H, W, 3]
        original_shape = tf.shape(image)
        
        # Verificar se tem 3 canais
        if len(image.shape) == 3 and image.shape[-1] != 3:
            # Se tiver 1 canal, converter para 3
            image = tf.image.grayscale_to_rgb(image)
        elif len(image.shape) == 2:
            # Se for [H, W], adicionar dimensão de canal
            image = tf.expand_dims(image, axis=-1)
            image = tf.image.grayscale_to_rgb(image)
        
        # 1. GEOMETRIC AUGMENTATIONS
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        
        # Rotação em múltiplos de 90° com validação de shape
        k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        image = tf.image.rot90(image, k=k)
        
        # CRÍTICO: Após rot90, garantir que ainda tem 3 canais
        # rot90 pode alterar ordem das dimensões em alguns casos
        current_shape = tf.shape(image)
        if len(image.shape) == 3:
            # Se shape ficou [C, H, W], converter para [H, W, C]
            if image.shape[0] == 3 or current_shape[0] == 3:
                image = tf.transpose(image, [1, 2, 0])
            # Se shape ficou [H, W, 1], converter para [H, W, 3]
            elif image.shape[-1] == 1 or current_shape[-1] == 1:
                image = tf.image.grayscale_to_rgb(image)
        
        # 2. COLOR AUGMENTATIONS (Sutis)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        image = tf.image.random_hue(image, max_delta=0.02)
        
        # 3. STAIN AUGMENTATION (CRÍTICO)
        if enable_stain_aug:
            # Garantir shape [H, W, 3] antes de aplicar RandStainNA
            if len(image.shape) == 3 and image.shape[-1] == 3:
                should_apply_stain = tf.random.uniform([]) < stain_probability
                image = tf.cond(
                    should_apply_stain,
                    lambda: apply_randstaina_tf(image, std_hyper=stain_intensity),
                    lambda: image
                )
        
        # 4. NORMALIZATION & CLIPPING
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image, label
    
    # Aplicar augmentation
    augmented_ds = train_ds.map(
        augment,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    return augmented_ds


def apply_mixup_to_dataset(
    dataset: tf.data.Dataset,
    alpha: float = 0.2,
    num_classes: int = 4
) -> tf.data.Dataset:
    def mixup(images, labels):
        batch_size = tf.shape(images)[0]
        
        # Sample lambda from Beta(alpha, alpha)
        lambda_value = tf.compat.v1.distributions.Beta(alpha, alpha).sample()
        lambda_value = tf.clip_by_value(lambda_value, 0.0, 1.0)
        
        # Reshape para broadcasting
        lambda_img = tf.reshape(lambda_value, [1, 1, 1, 1])
        lambda_lbl = tf.reshape(lambda_value, [1, 1])
        
        # Shuffle indices
        indices = tf.random.shuffle(tf.range(batch_size))
        images_shuffled = tf.gather(images, indices)
        labels_shuffled = tf.gather(labels, indices)
        
        # Converter labels para one-hot se necessário
        if len(labels.shape) == 1:
            labels = tf.one_hot(labels, num_classes)
            labels_shuffled = tf.one_hot(labels_shuffled, num_classes)
        
        # Aplicar mixup
        mixed_images = lambda_img * images + (1.0 - lambda_img) * images_shuffled
        mixed_labels = lambda_lbl * tf.cast(labels, tf.float32) + \
                       (1.0 - lambda_lbl) * tf.cast(labels_shuffled, tf.float32)
        
        mixed_labels = tf.argmax(mixed_labels, axis=-1)
        
        return mixed_images, mixed_labels
    
    # Aplicar mixup por batch
    return dataset.map(mixup, num_parallel_calls=tf.data.AUTOTUNE)


def train_single_phase(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    phase_config: Dict[str, Any],
    class_weight: Dict,
    output_dir: str,
    phase_name: str,
    enable_augmentation: bool = True,
    enable_mixup: bool = True,  # NOVO
    num_classes: int = 4  # NOVO
) -> keras.callbacks.History:
    """Treina uma única fase com regularização AVANÇADA."""
    
    if enable_augmentation:
        logger.info("Aplicando augmentation avançado (RandStainNA + Geometric)...")
        train_ds = add_data_augmentation(
            train_ds,
            enable_stain_aug=True,
            stain_intensity=0.4,  # Ajustar se necessário (0.3-0.5)
            stain_probability=0.8
        )
    
    alpha_value = 0.2
    
    if enable_mixup:
        logger.info(f"Aplicando Mixup (alpha={alpha_value})...")
        # Mixup deve ser aplicado APÓS batching
        train_ds = apply_mixup_to_dataset(
            train_ds,
            alpha=alpha_value,
            num_classes=num_classes
        )
        
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, f'{phase_name}_best.keras'),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=phase_config['patience'],
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=max(2, phase_config['patience'] - 2),
            min_lr=1e-8,
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            os.path.join(output_dir, f'{phase_name}_history.csv'),
            append=False
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(output_dir, 'tensorboard', phase_name),
            update_freq='epoch'
        )
    ]
    
    logger.info(
        f"Treinando {phase_name}: "
        f"LR={phase_config['learning_rate']:.2e}, "
        f"Épocas={phase_config['epochs']}, "
        f"Patience={phase_config['patience']}, "
        f"Augmentation={enable_augmentation}, "
        f"Mixup={enable_mixup}"
    )
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase_config['epochs'],
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )
    
    return history



def configure_tf_threading():
    """
    Configura threading do TensorFlow para evitar race conditions.
    """
    num_cpus = os.cpu_count() or 8
    tf.config.threading.set_inter_op_parallelism_threads(num_cpus)
    tf.config.threading.set_intra_op_parallelism_threads(num_cpus)
    
    logger.info(f"TensorFlow threading configurado: inter_op={num_cpus}, intra_op={num_cpus}")



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def compute_class_weights_dynamic(cfg: TrainConfig, label_map: Dict[str, int]):
    """
    Calcula pesos de classe dinâmicos baseados na frequência no banco de dados.
    """
    assert cfg.clickhouse is not None
    ch = ClickHouseClient(cfg.clickhouse)
    
    # Query otimizada para contar apenas o necessário
    query = f"SELECT stage_label, count(*) as cnt FROM {cfg.clickhouse.table_patches} WHERE split = 'train' GROUP BY stage_label"
    df = ch.query_df(query)
    
    # Converte para dict {label: count}
    counts = dict(zip(df["stage_label"], df["cnt"]))
    logger.info(f"Base stage count: {counts}")

    num_classes = len(cfg.class_names)
    freqs = np.zeros(num_classes, dtype=float)
    
    # Mapeia contagens para índices inteiros
    for name, index in label_map.items():
        freqs[index] = counts.get(name, 0)

    # Evita divisão por zero
    freqs[freqs == 0] = 1.0 
    
    # Inverso da frequência (balanceamento)
    inv = 1.0 / freqs
    
    inv = np.power(inv, 0.25)
    
    inv = inv / inv.sum()

    # Boost clínico opcional
    if cfg.stage_clinical_weights:
        boost = np.array(
            [cfg.stage_clinical_weights.get(name, 1.0) for name in cfg.class_names],
            dtype=float,
        )
        weights = inv * boost
    else:
        weights = inv

    # Normaliza para que a soma seja 1 (ou escala conforme preferência)
    weights = weights / weights.sum()
    
    return {int(i): float(w) for i, w in enumerate(weights)}

def main(config_path: str) -> None:
    configure_tf_threading()
    cfg = load_config(config_path)
    logger.info(f"Loaded config: {cfg}")
    set_seed(cfg.seed)

    # Configura Hardware
    device_str, device_desc = setup_tf_device()
    logger.info(f"TensorFlow device: {device_desc}")

    # Inicializa MLflow
    tracking_uri = Path(cfg.mlflow_tracking_uri)
    init_mlflow(tracking_uri, cfg.experiment_name)
    

    strategy_name = cfg.training_strategy
    
    try:
        strategy = TrainingStrategy(strategy_name)
    except ValueError:
        logger.warning(
            f"Estratégia '{strategy_name}' inválida. "
            f"Opções: {[s.value for s in TrainingStrategy]}. "
            f"Usando 'progressive' por padrão."
        )
        strategy = TrainingStrategy.PROGRESSIVE
    
    logger.info(f"=" * 80)
    logger.info(f"ESTRATÉGIA DE TREINAMENTO: {strategy.value.upper()}")
    logger.info(f"=" * 80)
    
    params: Dict[str, Any] = {
        "task_type": cfg.task_type,
        "class_names": ",".join(cfg.class_names),
        "batch_size": cfg.batch_size,
        "patch_size": cfg.patch_size,
        "model_type": "Ensemble (KimiaNet + ResNet)" if cfg.use_resnet else "KimiaNet",
        "training_strategy": strategy.value,  # NOVO
        "early_stopping_patience": cfg.early_stopping_patience,
    }
    start_run(cfg.run_name, params=params)

    with tf.device(device_str):
        train_ds, val_ds, test_ds, label_map = create_tf_datasets_from_clickhouse(cfg)
        num_classes = len(cfg.class_names)

        backbones: List[keras.Model] = []
        
        backbone_path = Path(cfg.backbone_kimianet_weights)
        kimianet = build_kimianet_backbone_tf(
            num_classes=num_classes,
            weights_path=backbone_path,
            dropout=cfg.dropout,
        )
        backbones.append(kimianet)

        if cfg.use_resnet:
            resnet = build_resnet_backbone_tf(
                num_classes=num_classes,
                dropout=cfg.dropout,
            )
            backbones.append(resnet)

        # Ensemble (média de logits)
        model = build_ensemble_model_tf(backbones, num_classes=num_classes)
        
        class_weight = None
        if cfg.dynamic_class_weights:
            class_weight = compute_class_weights_dynamic(cfg, label_map)
            logger.info(f"Class weights calculados: {class_weight}")
        
        output_dir = Path(os.path.join(cfg.output_dir, "training"))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        full_history = {}
        
        if strategy == TrainingStrategy.PROGRESSIVE:
            logger.info("=" * 80)
            logger.info("INICIANDO TREINAMENTO PROGRESSIVO (3 FASES)")
            logger.info("=" * 80)
            
            for phase in [1, 2, 3]:
                logger.info(f"\n{'=' * 80}")
                logger.info(f"FASE {phase}/3")
                logger.info(f"{'=' * 80}")
                
                # Aplicar estratégia de unfreezing
                model, trainable_params = apply_unfreezing_strategy(
                    model, strategy, phase=phase
                )
                
                # Obter configuração da fase
                phase_config = get_phase_config(strategy, phase=phase)
                
                # Compilar modelo com novo LR
                loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                optimizer = keras.optimizers.Adam(
                    learning_rate=phase_config["learning_rate"]
                )
                metrics = [keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
                
                model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
                
                if phase == 1:
                    model.summary(print_fn=logger.info)
                
                # Treinar fase
                history = train_single_phase(
                    model=model,
                    train_ds=train_ds,
                    val_ds=val_ds,
                    phase_config=phase_config,
                    class_weight=class_weight,
                    output_dir=str(output_dir),
                    phase_name=f'phase{phase}',
                    enable_mixup=True,
                    num_classes=num_classes,
                    enable_augmentation=True
                )
                
                full_history[f'phase{phase}'] = history.history
                
                # Log de métricas da fase
                best_val_acc = max(history.history['val_accuracy'])
                best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
                
                logger.info(
                    f"Fase {phase} concluída. "
                    f"Melhor val_accuracy: {best_val_acc:.4f} "
                    f"(época {best_epoch})"
                )
                
                # Log no MLflow
                log_metrics({
                    f"phase{phase}_best_val_accuracy": best_val_acc,
                    f"phase{phase}_best_epoch": best_epoch,
                    f"phase{phase}_trainable_params": trainable_params
                }, step=phase)
            
            # Melhor resultado geral
            best_phase = max(
                [1, 2, 3],
                key=lambda p: max(full_history[f'phase{p}']['val_accuracy'])
            )
            best_val_acc_overall = max(
                full_history[f'phase{best_phase}']['val_accuracy']
            )
            
            logger.info("=" * 80)
            logger.info("RESUMO DO TREINAMENTO PROGRESSIVO")
            logger.info("=" * 80)
            for phase in [1, 2, 3]:
                best_acc = max(full_history[f'phase{phase}']['val_accuracy'])
                logger.info(f"Fase {phase}: val_accuracy = {best_acc:.4f}")
            logger.info(f"Melhor resultado: Fase {best_phase} ({best_val_acc_overall:.4f})")
            logger.info("=" * 80)
            
        else:
            logger.info(f"Treinando com estratégia: {strategy.value}")
            
            # Aplicar estratégia de unfreezing
            model, trainable_params = apply_unfreezing_strategy(model, strategy)
            
            # Obter configuração
            train_config = get_phase_config(strategy)
            
            # Compilar
            loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            optimizer = keras.optimizers.Adam(
                learning_rate=train_config["learning_rate"]
            )
            metrics = [keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
            
            model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
            model.summary(print_fn=logger.info)
            
            # Treinar
            history = train_single_phase(
                model=model,
                train_ds=train_ds,
                val_ds=val_ds,
                phase_config=train_config,
                class_weight=class_weight,
                output_dir=str(output_dir),
                phase_name='standard',
                enable_augmentation=True,
                enable_mixup=True,
                num_classes=num_classes
            )
            
            full_history['standard'] = history.history
            
            # Log de métricas
            val_loss_history = history.history["val_loss"]
            best_epoch_idx = val_loss_history.index(min(val_loss_history))
            
            metrics_to_log = {}
            for k, v_list in history.history.items():
                metric_type = "val" if k.startswith("val_") else "train"
                clean_k = k.replace("val_", "")
                metrics_to_log[f"{metric_type}_{clean_k}"] = float(v_list[best_epoch_idx])
            
            log_metrics(metrics_to_log, step=best_epoch_idx)

        logger.info("=" * 80)
        logger.info("AVALIAÇÃO NO DATASET DE TESTE")
        logger.info("=" * 80)
        
        # Avaliação básica
        logger.info("Avaliando test_ds (métricas básicas)...")
        test_metrics = model.evaluate(test_ds, return_dict=True)
        log_metrics({f"test_{k}": float(v) for k, v in test_metrics.items()})

        # Métricas Avançadas
        logger.info("Gerando predições para métricas avançadas...")
        
        # Extração de Labels
        logger.info("Extraindo labels verdadeiros do dataset...")
        y_true_batches = []
        for _, batch_labels in test_ds:
            y_true_batches.append(batch_labels.numpy())
        y_true = np.concatenate(y_true_batches, axis=0)
        
        # Predições
        logger.info("Gerando predições...")
        y_proba = model.predict(test_ds, verbose=1)
        
        if len(y_true) != len(y_proba):
            raise RuntimeError(
                f"Desalinhamento! y_true tem {len(y_true)} amostras, "
                f"mas y_proba tem {len(y_proba)}."
            )
        
        # Patch-level Metrics
        logger.info("Calculando métricas Patch-Level...")
        patch_metrics = compute_slide_metrics_multiclass(
            y_proba=y_proba,
            y_true=y_true,
            class_names=cfg.class_names
        )
        log_metrics({f"patch_{k}": v for k, v in patch_metrics.items()})

        # Slide-level Metrics
        if cfg.clickhouse:
            logger.info("Consultando metadados de teste para agregação por slide...")
            ch = ClickHouseClient(cfg.clickhouse)
            
            df_test_meta = ch.query_df(
                f"SELECT slide_id, stage_label FROM {cfg.clickhouse.table_patches} "
                f"WHERE split = 'test'"
            )
            
            if len(df_test_meta) == len(y_true):
                slide_ids = df_test_meta["slide_id"].to_numpy()
                
                # Slide-level (Mean Prob)
                y_true_slide, y_proba_slide = aggregate_by_slide(
                    y_true=y_true,
                    y_proba=y_proba,
                    slide_ids=slide_ids,
                    method="mean_prob",
                )
                slide_metrics = compute_slide_metrics_multiclass(
                    y_true_slide, y_proba_slide, cfg.class_names
                )
                log_metrics({f"slide_soft_{k}": v for k, v in slide_metrics.items()})
                logger.info("Métricas de slide calculadas com sucesso.")
            else:
                logger.warning(
                    f"Tamanho do dataset de teste ({len(y_true)}) difere do "
                    f"metadata do banco ({len(df_test_meta)}). "
                    "Pulando métricas de slide para evitar desalinhamento."
                )
        
        models_dir = Path(os.path.join(cfg.output_dir, "models"))
        models_dir.mkdir(parents=True, exist_ok=True)
        save_path = models_dir / f"ensemble_model_{strategy.value}.keras"
        model.save(save_path)
        logger.info(f"Modelo salvo em: {save_path}")
        
        logger.info("=" * 80)
        logger.info("PIPELINE CONCLUÍDO COM SUCESSO!")
        logger.info("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)