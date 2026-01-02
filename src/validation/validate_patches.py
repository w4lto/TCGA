"""
Valida integridade das imagens referenciadas no ClickHouse.
Identifica e registra patches corrompidos ANTES do treinamento.
"""
import sys
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
import tensorflow as tf
from PIL import Image

# Importar suas classes
from src.data.ch_utils import ClickHouseClient
from src.training.train_tf import load_config

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/validation_clickhouse_patches.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ClickHouseValidator')


class ClickHouseImageValidator:
    """Valida integridade de patches do ClickHouse."""
    
    def __init__(self, clickhouse_client: ClickHouseClient, max_workers: int = 8):
        self.ch = clickhouse_client
        self.max_workers = max_workers
        self.corrupted_patches = []
        self.valid_patches = []
    
    def validate_single_patch(self, row: pd.Series) -> Tuple[bool, str, str, str]:
        """
        Valida um patch individual.
        
        Args:
            row: Linha do DataFrame com 'patch_path', 'stage_label', etc.
        
        Returns:
            (is_valid, patch_path, stage_label, error_message)
        """
        patch_path = row.get('patch_path') or row.get('image_path')
        stage_label = row.get('stage_label', 'UNKNOWN')
        
        if patch_path is None:
            return False, "N/A", stage_label, "Coluna patch_path ausente"
        
        patch_path_str = str(patch_path)
        
        # Verificar se arquivo existe
        if not Path(patch_path_str).exists():
            return False, patch_path_str, stage_label, "Arquivo não encontrado no filesystem"
        
        try:
            # Validação PIL
            with Image.open(patch_path_str) as img:
                img.verify()
            
            # Reabrir e carregar pixels
            with Image.open(patch_path_str) as img:
                img.load()
                width, height = img.size
                
                if width == 0 or height == 0:
                    return False, patch_path_str, stage_label, "Dimensões inválidas (0x0)"
            
            # Validação TensorFlow (simula pipeline real)
            raw_data = tf.io.read_file(patch_path_str)
            decoded = tf.image.decode_png(raw_data, channels=3)
            
            shape = decoded.shape
            if shape[0] < 32 or shape[1] < 32:
                return False, patch_path_str, stage_label, f"Imagem muito pequena: {shape}"
            
            return True, patch_path_str, stage_label, ""
            
        except tf.errors.InvalidArgumentError as e:
            return False, patch_path_str, stage_label, f"TF decode error: {str(e)[:100]}"
        except (IOError, SyntaxError, OSError) as e:
            return False, patch_path_str, stage_label, f"PIL error: {str(e)[:100]}"
        except Exception as e:
            return False, patch_path_str, stage_label, f"Unexpected: {str(e)[:100]}"
    
    def validate_all_splits(self, splits: List[str] = None) -> Dict:
        """
        Valida todos os patches de todos os splits.
        
        Args:
            splits: Lista de splits para validar (ex: ['train', 'val', 'test'])
        
        Returns:
            Dict com estatísticas por split
        """
        if splits is None:
            splits = ['train', 'val', 'test']
        
        logger.info(f"Carregando patches do ClickHouse para splits: {splits}")
        
        # Carregar DataFrames
        dfs = self.ch.load_patches(splits=splits, limit_per_split=None)
        
        results = {}
        all_corrupted = []
        
        for split in splits:
            if split not in dfs or dfs[split] is None or len(dfs[split]) == 0:
                logger.warning(f"Split '{split}' vazio, pulando validação")
                continue
            
            df = dfs[split]
            total = len(df)
            
            logger.info(f"Validando {total} patches do split '{split}'...")
            
            # Validação paralela
            split_corrupted = []
            split_valid = []
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.validate_single_patch, row): idx 
                    for idx, row in df.iterrows()
                }
                
                with tqdm(total=total, desc=f"Validando {split}") as pbar:
                    for future in as_completed(futures):
                        is_valid, path, label, error = future.result()
                        
                        if is_valid:
                            split_valid.append(path)
                        else:
                            split_corrupted.append((path, label, error))
                            all_corrupted.append((split, path, label, error))
                            logger.warning(f"[{split}] CORROMPIDO: {path} ({label}) - {error}")
                        
                        pbar.update(1)
            
            # Estatísticas do split
            results[split] = {
                'total': total,
                'valid': len(split_valid),
                'corrupted': len(split_corrupted),
                'corruption_rate': len(split_corrupted) / total * 100 if total > 0 else 0
            }
            
            logger.info(f"\n{'='*60}")
            logger.info(f"SPLIT: {split}")
            logger.info(f"Total: {results[split]['total']}")
            logger.info(f"Válidos: {results[split]['valid']}")
            logger.info(f"Corrompidos: {results[split]['corrupted']} ({results[split]['corruption_rate']:.2f}%)")
            logger.info(f"{'='*60}\n")
        
        # Relatório consolidado
        total_patches = sum(r['total'] for r in results.values())
        total_corrupted = sum(r['corrupted'] for r in results.values())
        
        logger.info(f"\n{'='*60}")
        logger.info(f"RELATÓRIO CONSOLIDADO")
        logger.info(f"{'='*60}")
        logger.info(f"Total de patches: {total_patches}")
        logger.info(f"Patches corrompidos: {total_corrupted} ({total_corrupted/total_patches*100:.2f}%)")
        
        # Salvar lista de corrompidos
        if all_corrupted:
            corrupted_df = pd.DataFrame(
                all_corrupted,
                columns=['split', 'patch_path', 'stage_label', 'error']
            )
            output_path = Path('logs/corrupted_patches.csv')
            corrupted_df.to_csv(output_path, index=False)
            logger.warning(f"Lista de patches corrompidos salva em: {output_path}")
            
            # Agrupar por erro
            logger.info("\nDistribuição de erros:")
            error_counts = corrupted_df['error'].value_counts()
            for error_type, count in error_counts.items():
                logger.info(f"  {error_type}: {count}")
        
        results['_summary'] = {
            'total_patches': total_patches,
            'total_corrupted': total_corrupted,
            'overall_corruption_rate': total_corrupted/total_patches*100 if total_patches > 0 else 0
        }
        
        return results


def main():
    """Executa validação de patches do ClickHouse."""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(
        description='Valida integridade de patches referenciados no ClickHouse'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Caminho para config.yaml'
    )
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        default=['train', 'val', 'test'],
        help='Splits para validar (default: train val test)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Número de workers paralelos'
    )
    parser.add_argument(
        '--fail-on-corruption',
        action='store_true',
        help='Retorna exit code 1 se encontrar patches corrompidos'
    )
    
    args = parser.parse_args()
    
    
    cfg = load_config(args.config)
    
    # Criar cliente ClickHouse
    ch = ClickHouseClient(cfg.clickhouse)
    
    # Executar validação
    validator = ClickHouseImageValidator(ch, max_workers=args.workers)
    results = validator.validate_all_splits(splits=args.splits)
    
    # Verificar se deve falhar
    summary = results['_summary']
    if args.fail_on_corruption and summary['total_corrupted'] > 0:
        logger.error(
            f"Validação FALHOU: {summary['total_corrupted']} patches corrompidos "
            f"({summary['overall_corruption_rate']:.2f}%)"
        )
        sys.exit(1)
    
    logger.info("Validação concluída com sucesso!")
    sys.exit(0)


if __name__ == '__main__':
    main()