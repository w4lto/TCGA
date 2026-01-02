"""
Script para validar integridade de todas as imagens antes do treinamento.
Identifica e registra arquivos corrompidos.
"""
import os
import sys
from pathlib import Path
from typing import List, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import tensorflow as tf
from PIL import Image
import io

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('data/validation_report.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ImageValidator')


class ImageValidator:
    """Valida integridade de imagens PNG para pipelines TensorFlow."""
    
    def __init__(self, data_dir: str, max_workers: int = 8):
        self.data_dir = Path(data_dir)
        self.max_workers = max_workers
        self.corrupted_files = []
        self.valid_files = []
        
    def validate_single_image(self, image_path: Path) -> Tuple[bool, str, str]:
        """
        Valida uma imagem usando múltiplos métodos.
        
        Returns:
            (is_valid, image_path_str, error_message)
        """
        image_path_str = str(image_path)
        
        try:
            # Método 1: Verificação com PIL
            with Image.open(image_path) as img:
                img.verify()  # Verifica integridade
                
            # Método 2: Reabrir e carregar pixels (verify() fecha o arquivo)
            with Image.open(image_path) as img:
                img.load()  # Força carregamento completo
                width, height = img.size
                
                if width == 0 or height == 0:
                    return False, image_path_str, "Dimensões inválidas (0x0)"
                    
            # Método 3: Decodificação TensorFlow (simula pipeline real)
            raw_data = tf.io.read_file(image_path_str)
            decoded = tf.image.decode_png(raw_data, channels=3)
            
            # Validações adicionais
            shape = decoded.shape
            if shape[0] < 32 or shape[1] < 32:
                return False, image_path_str, f"Imagem muito pequena: {shape}"
                
            return True, image_path_str, ""
            
        except tf.errors.InvalidArgumentError as e:
            return False, image_path_str, f"TF decode error: {str(e)[:100]}"
        except (IOError, SyntaxError, OSError) as e:
            return False, image_path_str, f"PIL error: {str(e)[:100]}"
        except Exception as e:
            return False, image_path_str, f"Unexpected error: {str(e)[:100]}"
    
    def validate_dataset(self, pattern: str = "**/*.png") -> dict:
        """
        Valida todas as imagens no dataset.
        
        Args:
            pattern: Glob pattern para encontrar imagens
            
        Returns:
            Dict com estatísticas de validação
        """
        logger.info(f"Buscando imagens em {self.data_dir} com pattern '{pattern}'")
        image_files = list(self.data_dir.glob(pattern))
        total_images = len(image_files)
        
        if total_images == 0:
            logger.error(f"Nenhuma imagem encontrada em {self.data_dir}")
            return {"total": 0, "valid": 0, "corrupted": 0}
        
        logger.info(f"Encontradas {total_images} imagens. Iniciando validação...")
        
        # Validação paralela
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.validate_single_image, img): img 
                for img in image_files
            }
            
            with tqdm(total=total_images, desc="Validando imagens") as pbar:
                for future in as_completed(futures):
                    is_valid, image_path, error_msg = future.result()
                    
                    if is_valid:
                        self.valid_files.append(image_path)
                    else:
                        self.corrupted_files.append((image_path, error_msg))
                        logger.warning(f"CORROMPIDA: {image_path} - {error_msg}")
                    
                    pbar.update(1)
        
        # Relatório final
        stats = {
            "total": total_images,
            "valid": len(self.valid_files),
            "corrupted": len(self.corrupted_files),
            "corruption_rate": len(self.corrupted_files) / total_images * 100
        }
        
        logger.info(f"\n{'='*60}")
        logger.info(f"RELATÓRIO DE VALIDAÇÃO")
        logger.info(f"{'='*60}")
        logger.info(f"Total de imagens: {stats['total']}")
        logger.info(f"Imagens válidas: {stats['valid']}")
        logger.info(f"Imagens corrompidas: {stats['corrupted']} ({stats['corruption_rate']:.2f}%)")
        logger.info(f"{'='*60}\n")
        
        # Salvar lista de arquivos corrompidos
        if self.corrupted_files:
            corrupted_list_path = self.data_dir / "corrupted_files.txt"
            with open(corrupted_list_path, 'w') as f:
                for path, error in self.corrupted_files:
                    f.write(f"{path}\t{error}\n")
            logger.warning(f"Lista de arquivos corrompidos salva em: {corrupted_list_path}")
        
        # Salvar lista de arquivos válidos
        valid_list_path = self.data_dir / "valid_files.txt"
        with open(valid_list_path, 'w') as f:
            for path in self.valid_files:
                f.write(f"{path}\n")
        logger.info(f"Lista de arquivos válidos salva em: {valid_list_path}")
        
        return stats


def main():
    """Executa validação do dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Valida integridade de imagens no dataset')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Diretório raiz com as imagens')
    parser.add_argument('--pattern', type=str, default='**/*.png',
                        help='Glob pattern para buscar imagens (default: **/*.png)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Número de workers paralelos (default: 8)')
    parser.add_argument('--fail-on-corruption', action='store_true',
                        help='Retorna exit code 1 se encontrar arquivos corrompidos')
    
    args = parser.parse_args()
    
    validator = ImageValidator(args.data_dir, max_workers=args.workers)
    stats = validator.validate_dataset(pattern=args.pattern)
    
    if args.fail_on_corruption and stats['corrupted'] > 0:
        logger.error(f"Validação falhou: {stats['corrupted']} arquivo(s) corrompido(s)")
        sys.exit(1)
    
    logger.info("Validação concluída com sucesso!")
    sys.exit(0)


if __name__ == '__main__':
    main()