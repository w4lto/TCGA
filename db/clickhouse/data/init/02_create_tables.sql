USE ml_histopath;

CREATE TABLE IF NOT EXISTS tcga_slides
(
    patient_id     String,                      -- TCGA-XX-XXXX
    slide_id       String,                      -- nome do arquivo sem extensão
    wsi_path       String,                      -- caminho local .svs
    stage_label    LowCardinality(String),      -- 'I','II','III','IV'
    ajcc_raw       String,                      -- texto original do estágio (opcional)
    dataset_source LowCardinality(String),      -- ex: 'TCGA_BRCA_WSI'
    split          LowCardinality(String),      -- 'train','val','test' (inicialmente 'train')
    created_at     DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY (patient_id, slide_id);


CREATE TABLE IF NOT EXISTS tcga_patches
(
    patient_id     String,
    slide_id       String,
    patch_id       String,                      -- ex: '<slide>_x1234_y5678_0'
    patch_path     String,                      -- caminho local do .png
    x              UInt32,
    y              UInt32,
    level          UInt8,
    patch_size     UInt16,
    tissue_score   Float32,
    cellularity    Float32,
    stage_label    LowCardinality(String),      -- herdado de tcga_slides
    dataset_source LowCardinality(String),      -- 'TCGA_BRCA_PATCH'
    split          LowCardinality(String),      -- 'train','val','test' (ainda não usado)
    created_at     DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY (patient_id, slide_id, patch_id);
