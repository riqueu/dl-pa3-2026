# Registro de Uso de IA (AI_LOG)

Uso de ferramentas de IA como suporte de programação, planejamento e divisão de atividades da dupla (`docs/`), automação de scripts e revisão teórica. Decisões técnicas, implementações e interpretações dos resultados foram validadas pela dupla.

---

## Etapas Registradas

1. **Planejamento e arquitetura paralela (`docs/plano.md`):**
   - Definição da Trilha C (Professor BioCLIP offline $\rightarrow$ Aluno `ResNet-18` $\le 12\text{M}$ parâmetros) e dos contratos compartilhados entre módulos (`src/dataset.py`, `src/metrics.py`, `src/models.py`, `src/saliency.py`, `src/visualization.py` e `src/teacher.py`, `src/pseudo_labels.py`, `src/distillation.py`).
2. **Integração final, pipeline unificado e treinamento em GPU (`RTX 4070 Laptop GPU`):**
   - Implementação dos scripts [`train.py`](train.py), [`evaluate.py`](evaluate.py), [`predict.py`](predict.py) e testes em [`tests/test_cli_and_pipeline.py`](tests/test_cli_and_pipeline.py), além de portabilidade automática de `data/` vs. `data/pa3_dados/` e tratamento de `orcamento=0` na escala logarítmica em [`src/visualization.py`](src/visualization.py).
   - Extração de embeddings do professor BioCLIP (`6.505` imagens de `train`, `811` de `test_public`), ablação controlada estritamente em `val` (`K=5, seed=42`) comparando congelamento/pesos de perda (`freeze="none", epochs=15, epoch_size=3072` obteve melhor `val_F1=0.5189`), treinamento completo do Piso (`SmallCNN` + `ResNet-18`) e da Trilha C em `K ∈ {1, 5, 20}` (`seeds 42, 43, 44` em `K=5`).
   - Consolidação de [`curva_orcamento.csv`](curva_orcamento.csv), geração e inspeção visual das galerias **Grad-CAM + Sensibilidade por Oclusão** (diagnóstico individual de 3 erros e 5 acertos incluindo detecção de atalhos de fundo/câmera fixa) e execução completa de [`notebooks/inferencia.ipynb`](notebooks/inferencia.ipynb).
