# Plano Definitivo de Ataque — PA3: Poucos Rótulos em Câmeras Trap (ENA24)

---

## 1. Decisões Estratégicas Fechadas (Trilha C)

Após análise técnica do edital (`PA3.pdf`), dos dados do ENA24-detection e das restrições de orçamento e entrega, as decisões oficiais foram fixadas:

| Componente | Escolha Oficial | Justificativa |
| :--- | :---: | :--- |
| **Parte 1 (Trilha)** | **Trilha C — Professor e Aluno** | Um *foundation model* biológico aberto rotula o pool de 6.505 imagens de `train` e transfere esse conhecimento para um aluno compacto ($\le 12\text{M}$ params). Maior estabilidade em 1 e 5 rótulos/classe e maior teto de F1 Macro. |
| **Professor (Trilha C)** | **BioCLIP (`hf-hub:imageomics/bioclip`)** | Modelo aberto (`open_clip`) treinado no TreeOfLife-10M (taxonomia/fauna). Só olha `train` e `test_public` (para diagnóstico zero-shot), **nunca toca no `val`** durante o treino/ajuste e **nunca entra no `predict.py`**. |
| **Aluno (Parte 0b, 1 e 2)** | **ResNet-18 (ImageNet, $\approx 11,18\text{M}$ params)** | Dentro do limite de $12\text{M}$ parâmetros, rápido em CPU e comparável diretamente ao **teto de referência (`0,686` de F1 macro)** definido pelo monitor. |
| **Piso do Zero (Parte 0a)** | **`SmallCNN` treinada do zero** | CNN leve (~0,4M params) treinada apenas nos 105 rótulos (`rotulos_5.csv`) para quantificar o ganho do pré-treino no ImageNet. |
| **Tratamento de Ruído (Trilha C)** | **Professor Melhorado + Soft Labels + Filtro Balanceado** | Usa os embeddings do BioCLIP + rótulos do orçamento $K$ para calibrar o professor, aplica filtragem de confiança sem colapsar classes raras (`horse`) e destila com temperatura $T$ combinando rótulos humanos e pseudo-rótulos. |
| **Métrica Principal** | **F1 Macro (`test_public`)** | Pesa igualmente classes raras (`horse`: 45 imagens no treino, 4 no teste) e frequentes (`american_black_bear`: 690 imagens). Seleção de época e limiares feita exclusivamente no `val`. |
| **Pergunta OOD (Capivara no Uruguai)** | **Energia Livre + Entropia (Aluno) / Cosseno Máximo (Professor)** | Sem implementar no treino, avaliamos via energia livre dos logits $E(x) = -T \log \sum_i e^{z_i/T}$ e entropia preditiva como sinal de espécie inédita (OOD). |
| **Parte 3 (Interpretabilidade)** | **Grad-CAM (`layer4`)** | Inspeção de 3 erros e 5 acertos no `test_public` para diagnosticar fotos noturnas IR, animais < 2% do quadro e atalhos de fundo das câmeras. |

---

## 2. Contratos Estritos de Interface

Fixar os formatos abaixo antes de começar a codificar garante que as duas branches se integrem na `main` sem atrito:

1. **Mapeamento das 21 Classes (`data/classes.csv`):**
   * Ordenado por `idx` de `0` (`american_black_bear`) a `20` (`woodchuck`).
   * O `predict.py` deve manter fallback interno dessa lista para funcionar em CPU offline mesmo fora da pasta `data/`.
2. **Saída do Dataset (`src/dataset.py`):**
   * Imagens convertidas para RGB, redimensionadas para `224 × 224` (na avaliação, usar `Resize((224, 224))` em vez de `CenterCrop` para não cortar animais próximos às bordas) e normalizadas com média/desvio padrão do ImageNet.
   * Retorno padrão: tupla `(image_tensor, label_idx, image_id)` com `image_tensor` `torch.float32` `(3, 224, 224)` e `label_idx = -1` quando não rotulado.
3. **Cache do Professor (`outputs/teacher_cache/`):**
   * Para não re-executar o BioCLIP a cada experimento, salvar `train_teacher.npz` e `test_public_teacher.npz` com as chaves:
     * `image_ids`: `(N,)` strings
     * `embeddings`: `(N, D)` float32 L2-normalizados
     * `logits`: `(N, 21)` float32
     * `probs`: `(N, 21)` float32
     * `preds`: `(N,)` int64
     * `is_infrared`: `(N,)` bool (opcional/útil para diagnóstico dia vs. noite)
4. **Pseudo-Rótulos Refinados (`src/pseudo_labels.py`):**
   * Dicionário alinhado a `train_image_ids` contendo:
     * `soft_labels`: `np.ndarray (N, 21)` float32 (distribuição que soma 1)
     * `hard_labels`: `np.ndarray (N,)` int64
     * `keep_mask`: `np.ndarray (N,)` bool
     * `sample_weights`: `np.ndarray (N,)` float32
     * `is_human_labeled`: `np.ndarray (N,)` bool
5. **Checkpoint do Aluno (`checkpoints/*.pth`):**
   * Dicionário padronizado: `{"model_state_dict": model.state_dict(), "arch": "resnet18", "num_classes": 21}`.
   * No carregamento para inferência (`predict.py`), instanciar o backbone sempre com `pretrained=False` (zero acesso à internet) e validar `sum(p.numel() for p in model.parameters()) <= 12_000_000`.
6. **Tabela de Orçamento (`curva_orcamento.csv`):**
   * Colunas obrigatórias do edital: `metodo,orcamento,seed,f1_macro,acuracia`.

---

## 3. Divisão de Tarefas em Paralelo

A divisão separa completamente os arquivos de `src/` entre as duas branches para evitar conflitos de merge:

```
                  ┌──────────────────────────────────────────┐
                  │      main (estrutura base e plano.md)    │
                  └────────────────────┬─────────────────────┘
                                       │
                  ┌────────────────────┴─────────────────────┐
                  ▼                                          ▼
   feature/data-metrics-baseline             feature/teacher-distillation
   (Membro 1 — Henrique)                     (Membro 2 — Isaias)
   • src/dataset.py                          • src/teacher.py
   • src/metrics.py                          • src/pseudo_labels.py
   • src/models.py                           • src/distillation.py
   • src/saliency.py                         • Diagnóstico zero-shot e ruído
   • src/visualization.py                    • Cache do BioCLIP (train/test_public)
                  │                                          │
                  └────────────────────┬─────────────────────┘
                                       ▼
                        Integração & Execução Conjunta
                        • train.py, evaluate.py e predict.py (CPU)
                        • Parte 0 (Piso) & Parte 1 (Trilha C)
                        • Parte 2: curva_orcamento.csv (1, 5 [3 seeds], 20)
                        • Parte 3: Galeria de falhas e atalhos (Grad-CAM)
                        • notebooks/inferencia.ipynb (notebook unificado)
                        • Slides da Apresentação & AI_LOG.md
```

### Membro 1 (Henrique) — Dados, Métricas, Piso & Saliência
* **Branch:** `feature/data-metrics-baseline`
* **Escopo em `src/`:**
  1. `src/dataset.py`:
     * Leitura de `data/classes.csv` e dos CSVs (`rotulos_1/5/20.csv`, `val.csv`, `test_public.csv`).
     * `ENA24Dataset` suportando modo rotulado, pool `train` (6.505 imagens) e pasta arbitrária sem CSV (para o `predict.py`).
     * Detector heurístico de fotos noturnas em infravermelho (`is_infrared_image`, baixa diferença entre canais R/G/B).
  2. `src/metrics.py`:
     * `compute_macro_f1_and_accuracy`, `compute_per_class_f1` e `compute_confusion_matrix` (21×21).
     * `compute_ood_scores`: MSP, entropia preditiva e energia livre (para a pergunta da capivara).
  3. `src/models.py`:
     * `SmallCNN` (Parte 0a — treinada do zero) e construtor de `ResNet-18` (Parte 0b/1/2) com checagem de $\le 12\text{M}$ parâmetros.
     * Loop de treino supervisionado do Piso (Parte 0 e curva base da Parte 2), escolhendo a melhor época exclusivamente no `val` e identificando a época em que o modelo começa a decorar os 105 exemplos.
  4. `src/saliency.py` & `src/visualization.py`:
     * Grad-CAM na última camada convolucional (`layer4` da ResNet-18) para extrair e plotar os 3 erros e 5 acertos da Parte 3.
     * Plots padronizados: curvas de treino/val, matriz 21×21, curva de orçamento em escala log (com linha do teto `0,686` e linha do professor) e gráfico de ganho por classe no orçamento 5.

### Membro 2 (Isaias) — Professor (BioCLIP), Ruído & Destilação
* **Branch:** `feature/teacher-distillation`
* **Escopo em `src/`:**
  1. `src/teacher.py`:
     * Definição dos prompts em linguagem natural das 21 espécies para o BioCLIP (`hf-hub:imageomics/bioclip` via `open_clip`).
     * Inferência única em GPU sobre `train` (6.505 imagens) e `test_public` (811 imagens) salvando embeddings e predições *zero-shot* em `outputs/teacher_cache/*.npz` (**sem tocar no `val`**).
  2. `src/pseudo_labels.py`:
     * Diagnóstico completo do zero-shot do professor no `test_public`: F1 macro geral, matriz de confusão/ruído, quebra de F1 entre fotos diurnas vs. noturnas (IR) e identificação das confusões sistemáticas (`bird` vs. `american_crow`, esquilos, etc.).
     * Estratégia de tratamento de ruído: **Professor Melhorado** (adaptador/protótipos treinados apenas nos embeddings dos rótulos humanos do orçamento $K$) + **Soft Labels** com temperatura $T$ + **Filtro de Confiança Balanceado por Classe** (com pesos inversamente proporcionais à frequência para não ignorar `horse`).
  3. `src/distillation.py`:
     * `DistillationLoss`: combinação de Cross-Entropy nos rótulos humanos reais (`rotulos_K.csv`) e perda suave (KL-Divergence / CE suave) nos pseudo-rótulos filtrados do pool `train`.
     * Loop de treino do aluno (`ResNet-18`) selecionando o melhor checkpoint pelo F1 macro no `val`.

### Integração Conjunta (Pós-Merge na `main`)
1. Conectar os módulos em **`train.py`**, **`evaluate.py`** e **`predict.py`** (CPU offline, flags `--imagens` / `--saida`).
2. Rodar a **Parte 0** (`SmallCNN` vs. `ResNet-18` em 105 rótulos), **Parte 1** (diagnóstico do BioCLIP + destilação no orçamento 5) e **Parte 2** (orçamentos `1`, `5` com 3 seeds e `20`, gerando `curva_orcamento.csv`).
3. Rodar a **Parte 3** (3 erros + 5 acertos com Grad-CAM e diagnósticos concretos).
4. Consolidar todas as execuções, tabelas e figuras no notebook unificado **`notebooks/inferencia.ipynb`** para a apresentação.

---

## 4. Cronograma Marco a Marco

1. **Marco 1 (Individual em paralelo):** Implementação dos módulos nas branches `feature/data-metrics-baseline` (Henrique) e `feature/teacher-distillation` (Isaias) respeitando os contratos da Seção 2.
2. **Marco 2 (Merge & Cache do Professor):** Merge na `main` sem conflitos; execução única do BioCLIP em `train` e `test_public`; avaliação zero-shot do professor e matriz de ruído.
3. **Marco 3 (Partes 0 e 1):** Treino e comparação do Piso (`SmallCNN` vs. `ResNet-18` em `K=5`, época de memorização) contra o Aluno destilado (Trilha C), incluindo análise de em quais classes o aluno supera o professor.
4. **Marco 4 (Parte 2 — Curva de Orçamento):** Execução das rodadas `K=1` (1 seed), `K=5` (3 seeds, média $\pm$ desvio) e `K=20` (1 seed) para Piso e Trilha C, salvando `curva_orcamento.csv` e quebra de ganho por classe.
5. **Marco 5 (Parte 3 — Galeria Grad-CAM):** Extração e diagnóstico dos 3 erros e 5 acertos no `test_public`.
6. **Marco 6 (Entregáveis Finais):** Preenchimento do `notebooks/inferencia.ipynb`, validação do `predict.py` em CPU estrita, atualização do `AI_LOG.md` e montagem dos slides da apresentação.
