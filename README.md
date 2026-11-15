# PA3 - Poucos Rótulos em Câmeras Trap (Trilha C: Professor e Aluno)

Programming Assignment 3 da disciplina de Aprendizado Profundo (FGV EMAp).  
Classificação de 21 espécies da fauna norte-americana (**ENA24-detection**) com orçamento reduzido de rótulos (1, 5 e 20 rótulos/classe) e 6.505 imagens não rotuladas, utilizando a **Trilha C - Professor (`BioCLIP`) e Aluno (`ResNet-18`)**.

## Autores
- [Henrique Coelho Beltrão](https://github.com/riqueu)
- [Isaias Gouvêa Gonçalves](https://github.com/isaiasgoncalves)

---

## 1. Visão Geral do Projeto

O trabalho avalia quanto da distância entre o **Piso** (supervisionado com poucos rótulos) e o **Teto** (`0,686` de F1 macro com todos os rótulos, medido pela monitoria) pode ser recuperado com dados não rotulados de câmeras trap:

- **Parte 0 - O Piso (`rotulos_5.csv`, 105 imagens):** Comparação entre CNN treinada do zero e `ResNet-18` pré-treinada no ImageNet, analisando matriz de confusão e época de sobreajuste.
- **Parte 1 - Trilha C (Professor e Aluno):** `BioCLIP` rotula o pool `train` e tem sua matriz de ruído diagnosticada no `test_public`; aplicamos tratamento de ruído para destilar o conhecimento para um aluno `ResNet-18` ($\le 12\text{M}$ parâmetros).
- **Parte 2 - Curva de Orçamento:** Comparação Piso vs. Trilha C nos orçamentos `1`, `5` (3 seeds: média $\pm$ desvio) e `20` rótulos/classe (`curva_orcamento.csv`) + ganho por classe em `K=5`.
- **Parte 3 - Galeria de Falhas e Atalhos:** Análise de 3 erros e 5 acertos com **Grad-CAM** (animal vs. atalhos de fundo, fotos noturnas IR e animais < 2% do quadro).

### Estrutura do Repositório
```text
dl-pa3-2026/
├── AI_LOG.md                  # Registro de uso de ferramentas de IA
├── PA3.pdf                    # Enunciado oficial do trabalho
├── README.md
├── curva_orcamento.csv        # Consolidação final da Parte 2 (test_public)
├── requirements.txt           # Dependências do ambiente
├── checkpoints/               # Pesos treinados (.pth ignorados no Git; *_history.json versionados)
├── data/                      # Dados ENA24 (ignorados no Git)
├── docs/
│   └── plano.md               # Decisões, contratos de interface e divisão em paralelo
├── notebooks/
│   └── inferencia.ipynb       # Notebook unificado e executado com análises das Partes 0–3
├── outputs/                   # Gráficos, matrizes, diagnósticos JSON e caches
├── runs/                      # Logs e scripts auxiliares por experimento
├── src/                       # Módulos da dupla (dataset, metrics, models, saliency, viz, teacher, pseudo_labels, distillation)
├── tests/                     # Suíte de testes unitários e de integração (pytest)
├── train.py                   # CLI unificado de treinamento (teacher_cache, scratch_cnn, piso, distill)
├── evaluate.py                # CLI de avaliação, consolidação do CSV final e figuras
└── predict.py                 # CLI oficial de entrega (roda em CPU, sem internet, <= 12M params)
```

---

## 2. Configuração do Ambiente

Python 3.10+ em ambiente virtual:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 3. Dados (ENA24-detection / `pa3_dados.zip`)

Os arquivos do dataset estão listados no `.gitignore` e não são versionados no repositório.

1. Baixe o pacote **`pa3_dados.zip` (415 MB)** no [Google Drive da disciplina](https://drive.google.com/file/d/1g7FGqtXOk_ziNKBwQAk8BxjPOaPCnbui/view?usp=sharing).
2. Extraia o conteúdo dentro de `data/`:
```bash
unzip pa3_dados.zip -d data/
```
3. Estrutura esperada dentro de `data/`:
```text
data/
├── classes.csv
├── LEIAME.md
├── images/
│   ├── train/          # 6.505 imagens (pool sem rótulo + subconjuntos rotulados)
│   ├── val/            # 202 imagens (seleção de modelo e hiperparâmetros)
│   └── test_public/    # 811 imagens (somente reporte final e diagnóstico do professor)
└── labels/
    ├── rotulos_1.csv   # 21 imagens (1 por classe)
    ├── rotulos_5.csv   # 105 imagens (5 por classe - orçamento padrão)
    ├── rotulos_20.csv  # 420 imagens (20 por classe)
    ├── val.csv
    └── test_public.csv
```

> **Regras de conformidade:** O professor nunca acessa o `val` durante o treino/ajuste; o `test_public` nunca é usado para escolher época, limiar ou arquitetura; o `predict.py` roda apenas o aluno ($\le 12\text{M}$ parâmetros) em CPU e sem internet.

---

## 4. Resultados Principais (`test_public`, F1 Macro)

Todos os modelos de produção (`ResNet-18`) têm **11.187.285 parâmetros** ($\le 12\text{M}$). Referência do teto (todos os rótulos de `train`, monitoria): **`0,686`**.

| Método | $K=1$ (`s=42`) | $K=5$ (`s=42,43,44`) | $K=20$ (`s=42`) |
|---|---|---|---|
| BioCLIP Zero-Shot (`orcamento=0`) | `0,3078` | `0,3078` | `0,3078` |
| Piso `SmallCNN` do zero (`394.293` params) | - | `0,1978` (`s=42`) | - |
| **Piso `ResNet-18`** (apenas rótulos humanos) | `0,1689` | `0,4334 ± 0,0130` | **`0,5953`** |
| Professor BioCLIP Melhorado (protótipos + refin. LOO) | `0,3267` | `0,5142` | `0,5860` |
| **Trilha C - Aluno `ResNet-18` (Destilado)** | **`0,3579`** (`+0,1890`) | **`0,5217 ± 0,0134`** (`+0,0883`) | `0,5579` |

O melhor modelo em validação (`checkpoints/best_student.pth` $\leftarrow$ `student_K5_s42.pth`, `val_F1 = 0,5189`) atinge **`0,5094`** de F1 Macro (`0,5672` de acurácia) em `test_public`. Os arquivos `.pth` ficam fora do Git (`.gitignore`), podem ser baixados diretamente pela Release [`v1.0.0`](https://github.com/riqueu/dl-pa3-2026/releases/tag/v1.0.0) ou reproduzidos integralmente pelos comandos abaixo.

### Checkpoints Oficiais (GitHub Release `v1.0.0`)

Para executar apenas a avaliação ou a inferência via `predict.py` após clonar o repositório, baixe o checkpoint principal diretamente para `checkpoints/best_student.pth`:

```bash
mkdir -p checkpoints
wget -O checkpoints/best_student.pth \
  https://github.com/riqueu/dl-pa3-2026/releases/download/v1.0.0/best_student.pth
```

Todos os pesos treinados na GPU (alunos destilados da Trilha C e baselines do piso) estão publicados na Release [`v1.0.0`](https://github.com/riqueu/dl-pa3-2026/releases/tag/v1.0.0):

| Arquivo | Experimento | SHA-256 | Download |
| :--- | :--- | :--- | :--- |
| `best_student.pth` | Alias do melhor aluno (`student_K5_s42.pth`) | `3c065d4665a0b6f752d80aa2fe5c3315e02d0a97349b1843bca9e20f01a4d915` | [Baixar](https://github.com/riqueu/dl-pa3-2026/releases/download/v1.0.0/best_student.pth) |
| `student_K1_s42.pth` | Aluno ResNet-18 (Trilha C, $K=1$, seed 42) | `501cff811097396f3d1cf70979f4586f4daba1e3e6bae59978fa68e3e4228927` | [Baixar](https://github.com/riqueu/dl-pa3-2026/releases/download/v1.0.0/student_K1_s42.pth) |
| `student_K5_s42.pth` | Aluno ResNet-18 (Trilha C, $K=5$, seed 42) | `3c065d4665a0b6f752d80aa2fe5c3315e02d0a97349b1843bca9e20f01a4d915` | [Baixar](https://github.com/riqueu/dl-pa3-2026/releases/download/v1.0.0/student_K5_s42.pth) |
| `student_K5_s43.pth` | Aluno ResNet-18 (Trilha C, $K=5$, seed 43) | `df0a8cf1383549b59d069bb8270d411a63bd0c8e4cde72107331f915a81e193e` | [Baixar](https://github.com/riqueu/dl-pa3-2026/releases/download/v1.0.0/student_K5_s43.pth) |
| `student_K5_s44.pth` | Aluno ResNet-18 (Trilha C, $K=5$, seed 44) | `5e38afdec208302e41c5ba5c9c459696364b24308918f8b12130f7004a81fa8b` | [Baixar](https://github.com/riqueu/dl-pa3-2026/releases/download/v1.0.0/student_K5_s44.pth) |
| `student_K20_s42.pth` | Aluno ResNet-18 (Trilha C, $K=20$, seed 42) | `3fb4d690a8b19021102990a1e7c72874430d4d6b62d6bfebae8fe674af1854c3` | [Baixar](https://github.com/riqueu/dl-pa3-2026/releases/download/v1.0.0/student_K20_s42.pth) |
| `piso_small_cnn_K5_s42.pth` | Baseline Piso `SmallCNN` ($K=5$, seed 42) | `a0122d7be174577f4830dd408715d90a995e3ef28d9184cac9bb9accec212bcd` | [Baixar](https://github.com/riqueu/dl-pa3-2026/releases/download/v1.0.0/piso_small_cnn_K5_s42.pth) |
| `piso_K1_s42.pth` | Baseline Piso ResNet-18 ($K=1$, seed 42) | `e3a4458388b8877edb6a3fe83b4e17b22eccff86332adc0490c47df4d1e70bed` | [Baixar](https://github.com/riqueu/dl-pa3-2026/releases/download/v1.0.0/piso_K1_s42.pth) |
| `piso_K5_s42.pth` | Baseline Piso ResNet-18 ($K=5$, seed 42) | `7ae1a3c4f0942d4a2fdff0ffef76304b22e211f0f2203d93647ef1c27aae48a2` | [Baixar](https://github.com/riqueu/dl-pa3-2026/releases/download/v1.0.0/piso_K5_s42.pth) |
| `piso_K5_s43.pth` | Baseline Piso ResNet-18 ($K=5$, seed 43) | `bf5f5bc55f5b2dd7cc5c78d750d55277f4f1918cf7d9e698962a88fb5374bd24` | [Baixar](https://github.com/riqueu/dl-pa3-2026/releases/download/v1.0.0/piso_K5_s43.pth) |
| `piso_K5_s44.pth` | Baseline Piso ResNet-18 ($K=5$, seed 44) | `309d66b79336bffe955ac8cab675dae1cfd08d6b1f5856d434da95cb16b92d48` | [Baixar](https://github.com/riqueu/dl-pa3-2026/releases/download/v1.0.0/piso_K5_s44.pth) |
| `piso_K20_s42.pth` | Baseline Piso ResNet-18 ($K=20$, seed 42) | `48811c5a383eaaca90e8b58bae7a84e04719f2bc03e2a64eefcaca1d81c18fdd` | [Baixar](https://github.com/riqueu/dl-pa3-2026/releases/download/v1.0.0/piso_K20_s42.pth) |

> **Verificação de integridade:** Após o download, confirme com `sha256sum -b checkpoints/best_student.pth`.

---

## 5. Comandos de Execução e Reprodução

* **1. Gerar cache de embeddings do BioCLIP (GPU/CPU) e pseudo-rótulos:**
  ```bash
  ./venv/bin/python train.py --mode teacher_cache
  for K in 1 5 20; do ./venv/bin/python -m src.pseudo_labels --budget $K --diagnose; done
  ```
* **2. Treinar o Piso (Parte 0 e Parte 2):**
  ```bash
  ./venv/bin/python train.py --mode scratch_cnn --budget 5 --seed 42 --epochs 35
  ./venv/bin/python train.py --mode piso --budget 5 --seed 42 --epochs 35
  ```
* **3. Treinar o Aluno destilado (Trilha C):**
  ```bash
  ./venv/bin/python train.py --mode distill --budget 5 --seed 42 --epochs 15 --epoch-size 3072 --freeze none
  ```
* **4. Consolidar `curva_orcamento.csv`, selecionar `best_student.pth` em `val` e gerar figuras:**
  ```bash
  ./venv/bin/python evaluate.py --mode collect_all
  ./venv/bin/python evaluate.py --mode figures
  ```
* **5. Predição oficial da monitoria (estritamente CPU e sem internet):**
  ```bash
  CUDA_VISIBLE_DEVICES="" ./venv/bin/python predict.py --imagens data/images/test_public --saida preds.csv
  ```
* **6. Rodar testes automatizados:**
  ```bash
  ./venv/bin/python -m pytest tests/ -v
  ```

---

## 6. Planejamento e Registro de IA

* [docs/plano.md](docs/plano.md): Decisões da Trilha C, contratos de interface, divisão de trabalho em paralelo (Henrique e Isaias) e cronograma.
* [notebooks/inferencia.ipynb](notebooks/inferencia.ipynb): Notebook completo com EDA, Partes 0–3, questão OOD (capivara) e verificação do `predict.py`.
* [AI_LOG.md](AI_LOG.md): Registro de uso de ferramentas de IA.
