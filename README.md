# PA3 - Poucos Rótulos em Câmeras Trap (Trilha C: Professor e Aluno)

Programming Assignment 3 da disciplina de Aprendizado Profundo (FGV EMAp).  
Classificação de 21 espécies da fauna norte-americana (**ENA24-detection**) com orçamento reduzido de rótulos (1, 5 e 20 rótulos/classe) e 6.505 imagens não rotuladas, utilizando a **Trilha C — Professor (`BioCLIP`) e Aluno (`ResNet-18`)**.

## Autores
- [Henrique Coelho Beltrão](https://github.com/riqueu)
- [Isaias Gouvêa Gonçalves](https://github.com/isaiasgoncalves)

---

## 1. Visão Geral do Projeto

O trabalho avalia quanto da distância entre o **Piso** (supervisionado com poucos rótulos) e o **Teto** (`0,686` de F1 macro com todos os rótulos, medido pela monitoria) pode ser recuperado com dados não rotulados de câmeras trap:

- **Parte 0 — O Piso (`rotulos_5.csv`, 105 imagens):** Comparação entre CNN treinada do zero e `ResNet-18` pré-treinada no ImageNet, analisando matriz de confusão e época de sobreajuste.
- **Parte 1 — Trilha C (Professor e Aluno):** `BioCLIP` rotula o pool `train` e tem sua matriz de ruído diagnosticada no `test_public`; aplicamos tratamento de ruído para destilar o conhecimento para um aluno `ResNet-18` ($\le 12\text{M}$ parâmetros).
- **Parte 2 — Curva de Orçamento:** Comparação Piso vs. Trilha C nos orçamentos `1`, `5` (3 seeds: média $\pm$ desvio) e `20` rótulos/classe (`curva_orcamento.csv`) + ganho por classe em `K=5`.
- **Parte 3 — Galeria de Falhas e Atalhos:** Análise de 3 erros e 5 acertos com **Grad-CAM** (animal vs. atalhos de fundo, fotos noturnas IR e animais < 2% do quadro).

### Estrutura do Repositório
```text
dl-pa3-2026/
├── AI_LOG.md                  # Registro de uso de ferramentas de IA
├── PA3.pdf                    # Enunciado oficial do trabalho
├── README.md
├── requirements.txt           # Dependências do ambiente
├── checkpoints/               # Pesos treinados (ignorados no Git se volumosos)
├── data/                      # Dados ENA24 (ignorados no Git)
├── docs/
│   └── plano.md               # Decisões, contratos de interface e divisão em paralelo
├── notebooks/
│   └── inferencia.ipynb       # Notebook unificado para consolidação e apresentação
├── outputs/                   # Gráficos, matrizes e caches de embeddings
├── runs/                      # Logs de execução por experimento
├── src/                       # Módulos da dupla (ver docs/plano.md)
├── train.py                   # CLI de treinamento (Parte 0, Trilha C e Parte 2)
├── evaluate.py                # CLI de avaliação (aluno e diagnóstico do professor)
└── predict.py                 # CLI oficial de entrega (roda em CPU, sem internet)
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

## 4. Comandos da Entrega (Interface Planejada)

* **Treinar:**
  ```bash
  python train.py --mode distill --budget 5 --seed 42 --checkpoint checkpoints/best_student.pth
  ```
* **Avaliar no `test_public`:**
  ```bash
  python evaluate.py --checkpoint checkpoints/best_student.pth --split test_public
  ```
* **Predição oficial da monitoria (CPU, sem internet):**
  ```bash
  python predict.py --imagens data/images/test_public --saida preds.csv
  ```

---

## 5. Planejamento e Registro de IA

* [docs/plano.md](docs/plano.md): Decisões da Trilha C, contratos de interface, divisão de trabalho em paralelo (Henrique e Isaias) e cronograma.
* [AI_LOG.md](AI_LOG.md): Registro de uso de ferramentas de IA.
