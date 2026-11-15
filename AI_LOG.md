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
3. **Professor, tratamento de ruído e destilação (Isaias, branch `feature/teacher-distillation`):**
   - Usamos o Claude Code (agente de programação) para ler o enunciado e o `docs/plano.md`, abrir as issues #1–#3 da Trilha C e implementar [`src/teacher.py`](src/teacher.py), [`src/pseudo_labels.py`](src/pseudo_labels.py) e [`src/distillation.py`](src/distillation.py), com testes sintéticos em [`tests/test_teacher_distillation.py`](tests/test_teacher_distillation.py). O código foi revisado e o desenho das escolhas discutido antes de cada rodada.

### Episódios da Trilha C

- **Escolher o professor melhorado sem tocar no `val`.** O edital proíbe o professor de ver o `val`, mas o professor melhorado tem hiperparâmetros (conjunto de prompts, peso do texto no protótipo, iterações de refinamento no pool). A IA sugeriu fazer a seleção por *leave-one-out* nos próprios rótulos humanos do orçamento K (cada imagem rotulada é classificada por um professor ajustado sem ela). Assim, nem `val` nem `test_public` entram na escolha; o `test_public` só reporta. Com K=5, o professor passou de F1 0,308 (zero-shot, só com os nomes das classes) para 0,514.
- **Classes raras sumindo no filtro de confiança.** Um filtro só por limiar descartaria quase todo `horse` e `bobcat`, que o professor prevê com pouca confiança. Pedimos um filtro balanceado: cota mínima de pseudo-rótulos por classe prevista e pesos de amostragem inversamente proporcionais à frequência. Um teste sintético garante que a classe rara não desaparece.
- **Crash do agendador de taxa de aprendizado.** O primeiro teste curto de treino quebrou com `ZeroDivisionError` dentro do `OneCycleLR` (aquecimento de 1 passo em uma rodada de 10 passos). A IA leu o traceback, identificou a causa e trocou por aquecimento linear + cosseno via `LambdaLR`.
- **Treino em CPU lento demais.** Sem GPU no notebook do Isaias, uma época completa levaria ~13 min. A IA mediu o throughput da ResNet-18 em CPU com variações (`channels_last`, `bfloat16`, resolução 160, congelar `conv1`+`layer1`): `bfloat16` piorou (1 img/s, sem suporte no processador) e congelar o tronco inicial deu ~1,5× sem mudar a resolução de 224. Com isso a primeira curva de orçamento rodou localmente; a versão final foi retreinada em GPU (etapa 2).
- **Professor vs. aluno por classe.** A IA montou a comparação de F1 por classe no `test_public` (K=5), que mostrou o aluno superando o professor em `dog`, `eastern_gray_squirrel`, `coyote` e `virginia_opossum` e perdendo em `bobcat`, `eastern_cottontail` e `horse`. A interpretação (o aluno aprende o padrão visual do domínio a partir de milhares de pseudo-rótulos, mas herda o ruído do professor nas classes onde ele erra de forma sistemática) é nossa.
- **Erros da própria IA.** Um laço de espera usando `pgrep -f` casava com o próprio comando e nunca terminava; foi corrigido à mão. O primeiro `git push` falhou porque a conta do Isaias ainda não era colaboradora do repositório; resolvemos pedindo acesso de escrita, sem fork.
