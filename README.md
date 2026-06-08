# Classificação de Gênero Musical com CNNs

Classificação automática de gênero musical a partir do áudio, usando **Redes Neurais
Convolucionais (CNN)** sobre **espectrogramas Mel** da base **GTZAN**. Projeto acadêmico de
IA (IFSC — ADS).

O projeto compara uma **CNN baseline** treinada do zero com **transferência de aprendizado**
(MobileNetV2 pré-treinada na ImageNet), e ainda compara duas **fontes de entrada**:
espectrogramas gerados via `librosa` (segmentos de 3 s) vs. as imagens PNG prontas do
dataset (30 s).

## Resultados

Métricas no conjunto de **teste**, no nível de segmento, com divisão **por música** (sem
vazamento de dados):

| Modelo | Entrada | Acurácia | F1 macro |
|--------|---------|:--------:|:--------:|
| CNN baseline (do zero) | Mel 3 s (librosa) | 0,550 | 0,533 |
| **Transfer MobileNetV2** | **Mel 3 s (librosa)** | **0,712** | **0,707** |
| Transfer MobileNetV2 | PNG 30 s (pronta) | 0,633 | 0,630 |

Acaso = 10% (10 classes). O melhor modelo é a transferência de aprendizado sobre
espectrogramas gerados via librosa.

## Estrutura do repositório

```
├── src/                # código reutilizável
│   ├── config.py       # parâmetros e caminhos (fonte única da verdade)
│   ├── audio.py        # carregar áudio, segmentar, espectrograma Mel
│   ├── data.py         # índice de faixas + divisão por música
│   ├── preprocess.py   # geração/cache dos datasets (mel e imagem)
│   ├── models.py       # CNN baseline + MobileNetV2 (extração de features)
│   └── evaluate.py     # métricas, matriz de confusão, gráficos
├── notebooks/
│   ├── 01_eda.ipynb                  # análise exploratória
│   ├── 02_preprocessing.ipynb        # gera e cacheia os espectrogramas
│   ├── 03_baseline_cnn.ipynb         # CNN baseline
│   ├── 04_transfer_learning.ipynb    # MobileNetV2
│   └── 05_evaluation_comparison.ipynb# comparação final
├── models/             # modelos treinados + results.json (gerados; não versionados)
├── data/               # arrays processados (gerados; não versionados)
├── requirements.txt    # dependências principais
└── requirements-lock.txt # versões exatas para reprodução fiel
```

## Dataset

Usa a base **GTZAN** (1000 clipes de 30 s, 10 gêneros × 100). Não é redistribuída neste
repositório. Baixe-a (por exemplo, no
[Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification))
e mantenha a estrutura original:

```
gtzan-dataset/
├── genres_original/<gênero>/<gênero>.000NN.wav
├── images_original/<gênero>/<gênero>000NN.png
├── features_30_sec.csv
└── features_3_sec.csv
```

Informe o caminho da base pela variável de ambiente `GTZAN_DIR` (ou ajuste o padrão em
`src/config.py`):

```powershell
$env:GTZAN_DIR = "C:\caminho\para\gtzan-dataset"
```

> O clipe `jazz.00054.wav` é corrompido na base original e é automaticamente excluído.

## Instalação

Requer **Python 3.13**. A partir da raiz do projeto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt        # ou requirements-lock.txt para versões exatas
```

Principais dependências: TensorFlow, librosa, scikit-learn, pandas, numpy, matplotlib,
seaborn, pillow, jupyter. **Sem GPU no Windows** (TensorFlow > 2.10 usa apenas CPU); o
treino da baseline leva ~30–40 min em CPU.

## Como reproduzir os resultados

Defina `GTZAN_DIR` e execute os notebooks **em ordem**:

```powershell
jupyter notebook
```

1. **`01_eda.ipynb`** — análise exploratória (distribuição, espectrogramas, vieses).
2. **`02_preprocessing.ipynb`** — gera e cacheia os espectrogramas Mel de 3 s (librosa) e os
   arrays das PNGs em `data/processed/`.
3. **`03_baseline_cnn.ipynb`** — treina a CNN baseline (ou carrega de `models/` se já existir).
4. **`04_transfer_learning.ipynb`** — extrai embeddings da MobileNetV2 e treina a cabeça densa.
5. **`05_evaluation_comparison.ipynb`** — experimento de fonte de entrada e comparação final.

Reprodutibilidade: sementes fixas (`SEED = 42`), divisão de dados determinística e padrão
*carregar-ou-treinar* (artefatos em cache evitam recomputar; apague o arquivo correspondente
em `models/` ou `data/processed/` para forçar a regeneração). Métricas consolidadas em
`models/results.json`.

## Decisões metodológicas

- **Segmentação em 3 s** (10 por faixa) para ampliar o número de amostras (~10×).
- **Divisão por música** (não por segmento), estratificada por gênero — evita que trechos da
  mesma música apareçam em treino e teste (vazamento).
- **Sem BatchNormalization na baseline**: provocava colapso na inferência; removê-la resolveu.
- **Transferência por extração de features**: backbone congelado gera embeddings uma única vez
  (viável em CPU), e apenas a cabeça densa é treinada.

## Limitações

- O GTZAN contém clipes duplicados, alguns rótulos ruidosos e artefatos de gravação
  compartilhados, o que limita a validade externa dos resultados.
- Cobre apenas 10 gêneros ocidentais, com faixas de 30 s coletadas por volta de 2002.
- A comparação de fonte de entrada (3 s vs. 30 s) mistura qualidade da entrada e granularidade.
- Classificador acadêmico/educacional — não deve embasar decisões comerciais sem validação
  adicional e supervisão humana.

## Trabalhos futuros

- *Fine-tuning* parcial do backbone MobileNetV2.
- Voto majoritário por música (agregando os 10 segmentos).
- *Data augmentation* de áudio (*time/frequency masking*).
- Validação em bases mais recentes e diversas.

## Licença

Distribuído sob a licença **MIT** — ver [LICENSE](LICENSE).
