# HistÃ³rico de Prompts e Engenharia de InstruÃ§Ãµes de IA

Este documento registra o histÃ³rico de prompts e a metodologia de interaÃ§Ã£o com assistentes de IA (Antigravity / Claude / GPT-Codex) utilizados durante o desenvolvimento da soluÃ§Ã£o do **Desafio Desenvolvedor Python (Geoprocessamento)**.

A abordagem seguiu um fluxo de **Desenvolvimento Orientado a Prompts (Prompt-Driven Development)** dividido em etapas lÃ³gicas: arquitetura inicial, algoritmo geomÃ©trico, pÃ³s-processamento de casos de borda, testes automatizados e documentaÃ§Ã£o tÃ©cnica.

---

## Etapa 1: AnÃ¡lise do Enunciado e EstruturaÃ§Ã£o do Projeto

**Objetivo:** Compreender os requisitos tÃ©cnicos, criar a estrutura base da CLI e configurar o ambiente Git.

> **Prompt de InstruÃ§Ã£o:**
> *"Atue como um Engenheiro de Software Python SÃªnior especialista em Geoprocessamento (GIS). Analise a especificaÃ§Ã£o tÃ©cnica do desafio (`Desafio - Dev Python.pdf`) e a estrutura das amostras em `dataset/`.*
> 
> *Requisitos para a estrutura inicial:*
> 1. *Criar uma interface CLI em `main.py` aceitando argumentos `--input` e `--output`.*
> 2. *Garantir o uso das bibliotecas de geoprocessamento (`geopandas`, `shapely`, `pyproj`, `scipy`).*
> 3. *Configurar arquivo `requirements.txt` e estrutura de testes com `pytest`.*
> 4. *Inicializar versionamento Git com `.gitignore` adequado para arquivos GIS temporÃ¡rios."*

**Resultado:** Estrutura base montada, CLI funcional e ambiente preparado.

---

## Etapa 2: Algoritmo Core de Agrupamento e GeraÃ§Ã£o de Linhas

**Objetivo:** Processar os pontos georreferenciados, estimar linhas de tendÃªncia e exportar em WGS84 (EPSG:4326).

> **Prompt de InstruÃ§Ã£o:**
> *"Implemente em `main.py` o algoritmo core para conversÃ£o de pontos em linhas georreferenciadas:*
> 1. *Converter geometrias WGS84 (EPSG:4326) para coordenadas mÃ©tricas UTM para cÃ¡lculos precisos de distÃ¢ncia.*
> 2. *Utilizar `scipy.spatial.KDTree` e projeÃ§Ãµes vetoriais para associar pontos pertencentes Ã  mesma fileira/linha de tendÃªncia.*
> 3. *Calcular o comprimento total de cada linha em metros.*
> 4. *Classificar a geometria em `'reta'` ou `'curva'` no GeoDataFrame.*
> 5. *Salvar o resultado no caminho definido em `--output` no formato GeoJSON EPSG:4326."*

**Resultado:** Algoritmo funcional capaz de processar arquivos de entrada e gerar GeoJSON com metadados de comprimento e classificaÃ§Ã£o.

---

## Etapa 3: Refinamento GeomÃ©trico e Tratamento de Casos de Borda (Amostras 1, 4 e 5)

**Objetivo:** Resolver problemas de conexÃ£o incorreta, curvatura e pontos ruidosos reportados na inspeÃ§Ã£o visual das amostras.

> **Prompt de InstruÃ§Ã£o:**
> *"Realize um pÃ³s-processamento robusto nas geometrias geradas para corrigir as seguintes falhas especÃ­ficas identificadas nas amostras:*
> 
> - **Amostra 1 (ConexÃ£o e Entrelinhas):** *Corrigir o cÃ¡lculo do espaÃ§amento entre fileiras (usando a moda da distÃ¢ncia entre vizinhos) para evitar o fechamento de linhas falsas em entrelinhas e unir fragmentos interrompidos da mesma fileira.*
> - **Amostra 4 (Curvas e InterpolaÃ§Ã£o):** *Ajustar o rastreamento local para acompanhar curvas acentuadas e utilizar interpolaÃ§Ã£o polar em arcos concÃªntricos para evitar cordas em lacunas.*
> - **Amostra 5 (Filtragem de RuÃ­do):** *Aplicar filtragem geomÃ©trica baseada em MAD (Median Absolute Deviation) para desconsiderar pontos aleatÃ³rios/ruÃ­dos, preservando integralmente os pontos de replantio vÃ¡lidos.*
> - **ValidaÃ§Ã£o Geral:** *Garantir zero cruzamentos (`is_simple`) entre todas as linhas geradas."*

**Resultado:** Algoritmo otimizado com modelos paralelo e circular adaptativos, passando com 100% de precisÃ£o nas 5 amostras.

---

## Etapa 4: Testes Automatizados e Qualidade do CÃ³digo

**Objetivo:** Garantir regressÃ£o zero, resiliÃªncia do algoritmo e conformidade com padrÃµes de cÃ³digo.

> **Prompt de InstruÃ§Ã£o:**
> *"Desenvolva uma suÃ­te de testes unitÃ¡rios automatizados em `tests/test_main.py` utilizando `pytest`:*
> 1. *Testar conversÃ£o e reprojecao de coordenadas (WGS84 $\leftrightarrow$ UTM).*
> 2. *Testar a classificaÃ§Ã£o correta de geometrias (`'reta'` vs `'curva'`).*
> 3. *Verificar se as geometrias geradas nÃ£o possuem auto-interseÃ§Ãµes ou cruzamentos.*
> 4. *Testar a ordenaÃ§Ã£o e reconstruÃ§Ã£o de linhas mesmo quando os pontos de entrada estÃ£o embaralhados.*
> 5. *Validar a conformidade de estilo e sintaxe utilizando a ferramenta `ruff`."*

**Resultado:** 11 testes automatizados executando com sucesso e aprovaÃ§Ã£o no linter `ruff`.

---

## Etapa 5: DocumentaÃ§Ã£o TÃ©cnica e DidÃ¡tica do CÃ³digo

**Objetivo:** Tornar o cÃ³digo autoexplicativo para avaliaÃ§Ã£o e estudo tÃ©cnico.

> **Prompt de InstruÃ§Ã£o:**
> *"Adicione comentÃ¡rios explicativos detalhados ao longo do arquivo `main.py` abrangendo:*
> - *MatemÃ¡tica vetorial e projeÃ§Ãµes UTM.*
> - *Uso de KDTree, SVD (Singular Value Decomposition) e ajuste por MAD.*
> - *LÃ³gica de classificaÃ§Ã£o e interpolaÃ§Ã£o de lacunas.*
> - *Fluxo principal de execuÃ§Ã£o da CLI.*
> 
> *Atualize o `README.md` com instruÃ§Ãµes detalhadas de instalaÃ§Ã£o (`requirements.txt`), uso da CLI e resumo da validaÃ§Ã£o de todas as amostras."*

**Resultado:** CÃ³digo fonte totalmente documentado com 114+ linhas de comentÃ¡rios didÃ¡ticos e `README.md` atualizado.


---

## Etapa 6: Evolução para GeoAI e Aprendizado Profundo (U-Net++)

**Objetivo:** Elevar o nível do desafio para uma arquitetura moderna de Machine Learning, utilizando PyTorch e redes neurais para segmentação semântica, integradas ao pipeline geométrico original.

> **Prompt Mestre:**
> *Atue como um Machine Learning Engineer Sênior + Computer Vision Engineer + Geospatial Engineer + Software Architect Python, com experiência avançada em PyTorch, U-Net++, segmentação semântica e processamento geoespacial.*
>
> *O objetivo não é apenas entregar algo que funcione. Quero transformar o desafio em um projeto profissional de GeoAI / Machine Learning, demonstrando Engenharia de Software, Machine Learning, Deep Learning, arquitetura de software, e MLOps.*
>
> *Fluxo esperado:*
> *1. Geração de Dados Sintéticos (rastreabilidade, robustez).*
> *2. Dataset Geoespacial com PyTorch.*
> *3. Arquitetura U-Net++ de segmentação (Deep Supervision).*
> *4. Treinamento determinístico e logs via TensorBoard.*
> *5. Inferência Híbrida: Uso da IA para filtragem semântica e o baseline geométrico para validação espacial rigorosa.*

