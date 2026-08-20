# Histórico de Prompts e Engenharia de Instruções de IA

Este documento registra o histórico de prompts e a metodologia de interação com assistentes de IA (Antigravity / Claude / GPT-Codex) utilizados durante o desenvolvimento da solução do **Desafio Desenvolvedor Python (Geoprocessamento)**.

A abordagem seguiu um fluxo de **Desenvolvimento Orientado a Prompts (Prompt-Driven Development)** dividido em etapas lógicas: arquitetura inicial, algoritmo geométrico, pós-processamento de casos de borda, testes automatizados e documentação técnica.

---

## Etapa 1: Análise do Enunciado e Estruturação do Projeto

**Objetivo:** Compreender os requisitos técnicos, criar a estrutura base da CLI e configurar o ambiente Git.

> **Prompt de Instrução:**
> *"Atue como um Engenheiro de Software Python Sênior especialista em Geoprocessamento (GIS). Analise a especificação técnica do desafio (`Desafio - Dev Python.pdf`) e a estrutura das amostras em `dataset/`.*
> 
> *Requisitos para a estrutura inicial:*
> 1. *Criar uma interface CLI em `main.py` aceitando argumentos `--input` e `--output`.*
> 2. *Garantir o uso das bibliotecas de geoprocessamento (`geopandas`, `shapely`, `pyproj`, `scipy`).*
> 3. *Configurar arquivo `requirements.txt` e estrutura de testes com `pytest`.*
> 4. *Inicializar versionamento Git com `.gitignore` adequado para arquivos GIS temporários."*

**Resultado:** Estrutura base montada, CLI funcional e ambiente preparado.

---

## Etapa 2: Algoritmo Core de Agrupamento e Geração de Linhas

**Objetivo:** Processar os pontos georreferenciados, estimar linhas de tendência e exportar em WGS84 (EPSG:4326).

> **Prompt de Instrução:**
> *"Implemente em `main.py` o algoritmo core para conversão de pontos em linhas georreferenciadas:*
> 1. *Converter geometrias WGS84 (EPSG:4326) para coordenadas métricas UTM para cálculos precisos de distância.*
> 2. *Utilizar `scipy.spatial.KDTree` e projeções vetoriais para associar pontos pertencentes à mesma fileira/linha de tendência.*
> 3. *Calcular o comprimento total de cada linha em metros.*
> 4. *Classificar a geometria em `'reta'` ou `'curva'` no GeoDataFrame.*
> 5. *Salvar o resultado no caminho definido em `--output` no formato GeoJSON EPSG:4326."*

**Resultado:** Algoritmo funcional capaz de processar arquivos de entrada e gerar GeoJSON com metadados de comprimento e classificação.

---

## Etapa 3: Refinamento Geométrico e Tratamento de Casos de Borda (Amostras 1, 4 e 5)

**Objetivo:** Resolver problemas de conexão incorreta, curvatura e pontos ruidosos reportados na inspeção visual das amostras.

> **Prompt de Instrução:**
> *"Realize um pós-processamento robusto nas geometrias geradas para corrigir as seguintes falhas específicas identificadas nas amostras:*
> 
> - **Amostra 1 (Conexão e Entrelinhas):** *Corrigir o cálculo do espaçamento entre fileiras (usando a moda da distância entre vizinhos) para evitar o fechamento de linhas falsas em entrelinhas e unir fragmentos interrompidos da mesma fileira.*
> - **Amostra 4 (Curvas e Interpolação):** *Ajustar o rastreamento local para acompanhar curvas acentuadas e utilizar interpolação polar em arcos concêntricos para evitar cordas em lacunas.*
> - **Amostra 5 (Filtragem de Ruído):** *Aplicar filtragem geométrica baseada em MAD (Median Absolute Deviation) para desconsiderar pontos aleatórios/ruídos, preservando integralmente os pontos de replantio válidos.*
> - **Validação Geral:** *Garantir zero cruzamentos (`is_simple`) entre todas as linhas geradas."*

**Resultado:** Algoritmo otimizado com modelos paralelo e circular adaptativos, passando com 100% de precisão nas 5 amostras.

---

## Etapa 4: Testes Automatizados e Qualidade do Código

**Objetivo:** Garantir regressão zero, resiliência do algoritmo e conformidade com padrões de código.

> **Prompt de Instrução:**
> *"Desenvolva uma suíte de testes unitários automatizados em `tests/test_main.py` utilizando `pytest`:*
> 1. *Testar conversão e reprojecao de coordenadas (WGS84 $\leftrightarrow$ UTM).*
> 2. *Testar a classificação correta de geometrias (`'reta'` vs `'curva'`).*
> 3. *Verificar se as geometrias geradas não possuem auto-interseções ou cruzamentos.*
> 4. *Testar a ordenação e reconstrução de linhas mesmo quando os pontos de entrada estão embaralhados.*
> 5. *Validar a conformidade de estilo e sintaxe utilizando a ferramenta `ruff`."*

**Resultado:** 11 testes automatizados executando com sucesso e aprovação no linter `ruff`.

---

## Etapa 5: Documentação Técnica e Didática do Código

**Objetivo:** Tornar o código autoexplicativo para avaliação e estudo técnico.

> **Prompt de Instrução:**
> *"Adicione comentários explicativos detalhados ao longo do arquivo `main.py` abrangendo:*
> - *Matemática vetorial e projeções UTM.*
> - *Uso de KDTree, SVD (Singular Value Decomposition) e ajuste por MAD.*
> - *Lógica de classificação e interpolação de lacunas.*
> - *Fluxo principal de execução da CLI.*
> 
> *Atualize o `README.md` com instruções detalhadas de instalação (`requirements.txt`), uso da CLI e resumo da validação de todas as amostras."*

**Resultado:** Código fonte totalmente documentado com 114+ linhas de comentários didáticos e `README.md` atualizado.


---

## Etapa 6: Evolu��o para GeoAI e Aprendizado Profundo (U-Net++)

**Objetivo:** Elevar o n�vel do desafio para uma arquitetura moderna de Machine Learning, utilizando PyTorch e redes neurais para segmenta��o sem�ntica, integradas ao pipeline geom�trico original.

> **Prompt Mestre:**
> *Atue como um Machine Learning Engineer S�nior + Computer Vision Engineer + Geospatial Engineer + Software Architect Python, com experi�ncia avan�ada em PyTorch, U-Net++, segmenta��o sem�ntica e processamento geoespacial.*
>
> *O objetivo n�o � apenas entregar algo que funcione. Quero transformar o desafio em um projeto profissional de GeoAI / Machine Learning, demonstrando Engenharia de Software, Machine Learning, Deep Learning, arquitetura de software, e MLOps.*
>
> *Fluxo esperado:*
> *1. Gera��o de Dados Sint�ticos (rastreabilidade, robustez).*
> *2. Dataset Geoespacial com PyTorch.*
> *3. Arquitetura U-Net++ de segmenta��o (Deep Supervision).*
> *4. Treinamento determin�stico e logs via TensorBoard.*
> *5. Infer�ncia H�brida: Uso da IA para filtragem sem�ntica e o baseline geom�trico para valida��o espacial rigorosa.*

