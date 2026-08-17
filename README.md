# Linhas a partir de pontos georreferenciados

Solução do desafio técnico para gerar linhas de tendência a partir de uma camada de pontos. O programa identifica automaticamente a orientação e o espaçamento das fileiras, conecta lacunas compatíveis, calcula o comprimento em metros, classifica cada geometria como `reta` ou `curva` e salva o resultado em WGS84 (`EPSG:4326`).

## Requisitos

- Python 3.10 ou superior;
- arquivo de entrada georreferenciado, com CRS definido;
- geometrias `Point` ou `MultiPoint` em SHP, KML, GeoJSON ou GPKG.

Crie um ambiente virtual e instale as dependências:

```bash
python -m venv .venv
```

No Windows/PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

No Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Uso

```bash
python main.py --input <caminho/dos/pontos> --output <resultado>.geojson
```

Exemplo:

```bash
python main.py --input dataset/amostra2.geojson --output resultados/amostra2_linhas.geojson
```

O diretório de saída é criado automaticamente. Em caso de entrada inválida, o programa termina com uma mensagem objetiva e código diferente de zero.

## Atributos da saída

| Coluna | Descrição |
|---|---|
| `id_linha` | Identificador sequencial da linha |
| `comprimento_m` | Comprimento calculado no CRS UTM local, em metros |
| `classificacao` | `reta` ou `curva` |
| `n_pontos` | Quantidade de pontos associados à linha |
| `desvio_relativo` | Maior afastamento da corda dividido pelo comprimento da corda |
| `geometry` | `LineString` em `EPSG:4326` |

## Como o algoritmo funciona

1. valida o arquivo, o CRS e o tipo das geometrias;
2. converte os pontos para a zona UTM local estimada, para trabalhar em metros;
3. estima a orientação dominante das fileiras com os vetores entre vizinhos;
4. cria conexões direcionais, limitadas a um antecessor e um sucessor por ponto, sem permitir cruzamentos;
5. estima e remove temporariamente a curvatura compartilhada pelas fileiras;
6. une fragmentos que possuem a mesma tendência e intervalos espaciais compatíveis, preenchendo as lacunas;
7. suaviza ruído local, calcula comprimento, desvio, mudança angular e classifica a linha;
8. converte o resultado para `EPSG:4326` e grava o GeoJSON.

Todos os parâmetros espaciais importantes são derivados dos dados. Assim, o mesmo código atende amostras com escalas, orientações e curvaturas diferentes.

## Testes

Execute:

```bash
python -m unittest discover -s tests -v
```

A suíte cobre fileiras retas e curvas, lacunas internas, classificação, ausência de CRS, cruzamentos e o contrato do GeoJSON final.

Validação realizada com os cinco arquivos fornecidos:

| Entrada | Linhas | Retas | Curvas | Cruzamentos |
|---|---:|---:|---:|---:|
| `amostra1.gpkg` | 44 | 40 | 4 | 0 |
| `amostra2.geojson` | 12 | 11 | 1 | 0 |
| `amostra3.geojson` | 31 | 31 | 0 | 0 |
| `amostra4.geojson` | 31 | 8 | 23 | 0 |
| `amostra5.geojson` | 30 | 29 | 1 | 0 |

Todas as geometrias geradas foram validadas, ficaram em `EPSG:4326` e apresentaram diferença inferior a `0,001 m` entre o comprimento gravado e o comprimento recalculado no CRS métrico.

## Limites conhecidos

- cada arquivo deve representar uma família predominante de fileiras; blocos com orientações totalmente independentes devem ser processados separadamente;
- são necessários ao menos três pontos distintos para formar uma linha;
- a leitura de KML depende do driver KML disponível na instalação GDAL/pyogrio do ambiente.

O histórico de uso de IA solicitado no enunciado está em [PROMPTS.md](PROMPTS.md).
