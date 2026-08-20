"""Baseline determinístico — encapsula o algoritmo geométrico de main.py.

Este módulo é um wrapper de isolamento. O algoritmo geométrico continua vivendo
em main.py para manter compatibilidade total com o desafio original. Aqui apenas
re-exportamos as funções públicas necessárias para o pipeline híbrido.

Uso dentro do pacote GeoAI:
    from src.baseline.geometric import run_geometric
    result = run_geometric(input_path, output_path)
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd


def run_geometric(input_path: Path, output_path: Path) -> gpd.GeoDataFrame:
    """Executa o pipeline geométrico completo e salva o GeoJSON de saída.

    Delega diretamente para ``main.run``, preservando todo o comportamento
    original: leitura, validação, agrupamento, geração de linhas, reprojeção
    para EPSG:4326 e gravação em disco.

    Args:
        input_path: Caminho para o arquivo de pontos (.shp, .kml, .geojson, .gpkg).
        output_path: Caminho onde o GeoJSON de saída será salvo.

    Returns:
        GeoDataFrame em EPSG:4326 com as colunas:
        ``id_linha, comprimento_m, classificacao, n_pontos, desvio_relativo, geometry``.
    """
    import main as _main  # import local evita circularidade

    return _main.run(input_path, output_path)


def generate_lines_geometric(points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Gera linhas a partir de um GeoDataFrame de pontos já lido.

    Útil para o pipeline híbrido e para testes, onde a leitura do arquivo
    já foi feita por outra camada.

    Args:
        points: GeoDataFrame com geometrias Point e CRS definido.

    Returns:
        GeoDataFrame em EPSG:4326 com as colunas padrão do desafio.
    """
    import main as _main  # import local evita circularidade

    return _main.generate_lines(points)
