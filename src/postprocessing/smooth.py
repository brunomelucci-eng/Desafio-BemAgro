"""Módulo de suavização geométrica de LineStrings."""

import logging
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString

logger = logging.getLogger(__name__)


def smooth_linestrings(
    gdf: gpd.GeoDataFrame,
    step_m: float = 1.0,
    window_size: int = 5
) -> gpd.GeoDataFrame:
    """Suaviza as linhas do GeoDataFrame usando média móvel e reamostragem.
    
    Aplica o seguinte pipeline em cada LineString:
    1. Reamostra a linha para ter um vértice exatamente a cada `step_m` metros.
    2. Aplica uma média móvel nas coordenadas X e Y para remover zigue-zagues.
    3. Preserva o primeiro e o último ponto exatos da linha original para 
       não encolher as extremidades.
       
    O GeoDataFrame de entrada DEVE estar em um CRS métrico (ex: UTM), para 
    que `step_m` corresponda a metros de fato.
    
    Parameters
    ----------
    gdf : GeoDataFrame
        DataFrame contendo as geometrias LineString.
    step_m : float
        Distância em metros entre os vértices (default: 1.0).
    window_size : int
        Tamanho da janela para a média móvel (default: 5). 
        Deve ser ímpar para o padding simétrico funcionar perfeitamente.
        
    Returns
    -------
    GeoDataFrame
        Cópia do GeoDataFrame original com as geometrias suavizadas.
    """
    if gdf.empty:
        return gdf.copy()
        
    smoothed_geometries = []
    
    for geom in gdf.geometry:
        if geom is None or geom.is_empty or geom.geom_type != "LineString":
            smoothed_geometries.append(geom)
            continue
            
        length = geom.length
        if length <= step_m:
            smoothed_geometries.append(geom)
            continue
            
        # 1. Reamostrar a cada `step_m` metros
        distances = np.arange(0, length, step_m)
        if distances[-1] != length:
            distances = np.append(distances, length)
            
        resampled_pts = [geom.interpolate(d) for d in distances]
        resampled_coords = np.array([(p.x, p.y) for p in resampled_pts])
        
        if len(resampled_coords) < window_size:
            smoothed_geometries.append(LineString(resampled_coords))
            continue
            
        # 2. Média móvel (Moving Average)
        window = np.ones(window_size) / window_size
        pad_size = window_size // 2
        
        # Fazemos um padding nas extremidades repetindo os pontos da ponta.
        # Assim o np.convolve não encolhe a linha.
        padded_coords = np.pad(resampled_coords, ((pad_size, pad_size), (0, 0)), mode='edge')
        
        smooth_x = np.convolve(padded_coords[:, 0], window, mode='valid')
        smooth_y = np.convolve(padded_coords[:, 1], window, mode='valid')
        smooth_coords = np.column_stack((smooth_x, smooth_y))
        
        # 3. Ancorar o início e o fim aos valores reamostrados (evita arredondamentos de borda)
        smooth_coords[0] = resampled_coords[0]
        smooth_coords[-1] = resampled_coords[-1]
        
        smoothed_geometries.append(LineString(smooth_coords))
        
    smoothed_gdf = gdf.copy()
    smoothed_gdf.geometry = smoothed_geometries
    
    # Atualiza as colunas de métricas, se existirem
    if "comprimento_m" in smoothed_gdf.columns:
        smoothed_gdf["comprimento_m"] = smoothed_gdf.geometry.length.round(3)
        
    return smoothed_gdf
