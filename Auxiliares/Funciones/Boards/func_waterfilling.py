import numpy as np
def WaterFilling(densidades, n_objetivo=200000, n_bins=150, rango=(0.05, 0.40)):
    """
    Selecciona exactamente n_objetivo muestras intentando maximizar la uniformidad.
    Usa estrategia de 'Water Filling' para compensar bins vacíos con bins llenos.
    """
    
    # 1. Identificar a qué bin pertenece cada tablero
    bins = np.linspace(rango[0], rango[1], n_bins + 1) 
    indices_bins = np.digitize(densidades, bins) - 1
    indices_bins = np.clip(indices_bins, 0, n_bins - 1)
    
    # 2. Contar disponibilidad real por bin
    # conteos_disponibles[i] = cuántos tableros existen realmente en el bin i
    conteos_disponibles = np.bincount(indices_bins, minlength=n_bins)
    
    # 3. Encontrar el "Nivel de Corte" óptimo (Búsqueda Binaria)
    # Buscamos un número L tal que si tomamos min(disponible, L) de cada bin,
    # la suma total se acerque lo más posible a n_objetivo sin pasarse.
    low = 0
    high = np.max(conteos_disponibles) # El bin más alto
    nivel_base = 0
    
    while low <= high:
        mid = (low + high) // 2
        # Cuántos tomaríamos si el límite fuera 'mid'
        suma_estimada = np.sum(np.minimum(conteos_disponibles, mid))
        
        if suma_estimada <= n_objetivo:
            nivel_base = mid
            low = mid + 1
        else:
            high = mid - 1
            
    # 4. Calcular asignación base y el faltante
    # asignacion_final[i] = cuántos vamos a tomar del bin i
    asignacion_final = np.minimum(conteos_disponibles, nivel_base)
    faltante = n_objetivo - np.sum(asignacion_final)
    
    # 5. Repartir el remanente (faltante)
    # El 'faltante' será pequeño (menor que n_bins). Lo repartimos uno a uno
    # entre los bins que todavía tienen tableros de sobra (donde disponible > nivel_base).
    indices_con_superavit = np.where(conteos_disponibles > nivel_base)[0]
    
    if faltante > 0 and len(indices_con_superavit) > 0:
        # Elegimos al azar quiénes ponen el extra para no sesgar
        bins_extra = np.random.choice(indices_con_superavit, size=int(faltante), replace=False)
        for bin_idx in bins_extra:
            asignacion_final[bin_idx] += 1
            
    # 6. Ejecutar la selección
    indices_seleccionados = []
    for b in range(n_bins):
        n_tomar = asignacion_final[b]
        if n_tomar > 0:
            # Índices originales que caen en este bin
            idx_en_bin = np.where(indices_bins == b)[0]
            # Muestreo aleatorio sin reemplazo
            seleccion = np.random.choice(idx_en_bin, size=n_tomar, replace=False)
            indices_seleccionados.append(seleccion)
            
    # Unir y mezclar
    indices = np.concatenate(indices_seleccionados)
    np.random.shuffle(indices)
    
    return indices



