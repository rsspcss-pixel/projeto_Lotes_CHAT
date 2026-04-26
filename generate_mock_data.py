from __future__ import annotations

from pathlib import Path
import random

import pandas as pd

OUTPUT_DIR = Path("data")
ROWS_PER_FILE = 5000
FILES_TO_GENERATE = 5
RANDOM_SEED = 42

CATEGORIAS = {
    "Emoliente": ["Oleo de jojoba", "Manteiga de karite", "Caprilico caprico triglicerideo", "Esqualano vegetal"],
    "Tensoativo": ["Cocamidopropil betaina", "Lauril eter sulfato de sodio", "Decil glucosideo", "Sodium cocoyl isethionate"],
    "Conservante": ["Fenoxietanol", "Benzoato de sodio", "Sorbato de potassio", "Acido dehidroacetico"],
    "Ativo funcional": ["Niacinamida", "Acido hialuronico", "Pantenol", "Aloe vera concentrado"],
    "Fragrancia": ["Fragrancia floral suave", "Fragrancia citrica", "Fragrancia herbal", "Fragrancia baunilha"],
    "Corante": ["Oxido de ferro amarelo", "Mica perolada", "Dioxido de titanio", "Carmin sintetico"],
    "Espessante": ["Goma xantana", "Carbopol", "Hidroxietilcelulose", "Goma guar cationica"],
    "Umectante": ["Glicerina bidestilada", "Propilenoglicol", "Sorbitol", "Butilenoglicol"],
}

FORNECEDORES = [
    "CosmoQuimica Brasil",
    "Insumos DermaTech",
    "NaturaPrima Quimicos",
    "BioAtivos Sul",
    "AromaBase Import",
    "CoreLab Ingredients",
    "Quimnova Trading",
    "GreenLab Mat-primas",
    "Essencia Pura Supply",
    "DermaLink Distribuidora",
]

UNIDADES = ["kg", "L", "g", "mL"]
REGIOES = ["SP", "RJ", "MG", "PR", "SC", "BA", "GO", "PE"]


def _make_rows(file_idx: int, rng: random.Random) -> list[dict]:
    rows = []
    for row_idx in range(ROWS_PER_FILE):
        categoria = rng.choice(list(CATEGORIAS.keys()))
        insumo = rng.choice(CATEGORIAS[categoria])
        fornecedor = rng.choice(FORNECEDORES)

        lote_minimo = rng.choice([25, 50, 100, 150, 200, 250, 500, 1000])
        consumo_4m = round(rng.uniform(80, 1500), 2)
        consumo_6m = round(consumo_4m * rng.uniform(1.25, 1.75), 2)
        consumo_12m = round(consumo_6m * rng.uniform(1.60, 2.30), 2)
        custo = round(rng.uniform(12.0, 820.0), 2)

        rows.append(
            {
                "id_insumo": f"INS-{file_idx:02d}-{row_idx + 1:05d}",
                "insumo": insumo,
                "categoria": categoria,
                "fornecedor": fornecedor,
                "regiao_fornecedor": rng.choice(REGIOES),
                "unidade_medida": rng.choice(UNIDADES),
                "lote_minimo_compra": lote_minimo,
                "consumo_medio_4m": consumo_4m,
                "consumo_medio_6m": consumo_6m,
                "consumo_medio_12m": consumo_12m,
                "custo_aquisicao_unitario": custo,
                "prazo_entrega_dias": rng.randint(5, 60),
                "indice_reajuste_12m": round(rng.uniform(-0.08, 0.25), 4),
                "criticidade_insumo": rng.choice(["baixa", "media", "alta"]),
            }
        )
    return rows


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for file_idx in range(1, FILES_TO_GENERATE + 1):
        rows = _make_rows(file_idx, rng)
        df = pd.DataFrame(rows)
        out_path = OUTPUT_DIR / f"insumos_cosmeticos_{file_idx:02d}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"Gerado: {out_path} ({len(df)} linhas)")


if __name__ == "__main__":
    main()
