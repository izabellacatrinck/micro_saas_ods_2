"""Tests for src.glossary_repair — PT-BR post-extraction fixes."""
from src.glossary_repair import repair_pt_text


class TestNumpyTerms:
    def test_matriz_becomes_array_in_body(self):
        pt = "As matrizes de entrada não precisam ter o mesmo número."
        assert "arrays de entrada" in repair_pt_text(pt)

    def test_singular_matriz_also_fixed(self):
        pt = "Uma matriz n-dimensional é o objeto central."
        assert "array n-dimensional" in repair_pt_text(pt)

    def test_matriz_in_title_case_preserved(self):
        pt = "Impressão de Matrizes NumPy# Esta página..."
        out = repair_pt_text(pt)
        # "NumPy arrays" replacement carries title fix
        assert "NumPy arrays" in out
        assert "Matrizes NumPy" not in out

    def test_linalg_matriz_de_covariancia_left_alone(self):
        """`matriz de covariância` is a real PT math term; don't break it."""
        pt = "A matriz de covariância é calculada com np.cov()."
        assert "matriz de covariância" in repair_pt_text(pt)

    def test_matriz_de_correlacao_left_alone(self):
        pt = "Calcule a matriz de correlação entre as colunas."
        assert "matriz de correlação" in repair_pt_text(pt)

    def test_transmissao_becomes_broadcasting(self):
        pt = "Suporte à Transmissão em Array, Tipo elenco."
        out = repair_pt_text(pt)
        assert "Broadcasting" in out
        assert "Transmissão" not in out

    def test_difusao_becomes_broadcasting(self):
        pt = "a difusão é feita sobre outra dimensão"
        assert "broadcasting é feita" in repair_pt_text(pt)

    def test_fatiamento_becomes_slicing(self):
        pt = "Use recursos básicos de indexação como fatiamento"
        assert "slicing" in repair_pt_text(pt)
        assert "fatiamento" not in repair_pt_text(pt)

    def test_fatia_becomes_slice_when_numpy_context(self):
        pt = "É um inteiro, ou uma tupla de objetos de fatia e inteiros."
        out = repair_pt_text(pt)
        assert "slice" in out

    def test_fatia_de_food_is_left_alone(self):
        """Negative lookahead protects 'fatia de pão' etc."""
        pt = "ele comeu uma fatia de pão"
        assert "fatia de pão" in repair_pt_text(pt)


class TestPandasTerms:
    def test_capital_Serie_becomes_Series(self):
        pt = "Uma Série contém os dados da coluna."
        assert "Series contém" in repair_pt_text(pt)

    def test_series_plural_becomes_Series(self):
        pt = "As Séries são os blocos fundamentais."
        out = repair_pt_text(pt)
        assert "Series" in out

    def test_serie_temporal_stays(self):
        """'série temporal' is the correct PT term for 'time series' and does
        not refer to the pandas Series class."""
        pt = "Uma série temporal de preços diários."
        assert "série temporal" in repair_pt_text(pt)

    def test_moldura_de_dados_becomes_DataFrame(self):
        pt = "Crie uma moldura de dados a partir de um dicionário."
        assert "DataFrame" in repair_pt_text(pt)
        assert "moldura" not in repair_pt_text(pt)

    def test_mesclar_becomes_merge(self):
        pt = "Mesclar múltiplos objetos ao longo de um índice."
        out = repair_pt_text(pt)
        assert "Merge" in out or "merge" in out
        assert "Mesclar" not in out

    def test_empilhar_in_dataframe_context_becomes_stack(self):
        pt = "Empilhar as colunas de um DataFrame com stack()"
        out = repair_pt_text(pt)
        assert "stack" in out.lower()

    def test_desempilhar_becomes_unstack(self):
        pt = "Use desempilhar para reverter a operação."
        assert "unstack" in repair_pt_text(pt)


class TestIdempotence:
    def test_repair_is_idempotent(self):
        pt = "Matrizes e Séries com transmissão e fatiamento."
        once = repair_pt_text(pt)
        twice = repair_pt_text(once)
        assert once == twice

    def test_pure_english_untouched(self):
        en = "Arrays and Series with broadcasting and slicing."
        assert repair_pt_text(en) == en
