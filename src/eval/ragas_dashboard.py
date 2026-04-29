"""
Dashboard Interativo para Resultados RAGAS

Visualiza:
- Distribuição de métricas (box plots)
- Evolução de performance
- Matriz de correlação
- Detalhes por amostra

Uso:
    python src/eval/ragas_dashboard.py ragas_results.json
    python src/eval/ragas_dashboard.py ragas_results.json --port 8501
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def load_results(path: Path | str) -> dict[str, Any]:
    """Carrega resultados RAGAS em JSON."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_percentage(value: float | None) -> str:
    """Formata métrica de 0-1 para porcentagem."""
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def print_metrics_table(results: dict) -> None:
    """Imprime tabela de métricas aggregadas."""
    samples = results.get("samples", [])
    
    if not samples:
        print("❌ Nenhuma amostra encontrada nos resultados")
        return
    
    # Coletar todas as métricas
    metrics = {
        "answer_semantic_similarity": [],
        "faithfulness": [],
        "answer_relevancy": [],
        "context_recall": [],
        "context_precision": [],
    }
    
    metric_labels = {
        "answer_semantic_similarity": "Similaridade Semântica",
        "faithfulness": "Fidelidade",
        "answer_relevancy": "Relevância",
        "context_recall": "Recall do Contexto",
        "context_precision": "Precisão do Contexto",
    }
    
    for sample in samples:
        for metric in metrics:
            value = sample.get(metric)
            if value is not None:
                metrics[metric].append(value)
    
    print(f"\n{'='*120}")
    print("📊 MÉTRICAS RAGAS - RESUMO ESTATÍSTICO")
    print(f"{'='*120}\n")
    
    # Header
    print(
        f"{'Métrica':<35} "
        f"{'Média':<15} "
        f"{'Desvio':<15} "
        f"{'Mín':<15} "
        f"{'Máx':<15} "
        f"{'Amostras':<10}"
    )
    print("-" * 120)
    
    # Rows
    for metric_name, values in metrics.items():
        label = metric_labels.get(metric_name, metric_name)
        
        if not values:
            print(
                f"{label:<35} "
                f"{'N/A':<15} "
                f"{'N/A':<15} "
                f"{'N/A':<15} "
                f"{'N/A':<15} "
                f"{0:<10}"
            )
            continue
        
        mean = np.mean(values)
        std = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        count = len(values)
        
        print(
            f"{label:<35} "
            f"{format_percentage(mean):<15} "
            f"{format_percentage(std):<15} "
            f"{format_percentage(min_val):<15} "
            f"{format_percentage(max_val):<15} "
            f"{count:<10}"
        )
    
    print()


def print_detailed_sample_report(results: dict, sample_idx: int = 0) -> None:
    """Imprime relatório detalhado de uma amostra."""
    samples = results.get("samples", [])
    
    if sample_idx >= len(samples):
        print(f"❌ Amostra {sample_idx} não encontrada (total: {len(samples)})")
        return
    
    sample = samples[sample_idx]
    
    print(f"\n{'='*100}")
    print(f"📋 RELATÓRIO DETALHADO - AMOSTRA {sample_idx + 1}/{len(samples)}")
    print(f"{'='*100}\n")
    
    # Pergunta
    print(f"❓ Pergunta:")
    print(f"   {sample.get('question', 'N/A')}\n")
    
    # Ground Truth
    print(f"📚 Ground Truth:")
    print(f"   {sample.get('ground_truth', 'N/A')}\n")
    
    # Resposta
    print(f"🤖 Resposta Gerada:")
    answer = sample.get('answer', 'N/A')
    if answer:
        print(f"   {answer}\n")
    else:
        print(f"   N/A\n")
    
    # Contextos
    contexts = sample.get('contexts', [])
    print(f"📖 Contextos Recuperados ({len(contexts)} chunks):")
    if contexts:
        for i, ctx in enumerate(contexts, 1):
            preview = ctx[:100].replace("\n", " ")
            print(f"   [{i}] {preview}...")
    else:
        print(f"   (nenhum contexto)")
    
    # Métricas
    print(f"\n📊 Métricas:")
    metrics_display = {
        "answer_semantic_similarity": "Similaridade Semântica",
        "faithfulness": "Fidelidade ao Contexto",
        "answer_relevancy": "Relevância da Resposta",
        "context_recall": "Recall do Contexto",
        "context_precision": "Precisão do Contexto",
    }
    
    for metric_key, metric_label in metrics_display.items():
        value = sample.get(metric_key)
        if value is not None:
            print(f"   - {metric_label}: {format_percentage(value)}")
        else:
            print(f"   - {metric_label}: N/A")


def print_top_and_bottom_samples(results: dict, k: int = 3) -> None:
    """Imprime top K e bottom K amostras por desempenho geral."""
    samples = results.get("samples", [])
    
    if not samples:
        print("❌ Nenhuma amostra encontrada")
        return
    
    # Calcular score geral para cada amostra
    scores = []
    for i, sample in enumerate(samples):
        metrics = [
            sample.get("answer_semantic_similarity"),
            sample.get("faithfulness"),
            sample.get("answer_relevancy"),
            sample.get("context_recall"),
            sample.get("context_precision"),
        ]
        
        valid_metrics = [m for m in metrics if m is not None]
        
        if valid_metrics:
            overall_score = np.mean(valid_metrics)
            scores.append((i, overall_score))
    
    if not scores:
        print("⚠️  Nenhuma métrica disponível para ranking")
        return
    
    # Ordenar
    scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n{'='*100}")
    print(f"🏆 TOP {k} AMOSTRAS COM MELHOR DESEMPENHO")
    print(f"{'='*100}\n")
    
    for rank, (idx, score) in enumerate(scores[:k], 1):
        sample = samples[idx]
        print(f"{rank}. Score: {format_percentage(score)} (Amostra {idx + 1})")
        print(f"   ❓ {sample['question'][:70]}...")
        print()
    
    print(f"\n{'='*100}")
    print(f"💔 BOTTOM {k} AMOSTRAS COM PIOR DESEMPENHO")
    print(f"{'='*100}\n")
    
    for rank, (idx, score) in enumerate(scores[-k:], 1):
        sample = samples[idx]
        print(f"{rank}. Score: {format_percentage(score)} (Amostra {idx + 1})")
        print(f"   ❓ {sample['question'][:70]}...")
        print()


def print_metric_distribution(results: dict) -> None:
    """Imprime distribuição de cada métrica em formato ASCII."""
    samples = results.get("samples", [])
    
    if not samples:
        print("❌ Nenhuma amostra encontrada")
        return
    
    metrics = {
        "answer_semantic_similarity": "Similaridade Semântica",
        "faithfulness": "Fidelidade",
        "answer_relevancy": "Relevância",
        "context_recall": "Context Recall",
        "context_precision": "Context Precision",
    }
    
    print(f"\n{'='*100}")
    print("📊 DISTRIBUIÇÃO DE MÉTRICAS (histograma ASCII)")
    print(f"{'='*100}\n")
    
    for metric_key, metric_label in metrics.items():
        values = [s.get(metric_key) for s in samples if s.get(metric_key) is not None]
        
        if not values:
            print(f"{metric_label}: N/A\n")
            continue
        
        bins = 5
        hist, bin_edges = np.histogram(values, bins=bins)
        
        print(f"{metric_label}:")
        for i, count in enumerate(hist):
            bar_width = int(count * 30 / max(hist))
            bar = "█" * bar_width
            
            bin_start = bin_edges[i] * 100
            bin_end = bin_edges[i + 1] * 100
            
            print(
                f"  [{bin_start:.1f}%-{bin_end:.1f}%] "
                f"{bar} ({int(count)})"
            )
        print()


def create_html_report(results: dict, output_path: Path = None) -> None:
    """Cria relatório HTML interativo."""
    if output_path is None:
        output_path = Path("ragas_report.html")
    
    samples = results.get("samples", [])
    metadata = results.get("metadata", {})
    
    # Calcular estatísticas
    stats = {}
    
    metrics_names = [
        "answer_semantic_similarity",
        "faithfulness",
        "answer_relevancy",
        "context_recall",
        "context_precision",
    ]
    
    for metric in metrics_names:
        values = [s.get(metric) for s in samples if s.get(metric) is not None]
        
        if values:
            stats[metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "count": len(values),
            }
    
    metric_labels = {
        "answer_semantic_similarity": "Similaridade Semântica",
        "faithfulness": "Fidelidade",
        "answer_relevancy": "Relevância",
        "context_recall": "Context Recall",
        "context_precision": "Context Precision",
    }
    
    # Template HTML
    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório RAGAS - RAG PT-BR</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f5f5;
            color: #333;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}

        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}

        h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        .metadata {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
        }}

        .stat-card h3 {{
            color: #667eea;
            margin-bottom: 10px;
        }}

        .stat-value {{
            font-size: 2em;
            font-weight: bold;
        }}

        .metric-table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            margin-bottom: 30px;
        }}

        .metric-table th {{
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
        }}

        .metric-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
        }}

        .sample {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}

        .sample-metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }}

        .sample-metric {{
            background: #f5f5f5;
            padding: 8px;
            border-radius: 4px;
        }}
    </style>
</head>

<body>
    <div class="container">

        <header>
            <h1>📊 Relatório RAGAS - RAG PT-BR</h1>
            <p>Avaliação completa de qualidade do sistema RAG</p>
        </header>

        <div class="metadata">
            <div class="stat-card">
                <h3>Total de Amostras</h3>
                <div class="stat-value">{metadata.get('total_samples', 'N/A')}</div>
            </div>

            <div class="stat-card">
                <h3>Variante</h3>
                <div class="stat-value">{metadata.get('variant', 'N/A')}</div>
            </div>

            <div class="stat-card">
                <h3>Timestamp</h3>
                <div class="stat-value">{metadata.get('timestamp', 'N/A')}</div>
            </div>
        </div>

        <h2>📈 Métricas Agregadas</h2>

        <table class="metric-table">
            <thead>
                <tr>
                    <th>Métrica</th>
                    <th>Média</th>
                    <th>Desvio Padrão</th>
                    <th>Mínimo</th>
                    <th>Máximo</th>
                    <th>Amostras</th>
                </tr>
            </thead>

            <tbody>
"""

    # Adicionar linhas de métricas
    for metric_key, metric_label in metric_labels.items():
        if metric_key in stats:
            s = stats[metric_key]

            html += f"""
                <tr>
                    <td>{metric_label}</td>
                    <td>{format_percentage(s['mean'])}</td>
                    <td>{format_percentage(s['std'])}</td>
                    <td>{format_percentage(s['min'])}</td>
                    <td>{format_percentage(s['max'])}</td>
                    <td>{s['count']}</td>
                </tr>
            """

    html += """
            </tbody>
        </table>

        <h2>Amostras Avaliadas</h2>
"""

    # Adicionar amostras
    for i, sample in enumerate(samples[:5], 1):
        html += f"""
        <div class="sample">
            <h3>Amostra {i}</h3>

            <p><strong>Pergunta:</strong> {sample.get('question', 'N/A')}</p>

            <p style="margin-top: 10px;">
                <strong>Resposta:</strong>
                {sample.get('answer', 'N/A')[:150]}...
            </p>

            <div class="sample-metrics">
"""

        for metric_key in metrics_names:
            value = sample.get(metric_key)

            if value is not None:
                label = metric_labels.get(metric_key, metric_key)

                html += f"""
                    <div class="sample-metric">
                        {label}: {format_percentage(value)}
                    </div>
                """

        html += """
            </div>
        </div>
"""

    html += """
    </div>
</body>
</html>
"""

    # Salvar arquivo
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✅ Relatório HTML salvo em: {output_path}")


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    
    if not argv:
        print("❌ Use: python src/eval/ragas_dashboard.py <arquivo_resultados.json>")
        return 1
    
    results_path = argv[0]
    
    try:
        results = load_results(results_path)
    except Exception as e:
        print(f"❌ Erro ao carregar resultados: {e}")
        return 1
    
    # Imprimir relatórios
    print_metrics_table(results)
    print_metric_distribution(results)
    print_top_and_bottom_samples(results, k=3)
    print_detailed_sample_report(results, sample_idx=0)
    
    # Gerar relatório HTML
    create_html_report(results)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())