"""
Integração de avaliação contínua com o backend FastAPI

Fornece:
- Endpoint de avaliação rápida
- Cache de resultados
- Monitoramento de performance em tempo real
- Comparação entre variantes (baseline vs new)

Uso:
    python src/eval/backend_eval_integration.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Any
from dataclasses import dataclass, asdict

from src import config
from src.rag_query import answer, retrieve, rerank
from src.eval.ragas_evaluator import load_eval_dataset, EvalSample, calculate_semantic_similarity


@dataclass
class EvalMetrics:
    """Container para métricas de avaliação."""
    question: str
    answer_text: str
    contexts_count: int
    latency_ms: float
    semantic_similarity: float
    variant: str


def evaluate_single_question(question: str, variant: str = "new") -> EvalMetrics:
    """Avalia uma pergunta única e retorna métricas."""
    start_time = time.time()
    
    try:
        # Executar pipeline
        result = answer(question, variant=variant)
        
        # Extrair dados
        answer_text = result["answer"]
        contexts = [c["content"] for c in result.get("retrieved_chunks", [])]
        contexts_count = len(contexts)
        
        latency_ms = (time.time() - start_time) * 1000
        
        return EvalMetrics(
            question=question,
            answer_text=answer_text,
            contexts_count=contexts_count,
            latency_ms=latency_ms,
            semantic_similarity=0.0,  # Sem ground truth
            variant=variant,
        )
    except Exception as e:
        print(f"❌ Erro ao avaliar: {e}")
        raise


def compare_variants(question: str) -> dict[str, Any]:
    """Compara performance entre variantes 'new' e 'baseline'."""
    results = {}
    
    for variant in ["new", "baseline"]:
        try:
            metrics = evaluate_single_question(question, variant=variant)
            results[variant] = asdict(metrics)
        except Exception as e:
            results[variant] = {"error": str(e)}
    
    return results


def benchmark_retrievers(num_samples: int = 10) -> dict[str, Any]:
    """Benchmark dos retriever/reranker."""
    dataset = load_eval_dataset()
    dataset = dataset[:num_samples]
    
    results = {
        "timestamp": str(datetime.now()),
        "samples": num_samples,
        "variants": {}
    }
    
    for variant in ["new", "baseline"]:
        print(f"\n🔄 Avaliando variante: {variant}")
        
        latencies = []
        context_counts = []
        
        for sample_data in dataset:
            try:
                question = sample_data["question"]
                start = time.time()
                
                if variant == "new":
                    from src.rag_query import (
                        retrieve as retrieve_fn,
                        rerank as rerank_fn,
                        _COLLECTION_NEW_PT,
                        _EMBEDDER_MODEL,
                    )
                    retrieved = retrieve_fn(question, variant=variant)
                    reranked = rerank_fn(question, retrieved, variant=variant)
                else:
                    from src.rag_query import (
                        retrieve as retrieve_fn,
                        rerank as rerank_fn,
                    )
                    retrieved = retrieve_fn(question, variant=variant)
                    reranked = rerank_fn(question, retrieved, variant=variant)
                
                latency = (time.time() - start) * 1000
                latencies.append(latency)
                context_counts.append(len(reranked))
                
            except Exception as e:
                print(f"⚠️  Erro em {variant}: {e}")
        
        if latencies:
            import numpy as np
            results["variants"][variant] = {
                "latency_mean_ms": float(np.mean(latencies)),
                "latency_std_ms": float(np.std(latencies)),
                "latency_min_ms": float(np.min(latencies)),
                "latency_max_ms": float(np.max(latencies)),
                "avg_contexts": float(np.mean(context_counts)),
            }
    
    return results


def generate_eval_badge(metrics: dict) -> str:
    """Gera badge SVG com score de avaliação."""
    avg_score = 0
    count = 0
    
    for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        if key in metrics and metrics[key] is not None:
            avg_score += metrics[key]
            count += 1
    
    if count > 0:
        avg_score /= count
    else:
        avg_score = 0
    
    # Cor baseada no score
    if avg_score >= 0.8:
        color = "#00aa00"  # Verde
    elif avg_score >= 0.6:
        color = "#ffaa00"  # Laranja
    else:
        color = "#aa0000"  # Vermelho
    
    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" 
         width="180" height="20" role="img" aria-label="RAGAS Score: {avg_score:.2f}">
        <title>RAGAS Score: {avg_score:.2f}</title>
        <linearGradient id="s" x2="0" y2="100%">
            <stop offset="0" stop-color="#bbb"/>
            <stop offset="1" stop-color="#999"/>
        </linearGradient>
        <clipPath id="r">
            <rect width="180" height="20" rx="3" fill="#fff"/>
        </clipPath>
        <g clip-path="url(#r)">
            <rect width="126" height="20" fill="#555"/>
            <rect x="126" width="54" height="20" fill="{color}"/>
            <rect width="180" height="20" fill="url(#s)"/>
        </g>
        <g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" 
           text-rendering="geometricPrecision" font-size="110">
            <text x="640" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)" 
                  textLength="1160">RAGAS Score</text>
            <text x="640" y="140" transform="scale(.1)" fill="#fff" textLength="1160">RAGAS Score</text>
            <text x="1515" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)" 
                  textLength="440">{avg_score:.2f}</text>
            <text x="1515" y="140" transform="scale(.1)" fill="#fff" textLength="440">{avg_score:.2f}</text>
        </g>
    </svg>
    """
    return svg.strip()


def save_eval_report(results: dict, output_path: Path = None) -> None:
    """Salva relatório de avaliação em JSON."""
    if output_path is None:
        output_path = config.EVAL_DIR / f"eval_report_{datetime.now().isoformat()}.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Relatório salvo em: {output_path}")


def print_benchmark_results(results: dict) -> None:
    """Imprime resultados do benchmark."""
    print(f"\n{'='*80}")
    print("⚡ BENCHMARK DE PERFORMANCE")
    print(f"{'='*80}\n")
    
    for variant, metrics in results.get("variants", {}).items():
        if "error" in metrics:
            print(f"❌ {variant}: {metrics['error']}")
            continue
        
        print(f"🔄 Variante: {variant}")
        print(f"   - Latência Média: {metrics['latency_mean_ms']:.1f} ms")
        print(f"   - Desvio Padrão: {metrics['latency_std_ms']:.1f} ms")
        print(f"   - Mínimo: {metrics['latency_min_ms']:.1f} ms")
        print(f"   - Máximo: {metrics['latency_max_ms']:.1f} ms")
        print(f"   - Contextos Médios: {metrics['avg_contexts']:.1f}")
        print()


if __name__ == "__main__":
    # Exemplo de uso
    print("🚀 Iniciando avaliação do backend...\n")
    
    # Benchmark
    benchmark_results = benchmark_retrievers(num_samples=5)
    print_benchmark_results(benchmark_results)
    
    # Salvar
    save_eval_report(benchmark_results)
    
    print("\n✅ Avaliação concluída!")