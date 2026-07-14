"""
Metrics Analysis Script

Example script for analyzing and visualizing metrics collected from the RAG system.
Loads metrics JSON files and generates reports.

Usage:
    python analyze_metrics.py metrics_20260714_103045.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
from datetime import datetime


class MetricsAnalyzer:
    """Analyze and report on collected metrics."""
    
    def __init__(self, metrics_file: str):
        """Load metrics from JSON file."""
        self.filepath = Path(metrics_file)
        
        if not self.filepath.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
        
        with open(self.filepath, 'r') as f:
            self.data = json.load(f)
    
    def print_executive_summary(self) -> None:
        """Print high-level summary of metrics."""
        print("\n" + "="*70)
        print("EXECUTIVE SUMMARY - RAG SYSTEM METRICS")
        print("="*70)
        
        # Session info
        session_start = self.data.get("session_start", "N/A")
        duration = self.data.get("session_duration_seconds", 0)
        print(f"\nSession Start: {session_start}")
        print(f"Duration: {duration:.2f} seconds")
        
        # Indexing
        index_stats = self.data.get("indexing_stats", {})
        if index_stats.get("total_reports_indexed"):
            print(f"\n{'INDEXING':-^70}")
            print(f"  Total Reports Indexed: {index_stats['total_reports_indexed']}")
            print(f"  Indexing Time: {index_stats['latest_indexing_duration_sec']:.2f}s")
            print(f"  Index Size: {index_stats['index_size_mb']:.2f} MB")
        
        # Retrieval
        retrieval_stats = self.data.get("retrieval_stats", {})
        if retrieval_stats.get("total_queries"):
            print(f"\n{'RETRIEVAL PERFORMANCE':-^70}")
            print(f"  Total Queries: {retrieval_stats['total_queries']}")
            print(f"  Avg Docs Retrieved: {retrieval_stats['avg_num_retrieved']:.1f}")
            print(f"  Avg Retrieval Distance: {retrieval_stats['avg_retrieval_distance']:.4f}")
            if retrieval_stats.get("avg_precision"):
                print(f"  Avg Precision: {retrieval_stats['avg_precision']:.3f}")
            if retrieval_stats.get("avg_recall"):
                print(f"  Avg Recall: {retrieval_stats['avg_recall']:.3f}")
        
        # Response Time
        response_stats = self.data.get("response_time_stats", {})
        if response_stats.get("total_queries"):
            print(f"\n{'RESPONSE TIME PERFORMANCE':-^70}")
            print(f"  Total Queries: {response_stats['total_queries']}")
            print(f"  Avg Response Time: {response_stats['avg_response_time_ms']:.2f}ms")
            print(f"  Median Response Time: {response_stats['median_response_time_ms']:.2f}ms")
            print(f"  P95 Response Time: {response_stats['p95_response_time_ms']:.2f}ms")
            print(f"  P99 Response Time: {response_stats['p99_response_time_ms']:.2f}ms")
        
        # Relevance
        relevance_stats = self.data.get("relevance_stats", {})
        if relevance_stats.get("total_feedback"):
            print(f"\n{'USER RELEVANCE FEEDBACK':-^70}")
            print(f"  Total Feedback Entries: {relevance_stats['total_feedback']}")
            print(f"  Relevant: {relevance_stats['relevant_count']}")
            print(f"  Irrelevant: {relevance_stats['irrelevant_count']}")
            print(f"  Precision: {relevance_stats['precision_from_feedback']:.3f}")
            if relevance_stats.get("avg_relevance_score"):
                print(f"  Avg Relevance Score: {relevance_stats['avg_relevance_score']:.2f}/1.0")
        
        print("\n" + "="*70 + "\n")
    
    def print_detailed_retrieval_analysis(self) -> None:
        """Detailed analysis of retrieval performance."""
        print("\n" + "="*70)
        print("DETAILED RETRIEVAL ANALYSIS")
        print("="*70)
        
        retrieval_queries = self.data.get("all_events", {}).get("retrieval_queries", [])
        
        if not retrieval_queries:
            print("\nNo retrieval queries to analyze.")
            return
        
        print(f"\nAnalyzing {len(retrieval_queries)} retrieval queries...\n")
        
        # Distance distribution
        all_distances = []
        for query in retrieval_queries:
            all_distances.extend(query.get("distances", []))
        
        if all_distances:
            all_distances = np.array(all_distances)
            print("DISTANCE DISTRIBUTION (L2 metric)")
            print(f"  Min: {np.min(all_distances):.4f}")
            print(f"  25th percentile: {np.percentile(all_distances, 25):.4f}")
            print(f"  Median: {np.median(all_distances):.4f}")
            print(f"  75th percentile: {np.percentile(all_distances, 75):.4f}")
            print(f"  Max: {np.max(all_distances):.4f}")
            print(f"  Mean: {np.mean(all_distances):.4f}")
            print(f"  Std Dev: {np.std(all_distances):.4f}")
        
        # Precision/Recall if available
        precisions = [q.get("precision") for q in retrieval_queries if "precision" in q]
        recalls = [q.get("recall") for q in retrieval_queries if "recall" in q]
        f1_scores = [q.get("f1_score") for q in retrieval_queries if "f1_score" in q]
        
        if precisions:
            print(f"\nPRECISION ANALYSIS ({len(precisions)} queries with ground truth)")
            print(f"  Min: {np.min(precisions):.3f}")
            print(f"  Median: {np.median(precisions):.3f}")
            print(f"  Max: {np.max(precisions):.3f}")
            print(f"  Mean: {np.mean(precisions):.3f}")
        
        if recalls:
            print(f"\nRECALL ANALYSIS ({len(recalls)} queries with ground truth)")
            print(f"  Min: {np.min(recalls):.3f}")
            print(f"  Median: {np.median(recalls):.3f}")
            print(f"  Max: {np.max(recalls):.3f}")
            print(f"  Mean: {np.mean(recalls):.3f}")
        
        if f1_scores:
            print(f"\nF1 SCORE ANALYSIS ({len(f1_scores)} queries with ground truth)")
            print(f"  Min: {np.min(f1_scores):.3f}")
            print(f"  Median: {np.median(f1_scores):.3f}")
            print(f"  Max: {np.max(f1_scores):.3f}")
            print(f"  Mean: {np.mean(f1_scores):.3f}")
        
        print("\n" + "="*70 + "\n")
    
    def print_response_time_analysis(self) -> None:
        """Detailed response time analysis."""
        print("\n" + "="*70)
        print("RESPONSE TIME ANALYSIS")
        print("="*70)
        
        response_times = self.data.get("all_events", {}).get("response_times", [])
        
        if not response_times:
            print("\nNo response times to analyze.")
            return
        
        times = [rt["duration_ms"] for rt in response_times]
        times = np.array(times)
        
        print(f"\nAnalyzing {len(times)} response times...\n")
        print(f"RESPONSE TIME STATISTICS")
        print(f"  Min: {np.min(times):.2f}ms")
        print(f"  25th percentile: {np.percentile(times, 25):.2f}ms")
        print(f"  Median: {np.median(times):.2f}ms")
        print(f"  75th percentile: {np.percentile(times, 75):.2f}ms")
        print(f"  Max: {np.max(times):.2f}ms")
        print(f"  Mean: {np.mean(times):.2f}ms")
        print(f"  Std Dev: {np.std(times):.2f}ms")
        
        # Performance bands
        fast = sum(1 for t in times if t < 2000)
        acceptable = sum(1 for t in times if 2000 <= t < 5000)
        slow = sum(1 for t in times if t >= 5000)
        
        print(f"\nPERFORMANCE BANDS")
        print(f"  Fast (<2000ms): {fast} ({100*fast/len(times):.1f}%)")
        print(f"  Acceptable (2-5s): {acceptable} ({100*acceptable/len(times):.1f}%)")
        print(f"  Slow (>5s): {slow} ({100*slow/len(times):.1f}%)")
        
        print("\n" + "="*70 + "\n")
    
    def print_relevance_feedback_analysis(self) -> None:
        """Detailed relevance feedback analysis."""
        print("\n" + "="*70)
        print("USER RELEVANCE FEEDBACK ANALYSIS")
        print("="*70)
        
        relevance_scores = self.data.get("all_events", {}).get("relevance_scores", [])
        
        if not relevance_scores:
            print("\nNo relevance feedback to analyze.")
            return
        
        print(f"\nAnalyzing {len(relevance_scores)} feedback entries...\n")
        
        # Binary relevance
        relevant = sum(1 for r in relevance_scores if r["is_relevant"])
        irrelevant = len(relevance_scores) - relevant
        
        print(f"BINARY RELEVANCE JUDGMENTS")
        print(f"  Relevant: {relevant} ({100*relevant/len(relevance_scores):.1f}%)")
        print(f"  Irrelevant: {irrelevant} ({100*irrelevant/len(relevance_scores):.1f}%)")
        
        # Scored feedback
        scored = [r for r in relevance_scores if r["relevance_score"] is not None]
        if scored:
            scores = np.array([r["relevance_score"] for r in scored])
            print(f"\nSCORED RELEVANCE ({len(scored)} entries)")
            print(f"  Min: {np.min(scores):.2f}")
            print(f"  25th percentile: {np.percentile(scores, 25):.2f}")
            print(f"  Median: {np.median(scores):.2f}")
            print(f"  75th percentile: {np.percentile(scores, 75):.2f}")
            print(f"  Max: {np.max(scores):.2f}")
            print(f"  Mean: {np.mean(scores):.2f}")
            
            # Score bands
            poor = sum(1 for s in scores if s < 0.3)
            fair = sum(1 for s in scores if 0.3 <= s < 0.6)
            good = sum(1 for s in scores if 0.6 <= s < 0.8)
            excellent = sum(1 for s in scores if s >= 0.8)
            
            print(f"\nSCORE DISTRIBUTION")
            print(f"  Poor (<0.3): {poor} ({100*poor/len(scores):.1f}%)")
            print(f"  Fair (0.3-0.6): {fair} ({100*fair/len(scores):.1f}%)")
            print(f"  Good (0.6-0.8): {good} ({100*good/len(scores):.1f}%)")
            print(f"  Excellent (≥0.8): {excellent} ({100*excellent/len(scores):.1f}%)")
        
        print("\n" + "="*70 + "\n")
    
    def generate_html_report(self, output_file: str = "metrics_report.html") -> str:
        """Generate an HTML report of metrics."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>RAG Metrics Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; border-bottom: 3px solid #2196F3; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; }
        .metric { background: white; padding: 15px; margin: 10px 0; border-radius: 5px; 
                  box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-name { font-weight: bold; color: #2196F3; }
        .metric-value { font-size: 1.3em; color: #333; }
        table { width: 100%; border-collapse: collapse; background: white; margin: 15px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background: #2196F3; color: white; }
        tr:nth-child(even) { background: #f9f9f9; }
        .good { color: #4CAF50; font-weight: bold; }
        .warning { color: #FF9800; font-weight: bold; }
        .critical { color: #F44336; font-weight: bold; }
    </style>
</head>
<body>
"""
        
        html += "<h1>📊 RAG System Metrics Report</h1>\n"
        html += f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n"
        
        # Summary section
        html += "<h2>Summary</h2>\n"
        session_start = self.data.get("session_start", "N/A")
        duration = self.data.get("session_duration_seconds", 0)
        html += f"<div class='metric'><span class='metric-name'>Session Start:</span> {session_start}</div>\n"
        html += f"<div class='metric'><span class='metric-name'>Duration:</span> <span class='metric-value'>{duration:.2f}s</span></div>\n"
        
        # Indexing metrics
        index_stats = self.data.get("indexing_stats", {})
        if index_stats.get("total_reports_indexed"):
            html += "<h2>Indexing Metrics</h2>\n"
            html += "<table>\n"
            html += "<tr><th>Metric</th><th>Value</th></tr>\n"
            html += f"<tr><td>Total Reports Indexed</td><td>{index_stats['total_reports_indexed']}</td></tr>\n"
            html += f"<tr><td>Indexing Duration</td><td>{index_stats['latest_indexing_duration_sec']:.2f}s</td></tr>\n"
            html += f"<tr><td>Index Size</td><td>{index_stats['index_size_mb']:.2f} MB</td></tr>\n"
            html += "</table>\n"
        
        # Retrieval metrics
        retrieval_stats = self.data.get("retrieval_stats", {})
        if retrieval_stats.get("total_queries"):
            html += "<h2>Retrieval Metrics</h2>\n"
            html += "<table>\n"
            html += "<tr><th>Metric</th><th>Value</th></tr>\n"
            html += f"<tr><td>Total Queries</td><td>{retrieval_stats['total_queries']}</td></tr>\n"
            html += f"<tr><td>Avg Documents Retrieved</td><td>{retrieval_stats['avg_num_retrieved']:.1f}</td></tr>\n"
            html += f"<tr><td>Avg Retrieval Distance</td><td>{retrieval_stats['avg_retrieval_distance']:.4f}</td></tr>\n"
            if retrieval_stats.get("avg_precision"):
                precision_class = "good" if retrieval_stats["avg_precision"] > 0.7 else "warning"
                html += f"<tr><td>Avg Precision</td><td class='{precision_class}'>{retrieval_stats['avg_precision']:.3f}</td></tr>\n"
            html += "</table>\n"
        
        # Response time metrics
        response_stats = self.data.get("response_time_stats", {})
        if response_stats.get("total_queries"):
            html += "<h2>Response Time Metrics</h2>\n"
            html += "<table>\n"
            html += "<tr><th>Metric</th><th>Value</th></tr>\n"
            html += f"<tr><td>Total Queries</td><td>{response_stats['total_queries']}</td></tr>\n"
            avg_response = response_stats['avg_response_time_ms']
            response_class = "good" if avg_response < 3000 else "warning" if avg_response < 5000 else "critical"
            html += f"<tr><td>Avg Response Time</td><td class='{response_class}'>{avg_response:.2f}ms</td></tr>\n"
            html += f"<tr><td>Median Response Time</td><td>{response_stats['median_response_time_ms']:.2f}ms</td></tr>\n"
            html += f"<tr><td>P95 Response Time</td><td>{response_stats['p95_response_time_ms']:.2f}ms</td></tr>\n"
            html += "</table>\n"
        
        html += "</body>\n</html>"
        
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            f.write(html)
        
        return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze metrics from RAG system"
    )
    parser.add_argument("metrics_file", help="Path to metrics JSON file")
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML report"
    )
    parser.add_argument(
        "--output",
        default="metrics_report.html",
        help="Output file for HTML report"
    )
    
    args = parser.parse_args()
    
    try:
        analyzer = MetricsAnalyzer(args.metrics_file)
        
        # Print all analyses
        analyzer.print_executive_summary()
        analyzer.print_detailed_retrieval_analysis()
        analyzer.print_response_time_analysis()
        analyzer.print_relevance_feedback_analysis()
        
        # Generate HTML if requested
        if args.html:
            html_file = analyzer.generate_html_report(args.output)
            print(f"\n✅ HTML report generated: {html_file}\n")
    
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
