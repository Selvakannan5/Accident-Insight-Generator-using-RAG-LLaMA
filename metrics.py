"""
Metrics instrumentation module for tracking RAG system performance.

Tracks:
- Retrieval accuracy and relevance
- Number of reports indexed
- Response time
- Precision/Recall scores
"""

import time
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np


class MetricsTracker:
    """Track and aggregate performance metrics for RAG system."""
    
    def __init__(self, log_dir: str = "metrics_logs"):
        """Initialize metrics tracker with logging directory."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.metrics = {
            "retrieval_queries": [],
            "indexing_events": [],
            "response_times": [],
            "relevance_scores": [],
            "indexed_reports_count": 0,
        }
        self.session_start = datetime.now()
    
    # ==================== Indexing Metrics ====================
    
    def log_indexing_start(self, num_reports: int) -> float:
        """Log start of indexing operation."""
        start_time = time.time()
        self.metrics["indexed_reports_count"] = num_reports
        
        event = {
            "timestamp": datetime.now().isoformat(),
            "event": "indexing_start",
            "num_reports": num_reports,
            "start_time": start_time
        }
        self.metrics["indexing_events"].append(event)
        return start_time
    
    def log_indexing_end(self, start_time: float, index_size_mb: float) -> Dict:
        """Log end of indexing operation and return statistics."""
        end_time = time.time()
        duration = end_time - start_time
        
        event = {
            "timestamp": datetime.now().isoformat(),
            "event": "indexing_end",
            "duration_seconds": duration,
            "index_size_mb": index_size_mb,
            "reports_indexed": self.metrics["indexed_reports_count"]
        }
        self.metrics["indexing_events"].append(event)
        
        return {
            "indexing_duration": duration,
            "index_size_mb": index_size_mb,
            "reports_indexed": self.metrics["indexed_reports_count"],
            "time_per_report_ms": (duration * 1000) / self.metrics["indexed_reports_count"]
        }
    
    def get_indexing_stats(self) -> Dict:
        """Get aggregated indexing statistics."""
        if not self.metrics["indexing_events"]:
            return {"status": "No indexing events"}
        
        indexing_events = [e for e in self.metrics["indexing_events"] if e["event"] == "indexing_end"]
        if not indexing_events:
            return {"status": "Indexing not completed"}
        
        latest = indexing_events[-1]
        return {
            "total_reports_indexed": latest.get("reports_indexed", 0),
            "latest_indexing_duration_sec": latest.get("duration_seconds", 0),
            "index_size_mb": latest.get("index_size_mb", 0),
            "timestamp": latest.get("timestamp", "")
        }
    
    # ==================== Response Time Metrics ====================
    
    def start_response_timer(self) -> float:
        """Start timer for response generation."""
        return time.time()
    
    def log_response_time(self, start_time: float, query: str, num_similar_docs: int = 5) -> Dict:
        """Log response generation time."""
        end_time = time.time()
        duration = end_time - start_time
        
        response_metric = {
            "timestamp": datetime.now().isoformat(),
            "query_length": len(query),
            "duration_ms": duration * 1000,
            "num_similar_docs": num_similar_docs
        }
        self.metrics["response_times"].append(response_metric)
        
        return {
            "response_time_ms": duration * 1000,
            "query_length": len(query),
            "num_similar_docs": num_similar_docs
        }
    
    def get_response_time_stats(self) -> Dict:
        """Get aggregated response time statistics."""
        if not self.metrics["response_times"]:
            return {"status": "No response times recorded"}
        
        times = [rt["duration_ms"] for rt in self.metrics["response_times"]]
        return {
            "total_queries": len(times),
            "avg_response_time_ms": np.mean(times),
            "median_response_time_ms": np.median(times),
            "min_response_time_ms": np.min(times),
            "max_response_time_ms": np.max(times),
            "p95_response_time_ms": np.percentile(times, 95),
            "p99_response_time_ms": np.percentile(times, 99)
        }
    
    # ==================== Retrieval Accuracy Metrics ====================
    
    def log_retrieval(self, query: str, retrieved_docs: List[str], distances: np.ndarray, 
                     ground_truth_relevant: Optional[List[int]] = None) -> Dict:
        """
        Log retrieval operation with optional ground truth for evaluation.
        
        Args:
            query: The query string
            retrieved_docs: List of retrieved document texts
            distances: Array of distances/scores from FAISS
            ground_truth_relevant: List of indices of relevant documents (if available)
        
        Returns:
            Dictionary with retrieval metrics
        """
        retrieval_metric = {
            "timestamp": datetime.now().isoformat(),
            "query": query[:100],  # Store first 100 chars
            "num_retrieved": len(retrieved_docs),
            "distances": distances.tolist() if isinstance(distances, np.ndarray) else distances,
            "ground_truth_available": ground_truth_relevant is not None
        }
        
        # Calculate relevance score if ground truth available
        precision = None
        recall = None
        if ground_truth_relevant is not None:
            relevant_retrieved = sum(1 for i in range(len(retrieved_docs)) 
                                    if i in ground_truth_relevant)
            precision = relevant_retrieved / len(retrieved_docs) if retrieved_docs else 0
            recall = relevant_retrieved / len(ground_truth_relevant) if ground_truth_relevant else 0
            
            retrieval_metric["precision"] = precision
            retrieval_metric["recall"] = recall
            retrieval_metric["f1_score"] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        self.metrics["retrieval_queries"].append(retrieval_metric)
        
        # Calculate distance-based relevance (lower is better with L2)
        mean_distance = np.mean(distances)
        
        return {
            "num_retrieved": len(retrieved_docs),
            "mean_retrieval_distance": mean_distance,
            "min_distance": np.min(distances),
            "max_distance": np.max(distances),
            "precision": precision,
            "recall": recall
        }
    
    def get_retrieval_stats(self) -> Dict:
        """Get aggregated retrieval statistics."""
        if not self.metrics["retrieval_queries"]:
            return {"status": "No retrieval queries recorded"}
        
        queries = self.metrics["retrieval_queries"]
        distances = []
        precisions = []
        recalls = []
        
        for q in queries:
            distances.extend(q.get("distances", []))
            if "precision" in q:
                precisions.append(q["precision"])
            if "recall" in q:
                recalls.append(q["recall"])
        
        stats = {
            "total_queries": len(queries),
            "avg_num_retrieved": np.mean([q["num_retrieved"] for q in queries]),
            "avg_retrieval_distance": np.mean(distances) if distances else None,
        }
        
        if precisions:
            stats["avg_precision"] = np.mean(precisions)
            stats["median_precision"] = np.median(precisions)
        
        if recalls:
            stats["avg_recall"] = np.mean(recalls)
            stats["median_recall"] = np.median(recalls)
        
        return stats
    
    # ==================== Relevance/Precision Scoring ====================
    
    def log_user_feedback(self, query: str, retrieved_doc_idx: int, is_relevant: bool, 
                         relevance_score: Optional[float] = None) -> None:
        """
        Log user feedback on retrieval relevance.
        
        Args:
            query: The query string
            retrieved_doc_idx: Index of the retrieved document
            is_relevant: Whether user found it relevant
            relevance_score: Optional 0-1 relevance score from user
        """
        feedback = {
            "timestamp": datetime.now().isoformat(),
            "query": query[:100],
            "doc_index": retrieved_doc_idx,
            "is_relevant": is_relevant,
            "relevance_score": relevance_score
        }
        self.metrics["relevance_scores"].append(feedback)
    
    def get_relevance_stats(self) -> Dict:
        """Get aggregated relevance statistics from user feedback."""
        if not self.metrics["relevance_scores"]:
            return {"status": "No relevance feedback recorded"}
        
        feedback = self.metrics["relevance_scores"]
        relevant_count = sum(1 for f in feedback if f["is_relevant"])
        
        scored_feedback = [f for f in feedback if f["relevance_score"] is not None]
        
        stats = {
            "total_feedback": len(feedback),
            "relevant_count": relevant_count,
            "irrelevant_count": len(feedback) - relevant_count,
            "precision_from_feedback": relevant_count / len(feedback) if feedback else 0
        }
        
        if scored_feedback:
            scores = [f["relevance_score"] for f in scored_feedback]
            stats["avg_relevance_score"] = np.mean(scores)
            stats["median_relevance_score"] = np.median(scores)
        
        return stats
    
    # ==================== Summary & Reporting ====================
    
    def get_session_summary(self) -> Dict:
        """Get complete session metrics summary."""
        return {
            "session_start": self.session_start.isoformat(),
            "session_duration_seconds": (datetime.now() - self.session_start).total_seconds(),
            "indexing_stats": self.get_indexing_stats(),
            "response_time_stats": self.get_response_time_stats(),
            "retrieval_stats": self.get_retrieval_stats(),
            "relevance_stats": self.get_relevance_stats()
        }
    
    def save_metrics_to_file(self, filename: Optional[str] = None) -> str:
        """Save all metrics to JSON file."""
        if filename is None:
            filename = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.log_dir / filename
        
        summary = self.get_session_summary()
        summary["all_events"] = self.metrics
        
        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)
        
        return str(filepath)
    
    def print_summary(self) -> None:
        """Print formatted metrics summary to console."""
        summary = self.get_session_summary()
        
        print("\n" + "="*60)
        print("RAG SYSTEM METRICS SUMMARY")
        print("="*60)
        
        print(f"\nSession Duration: {summary['session_duration_seconds']:.2f}s")
        
        # Indexing
        if summary["indexing_stats"].get("total_reports_indexed"):
            print(f"\n--- INDEXING METRICS ---")
            print(f"Total Reports Indexed: {summary['indexing_stats']['total_reports_indexed']}")
            print(f"Indexing Duration: {summary['indexing_stats']['latest_indexing_duration_sec']:.2f}s")
            print(f"Index Size: {summary['indexing_stats']['index_size_mb']:.2f} MB")
        
        # Response Times
        if summary["response_time_stats"].get("total_queries"):
            print(f"\n--- RESPONSE TIME METRICS ---")
            print(f"Total Queries: {summary['response_time_stats']['total_queries']}")
            print(f"Avg Response Time: {summary['response_time_stats']['avg_response_time_ms']:.2f}ms")
            print(f"Median Response Time: {summary['response_time_stats']['median_response_time_ms']:.2f}ms")
            print(f"P95 Response Time: {summary['response_time_stats']['p95_response_time_ms']:.2f}ms")
            print(f"P99 Response Time: {summary['response_time_stats']['p99_response_time_ms']:.2f}ms")
        
        # Retrieval
        if summary["retrieval_stats"].get("total_queries"):
            print(f"\n--- RETRIEVAL METRICS ---")
            print(f"Total Queries: {summary['retrieval_stats']['total_queries']}")
            print(f"Avg Documents Retrieved: {summary['retrieval_stats']['avg_num_retrieved']:.1f}")
            print(f"Avg Retrieval Distance: {summary['retrieval_stats']['avg_retrieval_distance']:.4f}")
            if summary["retrieval_stats"].get("avg_precision"):
                print(f"Avg Precision: {summary['retrieval_stats']['avg_precision']:.3f}")
            if summary["retrieval_stats"].get("avg_recall"):
                print(f"Avg Recall: {summary['retrieval_stats']['avg_recall']:.3f}")
        
        # Relevance
        if summary["relevance_stats"].get("total_feedback"):
            print(f"\n--- RELEVANCE METRICS (from user feedback) ---")
            print(f"Total Feedback: {summary['relevance_stats']['total_feedback']}")
            print(f"Relevant: {summary['relevance_stats']['relevant_count']}")
            print(f"Irrelevant: {summary['relevance_stats']['irrelevant_count']}")
            print(f"Precision: {summary['relevance_stats']['precision_from_feedback']:.3f}")
            if summary["relevance_stats"].get("avg_relevance_score"):
                print(f"Avg Relevance Score: {summary['relevance_stats']['avg_relevance_score']:.2f}/1.0")
        
        print("\n" + "="*60 + "\n")


# Singleton instance for global use
_metrics_tracker = None


def get_metrics_tracker() -> MetricsTracker:
    """Get or create global metrics tracker instance."""
    global _metrics_tracker
    if _metrics_tracker is None:
        _metrics_tracker = MetricsTracker()
    return _metrics_tracker
