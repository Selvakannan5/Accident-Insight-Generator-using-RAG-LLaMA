# Metrics Implementation Guide

## Overview

This document describes the metrics instrumentation added to the Accident Insight Generator to track RAG system performance across four key dimensions:

1. **Retrieval Accuracy** - Quality of document retrieval
2. **Number of Reports Indexed** - Dataset indexing statistics
3. **Response Time** - Generation latency
4. **Relevance/Precision Score** - User-validated relevance feedback

## Architecture

### Core Module: `metrics.py`

The `MetricsTracker` class provides comprehensive performance tracking:

```python
from metrics import get_metrics_tracker

metrics = get_metrics_tracker()  # Singleton instance
```

## Metrics Categories

### 1. Indexing Metrics

Tracks dataset indexing performance.

**Methods:**
- `log_indexing_start(num_reports)` - Mark indexing start
- `log_indexing_end(start_time, index_size_mb)` - Mark indexing completion
- `get_indexing_stats()` - Retrieve indexing statistics

**Example Output:**
```json
{
  "total_reports_indexed": 200,
  "latest_indexing_duration_sec": 2.45,
  "index_size_mb": 0.45,
  "timestamp": "2026-07-14T10:30:45.123456"
}
```

**What's Tracked:**
- Total reports indexed
- Indexing duration (seconds)
- Index file size (MB)
- Time per report (ms)

---

### 2. Response Time Metrics

Measures LLM response generation latency.

**Methods:**
- `start_response_timer()` - Start timer
- `log_response_time(start_time, query, num_similar_docs)` - Log completion

**Example Output:**
```json
{
  "total_queries": 12,
  "avg_response_time_ms": 3250.45,
  "median_response_time_ms": 3100.00,
  "min_response_time_ms": 2800.00,
  "max_response_time_ms": 4200.00,
  "p95_response_time_ms": 4050.00,
  "p99_response_time_ms": 4150.00
}
```

**What's Tracked:**
- Query count
- Average/median/min/max response times
- P95, P99 percentiles
- Query length
- Number of similar documents retrieved

---

### 3. Retrieval Accuracy Metrics

Evaluates semantic search quality.

**Methods:**
- `log_retrieval(query, retrieved_docs, distances, ground_truth_relevant)` - Log retrieval
- `get_retrieval_stats()` - Get aggregated stats

**Example Output:**
```json
{
  "total_queries": 12,
  "avg_num_retrieved": 5.0,
  "avg_retrieval_distance": 0.8234,
  "avg_precision": 0.75,
  "median_precision": 0.80,
  "avg_recall": 0.68,
  "median_recall": 0.70
}
```

**Distance Metrics:**
- Uses L2 distance from FAISS IndexFlatL2
- Lower distances = more similar documents
- Mean, min, max distances tracked

**Evaluation Metrics (if ground truth available):**
- **Precision**: Fraction of retrieved documents that are relevant
- **Recall**: Fraction of relevant documents that are retrieved
- **F1 Score**: Harmonic mean of precision and recall

---

### 4. Relevance/Precision Scoring

Collects user feedback on result quality.

**Methods:**
- `log_user_feedback(query, retrieved_doc_idx, is_relevant, relevance_score)` - Record feedback
- `get_relevance_stats()` - Get aggregated feedback

**Example Output:**
```json
{
  "total_feedback": 15,
  "relevant_count": 12,
  "irrelevant_count": 3,
  "precision_from_feedback": 0.80,
  "avg_relevance_score": 0.78,
  "median_relevance_score": 0.80
}
```

**What's Tracked:**
- Binary relevance judgments (relevant/irrelevant)
- Continuous relevance scores (0-1)
- Precision calculated from feedback
- Average/median relevance scores

---

## Integration with Streamlit App

### 1. Initialization

```python
from metrics import get_metrics_tracker

metrics = get_metrics_tracker()  # Global singleton
```

### 2. Indexing Tracking

```python
@st.cache_data
def load_dataset_and_index():
    indexing_start = metrics.log_indexing_start(num_reports=1000)
    
    # ... indexing code ...
    
    index_size_mb = os.path.getsize(DATASET_PATH) / (1024 * 1024)
    indexing_stats = metrics.log_indexing_end(indexing_start, index_size_mb)
    
    return df, embedder, index, indexing_stats
```

### 3. Response Generation Tracking

```python
def generate_precaution_ollama(...):
    # Start timer
    response_timer = metrics.start_response_timer()
    
    # ... retrieval code ...
    
    # Log retrieval
    retrieval_metrics = metrics.log_retrieval(
        query=new_report[:200],
        retrieved_docs=similar_reports,
        distances=D[0]
    )
    
    # ... LLM generation ...
    
    # Log response time
    response_time_metrics = metrics.log_response_time(
        response_timer, new_report, num_similar_docs=k
    )
    
    return output, retrieval_metrics, response_time_metrics
```

### 4. UI Integration

**Sidebar Metrics:**
```
📊 System Metrics
┌─ Show Indexing Stats
│  ├─ Reports Indexed: 200
│  ├─ Indexing Time (s): 2.45
│  └─ Time per Report (ms): 12.25
└─ Show Metrics Dashboard
```

**Expandable Results Metrics:**
```
📈 Performance Metrics
┌─ Retrieval Metrics
│  ├─ Documents Retrieved: 5
│  ├─ Mean Distance: 0.8234
│  ├─ Min Distance: 0.6123
│  └─ Max Distance: 0.9876
├─ Response Metrics
│  ├─ Response Time (ms): 3245.67
│  └─ Query Length: 350
└─ Rate Relevance
   └─ Slider: [0 ←→ 1]
```

---

## Usage Examples

### Example 1: Track a Complete Session

```python
from metrics import get_metrics_tracker

metrics = get_metrics_tracker()

# Indexing
start = metrics.log_indexing_start(200)
# ... indexing ...
metrics.log_indexing_end(start, 0.45)

# Query
response_timer = metrics.start_response_timer()
retrieval = metrics.log_retrieval(query="...", retrieved_docs=[...], distances=np.array([...]))
metrics.log_response_time(response_timer, "...", 5)

# Feedback
metrics.log_user_feedback(query="...", retrieved_doc_idx=0, is_relevant=True, relevance_score=0.85)

# Export
filepath = metrics.save_metrics_to_file()
metrics.print_summary()
```

### Example 2: Retrieve Specific Statistics

```python
# Get indexing stats
index_stats = metrics.get_indexing_stats()
print(f"Indexed: {index_stats['total_reports_indexed']} reports")

# Get response time stats
response_stats = metrics.get_response_time_stats()
print(f"Avg response: {response_stats['avg_response_time_ms']:.2f}ms")

# Get retrieval stats
retrieval_stats = metrics.get_retrieval_stats()
print(f"Avg precision: {retrieval_stats.get('avg_precision', 'N/A')}")

# Get relevance stats
relevance_stats = metrics.get_relevance_stats()
print(f"User-rated precision: {relevance_stats['precision_from_feedback']:.2f}")
```

### Example 3: Export Metrics

```python
# Save to JSON
filepath = metrics.save_metrics_to_file()  # Default: metrics_YYYYMMDD_HHMMSS.json
# Or custom filename
filepath = metrics.save_metrics_to_file("my_metrics.json")

# Print summary to console
metrics.print_summary()
```

---

## Metrics Output Format

### Session Summary JSON

```json
{
  "session_start": "2026-07-14T10:30:00.123456",
  "session_duration_seconds": 120.45,
  "indexing_stats": {
    "total_reports_indexed": 200,
    "latest_indexing_duration_sec": 2.45,
    "index_size_mb": 0.45,
    "timestamp": "2026-07-14T10:30:05.123456"
  },
  "response_time_stats": {
    "total_queries": 5,
    "avg_response_time_ms": 3100.45,
    "median_response_time_ms": 3050.00,
    "min_response_time_ms": 2800.00,
    "max_response_time_ms": 3450.00,
    "p95_response_time_ms": 3400.00,
    "p99_response_time_ms": 3420.00
  },
  "retrieval_stats": {
    "total_queries": 5,
    "avg_num_retrieved": 5.0,
    "avg_retrieval_distance": 0.8234,
    "avg_precision": 0.75,
    "avg_recall": 0.68
  },
  "relevance_stats": {
    "total_feedback": 3,
    "relevant_count": 2,
    "irrelevant_count": 1,
    "precision_from_feedback": 0.67,
    "avg_relevance_score": 0.75
  }
}
```

---

## Interpreting Metrics

### Retrieval Distance
- **Lower is better** (L2 distance metric)
- Typical range: 0.5 - 2.0
- < 0.8: Very similar document
- 0.8 - 1.2: Similar document
- > 1.2: Less relevant document

### Response Time
- **Target**: < 5000ms per query
- **Acceptable**: 3000 - 5000ms
- **Good**: < 3000ms
- **Excellent**: < 2000ms

### Precision/Recall
- **Precision**: How many retrieved docs are relevant?
  - Target: > 0.75 (75% of retrieved docs are relevant)
- **Recall**: How many relevant docs were retrieved?
  - Target: > 0.70 (70% of relevant docs were retrieved)

### User Relevance Feedback
- Aggregated from 0-1 slider ratings
- Target: > 0.70 average relevance score
- Precision from binary feedback: target > 0.75

---

## Best Practices

1. **Always log user feedback** after queries for quality monitoring
2. **Check metrics regularly** to identify performance regressions
3. **Export metrics periodically** for analysis and trend tracking
4. **Use ground truth** when available to evaluate precision/recall
5. **Monitor response times** to ensure acceptable UX
6. **Track distance distributions** to optimize retrieval thresholds

---

## Troubleshooting

**No metrics appearing?**
- Ensure `metrics.log_*()` calls are placed correctly
- Check that `get_metrics_tracker()` returns the same instance
- Verify metrics are being saved before application restart

**Metrics seem wrong?**
- Check that distances are from FAISS search (not transformed)
- Verify query/document encoding is consistent
- Ensure feedback is logged with correct parameters

**Performance bottlenecks?**
- High response time? Profile LLM generation
- High retrieval distance? Check embedding model quality
- Low precision? Review query formulation

---

## Future Enhancements

1. Real-time metrics dashboard with Plotly visualizations
2. Automated alerting for metric anomalies
3. A/B testing framework for prompt optimization
4. Ground truth dataset with pre-labeled relevant documents
5. Distributed metrics collection across multiple deployments
6. Integration with monitoring platforms (Prometheus, DataDog)
