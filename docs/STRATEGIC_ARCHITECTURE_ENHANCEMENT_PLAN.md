# Strategic Architecture Enhancement Plan (3-6 months)

**Context**: Advanced production-scale enhancements for LLM evaluation pipeline  
**Timeline**: 3-6 months implementation roadmap  
**Scope**: Database integration, API layer, and real-time monitoring dashboard  

---

## 🗄️ **DATABASE INTEGRATION FOR PRODUCTION-SCALE DEPLOYMENT**

### **Current State vs. Future State**

#### **Current File-Based Approach**
```python
# Current: Individual JSON files per evaluation session
category_evaluation_results/
├── evaluation_session_20250919_042241.json    # 15MB session data
├── predictions_20250919_042241_task1.json     # 8MB predictions
├── predictions_20250919_042241_task2.json     # 12MB predictions
└── ...                                        # Hundreds of files

# Challenges:
- Linear file scanning for queries
- No concurrent access control
- Manual aggregation across sessions
- Limited search capabilities
- No data relationships/joins
```

#### **Proposed Database Architecture**
```sql
-- PostgreSQL Schema Design for Production Scale

-- Core evaluation metadata
CREATE TABLE evaluation_sessions (
    session_id VARCHAR(50) PRIMARY KEY,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    category VARCHAR(50),
    total_tasks INTEGER,
    success_count INTEGER,
    slurm_job_id INTEGER,
    node_name VARCHAR(50),
    gpu_type VARCHAR(20),
    status VARCHAR(20) -- running, completed, failed
);

-- Individual evaluation tasks
CREATE TABLE evaluation_tasks (
    task_id SERIAL PRIMARY KEY,
    session_id VARCHAR(50) REFERENCES evaluation_sessions(session_id),
    model_name VARCHAR(100),
    dataset_name VARCHAR(100),
    preset VARCHAR(50),
    sample_limit INTEGER,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    success BOOLEAN,
    error_message TEXT
);

-- Model metadata and configurations
CREATE TABLE models (
    model_id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) UNIQUE,
    model_family VARCHAR(50),
    parameter_count BIGINT,
    specialization VARCHAR(50),
    backend VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Dataset metadata
CREATE TABLE datasets (
    dataset_id SERIAL PRIMARY KEY,
    dataset_name VARCHAR(100) UNIQUE,
    task_type VARCHAR(50),
    category VARCHAR(50),
    sample_count INTEGER,
    file_path VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Evaluation results with metrics
CREATE TABLE evaluation_results (
    result_id SERIAL PRIMARY KEY,
    task_id INTEGER REFERENCES evaluation_tasks(task_id),
    samples_processed INTEGER,
    evaluation_metrics JSONB,  -- Flexible metrics storage
    average_execution_time FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Predictions storage (for large datasets, consider separate blob storage)
CREATE TABLE predictions (
    prediction_id SERIAL PRIMARY KEY,
    task_id INTEGER REFERENCES evaluation_tasks(task_id),
    sample_id INTEGER,
    prompt_hash VARCHAR(64),
    prediction TEXT,
    ground_truth TEXT,
    execution_time FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Performance metrics for monitoring
CREATE TABLE performance_metrics (
    metric_id SERIAL PRIMARY KEY,
    session_id VARCHAR(50) REFERENCES evaluation_sessions(session_id),
    timestamp TIMESTAMP,
    gpu_utilization FLOAT,
    memory_usage FLOAT,
    throughput FLOAT,
    node_name VARCHAR(50)
);

-- Indexes for fast queries
CREATE INDEX idx_sessions_category ON evaluation_sessions(category);
CREATE INDEX idx_sessions_status ON evaluation_sessions(status);
CREATE INDEX idx_tasks_model ON evaluation_tasks(model_name);
CREATE INDEX idx_tasks_dataset ON evaluation_tasks(dataset_name);
CREATE INDEX idx_results_metrics ON evaluation_results USING GIN(evaluation_metrics);
CREATE INDEX idx_predictions_task ON predictions(task_id);
CREATE INDEX idx_performance_session ON performance_metrics(session_id);
```

### **Database Integration Benefits**

#### **1. Advanced Query Capabilities**
```sql
-- Compare model performance across datasets
SELECT 
    t.model_name,
    t.dataset_name,
    AVG((r.evaluation_metrics->>'accuracy')::FLOAT) as avg_accuracy,
    COUNT(*) as evaluation_count
FROM evaluation_tasks t
JOIN evaluation_results r ON t.task_id = r.task_id
WHERE t.success = true
GROUP BY t.model_name, t.dataset_name
ORDER BY avg_accuracy DESC;

-- Find best performing models by category
SELECT 
    m.specialization,
    t.model_name,
    AVG((r.evaluation_metrics->>'pass_at_1')::FLOAT) as avg_pass_rate
FROM models m
JOIN evaluation_tasks t ON m.model_name = t.model_name
JOIN evaluation_results r ON t.task_id = r.task_id
WHERE t.dataset_name IN ('humaneval', 'mbpp', 'bigcodebench')
GROUP BY m.specialization, t.model_name
HAVING COUNT(*) >= 3
ORDER BY m.specialization, avg_pass_rate DESC;

-- Time-series analysis of evaluation performance
SELECT 
    DATE_TRUNC('day', s.start_time) as evaluation_date,
    COUNT(t.task_id) as total_tasks,
    AVG(t.end_time - t.start_time) as avg_task_duration,
    SUM(CASE WHEN t.success THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as success_rate
FROM evaluation_sessions s
JOIN evaluation_tasks t ON s.session_id = t.session_id
GROUP BY DATE_TRUNC('day', s.start_time)
ORDER BY evaluation_date;
```

#### **2. Concurrent Access and Data Integrity**
```python
# Database-backed evaluation orchestrator
class DatabaseEvaluationOrchestrator:
    def __init__(self, db_connection):
        self.db = db_connection
    
    def start_evaluation_session(self, category, tasks):
        """Start new evaluation session with database tracking"""
        with self.db.transaction():
            session_id = self.generate_session_id()
            
            # Insert session record
            self.db.execute("""
                INSERT INTO evaluation_sessions 
                (session_id, start_time, category, total_tasks, status)
                VALUES (%s, NOW(), %s, %s, 'running')
            """, (session_id, category, len(tasks)))
            
            # Insert task records
            for task in tasks:
                self.db.execute("""
                    INSERT INTO evaluation_tasks
                    (session_id, model_name, dataset_name, preset, sample_limit)
                    VALUES (%s, %s, %s, %s, %s)
                """, (session_id, task['model'], task['dataset'], 
                      task['preset'], task['samples']))
            
            return session_id
    
    def update_task_completion(self, task_id, success, results=None, error=None):
        """Update task completion status atomically"""
        with self.db.transaction():
            self.db.execute("""
                UPDATE evaluation_tasks 
                SET end_time = NOW(), success = %s, error_message = %s
                WHERE task_id = %s
            """, (success, error, task_id))
            
            if success and results:
                self.db.execute("""
                    INSERT INTO evaluation_results
                    (task_id, samples_processed, evaluation_metrics, average_execution_time)
                    VALUES (%s, %s, %s, %s)
                """, (task_id, results['samples_processed'], 
                      json.dumps(results['metrics']), results['avg_time']))
```

#### **3. Advanced Analytics and Reporting**
```python
class EvaluationAnalytics:
    def __init__(self, db_connection):
        self.db = db_connection
    
    def generate_model_leaderboard(self, dataset_category=None):
        """Generate comprehensive model rankings"""
        query = """
        WITH model_stats AS (
            SELECT 
                t.model_name,
                m.specialization,
                COUNT(*) as total_evaluations,
                AVG((r.evaluation_metrics->>'accuracy')::FLOAT) as avg_accuracy,
                STDDEV((r.evaluation_metrics->>'accuracy')::FLOAT) as accuracy_stddev,
                AVG(r.average_execution_time) as avg_response_time
            FROM evaluation_tasks t
            JOIN models m ON t.model_name = m.model_name
            JOIN evaluation_results r ON t.task_id = r.task_id
            JOIN datasets d ON t.dataset_name = d.dataset_name
            WHERE t.success = true
            """ + ("AND d.category = %s" if dataset_category else "") + """
            GROUP BY t.model_name, m.specialization
            HAVING COUNT(*) >= 5
        )
        SELECT 
            model_name,
            specialization,
            total_evaluations,
            ROUND(avg_accuracy::NUMERIC, 3) as accuracy,
            ROUND(accuracy_stddev::NUMERIC, 3) as accuracy_std,
            ROUND(avg_response_time::NUMERIC, 3) as response_time_sec,
            RANK() OVER (ORDER BY avg_accuracy DESC) as accuracy_rank
        FROM model_stats
        ORDER BY avg_accuracy DESC;
        """
        
        params = [dataset_category] if dataset_category else []
        return self.db.execute(query, params).fetchall()
    
    def analyze_evaluation_trends(self, days=30):
        """Analyze evaluation trends over time"""
        return self.db.execute("""
        SELECT 
            DATE_TRUNC('day', s.start_time) as date,
            COUNT(DISTINCT s.session_id) as sessions_count,
            COUNT(t.task_id) as tasks_count,
            AVG(CASE WHEN t.success THEN 1.0 ELSE 0.0 END) as success_rate,
            AVG(EXTRACT(EPOCH FROM (t.end_time - t.start_time))/60) as avg_duration_minutes
        FROM evaluation_sessions s
        JOIN evaluation_tasks t ON s.session_id = t.session_id
        WHERE s.start_time >= NOW() - INTERVAL '%s days'
        GROUP BY DATE_TRUNC('day', s.start_time)
        ORDER BY date;
        """, (days,)).fetchall()
```

---

## 🌐 **RESTful API LAYER FOR EVALUATION SERVICES**

### **API Architecture Design**

#### **FastAPI-Based Evaluation Service**
```python
# api/main.py - Production-ready FastAPI service
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import uuid

app = FastAPI(
    title="LLM Evaluation API",
    description="Production-scale LLM model evaluation service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for web dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Pydantic models for request/response validation
class EvaluationTaskRequest(BaseModel):
    model_name: str
    dataset_name: str
    preset: str = "balanced"
    sample_limit: Optional[int] = None
    priority: str = "normal"  # low, normal, high

class EvaluationSessionRequest(BaseModel):
    category: str
    tasks: List[EvaluationTaskRequest]
    metadata: Optional[Dict[str, Any]] = {}

class EvaluationSessionResponse(BaseModel):
    session_id: str
    status: str
    created_at: str
    estimated_duration_minutes: Optional[int]
    tasks_count: int

class TaskStatusResponse(BaseModel):
    task_id: int
    session_id: str
    model_name: str
    dataset_name: str
    status: str  # pending, running, completed, failed
    progress_percent: Optional[float]
    start_time: Optional[str]
    end_time: Optional[str]
    results: Optional[Dict[str, Any]]

# Dependency injection
async def get_db():
    """Database connection dependency"""
    # Return database connection pool
    pass

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API authentication token"""
    # Implement JWT or API key verification
    return credentials.credentials
```

#### **Core API Endpoints**
```python
# Evaluation Management Endpoints
@app.post("/api/v1/evaluations", response_model=EvaluationSessionResponse)
async def create_evaluation_session(
    request: EvaluationSessionRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token),
    db = Depends(get_db)
):
    """Create a new evaluation session"""
    try:
        # Validate models and datasets exist
        await validate_evaluation_request(request, db)
        
        # Create session in database
        session_id = await create_session_record(request, db)
        
        # Estimate duration based on historical data
        estimated_duration = await estimate_evaluation_duration(request.tasks, db)
        
        # Queue evaluation tasks
        background_tasks.add_task(execute_evaluation_session, session_id, request.tasks)
        
        return EvaluationSessionResponse(
            session_id=session_id,
            status="queued",
            created_at=datetime.utcnow().isoformat(),
            estimated_duration_minutes=estimated_duration,
            tasks_count=len(request.tasks)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/v1/evaluations/{session_id}")
async def get_evaluation_session(
    session_id: str,
    token: str = Depends(verify_token),
    db = Depends(get_db)
):
    """Get evaluation session status and results"""
    session = await db.fetch_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    tasks = await db.fetch_session_tasks(session_id)
    
    return {
        "session": session,
        "tasks": tasks,
        "overall_progress": calculate_session_progress(tasks),
        "results_summary": await generate_session_summary(session_id, db)
    }

@app.get("/api/v1/evaluations/{session_id}/tasks", response_model=List[TaskStatusResponse])
async def get_session_tasks(
    session_id: str,
    status: Optional[str] = None,
    token: str = Depends(verify_token),
    db = Depends(get_db)
):
    """Get all tasks for a session with optional status filtering"""
    tasks = await db.fetch_session_tasks(session_id, status_filter=status)
    return [TaskStatusResponse(**task) for task in tasks]

@app.get("/api/v1/evaluations/{session_id}/results")
async def get_session_results(
    session_id: str,
    format: str = "json",  # json, csv, excel
    token: str = Depends(verify_token),
    db = Depends(get_db)
):
    """Get detailed results for completed evaluation session"""
    results = await db.fetch_session_results(session_id)
    
    if format == "csv":
        return generate_csv_response(results)
    elif format == "excel":
        return generate_excel_response(results)
    else:
        return results

@app.delete("/api/v1/evaluations/{session_id}")
async def cancel_evaluation_session(
    session_id: str,
    token: str = Depends(verify_token),
    db = Depends(get_db)
):
    """Cancel a running evaluation session"""
    success = await cancel_session_tasks(session_id)
    if success:
        return {"message": "Session cancelled successfully"}
    else:
        raise HTTPException(status_code=400, detail="Cannot cancel session")
```

#### **Analytics and Reporting Endpoints**
```python
# Analytics Endpoints
@app.get("/api/v1/analytics/leaderboard")
async def get_model_leaderboard(
    category: Optional[str] = None,
    dataset: Optional[str] = None,
    days: int = 30,
    token: str = Depends(verify_token),
    db = Depends(get_db)
):
    """Get model performance leaderboard"""
    analytics = EvaluationAnalytics(db)
    leaderboard = await analytics.generate_model_leaderboard(
        category=category, 
        dataset=dataset, 
        days=days
    )
    return {
        "leaderboard": leaderboard,
        "metadata": {
            "category": category,
            "dataset": dataset,
            "evaluation_period_days": days,
            "generated_at": datetime.utcnow().isoformat()
        }
    }

@app.get("/api/v1/analytics/trends")
async def get_evaluation_trends(
    days: int = 30,
    granularity: str = "daily",  # hourly, daily, weekly
    token: str = Depends(verify_token),
    db = Depends(get_db)
):
    """Get evaluation trends and statistics"""
    trends = await analyze_evaluation_trends(days, granularity, db)
    return {
        "trends": trends,
        "summary": {
            "total_evaluations": sum(t["tasks_count"] for t in trends),
            "average_success_rate": sum(t["success_rate"] for t in trends) / len(trends),
            "period_days": days
        }
    }

@app.get("/api/v1/analytics/comparison")
async def compare_models(
    model_names: List[str],
    datasets: Optional[List[str]] = None,
    metrics: List[str] = ["accuracy", "response_time"],
    token: str = Depends(verify_token),
    db = Depends(get_db)
):
    """Compare performance between multiple models"""
    comparison = await generate_model_comparison(
        model_names, datasets, metrics, db
    )
    return comparison
```

#### **Resource Management Endpoints**
```python
# Resource Management
@app.get("/api/v1/resources/models")
async def list_available_models(
    category: Optional[str] = None,
    token: str = Depends(verify_token),
    db = Depends(get_db)
):
    """List all available models with metadata"""
    models = await db.fetch_models(category_filter=category)
    return {
        "models": models,
        "count": len(models),
        "categories": list(set(m["specialization"] for m in models))
    }

@app.get("/api/v1/resources/datasets")
async def list_available_datasets(
    category: Optional[str] = None,
    token: str = Depends(verify_token),
    db = Depends(get_db)
):
    """List all available datasets with metadata"""
    datasets = await db.fetch_datasets(category_filter=category)
    return {
        "datasets": datasets,
        "count": len(datasets),
        "categories": list(set(d["category"] for d in datasets))
    }

@app.get("/api/v1/resources/cluster-status")
async def get_cluster_status(
    token: str = Depends(verify_token)
):
    """Get current cluster resource availability"""
    status = await check_slurm_cluster_status()
    return {
        "nodes": status["nodes"],
        "queue": status["queue"],
        "available_gpus": status["gpus"],
        "estimated_wait_time": status["wait_time_minutes"]
    }
```

### **API Client Libraries**

#### **Python Client**
```python
# api_client/python/llm_evaluation_client.py
import requests
from typing import List, Dict, Any, Optional
import time

class LLMEvaluationClient:
    def __init__(self, base_url: str, api_token: str):
        self.base_url = base_url.rstrip('/')
        self.headers = {
            'Authorization': f'Bearer {api_token}',
            'Content-Type': 'application/json'
        }
    
    def create_evaluation(self, category: str, tasks: List[Dict]) -> str:
        """Create new evaluation session"""
        response = requests.post(
            f"{self.base_url}/api/v1/evaluations",
            json={"category": category, "tasks": tasks},
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()["session_id"]
    
    def wait_for_completion(self, session_id: str, 
                          poll_interval: int = 30) -> Dict[str, Any]:
        """Wait for evaluation to complete and return results"""
        while True:
            status = self.get_evaluation_status(session_id)
            if status["status"] in ["completed", "failed"]:
                return status
            time.sleep(poll_interval)
    
    def get_evaluation_status(self, session_id: str) -> Dict[str, Any]:
        """Get current evaluation status"""
        response = requests.get(
            f"{self.base_url}/api/v1/evaluations/{session_id}",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def get_results(self, session_id: str, format: str = "json") -> Any:
        """Get evaluation results in specified format"""
        response = requests.get(
            f"{self.base_url}/api/v1/evaluations/{session_id}/results",
            params={"format": format},
            headers=self.headers
        )
        response.raise_for_status()
        
        if format == "json":
            return response.json()
        else:
            return response.content
    
    def get_leaderboard(self, category: Optional[str] = None) -> List[Dict]:
        """Get model performance leaderboard"""
        params = {"category": category} if category else {}
        response = requests.get(
            f"{self.base_url}/api/v1/analytics/leaderboard",
            params=params,
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()["leaderboard"]

# Usage example
client = LLMEvaluationClient("https://eval-api.example.com", "your-api-token")

# Create evaluation
session_id = client.create_evaluation("coding_specialists", [
    {"model_name": "qwen3_8b", "dataset_name": "humaneval", "preset": "balanced"},
    {"model_name": "qwen3_14b", "dataset_name": "humaneval", "preset": "balanced"}
])

# Wait for completion
results = client.wait_for_completion(session_id)
print(f"Evaluation completed with {results['overall_progress']}% success rate")

# Get detailed results
detailed_results = client.get_results(session_id)
```

---

## 📊 **REAL-TIME MONITORING DASHBOARD**

### **Dashboard Architecture**

#### **Frontend Technology Stack**
```typescript
// dashboard/src/types/evaluation.ts - TypeScript type definitions
export interface EvaluationSession {
  session_id: string;
  category: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  start_time: string;
  end_time?: string;
  total_tasks: number;
  completed_tasks: number;
  success_rate: number;
  estimated_duration_minutes?: number;
}

export interface EvaluationTask {
  task_id: number;
  session_id: string;
  model_name: string;
  dataset_name: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress_percent: number;
  start_time?: string;
  end_time?: string;
  results?: EvaluationResults;
}

export interface EvaluationResults {
  samples_processed: number;
  evaluation_metrics: Record<string, number>;
  average_execution_time: number;
  predictions_count: number;
}

export interface ClusterStatus {
  total_nodes: number;
  active_nodes: number;
  available_gpus: number;
  queue_length: number;
  estimated_wait_time: number;
}
```

#### **Real-Time Dashboard Components**
```tsx
// dashboard/src/components/EvaluationDashboard.tsx
import React, { useState, useEffect } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import { EvaluationSession, ClusterStatus } from '../types/evaluation';

const EvaluationDashboard: React.FC = () => {
  const [sessions, setSessions] = useState<EvaluationSession[]>([]);
  const [clusterStatus, setClusterStatus] = useState<ClusterStatus | null>(null);
  const [selectedSession, setSelectedSession] = useState<string | null>(null);
  
  // WebSocket connection for real-time updates
  const { lastMessage, sendMessage } = useWebSocket('ws://localhost:8000/ws');
  
  useEffect(() => {
    if (lastMessage) {
      const data = JSON.parse(lastMessage.data);
      
      switch (data.type) {
        case 'session_update':
          updateSession(data.session);
          break;
        case 'cluster_status':
          setClusterStatus(data.status);
          break;
        case 'task_progress':
          updateTaskProgress(data.task_id, data.progress);
          break;
      }
    }
  }, [lastMessage]);
  
  return (
    <div className="dashboard">
      <Header clusterStatus={clusterStatus} />
      
      <div className="dashboard-grid">
        {/* Live Sessions Overview */}
        <SessionsOverview 
          sessions={sessions}
          onSessionSelect={setSelectedSession}
        />
        
        {/* Detailed Session View */}
        {selectedSession && (
          <SessionDetails sessionId={selectedSession} />
        )}
        
        {/* Performance Metrics */}
        <PerformanceMetrics />
        
        {/* Resource Utilization */}
        <ResourceUtilization clusterStatus={clusterStatus} />
        
        {/* Recent Results */}
        <RecentResults />
      </div>
    </div>
  );
};

// Real-time session monitoring component
const SessionsOverview: React.FC<{
  sessions: EvaluationSession[];
  onSessionSelect: (sessionId: string) => void;
}> = ({ sessions, onSessionSelect }) => {
  return (
    <div className="sessions-overview">
      <h2>Active Evaluation Sessions</h2>
      
      {sessions.map(session => (
        <div 
          key={session.session_id}
          className={`session-card ${session.status}`}
          onClick={() => onSessionSelect(session.session_id)}
        >
          <div className="session-header">
            <span className="session-id">{session.session_id}</span>
            <span className={`status-badge ${session.status}`}>
              {session.status}
            </span>
          </div>
          
          <div className="session-details">
            <p>Category: {session.category}</p>
            <p>Tasks: {session.completed_tasks}/{session.total_tasks}</p>
            <p>Success Rate: {session.success_rate.toFixed(1)}%</p>
          </div>
          
          <div className="progress-bar">
            <div 
              className="progress-fill"
              style={{ 
                width: `${(session.completed_tasks / session.total_tasks) * 100}%`
              }}
            />
          </div>
          
          {session.status === 'running' && session.estimated_duration_minutes && (
            <p className="eta">
              ETA: {session.estimated_duration_minutes} minutes
            </p>
          )}
        </div>
      ))}
    </div>
  );
};

// Detailed task monitoring
const SessionDetails: React.FC<{ sessionId: string }> = ({ sessionId }) => {
  const [tasks, setTasks] = useState<EvaluationTask[]>([]);
  const [sessionResults, setSessionResults] = useState<any>(null);
  
  useEffect(() => {
    // Fetch session details and tasks
    fetchSessionDetails(sessionId);
  }, [sessionId]);
  
  return (
    <div className="session-details">
      <h3>Session: {sessionId}</h3>
      
      {/* Task Progress Table */}
      <div className="tasks-table">
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Dataset</th>
              <th>Status</th>
              <th>Progress</th>
              <th>Duration</th>
              <th>Results</th>
            </tr>
          </thead>
          <tbody>
            {tasks.map(task => (
              <tr key={task.task_id}>
                <td>{task.model_name}</td>
                <td>{task.dataset_name}</td>
                <td>
                  <span className={`status-badge ${task.status}`}>
                    {task.status}
                  </span>
                </td>
                <td>
                  <div className="progress-bar">
                    <div 
                      className="progress-fill"
                      style={{ width: `${task.progress_percent}%` }}
                    />
                  </div>
                  {task.progress_percent.toFixed(1)}%
                </td>
                <td>
                  {task.start_time && task.end_time 
                    ? calculateDuration(task.start_time, task.end_time)
                    : task.start_time 
                      ? `${calculateDuration(task.start_time, new Date().toISOString())} (running)`
                      : 'Not started'
                  }
                </td>
                <td>
                  {task.results && (
                    <ResultsPreview results={task.results} />
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      {/* Live Log Streaming */}
      <LogViewer sessionId={sessionId} />
    </div>
  );
};
```

#### **Real-Time Metrics and Visualization**
```tsx
// Performance monitoring with real-time charts
const PerformanceMetrics: React.FC = () => {
  const [metrics, setMetrics] = useState<any[]>([]);
  
  return (
    <div className="performance-metrics">
      <h3>Live Performance Metrics</h3>
      
      {/* GPU Utilization Chart */}
      <div className="metric-chart">
        <h4>GPU Utilization</h4>
        <LineChart
          data={metrics}
          xField="timestamp"
          yField="gpu_utilization"
          color="#00ff00"
          animation={{ duration: 300 }}
        />
      </div>
      
      {/* Memory Usage Chart */}
      <div className="metric-chart">
        <h4>Memory Usage</h4>
        <LineChart
          data={metrics}
          xField="timestamp"
          yField="memory_usage"
          color="#ff6b6b"
          animation={{ duration: 300 }}
        />
      </div>
      
      {/* Throughput Chart */}
      <div className="metric-chart">
        <h4>Evaluation Throughput (samples/sec)</h4>
        <LineChart
          data={metrics}
          xField="timestamp"
          yField="throughput"
          color="#4ecdc4"
          animation={{ duration: 300 }}
        />
      </div>
    </div>
  );
};

// Cluster resource monitoring
const ResourceUtilization: React.FC<{
  clusterStatus: ClusterStatus | null;
}> = ({ clusterStatus }) => {
  if (!clusterStatus) return <div>Loading cluster status...</div>;
  
  return (
    <div className="resource-utilization">
      <h3>Cluster Resources</h3>
      
      <div className="resource-grid">
        <div className="resource-card">
          <h4>Nodes</h4>
          <div className="resource-value">
            {clusterStatus.active_nodes}/{clusterStatus.total_nodes}
          </div>
          <div className="resource-label">Active/Total</div>
        </div>
        
        <div className="resource-card">
          <h4>GPUs Available</h4>
          <div className="resource-value">
            {clusterStatus.available_gpus}
          </div>
          <div className="resource-label">H100 GPUs</div>
        </div>
        
        <div className="resource-card">
          <h4>Queue Length</h4>
          <div className="resource-value">
            {clusterStatus.queue_length}
          </div>
          <div className="resource-label">Pending Jobs</div>
        </div>
        
        <div className="resource-card">
          <h4>Wait Time</h4>
          <div className="resource-value">
            {clusterStatus.estimated_wait_time}m
          </div>
          <div className="resource-label">Estimated</div>
        </div>
      </div>
    </div>
  );
};
```

### **WebSocket Integration for Real-Time Updates**

```python
# api/websocket.py - Real-time WebSocket updates
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import json
from typing import List

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove dead connections
                self.active_connections.remove(connection)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            # Handle client requests (e.g., subscribe to specific sessions)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Background task to push updates
async def broadcast_updates():
    while True:
        # Get latest cluster status
        cluster_status = await get_cluster_status()
        await manager.broadcast(json.dumps({
            "type": "cluster_status",
            "status": cluster_status
        }))
        
        # Get session updates
        active_sessions = await get_active_sessions()
        for session in active_sessions:
            await manager.broadcast(json.dumps({
                "type": "session_update",
                "session": session
            }))
        
        await asyncio.sleep(5)  # Update every 5 seconds
```

---

## 🎯 **IMPLEMENTATION ROADMAP**

### **Phase 1: Database Integration (Month 1-2)**
- Set up PostgreSQL with proper schemas and indexes
- Migrate existing JSON data to database
- Implement database-backed evaluation orchestrator
- Add concurrent access control and data integrity checks

### **Phase 2: API Layer Development (Month 2-3)**
- Build FastAPI service with authentication
- Implement core evaluation endpoints
- Add analytics and reporting endpoints
- Create Python client library
- Set up API documentation and testing

### **Phase 3: Real-Time Dashboard (Month 3-4)**
- Build React-based dashboard with TypeScript
- Implement WebSocket for real-time updates
- Add performance monitoring and visualization
- Create responsive design for mobile access

### **Phase 4: Production Deployment (Month 4-5)**
- Set up container orchestration (Docker/Kubernetes)
- Configure load balancing and auto-scaling
- Implement monitoring and alerting
- Add backup and disaster recovery

### **Phase 5: Advanced Features (Month 5-6)**
- Add machine learning for evaluation optimization
- Implement intelligent resource scheduling
- Add advanced analytics and predictive modeling
- Create multi-tenant support for organizations

---

## 🏆 **BENEFITS OF STRATEGIC ARCHITECTURE**

### **Scalability Benefits**
- **10x throughput increase**: Database queries vs. file scanning
- **Concurrent access**: Multiple users/systems can access simultaneously
- **Resource optimization**: Intelligent job scheduling and resource allocation
- **Horizontal scaling**: Can distribute across multiple clusters

### **Operational Benefits**
- **Real-time monitoring**: Live visibility into evaluation progress
- **Automated reporting**: Scheduled analytics and alerting
- **API integration**: Programmatic access for CI/CD pipelines
- **Multi-user support**: Team collaboration and role-based access

### **Research Benefits**
- **Historical analysis**: Long-term trend analysis across evaluations
- **Comparative studies**: Easy comparison between models and datasets
- **Reproducibility**: Complete provenance tracking and versioning
- **Publication ready**: Automated report generation for research papers

### **Business Benefits**
- **Cost optimization**: Better resource utilization and scheduling
- **Time savings**: Automated workflows and batch processing
- **Quality assurance**: Comprehensive validation and error detection
- **Competitive advantage**: Faster model evaluation and deployment cycles

This strategic architecture transforms our current research-grade pipeline into an enterprise-scale evaluation platform capable of supporting large organizations, multi-institutional collaborations, and commercial model development workflows.
