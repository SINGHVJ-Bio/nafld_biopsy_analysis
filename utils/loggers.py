"""
Custom loggers for pipeline tracking and results
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

class PipelineLogger:
    """Logger for tracking pipeline execution"""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create main log file
        self.log_file = log_dir / "pipeline_execution.log"
        self.logger = self._setup_logger()
        
        # Execution tracking
        self.execution_history = []
    
    def _setup_logger(self) -> logging.Logger:
        """Setup the pipeline logger"""
        logger = logging.getLogger('PipelineLogger')
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_step_start(self, step_name: str, step_details: Dict = None):
        """Log the start of a pipeline step"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step_name,
            'status': 'started',
            'details': step_details or {}
        }
        
        self.execution_history.append(log_entry)
        self.logger.info(f"START: {step_name}")
        
        if step_details:
            self.logger.info(f"Details: {step_details}")
    
    def log_step_completion(self, step_name: str, results: Dict = None, 
                          duration: float = None):
        """Log the completion of a pipeline step"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step_name,
            'status': 'completed',
            'results': results or {},
            'duration': duration
        }
        
        self.execution_history.append(log_entry)
        
        message = f"COMPLETED: {step_name}"
        if duration is not None:
            message += f" (duration: {duration:.2f}s)"
        
        self.logger.info(message)
        
        if results:
            self.logger.info(f"Results: {results}")
    
    def log_step_error(self, step_name: str, error: Exception, 
                      context: Dict = None):
        """Log an error during pipeline execution"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step_name,
            'status': 'error',
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context or {}
        }
        
        self.execution_history.append(log_entry)
        self.logger.error(f"ERROR in {step_name}: {error}")
    
    def log_metrics(self, step_name: str, metrics: Dict):
        """Log performance metrics"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step_name,
            'type': 'metrics',
            'metrics': metrics
        }
        
        self.execution_history.append(log_entry)
        self.logger.info(f"METRICS for {step_name}: {metrics}")
    
    def save_execution_report(self):
        """Save execution report to file"""
        report_file = self.log_dir / "execution_report.json"
        
        with open(report_file, 'w') as f:
            json.dump(self.execution_history, f, indent=2, default=str)
        
        # Also save as CSV for easier analysis
        df_data = []
        for entry in self.execution_history:
            row = {
                'timestamp': entry['timestamp'],
                'step': entry['step'],
                'status': entry['status']
            }
            
            if 'duration' in entry and entry['duration'] is not None:
                row['duration'] = entry['duration']
            
            if 'metrics' in entry:
                for metric_name, metric_value in entry['metrics'].items():
                    row[f'metric_{metric_name}'] = metric_value
            
            df_data.append(row)
        
        if df_data:
            df = pd.DataFrame(df_data)
            csv_file = self.log_dir / "execution_report.csv"
            df.to_csv(csv_file, index=False)
        
        self.logger.info(f"Execution report saved to {report_file}")

class ResultsLogger:
    """Logger for tracking analysis results"""
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_data = {}
        self.current_experiment = None
    
    def start_experiment(self, experiment_name: str, parameters: Dict):
        """Start a new experiment"""
        self.current_experiment = experiment_name
        self.results_data[experiment_name] = {
            'parameters': parameters,
            'start_time': datetime.now().isoformat(),
            'results': {},
            'metrics': {}
        }
    
    def log_result(self, result_name: str, result_data: Any):
        """Log a result for current experiment"""
        if self.current_experiment is None:
            raise ValueError("No active experiment. Call start_experiment first.")
        
        self.results_data[self.current_experiment]['results'][result_name] = result_data
    
    def log_metric(self, metric_name: str, metric_value: float):
        """Log a metric for current experiment"""
        if self.current_experiment is None:
            raise ValueError("No active experiment. Call start_experiment first.")
        
        self.results_data[self.current_experiment]['metrics'][metric_name] = metric_value
    
    def complete_experiment(self, summary: Dict = None):
        """Complete the current experiment"""
        if self.current_experiment is None:
            raise ValueError("No active experiment.")
        
        self.results_data[self.current_experiment]['end_time'] = datetime.now().isoformat()
        self.results_data[self.current_experiment]['summary'] = summary or {}
        
        # Save results
        self.save_results()
        
        self.current_experiment = None
    
    def save_results(self):
        """Save all results to file"""
        # Save as JSON
        json_file = self.results_dir / "all_results.json"
        with open(json_file, 'w') as f:
            json.dump(self.results_data, f, indent=2, default=str)
        
        # Save metrics as CSV
        metrics_data = []
        for exp_name, exp_data in self.results_data.items():
            metric_row = {'experiment': exp_name}
            metric_row.update(exp_data.get('metrics', {}))
            metric_row.update(exp_data.get('parameters', {}))
            metrics_data.append(metric_row)
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            csv_file = self.results_dir / "experiment_metrics.csv"
            metrics_df.to_csv(csv_file, index=False)
    
    def load_results(self) -> Dict:
        """Load results from file"""
        json_file = self.results_dir / "all_results.json"
        if json_file.exists():
            with open(json_file, 'r') as f:
                self.results_data = json.load(f)
        return self.results_data
    
    def get_best_experiment(self, metric_name: str, maximize: bool = True) -> str:
        """Get the best experiment based on a metric"""
        best_value = -float('inf') if maximize else float('inf')
        best_experiment = None
        
        for exp_name, exp_data in self.results_data.items():
            metrics = exp_data.get('metrics', {})
            if metric_name in metrics:
                value = metrics[metric_name]
                if (maximize and value > best_value) or (not maximize and value < best_value):
                    best_value = value
                    best_experiment = exp_name
        
        return best_experiment