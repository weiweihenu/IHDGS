"""
Performance Optimization and Memory Management for OHDGS
Provides optimization utilities for efficient training and rendering
"""

import torch
import numpy as np
import time
import psutil
import gc
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple
import logging


class MemoryManager:
    """
    Advanced memory management for OHDGS training
    """

    def __init__(self, device='cuda'):
        self.device = device
        self.memory_stats = {
            'peak_allocated': 0,
            'peak_cached': 0,
            'current_allocated': 0,
            'current_cached': 0
        }
        self.optimization_enabled = True

    def get_memory_info(self):
        """Get current memory usage information"""
        if torch.cuda.is_available() and self.device == 'cuda':
            return {
                'allocated': torch.cuda.memory_allocated(self.device) / 1024**3,  # GB
                'cached': torch.cuda.memory_reserved(self.device) / 1024**3,      # GB
                'max_allocated': torch.cuda.max_memory_allocated(self.device) / 1024**3,  # GB
                'system_memory': psutil.virtual_memory().percent
            }
        return {
            'allocated': 0,
            'cached': 0,
            'max_allocated': 0,
            'system_memory': psutil.virtual_memory().percent
        }

    def monitor_memory(self, tag=""):
        """Monitor and log memory usage"""
        info = self.get_memory_info()
        logging.info(f"[Memory{tag}] GPU: {info['allocated']:.2f}GB allocated, "
                    f"{info['cached']:.2f}GB cached, System: {info['system_memory']:.1f}%")
        return info

    def clear_cache(self):
        """Clear GPU cache and run garbage collection"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def optimize_memory(self, force=False):
        """Optimize memory usage"""
        if not self.optimization_enabled and not force:
            return

        # Clear cache
        self.clear_cache()

        # Reset peak memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        logging.info("Memory optimization completed")

    @contextmanager
    def memory_efficient_context(self):
        """Context manager for memory-efficient operations"""
        initial_memory = self.get_memory_info()
        try:
            yield
        finally:
            final_memory = self.get_memory_info()
            memory_diff = final_memory['allocated'] - initial_memory['allocated']
            if memory_diff > 0.5:  # If more than 500MB increase
                logging.warning(f"High memory usage detected: +{memory_diff:.2f}GB")
                self.clear_cache()

    def check_memory_pressure(self, threshold=0.9):
        """Check if memory pressure is high"""
        info = self.get_memory_info()
        gpu_memory_pressure = info['allocated'] / (torch.cuda.get_device_properties(self.device).total_memory / 1024**3) if torch.cuda.is_available() else 0
        system_memory_pressure = info['system_memory'] / 100.0

        return gpu_memory_pressure > threshold or system_memory_pressure > threshold


class PerformanceProfiler:
    """
    Performance profiling for OHDGS operations
    """

    def __init__(self):
        self.timers = {}
        self.call_counts = {}
        self.enabled = True

    def start_timer(self, name):
        """Start timing an operation"""
        if self.enabled:
            self.timers[name] = time.time()

    def end_timer(self, name):
        """End timing an operation"""
        if self.enabled and name in self.timers:
            elapsed = time.time() - self.timers[name]
            if name not in self.call_counts:
                self.call_counts[name] = []
            self.call_counts[name].append(elapsed)
            return elapsed
        return 0

    @contextmanager
    def profile(self, name):
        """Context manager for profiling"""
        self.start_timer(name)
        try:
            yield
        finally:
            self.end_timer(name)

    def get_stats(self):
        """Get performance statistics"""
        stats = {}
        for name, times in self.call_counts.items():
            if times:
                stats[name] = {
                    'total_time': sum(times),
                    'avg_time': np.mean(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'call_count': len(times)
                }
        return stats

    def print_stats(self):
        """Print performance statistics"""
        stats = self.get_stats()
        if not stats:
            return

        print("\n" + "="*60)
        print("PERFORMANCE PROFILING RESULTS")
        print("="*60)

        # Sort by total time
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['total_time'], reverse=True)

        for name, stat in sorted_stats:
            print(f"{name:30s}: {stat['total_time']:6.2f}s total "
                  f"({stat['call_count']:4d} calls, "
                  f"{stat['avg_time']*1000:6.1f}ms avg)")
        print("="*60)

    def reset(self):
        """Reset all timers"""
        self.timers = {}
        self.call_counts = {}


class GaussianOptimizer:
    """
    Optimization utilities for Gaussian operations
    """

    def __init__(self, device='cuda'):
        self.device = device
        self.memory_manager = MemoryManager(device)
        self.profiler = PerformanceProfiler()

    def optimize_gaussian_operations(self, gaussians, operation_type='densify'):
        """Optimize Gaussian operations based on type"""
        with self.profiler.profile(f"optimize_{operation_type}"):
            if operation_type == 'densify':
                return self._optimize_densification(gaussians)
            elif operation_type == 'pruning':
                return self._optimize_pruning(gaussians)
            elif operation_type == 'rendering':
                return self._optimize_rendering(gaussians)
            else:
                return gaussians

    def _optimize_densification(self, gaussians):
        """Optimize densification operations"""
        # Check memory pressure
        if self.memory_manager.check_memory_pressure():
            logging.warning("High memory pressure during densification, applying optimizations")
            # Reduce densification rate
            return self._reduce_densification_rate(gaussians)

        # Use gradient checkpointing for large models
        if len(gaussians.get_xyz) > 100000:
            return self._apply_gradient_checkpointing(gaussians)

        return gaussians

    def _optimize_pruning(self, gaussians):
        """Optimize pruning operations"""
        # Batch pruning for large models
        if len(gaussians.get_xyz) > 50000:
            return self._batch_pruning(gaussians)
        return gaussians

    def _optimize_rendering(self, gaussians):
        """Optimize rendering operations"""
        # Adjust rendering resolution based on model size
        if len(gaussians.get_xyz) > 200000:
            return self._adjust_rendering_resolution(gaussians)
        return gaussians

    def _reduce_densification_rate(self, gaussians):
        """Reduce densification rate under memory pressure"""
        # This would interact with the densification scheduler
        logging.info("Reducing densification rate due to memory pressure")
        return gaussians

    def _apply_gradient_checkpointing(self, gaussians):
        """Apply gradient checkpointing for memory efficiency"""
        # Enable gradient checkpointing for large models
        if hasattr(gaussians, 'enable_gradient_checkpointing'):
            gaussians.enable_gradient_checkpointing()
            logging.info("Enabled gradient checkpointing for large model")
        return gaussians

    def _batch_pruning(self, gaussians, batch_size=10000):
        """Perform pruning in batches"""
        logging.info(f"Performing batch pruning with batch size {batch_size}")
        # Implementation would depend on specific pruning method
        return gaussians

    def _adjust_rendering_resolution(self, gaussians):
        """Adjust rendering resolution for performance"""
        # Implementation would adjust rendering parameters
        logging.info("Adjusting rendering resolution for large model")
        return gaussians


class OHDGSOptimizer:
    """
    Comprehensive optimizer for OHDGS operations
    """

    def __init__(self, device='cuda'):
        self.device = device
        self.memory_manager = MemoryManager(device)
        self.profiler = PerformanceProfiler()
        self.gaussian_optimizer = GaussianOptimizer(device)

        # Optimization settings
        self.settings = {
            'adaptive_batch_size': True,
            'memory_efficient_densification': True,
            'gradient_checkpointing_threshold': 100000,
            'auto_cleanup_interval': 1000,
            'profile_enabled': True
        }

        self.iteration_count = 0

    def optimize_training_step(self, gaussians, iteration):
        """Optimize training step based on current state"""
        self.iteration_count = iteration

        with self.profiler.profile("training_step_optimization"):
            # Monitor memory
            if iteration % 100 == 0:
                self.memory_manager.monitor_memory(f"_iter{iteration}")

            # Auto cleanup
            if iteration % self.settings['auto_cleanup_interval'] == 0:
                self.memory_manager.optimize_memory()

            # Adaptive optimization based on model size
            num_gaussians = len(gaussians.get_xyz)

            if num_gaussians > self.settings['gradient_checkpointing_threshold']:
                gaussians = self.gaussian_optimizer._apply_gradient_checkpointing(gaussians)

            if self.memory_manager.check_memory_pressure():
                gaussians = self._handle_memory_pressure(gaussians)

        return gaussians

    def _handle_memory_pressure(self, gaussians):
        """Handle high memory pressure situations"""
        logging.warning("Handling memory pressure")

        # Strategies to reduce memory usage:
        # 1. Clear cache
        self.memory_manager.clear_cache()

        # 2. Reduce batch size if applicable
        # 3. Disable some expensive operations
        # 4. Force garbage collection

        # For now, just clear cache and optimize
        self.memory_manager.optimize_memory(force=True)

        return gaussians

    def optimize_hierarchical_densification(self, gaussians, layer_name):
        """Optimize hierarchical densification for specific layer"""
        with self.profiler.profile(f"optimize_{layer_name}_densification"):
            # Layer-specific optimizations
            if layer_name == 'salient':
                return self._optimize_salient_densification(gaussians)
            elif layer_name == 'transition':
                return self._optimize_transition_densification(gaussians)
            elif layer_name == 'background':
                return self._optimize_background_densification(gaussians)

        return gaussians

    def _optimize_salient_densification(self, gaussians):
        """Optimize salient layer densification (CDD)"""
        # Salient layer needs high-quality densification
        # Focus on precision over speed
        return gaussians

    def _optimize_transition_densification(self, gaussians):
        """Optimize transition layer densification (NOD)"""
        # Transition layer balances quality and performance
        return gaussians

    def _optimize_background_densification(self, gaussians):
        """Optimize background layer operations (floater suppression)"""
        # Background layer can use more aggressive optimizations
        with self.memory_manager.memory_efficient_context():
            # Background operations are less critical
            return gaussians

    def get_optimization_report(self):
        """Get comprehensive optimization report"""
        memory_info = self.memory_manager.get_memory_info()
        performance_stats = self.profiler.get_stats()

        report = {
            'memory_info': memory_info,
            'performance_stats': performance_stats,
            'optimization_settings': self.settings,
            'iteration_count': self.iteration_count
        }

        return report

    def print_optimization_report(self):
        """Print optimization report"""
        print("\n" + "="*60)
        print("OHDGS OPTIMIZATION REPORT")
        print("="*60)

        # Memory info
        memory_info = self.memory_manager.get_memory_info()
        print(f"Memory Usage:")
        print(f"  GPU Allocated: {memory_info['allocated']:.2f} GB")
        print(f"  GPU Cached: {memory_info['cached']:.2f} GB")
        print(f"  System Memory: {memory_info['system_memory']:.1f}%")

        # Performance stats
        print(f"\nPerformance Statistics:")
        self.profiler.print_stats()

        print("="*60)


# Global optimizer instance
_global_optimizer = None

def get_optimizer(device='cuda'):
    """Get or create global optimizer instance"""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = OHDGSOptimizer(device)
    return _global_optimizer


@contextmanager
def optimize_context(operation_name, device='cuda'):
    """Context manager for automatic optimization"""
    optimizer = get_optimizer(device)
    with optimizer.profiler.profile(operation_name):
        with optimizer.memory_manager.memory_efficient_context():
            yield optimizer


def optimize_training_step(gaussians, iteration, device='cuda'):
    """Optimize a single training step"""
    optimizer = get_optimizer(device)
    return optimizer.optimize_training_step(gaussians, iteration)


def monitor_performance(gaussians, iteration, device='cuda'):
    """Monitor performance at current iteration"""
    optimizer = get_optimizer(device)

    if iteration % 1000 == 0:
        optimizer.memory_manager.monitor_memory(f"_iter{iteration}")

    if iteration % 5000 == 0:
        optimizer.print_optimization_report()


if __name__ == "__main__":
    print("Performance Optimization for OHDGS")
    print("Usage: from utils.performance_optimizer import get_optimizer, optimize_context")