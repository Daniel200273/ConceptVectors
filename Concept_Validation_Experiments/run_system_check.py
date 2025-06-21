#!/usr/bin/env python3
"""
System Capability Assessment for ConceptVectors Experiment
This script checks if your system can handle the concept validation experiment.
"""

import torch
import psutil
import platform
import subprocess

def get_system_info():
    """Get comprehensive system information"""
    info = {}
    
    # Basic system info
    info['platform'] = platform.platform()
    info['machine'] = platform.machine()
    info['processor'] = platform.processor()
    
    # Memory info
    memory = psutil.virtual_memory()
    info['total_ram_gb'] = round(memory.total / (1024**3), 2)
    info['available_ram_gb'] = round(memory.available / (1024**3), 2)
    info['ram_usage_percent'] = memory.percent
    
    # CPU info
    info['cpu_count'] = psutil.cpu_count()
    info['cpu_count_physical'] = psutil.cpu_count(logical=False)
    
    # GPU info
    info['gpu_available'] = torch.cuda.is_available()
    if info['gpu_available']:
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        # Get GPU memory for each device
        info['gpu_memory_gb'] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / (1024**3)
            info['gpu_memory_gb'].append(round(total_memory, 2))
    else:
        info['gpu_count'] = 0
        info['gpu_names'] = []
        info['gpu_memory_gb'] = []
    
    return info

def assess_capability(info):
    """Assess if system can handle the experiment"""
    recommendations = []
    warnings = []
    
    # Model memory requirements (approximate)
    llama_7b_memory_requirements = {
        'fp32': 28,  # GB
        'fp16': 14,  # GB
        'bfloat16': 14,  # GB
        'int8': 7,   # GB
        'int4': 4    # GB
    }
    
    print("=== SYSTEM CAPABILITY ASSESSMENT ===")
    print(f"Platform: {info['platform']}")
    print(f"Processor: {info['processor']}")
    print(f"CPU Cores: {info['cpu_count']} logical, {info['cpu_count_physical']} physical")
    print(f"Total RAM: {info['total_ram_gb']} GB")
    print(f"Available RAM: {info['available_ram_gb']} GB ({100-info['ram_usage_percent']:.1f}% free)")
    
    if info['gpu_available']:
        print(f"\n🎮 GPU Information:")
        for i, (name, memory) in enumerate(zip(info['gpu_names'], info['gpu_memory_gb'])):
            print(f"  GPU {i}: {name} ({memory} GB VRAM)")
        
        # GPU recommendations
        max_gpu_memory = max(info['gpu_memory_gb']) if info['gpu_memory_gb'] else 0
        if max_gpu_memory >= llama_7b_memory_requirements['bfloat16']:
            print(f"\n✅ GPU CAPABLE: Can run LLaMA-7B with bfloat16 precision")
            recommendations.append("Use GPU with bfloat16 precision for optimal performance")
        elif max_gpu_memory >= llama_7b_memory_requirements['int8']:
            print(f"\n⚠️  GPU LIMITED: Can run LLaMA-7B with 8-bit quantization")
            recommendations.append("Use GPU with 8-bit quantization to fit in memory")
            warnings.append("Limited GPU memory may cause slower performance")
        elif max_gpu_memory >= llama_7b_memory_requirements['int4']:
            print(f"\n⚠️  GPU VERY LIMITED: Can run LLaMA-7B with 4-bit quantization")
            recommendations.append("Use GPU with 4-bit quantization (may affect accuracy)")
            warnings.append("Very limited GPU memory may significantly impact performance")
        else:
            print(f"\n❌ GPU INSUFFICIENT: {max_gpu_memory} GB VRAM is insufficient for LLaMA-7B")
            recommendations.append("Fall back to CPU execution (very slow)")
            warnings.append("GPU memory insufficient - will be very slow on CPU")
    else:
        print(f"\n💻 No GPU detected - using CPU only")
        
    # CPU/RAM assessment
    if info['available_ram_gb'] >= llama_7b_memory_requirements['fp32']:
        print(f"\n✅ RAM CAPABLE: Can run LLaMA-7B on CPU with full precision")
        if not info['gpu_available']:
            recommendations.append("CPU execution possible but will be slow (10-100x slower than GPU)")
    elif info['available_ram_gb'] >= llama_7b_memory_requirements['bfloat16']:
        print(f"\n⚠️  RAM LIMITED: Can run LLaMA-7B on CPU with reduced precision")
        recommendations.append("Use CPU with bfloat16 or int8 quantization")
        warnings.append("Limited RAM may require quantization")
    else:
        print(f"\n❌ RAM INSUFFICIENT: {info['available_ram_gb']} GB RAM insufficient for LLaMA-7B")
        recommendations.append("Consider using a smaller model or cloud GPU")
        warnings.append("Insufficient RAM for running LLaMA-7B locally")
    
    # Experiment scope recommendations
    print(f"\n📊 EXPERIMENT RECOMMENDATIONS:")
    for rec in recommendations:
        print(f"  • {rec}")
    
    if warnings:
        print(f"\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  • {warning}")
    
    # Specific recommendations based on capability
    print(f"\n🎯 SUGGESTED CONFIGURATION:")
    max_gpu_memory = max(info['gpu_memory_gb']) if info['gpu_memory_gb'] else 0
    if info['gpu_available'] and max_gpu_memory >= 14:
        print(f"  • Model: LLaMA-7B with bfloat16 on GPU")
        print(f"  • Sample size: Start with 3 concepts, can scale to full dataset")
        print(f"  • Expected runtime: ~5-15 minutes for 3 concepts")
    elif info['gpu_available'] and max_gpu_memory >= 7:
        print(f"  • Model: LLaMA-7B with 8-bit quantization on GPU")
        print(f"  • Sample size: Start with 3 concepts, can scale gradually")
        print(f"  • Expected runtime: ~10-20 minutes for 3 concepts")
    elif info['available_ram_gb'] >= 14:
        print(f"  • Model: LLaMA-7B with bfloat16 on CPU")
        print(f"  • Sample size: Start with 1 concept only")
        print(f"  • Expected runtime: ~30-60 minutes for 1 concept")
    else:
        print(f"  • Model: Consider using a smaller model or cloud service")
        print(f"  • Alternative: Use Hugging Face Inference API")
        print(f"  • Sample size: Not recommended for local execution")
    
    return recommendations, warnings

if __name__ == "__main__":
    print("ConceptVectors System Capability Assessment")
    print("=" * 50)
    
    # Get system information
    system_info = get_system_info()
    
    # Assess capabilities
    recommendations, warnings = assess_capability(system_info)
    
    print(f"\n" + "=" * 50)
    print("Assessment complete!")
    
    if warnings:
        print(f"\n⚠️  Please address the following warnings before proceeding:")
        for warning in warnings:
            print(f"  • {warning}")
