#!/usr/bin/env python3
"""
Concurrent trajectory and query generation script.
Runs trajectory generation and query generation simultaneously to optimize performance.
"""

import os
import json
import threading
import time
import argparse
import subprocess
from pathlib import Path

def monitor_trajectory_files(domains, traj_file_pattern, query_script_args, check_interval=10):
    """
    Monitor trajectory files and trigger query generation for new records.
    
    Args:
        domains: List of domains to monitor
        traj_file_pattern: Pattern for trajectory files (e.g., "simple_traj_parallel_gemini-2.5-pro_v2.json")
        query_script_args: Arguments for query generation script
        check_interval: How often to check for new records (seconds)
    """
    print("Starting query generation monitor...")
    
    # Track last processed count for each domain
    last_processed = {}
    
    while True:
        for domain in domains:
            traj_file = f"/home/ec2-user/mountS3/newToolData/simple_query/{domain}/{traj_file_pattern}"
            
            if os.path.exists(traj_file):
                try:
                    with open(traj_file, 'r') as f:
                        records = json.load(f)
                    
                    current_count = len(records)
                    domain_last = last_processed.get(domain, 0)
                    
                    if current_count > domain_last:
                        print(f"Domain {domain}: Found {current_count - domain_last} new trajectory records")
                        
                        # Run query generation for this domain
                        cmd = [
                            'python', '/home/ec2-user/mountS3/newToolData/query_gen/traj_to_query_gen.py',
                            '-model', query_script_args['model'],
                            '-traj_file', query_script_args['traj_file'],
                            '-save_dir', query_script_args['save_dir'],
                            '-chk_dir', query_script_args['chk_dir']
                        ]
                        
                        # Set environment to process only this domain
                        env = os.environ.copy()
                        env['PROCESS_DOMAIN'] = domain
                        
                        print(f"Starting query generation for domain {domain}...")
                        subprocess.run(cmd, env=env, cwd='/home/ec2-user/agent-tool')
                        
                        last_processed[domain] = current_count
                        
                except Exception as e:
                    print(f"Error monitoring domain {domain}: {e}")
        
        time.sleep(check_interval)

def run_trajectory_generation(traj_args):
    """Run trajectory generation script with given arguments."""
    print("Starting trajectory generation...")
    
    cmd = [
        'python', '/home/ec2-user/mountS3/newToolData/query_gen/simple_traj_gen_v2.py',
        '-model', traj_args['model'],
        '-save_dir', traj_args['save_dir'],
        '-chk_dir', traj_args['chk_dir'],
        '-traj_type', traj_args['traj_type'],
        '-num_query', str(traj_args['num_query']),
        '-min_tools', str(traj_args['min_tools']),
        '-max_tools', str(traj_args['max_tools']),
        '-max_retries', str(traj_args['max_retries']),
        '-retry_delay', str(traj_args['retry_delay'])
    ]
    
    if traj_args.get('enable_checking'):
        cmd.append('-enable_checking')
    if traj_args.get('check_model'):
        cmd.extend(['-check_model', traj_args['check_model']])
    if traj_args.get('num_tools'):
        cmd.extend(['-num_tools', str(traj_args['num_tools'])])
    if traj_args.get('queries_per_tool_count'):
        cmd.extend(['-queries_per_tool_count', str(traj_args['queries_per_tool_count'])])
    
    subprocess.run(cmd, cwd='/home/ec2-user/agent-tool')
    print("Trajectory generation completed!")

def main():
    parser = argparse.ArgumentParser('Concurrent trajectory and query generation')
    
    # Trajectory generation arguments
    parser.add_argument('-traj_model', type=str, default='gemini-2.5-pro', 
                       help='Model for trajectory generation',
                       choices=['qwen-8b', 'qwen-32b', 'qwen-30b-A3B', 'gemini-2.5-pro', 
                               'gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-2.0-flash-lite', 
                               'gemini-1.5-flash-8b', 'claude_v4', 'claude_v37', 'nova_pro', 'nova_lite'])
    parser.add_argument('-query_model', type=str, default='claude_v37', 
                       help='Model for query generation',
                       choices=['qwen-8b', 'qwen-32b', 'qwen-30b-A3B', 'gemini-2.5-pro', 
                               'gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-2.0-flash-lite', 
                               'gemini-1.5-flash-8b', 'claude_v4', 'claude_v37', 'nova_pro', 'nova_lite'])
    parser.add_argument('-traj_type', type=str, default='parallel', 
                       choices=['parallel', 'sequential', 'mixed'])
    parser.add_argument('-save_dir', type=str, default='/home/ec2-user/mountS3/newToolData/simple_query')
    parser.add_argument('-traj_chk_dir', type=str, default='./chk/traj_gen')
    parser.add_argument('-query_chk_dir', type=str, default='./chk/traj_to_query')
    parser.add_argument('-num_query', type=int, default=10)
    parser.add_argument('-min_tools', type=int, default=3)
    parser.add_argument('-max_tools', type=int, default=10)
    parser.add_argument('-num_tools', type=int, default=None)
    parser.add_argument('-queries_per_tool_count', type=int, default=None)
    parser.add_argument('-max_retries', type=int, default=3)
    parser.add_argument('-retry_delay', type=float, default=1.0)
    parser.add_argument('-enable_checking', action='store_true')
    parser.add_argument('-check_model', type=str, default=None)
    parser.add_argument('-monitor_interval', type=int, default=30, 
                       help='Seconds between checking for new trajectory records')
    
    args = parser.parse_args()
    
    # Load domain list
    with open('/home/ec2-user/mountS3/newToolData/selected_category.json', 'r') as f:
        select_cate = json.load(f)
    domains = list(select_cate.keys())
    
    # Create checkpoint directories
    os.makedirs(args.traj_chk_dir, exist_ok=True)
    os.makedirs(args.query_chk_dir, exist_ok=True)
    
    # Configure trajectory generation arguments
    traj_args = {
        'model': args.traj_model,
        'save_dir': args.save_dir,
        'chk_dir': args.traj_chk_dir,
        'traj_type': args.traj_type,
        'num_query': args.num_query,
        'min_tools': args.min_tools,
        'max_tools': args.max_tools,
        'max_retries': args.max_retries,
        'retry_delay': args.retry_delay,
        'enable_checking': args.enable_checking,
        'check_model': args.check_model,
        'num_tools': args.num_tools,
        'queries_per_tool_count': args.queries_per_tool_count
    }
    
    # Configure query generation arguments
    traj_file_name = f"simple_traj_{args.traj_type}_{args.traj_model}_v2.json"
    query_args = {
        'model': args.query_model,
        'traj_file': f"simple_traj_{args.traj_type}_{args.traj_model}",
        'save_dir': args.save_dir,
        'chk_dir': args.query_chk_dir
    }
    
    print("="*80)
    print("CONCURRENT TRAJECTORY AND QUERY GENERATION")
    print("="*80)
    print(f"Trajectory model: {args.traj_model}")
    print(f"Query model: {args.query_model}")
    print(f"Trajectory type: {args.traj_type}")
    print(f"Domains: {len(domains)}")
    print(f"Monitor interval: {args.monitor_interval}s")
    print("="*80)
    
    # Start query generation monitor in separate thread
    monitor_thread = threading.Thread(
        target=monitor_trajectory_files,
        args=(domains, traj_file_name, query_args, args.monitor_interval),
        daemon=True
    )
    monitor_thread.start()
    
    # Give monitor thread time to start
    time.sleep(2)
    
    # Run trajectory generation in main thread
    try:
        run_trajectory_generation(traj_args)
    except KeyboardInterrupt:
        print("\nReceived interrupt signal. Shutting down...")
    except Exception as e:
        print(f"Error in trajectory generation: {e}")
    
    print("Main trajectory generation completed. Monitor thread will continue running...")
    print("Press Ctrl+C to stop the query generation monitor.")
    
    # Keep main thread alive to let monitor continue
    try:
        while monitor_thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down completely...")

if __name__ == "__main__":
    main()