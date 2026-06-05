# Copyright (c) jiamingda (https://github.com/Luyitas)
"""
Legacy batch helper: split a clip list into worker shards.

Not used by the single-clip annotation pipeline. Kept for reference only.
"""
import os
import sys
import math

def main():
    if len(sys.argv) < 6:
        print("Usage: python split_data.py <input_path> <shard_id> <total_shards> <worker_count> <output_dir>")
        sys.exit(1)

    input_path = sys.argv[1]
    shard_id = int(sys.argv[2])
    total_shards = int(sys.argv[3])
    worker_count = int(sys.argv[4])
    output_dir = sys.argv[5]

    if shard_id < 1 or shard_id > total_shards:
        print(f"Error: shard_id {shard_id} out of range [1, {total_shards}]")
        sys.exit(1)

    # 1. Collect all clips
    all_clips = []
    if os.path.isdir(input_path):
        # Scan directory
        try:
            entries = sorted(os.listdir(input_path))
            # Filter only directories if needed, typically inpainting root contains clip folders
            all_clips = [e for e in entries if os.path.isdir(os.path.join(input_path, e))]
        except Exception as e:
            print(f"Error reading directory {input_path}: {e}")
            sys.exit(1)
    elif os.path.isfile(input_path):
        # Read file list
        try:
            with open(input_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    # Similar logic to bash: take basename if it looks like a path
                    if '/' in line:
                        line = os.path.basename(line.rstrip(os.sep))
                    all_clips.append(line)
        except Exception as e:
            print(f"Error reading file {input_path}: {e}")
            sys.exit(1)
    else:
        print(f"Error: input_path not found: {input_path}")
        sys.exit(1)

    total_clips = len(all_clips)
    print(f"[python] Found {total_clips} total clips")

    # 2. Select clips for this shard
    # Round-robin assignment matches the awk logic: (index % total_shards) == (shard_id - 1)
    shard_clips = []
    for i, clip in enumerate(all_clips):
        if i % total_shards == (shard_id - 1):
            shard_clips.append(clip)
    
    num_shard_clips = len(shard_clips)
    print(f"[python] Shard {shard_id}/{total_shards} has {num_shard_clips} clips")

    if num_shard_clips == 0:
        print("[python] No clips for this shard")
        # Ensure worker files exist but are empty
        os.makedirs(output_dir, exist_ok=True)
        for i in range(worker_count):
            with open(os.path.join(output_dir, f"worker_{i:02d}.txt"), 'w') as f:
                pass
        return

    # 3. Distribute to workers
    os.makedirs(output_dir, exist_ok=True)
    
    # We open all worker files at once if count is reasonable, or append.
    # 64 files is fine.
    worker_files = []
    try:
        for i in range(worker_count):
            path = os.path.join(output_dir, f"worker_{i:02d}.txt")
            worker_files.append(open(path, 'w'))
        
        for i, clip in enumerate(shard_clips):
            worker_idx = i % worker_count
            worker_files[worker_idx].write(clip + '\n')
            
    finally:
        for f in worker_files:
            f.close()

    print(f"[python] Successfully split into {worker_count} worker files in {output_dir}")

if __name__ == "__main__":
    main()
