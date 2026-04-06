# Multi-Node DCP Checkpoint Bugs

Date: 2026-04-07
Affected: FSDP2 + activation checkpointing training, multi-node DCP saving
Severity: **Critical** -- silently saves empty/incomplete checkpoints

---

## Bug 1: `ModelWrapper.state_dict()` drops all block weights when activation checkpointing is enabled

### File

`fastvideo/training/checkpointing_utils.py` -- `ModelWrapper.state_dict()`

### Symptom

- Training loss decreases normally, everything appears fine during training.
- After converting DCP checkpoint to diffusers format via `dcp_to_diffusers`, the resulting model produces outputs identical to the untrained base model.
- Only 15 out of 825 parameters (for Wan 1.3B) differ from the base model -- all 810 block parameters are missing.

### Root Cause

When `enable_gradient_checkpointing_type: full` is set, `apply_activation_checkpointing()` wraps each transformer block with `checkpoint_wrapper`, which inserts `_checkpoint_wrapped_module` into the module hierarchy.

This causes a key naming mismatch:

| API | Example key |
|-----|------------|
| `model.named_parameters()` | `blocks.0._checkpoint_wrapped_module.linear.weight` |
| `get_model_state_dict()` | `blocks.0.linear.weight` |

The original code:

```python
def state_dict(self):
    state_dict = get_model_state_dict(self.model)
    param_requires_grad = set([
        k for k, v in dict(self.model.named_parameters()).items()
        if v.requires_grad
    ])
    # BUG: keys don't match, so ALL block params are filtered out
    state_dict = {k: v for k, v in state_dict.items()
                  if k in param_requires_grad}
    return state_dict
```

The `param_requires_grad` set contains keys with `_checkpoint_wrapped_module`, while `state_dict` keys don't have it. The intersection is only the non-block parameters (patch_embedding, head, time_embedding, etc.), so all block weights are silently dropped.

### Fix

Strip `_checkpoint_wrapped_module` from `named_parameters()` keys before comparison:

```python
def state_dict(self):
    state_dict = get_model_state_dict(self.model)
    param_requires_grad = {
        k.replace("._checkpoint_wrapped_module.", ".")
        for k, v in self.model.named_parameters() if v.requires_grad
    }
    filtered_state_dict = {k: v for k, v in state_dict.items()
                           if k in param_requires_grad}
    return filtered_state_dict
```

This is backward-compatible: if activation checkpointing is not used, there's no `_checkpoint_wrapped_module` in the keys, and `.replace()` is a no-op.

### Impact

All checkpoints saved with `enable_gradient_checkpointing_type: full` prior to this fix are **irrecoverable** -- they do not contain trained block weights. Training must be re-run after applying the fix.

### Verification

Quick check to confirm whether a DCP checkpoint is affected:

```python
import torch.distributed.checkpoint as dcp

# Load DCP state and count keys
# If only ~15 keys exist (instead of ~825 for Wan 1.3B), the checkpoint is broken
```

Or after conversion to diffusers:

```python
from safetensors.torch import load_file
base = load_file("/path/to/base/model/model.safetensors")
finetuned = load_file("/path/to/finetuned/model/model.safetensors")

diff_keys = [k for k in base if not torch.equal(base[k], finetuned[k])]
print(f"Changed keys: {len(diff_keys)} / {len(base)}")
# If only ~15: BUG. Should be ~825 for a fully trained model.
```

---

## Bug 2: `run_multinode_simple.sh` generates different `output_dir` per node

### File

`examples/train/run_multinode_simple.sh` -- line 44

### Symptom

- DCP shards from different nodes are saved in separate directories on shared storage.
- Example (4-node, 8 GPU/node):

```
/outputs/finetune_..._4x8_20260401_175347/   # Node 0, ranks 0-7
/outputs/finetune_..._4x8_20260401_175355/   # Node 2, ranks 16-23
/outputs/finetune_..._4x8_20260401_175359/   # Node 3, ranks 24-31
/outputs/finetune_..._4x8_20260401_175415/   # Node 1, ranks 8-15
```

- `dcp_to_diffusers` conversion fails or produces incomplete model because it only sees one node's shards.
- `_cleanup_old_checkpoints` only deletes from the local node's output_dir, leaving orphan directories on other nodes.

### Root Cause

```bash
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)  # Executed independently on each node
OUTPUT_DIR="/pfs_root/sc/outputs/${CONFIG_NAME}_${NNODES}x${NUM_GPUS}_${RUN_TIMESTAMP}"
```

Each node calls `date` at slightly different times, producing different timestamps and therefore different `OUTPUT_DIR` paths. DCP's `dcp.save()` writes each rank's shard to its local `OUTPUT_DIR`, scattering them across 4 directories.

### Fix

Node 0 generates the timestamp and writes it to a beacon file on shared storage. Worker nodes wait to read the same timestamp:

```bash
COORD_DIR="${COORD_DIR:-/pfs_root/sc/torch_coord}"
mkdir -p "$COORD_DIR"
_TS_BEACON="$COORD_DIR/${CONFIG_NAME}.run_ts"

if [ "$NODE_RANK" -eq 0 ]; then
    RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    echo "$RUN_TIMESTAMP" > "$_TS_BEACON"
else
    # Wait up to 120s for Node 0's beacon
    for _i in $(seq 1 120); do
        if [ -f "$_TS_BEACON" ]; then
            RUN_TIMESTAMP=$(cat "$_TS_BEACON" | tr -d '[:space:]')
            [ -n "$RUN_TIMESTAMP" ] && break
        fi
        sleep 1
    done
fi

# Cleanup after training (Node 0 only)
[ "$NODE_RANK" -eq 0 ] && rm -f "$_TS_BEACON"
```

### Note

`longvid_sft_train_161f_multinode.sh` already had a similar coordination mechanism (writing `${MY_IP} ${RUN_TIMESTAMP}` to a master file). `run_multinode_simple.sh` was missing this logic.

---

## Checklist for Future Multi-Node Training

1. Verify `output_dir` is identical across all nodes (check logs for `[INFO] Output dir:`)
2. After first checkpoint save, verify all ranks' shards exist in the same `dcp/` directory (should see `__0_0.distcp` through `__31_0.distcp` for 32 GPUs)
3. If using `enable_gradient_checkpointing_type: full`, verify checkpoint completeness by checking parameter count in the saved DCP state
4. Before converting with `dcp_to_diffusers`, confirm all shards are in one directory -- if not, merge them first:
   ```bash
   # Merge scattered shards into one directory
   for shard_dir in /outputs/finetune_*_175{355,359,415}/checkpoint-STEP/dcp/; do
       cp "$shard_dir"/*.distcp /outputs/finetune_*_175347/checkpoint-STEP/dcp/
   done
   ```
