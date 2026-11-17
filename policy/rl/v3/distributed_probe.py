"""Two-step hardware/gradient contract, not a model or hyperparameter comparison."""
import argparse,os,json
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from .model import Policy,checkpoint
from .data import Sequences,atomic_save

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--data',type=Path,nargs='+',required=True);a=p.parse_args()
    rank=int(os.environ['LOCAL_RANK']);torch.set_num_threads(1);torch.cuda.set_device(rank);torch.backends.mha.set_fastpath_enabled(False)
    dist.init_process_group('nccl');torch.manual_seed(41);model=Policy().to(rank);model.log_std.requires_grad_(False);wrapped=DDP(model,device_ids=[rank])
    batch=next(iter(DataLoader(Sequences(a.data,learn=16,burn=16,seed=rank+11),batch_size=2,num_workers=0)))
    batch=[v.to(rank) for v in batch];optimizer=torch.optim.AdamW(wrapped.parameters(),lr=2e-4)
    for _ in range(2):
        loss,_=wrapped(*batch);optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    if rank==0:
        a.out.mkdir(parents=True,exist_ok=True);atomic_save(checkpoint(model,step=2,contract_only=True),a.out/'probe.pt')
        print(json.dumps(dict(passed=True,gpus=dist.get_world_size(),parameters=sum(p.numel() for p in model.parameters()),loss=float(loss),peak_memory=torch.cuda.max_memory_allocated(rank))),flush=True)
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
