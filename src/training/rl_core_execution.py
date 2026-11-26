import copy, os, random, hashlib, json
from pathlib import Path
import numpy as np
import torch
from torch import nn

class DynamicQNet(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m=int(m)
        self.net=nn.Sequential(nn.Linear(4+6*m,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,m))
    def forward(self,x): return self.net(x)

class BatchQueue:
    """Vectorized fixed-trace environment; profile observation excludes latent demand."""
    def __init__(self, trace, speeds, workers, scales, nenv, episode_len, device, generator, mask='full'):
        self.device=device; self.speeds=torch.as_tensor(speeds,device=device,dtype=torch.float32)
        self.workers=torch.as_tensor(workers,device=device,dtype=torch.long)
        self.m=len(speeds); self.maxw=int(max(workers)); self.b=nenv; self.sc=scales; self.mask=mask
        self.gen=generator; self.n=len(trace); self.L=min(episode_len,self.n-1)
        if self.L<2: raise ValueError('trace too short')
        self.d={k:torch.tensor(trace[col].to_numpy(np.float32),device=device) for k,col in
                [('a','arrival'),('p50','est_p50'),('p99','est_p99'),('d','service_demand'),('slo','slo')]}
        self.maskw=torch.arange(self.maxw,device=device)[None,:]<self.workers[:,None]
        self.batch=torch.arange(nenv,device=device)
        self.idx=torch.zeros(nenv,dtype=torch.long,device=device); self.steps=self.idx.clone(); self.last=torch.zeros(nenv,device=device)
        self.avail=torch.full((nenv,self.m,self.maxw),float('inf'),device=device); self.reset(torch.ones(nenv,dtype=torch.bool,device=device))
    def reset(self, selected):
        count=int(selected.sum().item())
        if not count:return
        starts=torch.randint(0,max(self.n-self.L-1,1),(count,),generator=self.gen,device=self.device)
        self.idx[selected]=starts; self.steps[selected]=0; base=self.d['a'][starts]; self.last[selected]=base
        z=torch.full((count,self.m,self.maxw),float('inf'),device=self.device)
        for j in range(self.m): z[:,j,:int(self.workers[j])]=base[:,None]
        self.avail[selected]=z
    def observe(self):
        a=self.d['a'][self.idx]; p50=self.d['p50'][self.idx].clamp_min(1e-6); p99=torch.maximum(self.d['p99'][self.idx],p50)
        ia=(a-self.last).clamp_min(0)
        req=torch.stack([(torch.log1p(p50)/self.sc['p50']).clamp(0,1),(torch.log1p(p99)/self.sc['p99']).clamp(0,1),
                         ((p99/p50)/self.sc['unc']).clamp(0,1),(torch.log1p(ia)/self.sc['ia']).clamp(0,1)],1)
        residual=self.residual(a); valid=self.maskw[None,:,:]
        rz=torch.where(valid,residual,0.); ri=torch.where(valid,residual,float('inf'))
        s=torch.stack([(self.speeds/self.speeds.max())[None,:].expand(self.b,-1),
                       (self.workers/self.workers.max())[None,:].expand(self.b,-1),
                       (ri.min(2).values/self.sc['res']).clamp(0,1),
                       (rz.sum(2)/self.workers[None,:]/self.sc['res']).clamp(0,1),
                       (rz.max(2).values/(2*self.sc['res'])).clamp(0,1),
                       ((residual>0)&valid).sum(2)/self.workers[None,:]],2)
        if self.mask=='no_profile': req[:,:4]=0
        elif self.mask=='no_busy_residual': s[:,:,2:5]=0
        elif self.mask=='no_interarrival': req[:,3]=0
        elif self.mask!='full': raise ValueError('unknown ablation')
        return torch.cat((req,s.reshape(self.b,-1)),1)
    def residual(self, arrivals):
        return (self.avail-arrivals[:,None,None]).clamp_min(0)
    def step(self,action):
        a=self.d['a'][self.idx]; chosen=self.avail[self.batch,action]; worker=chosen.argmin(1)
        start=torch.maximum(a,chosen[self.batch,worker]); end=start+self.d['d'][self.idx]/self.speeds[action]
        self.avail[self.batch,action,worker]=end; latency=end-a; violation=(latency>self.d['slo'][self.idx]).float()
        self.last=a; self.idx+=1; self.steps+=1; done=(self.steps>=self.L)|(self.idx>=self.n-1)
        self.reset(done); return self.observe(),latency,violation,done
    def state_dict(self): return {k:getattr(self,k).detach().cpu() for k in ('idx','steps','last','avail')}
    def load_state_dict(self,state):
        for k,v in state.items(): setattr(self,k,v.to(self.device))

def atomic_torch_save(obj,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp'); torch.save(obj,tmp); os.replace(tmp,path)

def capture_rng(gens):
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'generators':{k:g.get_state() for k,g in gens.items()}}

def restore_rng(state,gens):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state['cuda'])
    for k,g in gens.items(): g.set_state(state['generators'][k])

def replay_init(cap,dim,device):
    return {'obs':torch.zeros((cap,dim),device=device),'next':torch.zeros((cap,dim),device=device),
            'act':torch.zeros(cap,dtype=torch.long,device=device),'lat':torch.zeros(cap,device=device),
            'vio':torch.zeros(cap,device=device),'done':torch.zeros(cap,device=device)}

def train_dynamic(train,val,scales,config, checkpoint, initial=None, resume=False, stop_after_batches=None, queue_factory=BatchQueue):
    """Same initialization and batch budget for latency, fixed and adaptive modes.

    Resumable checkpoint includes replay, both environments, counters and all RNG streams.
    Only resume when caller-provided data/config hash matches; no bitwise promise across devices.
    """
    device=torch.device(config['device']); m=len(config['speeds']); n=config['nenv']; cap=config['buffer']
    if config['method'] not in ('latency','fixed','adaptive'): raise ValueError('unknown training method')
    if not (0<n<=cap) or config['batch']<=0: raise ValueError('invalid replay configuration')
    seed=config['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    gens={k:torch.Generator(device=device).manual_seed(seed+v) for k,v in [('train',101),('dual',202),('action',303),('replay',404)]}
    env=queue_factory(train,config['speeds'],config['workers'],scales,n,config['episode_len'],device,gens['train'],config['mask'])
    dual=BatchQueue(val,config['speeds'],config['workers'],scales,config['dual_envs'],config['dual_warmup']+config['dual_measure']+1,device,gens['dual'],config['mask'])
    net=DynamicQNet(m).to(device); target=DynamicQNet(m).to(device)
    if initial is not None: net.load_state_dict(initial)
    target.load_state_dict(net.state_dict()); optimizer=torch.optim.AdamW(net.parameters(),lr=config['lr'],weight_decay=1e-5)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda',init_scale=config.get('amp_init_scale',1024.),growth_interval=config.get('amp_growth_interval',1000000000)); replay=replay_init(cap,4+6*m,device)
    ptr=size=step=updates=0; lam=float(config['lambda_init']); ema=None; history=[]; obs=env.observe()
    if resume and Path(checkpoint).exists():
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if state['identity']!=config['identity']: raise ValueError('checkpoint data/config identity mismatch')
        net.load_state_dict(state['net']); target.load_state_dict(state['target']); optimizer.load_state_dict(state['optimizer']); scaler.load_state_dict(state['scaler'])
        for k in replay: replay[k].copy_(state['replay'][k].to(device))
        env.load_state_dict(state['env']); dual.load_state_dict(state['dual']); obs=state['obs'].to(device)
        ptr,size,step,updates,lam,ema,history=(state[k] for k in ['ptr','size','step','updates','lam','ema','history'])
        restore_rng(state['rng'],gens)
    def save():
        atomic_torch_save({'identity':config['identity'],'net':net.state_dict(),'target':target.state_dict(),'optimizer':optimizer.state_dict(),
          'scaler':scaler.state_dict(),'replay':{k:v.cpu() for k,v in replay.items()},'env':env.state_dict(),'dual':dual.state_dict(),
          'obs':obs.cpu(),'ptr':ptr,'size':size,'step':step,'updates':updates,'lam':lam,'ema':ema,'history':history,
          'rng':capture_rng(gens),'scales':scales,'config':config},checkpoint)
    @torch.no_grad()
    def measure():
        dual.reset(torch.ones(dual.b,dtype=torch.bool,device=device)); o=dual.observe(); violations=[]
        for i in range(config['dual_warmup']+config['dual_measure']):
            o,_,v,_=dual.step(net(o).argmax(1))
            if i>=config['dual_warmup']: violations.append(v)
        return float(torch.cat(violations).mean())
    batches=(config['transitions']+n-1)//n; warm=(config['learning_starts']+n-1)//n
    from tqdm.auto import tqdm
    progress=tqdm(total=batches,initial=step,desc=f"{config['method']} seed{seed} {config.get('mask','full')}",unit='batch',leave=False,mininterval=2)
    while step<batches:
        eps=.02+(config['eps_start']-.02)*max(0.,1-step/max(1,int(batches*config['eps_fraction'])))
        with torch.no_grad():
            greedy=net(obs).argmax(1); rand=torch.randint(m,(n,),generator=gens['action'],device=device)
            acts=torch.where(torch.rand(n,generator=gens['action'],device=device)<eps,rand,greedy)
            nxt,lat,vio,done=env.step(acts); ix=(torch.arange(n,device=device)+ptr)%cap
            replay['obs'][ix]=obs; replay['next'][ix]=nxt; replay['act'][ix]=acts
            replay['lat'][ix]=lat; replay['vio'][ix]=vio; replay['done'][ix]=done.float(); obs=nxt
            ptr=(ptr+n)%cap; size=min(size+n,cap)
        if step>=warm:
            ix=torch.randint(size,(config['batch'],),generator=gens['replay'],device=device)
            b={k:v[ix] for k,v in replay.items()}; penalty=0. if config['method']=='latency' else (config['fixed_lambda'] if config['method']=='fixed' else lam)
            costs=b['lat']/scales['lat_ref']; costs=costs.clamp(max=10) if config['clip_reward'] else costs
            with torch.no_grad():
                na=net(b['next']).argmax(1); y=-(costs+penalty*b['vio'])+config['gamma']*(1-b['done'])*target(b['next']).gather(1,na[:,None]).squeeze(1)
            # Retry this same replay batch on AMP overflow; transitions and applied updates stay matched.
            for amp_attempt in range(16):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                    loss=nn.functional.smooth_l1_loss(net(b['obs']).gather(1,b['act'][:,None]).squeeze(1),y)
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite training loss')
                before_scale=scaler.get_scale()
                scaler.scale(loss).backward(); scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(net.parameters(),10.)
                scaler.step(optimizer); scaler.update()
                if scaler.get_scale()>=before_scale: break
            else: raise FloatingPointError('AMP overflow persists after 16 retries')
            updates+=1
            if step%config['target_update']==0: target.load_state_dict(net.state_dict())
            if config['method']=='adaptive' and step%config['dual_update']==0 and step>warm:
                v=measure(); ema=v if ema is None else config['ema']*ema+(1-config['ema'])*v
                lam=float(np.clip(lam+config['dual_lr']*(ema-config['target']),0,config['lambda_max']))
                history.append({'step':step,'violation':v,'ema':ema,'lambda':lam})
        step+=1; progress.update(1)
        if step%200==0: progress.set_postfix(updates=updates,lambda_=round(lam,3))
        if config['checkpoint_every'] and step%config['checkpoint_every']==0: save()
        if stop_after_batches is not None and step>=stop_after_batches:
            save(); progress.close(); return None, {'interrupted':True,'steps':step}
    save(); progress.close()
    actual={int(float(v['step'])) for v in optimizer.state.values()} or {0}
    if actual!={updates}: raise RuntimeError('applied optimizer step counter mismatch')
    return {k:v.detach().cpu() for k,v in net.state_dict().items()}, {'steps':step,'updates':updates,'lambda':lam,'history':history}
