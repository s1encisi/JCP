"""Probe: verify PPO policy inference (load PT -> infer -> apply delta -> re-predict)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

ESRL = Path(r"E:/postgraduate1/Code/ESRLCMO")
OPT = ESRL / "optimization"


def load_ppo():
    spec = importlib.util.spec_from_file_location("esrl_ppo", OPT / "ppo_lagrangian.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ppo = load_ppo()
    for cond in (1, 2, 3):
        ppo.make_cfg(cond, fast=False)
        net = ppo.ActorCritic(ppo.CFG["state_dim"], ppo.CFG["action_dim"],
                              ppo.CFG["weight_dim"], n_constraints=ppo.CFG["n_constraints"])
        net.load_state_dict(torch.load(OPT / "pt" / f"ppo_actor_condition{cond}.pt",
                                       map_location="cpu"))
        net.eval()

        if cond in (1, 2):
            state = np.array([40.0, 58.0, 15000.0, 117.0, 4.0], dtype=np.float32)
        else:
            state = np.array([40.0, 58.0, 15000.0, 117.0, 55.0, 12000.0, 118.0, 4.0],
                             dtype=np.float32)
        weight = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        s = torch.FloatTensor(state).unsqueeze(0)
        w = torch.FloatTensor(weight).unsqueeze(0)
        with torch.no_grad():
            dist, _, _ = net(s, w)
        mu = dist.mean.cpu().numpy()[0]
        delta = mu * ppo._DELTA_SC
        # Cu_in is externally fixed (feed disturbance): zero the Cu_in increment
        delta[0] = 0.0
        new_s = state.copy()
        dim = len(delta)
        new_s[:dim] = np.clip(state[:dim] + delta, ppo._S_LO[:dim], ppo._S_HI[:dim])
        print(f"Condition {cond}: state_dim={ppo.CFG['state_dim']} action_dim={ppo.CFG['action_dim']}")
        print(f"  state  = {state}")
        print(f"  mu     = {np.round(mu, 3)}")
        print(f"  delta  = {np.round(delta, 1)}")
        print(f"  new_s  = {np.round(new_s, 1)}")
        # re-predict with surrogate
        models = ppo.load_models(str(OPT / "joblib"), n_jobs=1)
        if cond in (1, 2):
            Cu, As, V = ppo.predict_all_batch(
                models, new_s[0:1], new_s[1:2], new_s[2:3], new_s[3:4], new_s[4:5])
            E, profit = ppo.compute_objectives_batch(
                (new_s[0:1], new_s[1:2], new_s[2:3], new_s[3:4], new_s[4:5]), Cu, As, V)
            costs = ppo.compute_costs_batch(Cu, As, new_s[2:3], V)
        else:
            Cu, As, Vm, VA, VB = ppo.predict_all_batch(
                models, new_s[0:1], new_s[1:2], new_s[2:3], new_s[3:4],
                new_s[4:5], new_s[5:6], new_s[6:7], new_s[7:8])
            E, profit = ppo.compute_objectives_batch(
                (new_s[0:1], new_s[1:2], new_s[2:3], new_s[3:4],
                 new_s[4:5], new_s[5:6], new_s[6:7], new_s[7:8]), Cu, As, (Vm, VA, VB))
            costs = ppo.compute_costs_batch(Cu, As, (new_s[2:3], new_s[5:6]), (VA, VB))
        print(f"  Cu_out={float(Cu[0]):.3f} As_out={float(As[0]):.3f} E={float(E[0]):.1f} "
              f"profit={float(profit[0]):.2f} feasible={np.asarray(costs).max() < 1e-3}")
        print()


if __name__ == "__main__":
    main()
