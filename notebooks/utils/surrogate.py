"""Design space, replicate-noise GP surrogate, prediction and qLogEHVI proposal.

DesignSpace        input scaling to [0, 1], BoTorch bounds, fixed HV reference point, Pareto/HV
fit_surrogate      one fixed-noise SingleTaskGP per objective (train_Yvar = s²/n)
predict            posterior → mean, latent CI, and 95% PI for one new replicate
propose            seeded qLogEHVI maximization → next sample
acquisition_map    qLogEHVI on a grid (for the landscape figure)
gp_hyperparameters, fit_quality   model diagnostics
"""
import numpy as np
import pandas as pd
import torch
import gpytorch
from sklearn.preprocessing import MinMaxScaler
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.optim import optimize_acqf
from .config import SEED, DTYPE, X_COLS, Y_COLS, Y_SHORT, X_BOUNDS, GRID_N


class DesignSpace:
    """Input scaling to [0, 1], BoTorch bounds, and the hypervolume reference point —
    the nadir of the initial DOE plus `slack` (10 %) in the negated (maximization)
    space, kept fixed for the whole campaign so hypervolumes are comparable."""

    def __init__(self, stats0, x_bounds=X_BOUNDS, slack=0.10):
        self.x_bounds = np.asarray(x_bounds, dtype=float)
        self.scaler = MinMaxScaler().fit(self.x_bounds)
        self.bounds01 = torch.stack([torch.zeros(2, dtype=DTYPE), torch.ones(2, dtype=DTYPE)])
        Yneg0 = -torch.tensor(stats0[Y_COLS].values, dtype=DTYPE)
        self.ref_point = Yneg0.min(0).values - slack * (Yneg0.max(0).values - Yneg0.min(0).values)
        self._hv = Hypervolume(ref_point=self.ref_point)

    def to01(self, x_orig):
        return torch.tensor(self.scaler.transform(np.atleast_2d(x_orig)), dtype=DTYPE)

    def from01(self, x01):
        x01 = x01.detach().numpy() if torch.is_tensor(x01) else x01
        return self.scaler.inverse_transform(np.atleast_2d(x01))

    def ref_point_orig(self):
        return -self.ref_point.numpy()

    def pareto(self, Y_orig):
        """(non-dominated mask, hypervolume) for original-scale objective rows (all minimized)."""
        Yneg = -torch.tensor(np.asarray(Y_orig, dtype=float), dtype=DTYPE)
        pm = is_non_dominated(Yneg)
        return pm.numpy(), float(self._hv.compute(Yneg[pm]))

    def grid(self, n=GRID_N):
        """([0,1]² grid tensor, X1 mesh, X2 mesh) in original units."""
        g = torch.linspace(0, 1, n, dtype=DTYPE)
        G1, G2 = torch.meshgrid(g, g, indexing='ij')
        X01 = torch.stack([G1.ravel(), G2.ravel()], dim=1)
        lb, ub = self.x_bounds
        return X01, G1.numpy() * (ub[0] - lb[0]) + lb[0], G2.numpy() * (ub[1] - lb[1]) + lb[1]


def fit_surrogate(space, stats, Yvar):
    """One fixed-noise GP per objective on condition means (negated → BoTorch maximizes).
    Returns (ModelListGP in eval mode, negated training targets)."""
    X01 = space.to01(stats[X_COLS].values)
    Yneg = -torch.tensor(stats[Y_COLS].values, dtype=DTYPE)
    Yv = torch.tensor(Yvar, dtype=DTYPE)
    gps = [SingleTaskGP(X01, Yneg[:, k:k + 1], train_Yvar=Yv[:, k:k + 1], outcome_transform=Standardize(m=1))
           for k in range(len(Y_COLS))]
    model = ModelListGP(*gps)
    fit_gpytorch_mll(gpytorch.mlls.SumMarginalLogLikelihood(model.likelihood, model))
    model.eval()
    return model, Yneg


def predict(space, model, x_orig, s2_pooled):
    """GP posterior at an original-scale point → DataFrame with the mean, latent sd and
    95% CI, and the predictive sd / 95% PI for one new replicate (latent + pooled s̄²)."""
    with torch.no_grad():
        post = model.posterior(space.to01(x_orig))
        mu, var = -post.mean.numpy().ravel(), post.variance.numpy().ravel()
    sd, sdp = np.sqrt(var), np.sqrt(var + s2_pooled)
    return pd.DataFrame({'Objective': [Y_SHORT[c] for c in Y_COLS], 'mean': mu,
                         'sd_latent': sd, 'ci_low': mu - 1.96 * sd, 'ci_high': mu + 1.96 * sd,
                         'sd_pred': sdp, 'pi_low': mu - 1.96 * sdp, 'pi_high': mu + 1.96 * sdp})


def propose(space, model, Yneg, seed=SEED, num_restarts=20, raw_samples=256):
    """Maximize qLogEHVI for one new sample → (x_orig (1×2), acquisition value).
    Reseeded on every call, so the result depends only on the data and the model."""
    torch.manual_seed(seed)
    part = FastNondominatedPartitioning(ref_point=space.ref_point, Y=Yneg)
    acqf = qLogExpectedHypervolumeImprovement(model=model, ref_point=space.ref_point, partitioning=part)
    cand, val = optimize_acqf(acqf, bounds=space.bounds01, q=1, num_restarts=num_restarts,
                              raw_samples=raw_samples, options={'maxiter': 200, 'batch_limit': 5})
    return space.from01(cand), float(val)


def acquisition_map(space, model, Yneg, n=GRID_N):
    """qLogEHVI on an n×n grid → (X1 mesh, X2 mesh, values)."""
    X01, X1g, X2g = space.grid(n)
    with torch.no_grad():
        part = FastNondominatedPartitioning(ref_point=space.ref_point, Y=Yneg)
        acqf = qLogExpectedHypervolumeImprovement(model=model, ref_point=space.ref_point, partitioning=part)
        vals = [acqf(X01[s:s + 500].unsqueeze(1)) for s in range(0, len(X01), 500)]
    return X1g, X2g, torch.cat(vals).numpy().reshape(n, n)


def gp_hyperparameters(model):
    rows = []
    for c, m in zip(Y_COLS, model.models):
        ls = m.covar_module.lengthscale.detach().numpy().ravel()
        rows.append({'Objective': Y_SHORT[c], 'lengthscale X1 (scaled)': ls[0], 'lengthscale X2 (scaled)': ls[1],
                     'fixed noise (standardised, mean)': float(m.likelihood.noise.mean())})
    return pd.DataFrame(rows)


def fit_quality(space, model, stats):
    """In-sample check: posterior mean at the training conditions vs. the condition means."""
    with torch.no_grad():
        mu = -model.posterior(space.to01(stats[X_COLS].values)).mean.numpy()
    rows = []
    for k, c in enumerate(Y_COLS):
        y = stats[c].values
        rows.append({'Objective': Y_SHORT[c],
                     'R² (train)': 1 - ((y - mu[:, k]) ** 2).sum() / ((y - y.mean()) ** 2).sum(),
                     'RMSE (train)': np.sqrt(((y - mu[:, k]) ** 2).mean())})
    return pd.DataFrame(rows)
